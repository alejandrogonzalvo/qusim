"""
DSE computation engine.

Two-stage execution model:
  Cold path – full circuit transpilation + HQA / TeleSABRE routing.
              Required when circuit structure or topology changes (~1-10 s).
  Hot path  – fidelity estimation only, reusing cached placements/swaps.
              Required when only noise parameters change (<1 ms per point).

This module owns the public :class:`DSEEngine` façade. The actual work is
distributed across focused leaf modules:

* ``memory``    — RAM ceilings, cold-MB estimate, thread-pool capping
* ``circuits``  — qiskit circuit + transpile pass
* ``topology``  — inter/intra-core graph, slot layout, distance matrix
* ``noise``     — ``_merge_noise``, derived teleportation cost
* ``results``   — ``CachedMapping``, ``SweepResult``, ``SweepProgress``
* ``config``    — alias expansion + clamp_cfg
* ``backends/`` — routing-algorithm strategies (HQA+SABRE, TeleSABRE)
* ``sweep``     — parallel pool, ``_compile_one``, ``_eval_cold_batch``
"""

# Memory module pins per-process thread pools at import time. It MUST be
# imported before numpy/qiskit/qusim so Rayon/OpenMP/BLAS pools
# initialize at 1 thread (otherwise parallel cold-path workers
# oversubscribe the CPU and the scheduler stalls).
from .memory import (  # noqa: F401 — re-export for back-compat
    _RESERVED_RAM_MB,
    _BYTES_PER_HOT_POINT,
    _RESERVED_RAM_MB_HOT,
    _EMPIRICAL_COLD_MB,
    _estimate_cold_mb,
    _max_hot_points_for_memory,
    _mem_budget_mb,
)

import hashlib
from itertools import islice
from typing import Callable, Iterable, Optional

import numpy as np

import qusim

# Public/back-compat re-exports from leaf modules so external code that
# does ``from qusim.dse.engine import X`` keeps working.
from .axes import (
    CAT_METRIC_BY_KEY,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    SWEEP_POINTS_1D,
    SWEEP_POINTS_2D,
    SWEEP_POINTS_3D,
    SWEEP_POINTS_COLD_1D,
    SWEEP_POINTS_COLD_2D,
    SWEEP_POINTS_COLD_3D,
    MAX_TOTAL_POINTS_HOT,
    MAX_TOTAL_POINTS_COLD,
    MAX_COLD_COMPILATIONS,
    MIN_POINTS_PER_AXIS,
)
from .circuits import _build_circuit, _transpile_circuit  # noqa: F401
from .topology import (  # noqa: F401
    _add_intracore_edges,
    _build_topology,
    _compute_distance_matrix,
    _grid_neighbours,
    _grid_side,
    _max_B_for_layout,
    _max_K_for_layout,
    assign_core_slots,
    core_groups_for,
    core_slot_layout,
    deduce_num_cores,
    deduce_qubits_per_core,
    g_max,
    idle_reserved_qubits,
    inter_core_edges,
    inter_core_neighbors,
    max_data_slots,
    num_comm_groups,
    total_reserved_slots,
)
from .noise import (  # noqa: F401
    _TELE_PROTOCOL_MEAS_COUNT,
    _TELE_PROTOCOL_SINGLE_GATE_COUNT,
    _TELE_PROTOCOL_TWO_GATE_COUNT,
    _derived_tele_error,
    _derived_tele_time,
    _make_gate_arrays,
    _merge_noise,
)
from .results import (
    _PER_QUBIT_GRID_KEYS,
    _RESULT_DTYPE,
    _RESULT_SCALAR_KEYS,
    _extract_per_qubit,
    _nan_result_row,
    _result_to_row,
    _row_to_dict,
    _strip_for_grid,
    CachedMapping,
    SweepProgress,
    SweepResult,
)
from .config import (
    COLD_PATH_KEYS as _CFG_COLD_PATH_KEYS,
    DEFAULT_PIN_AXIS,
    INTEGER_KEYS as _CFG_INTEGER_KEYS,
    PIN_CORES,
    PIN_QPC,
    _resolve_architecture,
    _resolve_cell_cold_cfg,
)
from .sweep import (
    _build_config_key,
    _compile_one,
    _eval_cold_batch,  # noqa: F401 — re-export for tests / shim
    _memory_capped_max_hot as _sweep_memory_capped_max_hot,
    _parallel_cold_sweep as _sweep_parallel_cold_sweep,
)


# ---------------------------------------------------------------------------
# DSE Engine
# ---------------------------------------------------------------------------

class DSEEngine:
    """Wraps qusim's mapper+estimator pipeline for Design Space Exploration.

    Cold path — build circuit + topology, route via the selected backend,
                cache structural results.
    Hot path  — call ``estimate_hardware_fidelity`` directly with cached
                data and a new noise dict.

    Sweeps  — :meth:`sweep_nd` is the canonical entry point. The 1/2/3-D
              variants are kept as legacy wrappers around it.
    """

    # Single source of truth for cold-path / integer keys lives in
    # :mod:`qusim.dse.config`. Re-exported on the class for callers that
    # introspect ``DSEEngine.COLD_PATH_KEYS``.
    COLD_PATH_KEYS = _CFG_COLD_PATH_KEYS
    INTEGER_KEYS = _CFG_INTEGER_KEYS

    # Max noise configs per Rust batch call (peak-memory bound).
    _HOT_BATCH_CHUNK = 5_000

    def __init__(self) -> None:
        self._cache: Optional[CachedMapping] = None

    # -- Cold path ------------------------------------------------------------

    def run_cold(
        self,
        circuit_type: str,
        num_logical_qubits: int,
        num_cores: int,
        qubits_per_core: int,
        topology_type: str,
        placement_policy: str,
        seed: int,
        noise: Optional[dict] = None,
        intracore_topology: str = "all_to_all",
        routing_algorithm: str = "hqa_sabre",
        communication_qubits: int = 1,
        buffer_qubits: int = 1,
        pin_axis: str = DEFAULT_PIN_AXIS,
        custom_qasm: Optional[str] = None,
    ) -> CachedMapping:
        """Build circuit + topology, route, cache. Returns a CachedMapping
        that can be reused for many hot-path fidelity evaluations.

        Logical-first parameterization: ``num_logical_qubits`` is the
        algorithm size and is held constant during a sweep. Either
        ``num_cores`` or ``qubits_per_core`` is pinned (per ``pin_axis``);
        the unpinned axis is deduced via :func:`_resolve_architecture`
        so the device fits the circuit. ``num_qubits`` (= ``num_cores ·
        qubits_per_core``) is a derived field, written into the cold
        config below.

        Routing dispatches via :mod:`qusim.dse.backends` — pass
        ``routing_algorithm="telesabre"`` to use the C-library mapper
        instead of HQA+SABRE.

        Raises ``ValueError`` when the configuration is infeasible
        (e.g. B > K, or no ``nc`` satisfies the fixpoint with qpc
        pinned). Sweep cells catch this and write a NaN row so the
        whole sweep continues.
        """
        # Custom circuit overrides logical_qubits from the QASM payload.
        if circuit_type == "custom":
            if not custom_qasm:
                raise ValueError("circuit_type='custom' requires a custom_qasm string")
            from qiskit import qasm2
            _custom_circ = qasm2.loads(custom_qasm)
            num_logical_qubits = int(_custom_circ.num_qubits)

        # Build a working config and run the deduction layer to fill in
        # the unpinned axis + num_qubits.
        cold_cfg = {
            "circuit_type": circuit_type,
            "num_logical_qubits": int(num_logical_qubits),
            "num_cores": int(num_cores),
            "qubits_per_core": int(qubits_per_core),
            "topology_type": topology_type,
            "intracore_topology": intracore_topology,
            "placement_policy": placement_policy,
            "seed": int(seed),
            "routing_algorithm": routing_algorithm,
            "communication_qubits": int(communication_qubits),
            "buffer_qubits": int(buffer_qubits),
            "pin_axis": pin_axis or DEFAULT_PIN_AXIS,
            "custom_qasm": custom_qasm,
        }
        feasibility = _resolve_architecture(cold_cfg)
        if not feasibility["feasible"]:
            raise ValueError(
                f"Infeasible architecture: {feasibility['reason']}"
            )

        # Cache key built from the resolved cfg so changing the *pinned*
        # axis vs the *derived* axis produces the same key when they
        # land on the same device.
        qasm_fp = (
            hashlib.sha256(custom_qasm.encode("utf-8")).hexdigest()[:16]
            if custom_qasm else None
        )
        config_key = (
            circuit_type,
            cold_cfg["num_qubits"],
            cold_cfg["num_cores"],
            cold_cfg["qubits_per_core"],
            topology_type,
            intracore_topology,
            placement_policy,
            int(seed),
            routing_algorithm,
            cold_cfg["communication_qubits"],
            cold_cfg["buffer_qubits"],
            cold_cfg["num_logical_qubits"],
            qasm_fp,
        )
        if self._cache is not None and self._cache.config_key == config_key:
            return self._cache

        cached = _compile_one(cold_cfg, noise, config_key)
        # Stamp the resolved architectural metrics on the cache so
        # ``run_hot`` (called directly or via the sweep grid) can ride
        # them through into every result dict.
        cached.num_qubits = cold_cfg["num_qubits"]
        cached.derived_num_cores = cold_cfg["num_cores"]
        cached.derived_qubits_per_core = cold_cfg["qubits_per_core"]
        cached.idle_reserved_qubits = cold_cfg.get("idle_reserved_qubits", 0)
        self._cache = cached
        return self._cache

    # -- Hot path -------------------------------------------------------------

    def run_hot(self, cached: CachedMapping, noise: dict) -> dict:
        """Fast fidelity estimation reusing cached placements + distance matrix."""
        merged = _merge_noise(noise)
        gate_error_arr, gate_time_arr = _make_gate_arrays(cached.gate_names, merged)
        result = qusim.estimate_fidelity_from_cache(
            gs_sparse=cached.gs_sparse,
            placements=cached.placements,
            distance_matrix=cached.distance_matrix,
            sparse_swaps=cached.sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            single_gate_error=merged["single_gate_error"],
            two_gate_error=merged["two_gate_error"],
            teleportation_error_per_hop=merged["teleportation_error_per_hop"],
            single_gate_time=merged["single_gate_time"],
            two_gate_time=merged["two_gate_time"],
            teleportation_time_per_hop=merged["teleportation_time_per_hop"],
            epr_error_per_hop=merged["epr_error_per_hop"],
            measurement_error=merged["measurement_error"],
            t1=merged["t1"],
            t2=merged["t2"],
            dynamic_decoupling=merged.get("dynamic_decoupling", False),
            readout_mitigation_factor=merged["readout_mitigation_factor"],
            classical_link_width=int(merged["classical_link_width"]),
            classical_clock_freq_hz=merged["classical_clock_freq_hz"],
            classical_routing_cycles=int(merged["classical_routing_cycles"]),
        )
        result["total_epr_pairs"] = cached.total_epr_pairs
        result["total_swaps"] = cached.total_swaps
        result["total_teleportations"] = cached.total_teleportations
        result["total_network_distance"] = cached.total_network_distance
        # Derived architectural metrics ride along on the cached mapping
        # so each hot-path call can stamp them onto the result dict
        # without re-deriving from cfg.
        for k in ("num_qubits", "derived_num_cores",
                  "derived_qubits_per_core", "idle_reserved_qubits"):
            v = getattr(cached, k, None)
            if v is not None:
                result[k] = v
        return result

    def run_hot_batch(
        self,
        cached: CachedMapping,
        noise_dicts: list[dict],
        keep_grids: bool = False,
    ) -> list[dict]:
        """Batch fidelity estimation: chunked Rust calls for many noise configs.

        Structural data is parsed once per chunk; chunks bound peak
        memory instead of allocating all results at once. With
        ``keep_grids=True`` the per-qubit grids are retained — the Rust
        batch entry point is scalar-only, so this branch falls back to a
        per-cell ``run_hot`` loop.
        """
        if keep_grids:
            return [self.run_hot(cached, n) for n in noise_dicts]

        gate_error_arr, gate_time_arr = _make_gate_arrays(
            cached.gate_names, _merge_noise(noise_dicts[0]),
        )
        all_results: list[dict] = []

        for start in range(0, len(noise_dicts), self._HOT_BATCH_CHUNK):
            chunk = noise_dicts[start : start + self._HOT_BATCH_CHUNK]
            merged_chunk = [_merge_noise(n) for n in chunk]

            chunk_results = qusim.estimate_fidelity_from_cache_batch(
                gs_sparse=cached.gs_sparse,
                placements=cached.placements,
                distance_matrix=cached.distance_matrix,
                sparse_swaps=cached.sparse_swaps,
                gate_error_arr=gate_error_arr,
                gate_time_arr=gate_time_arr,
                noise_dicts=merged_chunk,
            )
            for r in chunk_results:
                r["total_epr_pairs"] = cached.total_epr_pairs
                r["total_swaps"] = cached.total_swaps
                r["total_teleportations"] = cached.total_teleportations
                r["total_network_distance"] = cached.total_network_distance
                # Stamp resolved architectural metrics so each cell in
                # the sweep grid carries (cores, qpc, num_qubits, idle)
                # alongside the fidelity numbers.
                for k in ("num_qubits", "derived_num_cores",
                          "derived_qubits_per_core", "idle_reserved_qubits"):
                    v = getattr(cached, k, None)
                    if v is not None:
                        r[k] = v
            all_results.extend(_strip_for_grid(r) for r in chunk_results)

        return all_results

    # -- Sweep helpers (axis count math) --------------------------------------

    def _memory_capped_max_hot(self, max_hot: int | None) -> tuple[int, int]:
        """Clamp ``max_hot`` to what current RAM can hold.

        Returns ``(effective, requested)`` so callers can report when
        the cap kicked in. ``requested`` mirrors the user's configured
        value (or the registry default when ``None``); ``effective`` is
        what the sweep should actually respect.

        Calls ``_max_hot_points_for_memory`` via this module's binding
        so test monkeypatches against ``qusim.dse.engine`` apply here.
        """
        requested = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT
        return min(requested, _max_hot_points_for_memory()), requested

    def _metric_values(self, metric_key: str, low, high, n: int) -> np.ndarray:
        if metric_key in CAT_METRIC_BY_KEY:
            return np.array(low, dtype=object)
        m = METRIC_BY_KEY[metric_key]
        if m.log_scale:
            return np.logspace(low, high, n)
        vals = np.linspace(low, high, n)
        if metric_key in self.INTEGER_KEYS:
            vals = np.unique(np.round(vals).astype(int))
        return vals

    def _has_cold(self, *keys: str) -> bool:
        return any(k in self.COLD_PATH_KEYS for k in keys)

    def _sweep_points(self, ndim: int, has_cold: bool) -> int:
        if has_cold:
            return [SWEEP_POINTS_COLD_1D, SWEEP_POINTS_COLD_2D, SWEEP_POINTS_COLD_3D][ndim - 1]
        return [SWEEP_POINTS_1D, SWEEP_POINTS_2D, SWEEP_POINTS_3D][ndim - 1]

    def _points_per_axis(self, ndim: int, has_cold: bool) -> int:
        budget = MAX_TOTAL_POINTS_COLD if has_cold else MAX_TOTAL_POINTS_HOT
        if ndim <= 3:
            return self._sweep_points(ndim, has_cold)
        n = max(MIN_POINTS_PER_AXIS, int(budget ** (1.0 / ndim)))
        while n ** ndim > budget and n > MIN_POINTS_PER_AXIS:
            n -= 1
        return n

    def _compute_axis_counts(
        self, sweep_axes: list, has_cold: bool,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> list[int]:
        """Compute per-axis point counts using a split budget model.

        Categorical axes have a fixed count (len of selected values).
        Numeric cold-path axes are capped by ``max_cold``. Hot-path axes
        share the remaining hot budget.
        """
        ndim = len(sweep_axes)
        cold_budget = max_cold if max_cold is not None else MAX_COLD_COMPILATIONS
        hot_budget = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT

        cat_indices, cold_indices, hot_indices = [], [], []
        for i, ax in enumerate(sweep_axes):
            key = ax[0]
            if key in CAT_METRIC_BY_KEY:
                cat_indices.append(i)
            elif key in self.COLD_PATH_KEYS:
                cold_indices.append(i)
            else:
                hot_indices.append(i)

        counts: list[int] = [0] * ndim
        for i in cat_indices:
            counts[i] = len(sweep_axes[i][1])

        cat_product = 1
        for i in cat_indices:
            cat_product *= counts[i]

        effective_cold_budget = max(1, cold_budget // max(1, cat_product))
        if cold_indices:
            n_cold = len(cold_indices)
            per_cold = max(MIN_POINTS_PER_AXIS, int(effective_cold_budget ** (1.0 / n_cold)))
            for i in cold_indices:
                key, low, high = sweep_axes[i]
                natural = int(high) - int(low) + 1
                counts[i] = max(MIN_POINTS_PER_AXIS, min(natural, per_cold))
            cold_product = 1
            for i in cold_indices:
                cold_product *= counts[i]
            while cold_product > effective_cold_budget:
                largest = max(cold_indices, key=lambda i: counts[i])
                counts[largest] = max(MIN_POINTS_PER_AXIS, counts[largest] - 1)
                cold_product = 1
                for i in cold_indices:
                    cold_product *= counts[i]
                if all(counts[i] == MIN_POINTS_PER_AXIS for i in cold_indices):
                    break

        cold_product = cat_product
        for i in cold_indices:
            cold_product *= counts[i]

        if hot_indices:
            n_hot = len(hot_indices)
            min_total = (MIN_POINTS_PER_AXIS ** n_hot) * cold_product
            if min_total > hot_budget:
                raise RuntimeError(
                    f"Hot budget too tight for this sweep: "
                    f"{n_hot} hot axes at {MIN_POINTS_PER_AXIS} points each × "
                    f"{cold_product} cold combinations = {min_total:,} points, "
                    f"but max_hot={hot_budget:,}. "
                    f"Raise Max hot evaluations to at least {min_total:,}, "
                    f"reduce dimensions, or lower Max cold compilations."
                )
            effective_hot_budget = max(1, hot_budget // max(1, cold_product))
            per_hot = max(MIN_POINTS_PER_AXIS, int(effective_hot_budget ** (1.0 / n_hot)))
            while per_hot ** n_hot * cold_product > hot_budget and per_hot > MIN_POINTS_PER_AXIS:
                per_hot -= 1
            for i in hot_indices:
                counts[i] = per_hot
        elif not cold_indices and not cat_indices:
            n = max(MIN_POINTS_PER_AXIS, int(hot_budget ** (1.0 / ndim)))
            counts = [n] * ndim

        return counts

    # -- Per-cell evaluation --------------------------------------------------

    def _eval_point(
        self,
        cold_config: dict,
        noise: dict,
        swept: dict[str, float],
    ) -> dict:
        """Evaluate one design point, re-running cold path if needed.

        Returns the full result dict (per-qubit grids included so callers
        that need them can capture before stripping). Returns a NaN row
        when the resolved architecture is infeasible.
        """
        cfg = dict(cold_config)
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k in self.COLD_PATH_KEYS:
                cfg[k] = int(v) if k in self.INTEGER_KEYS else v
            else:
                hot_noise[k] = v
        feasibility = _resolve_architecture(cfg)
        if not feasibility["feasible"]:
            return _nan_result_row(reason=feasibility["reason"])
        # Strip keys run_cold doesn't accept (the resolver writes some
        # bookkeeping fields onto cfg).
        accepted = {
            "circuit_type", "num_logical_qubits", "num_cores", "qubits_per_core",
            "topology_type", "intracore_topology", "placement_policy", "seed",
            "routing_algorithm", "communication_qubits", "buffer_qubits",
            "pin_axis", "custom_qasm",
        }
        run_kwargs = {k: cfg[k] for k in accepted if k in cfg}
        try:
            cached = self.run_cold(**run_kwargs, noise=hot_noise)
        except ValueError as e:
            return _nan_result_row(reason=str(e))
        # Stamp derived architectural metrics onto the cache so run_hot
        # can ride them through into the result dict.
        cached.num_qubits = cfg["num_qubits"]
        cached.derived_num_cores = cfg["num_cores"]
        cached.derived_qubits_per_core = cfg["qubits_per_core"]
        cached.idle_reserved_qubits = cfg.get("idle_reserved_qubits", 0)
        return self.run_hot(cached, hot_noise)

    def _parallel_cold_sweep(
        self,
        cold_config: dict,
        noise: dict,
        indexed_points: Iterable[tuple[tuple, dict]],
        total: int,
        progress_callback: Callable[[SweepProgress], None] | None,
        max_workers: int | None,
        keep_grids: bool = False,
    ) -> dict[tuple, dict]:
        """Thin wrapper around :func:`qusim.dse.sweep._parallel_cold_sweep`."""
        return _sweep_parallel_cold_sweep(
            cold_config, noise, indexed_points, total,
            progress_callback, max_workers, keep_grids,
        )

    # -- N-D sweep (canonical entry point) ------------------------------------

    def sweep_nd(
        self,
        cached: CachedMapping | None,
        sweep_axes: list[tuple[str, float, float]],
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
        keep_per_qubit_grids: bool = False,
    ) -> SweepResult:
        """N-dimensional sweep over arbitrary axes.

        ``sweep_axes`` entries are either ``(metric_key, low, high)`` for
        numeric axes, or ``(metric_key, values_list)`` for categorical axes.

        ``keep_per_qubit_grids=True`` populates ``SweepResult.per_qubit_data``
        with per-cell per-qubit fidelity grids + placements + effective
        cold config so the topology view can colour nodes / replot the
        device for any cell without re-running the engine.  Adds memory
        cost roughly proportional to ``num_cells × num_layers × num_qubits``.
        """
        ndim = len(sweep_axes)
        keys = [ax[0] for ax in sweep_axes]
        has_cold = self._has_cold(*keys)

        # Clamp user-requested max_hot to what RAM can hold before
        # computing per-axis counts. Without this, an oversized budget
        # silently produces a grid the kernel OOM-kills mid-sweep.
        effective_max_hot, requested_max_hot = self._memory_capped_max_hot(max_hot)
        try:
            axis_counts = self._compute_axis_counts(
                sweep_axes, has_cold, max_cold, effective_max_hot,
            )
        except RuntimeError as e:
            if effective_max_hot < requested_max_hot:
                raise RuntimeError(
                    f"{e} Memory cap: hot budget reduced from requested "
                    f"{requested_max_hot:,} to {effective_max_hot:,} to fit in "
                    f"available RAM."
                ) from e
            raise

        # Per-axis value arrays.
        # Categorical axes: ax = (key, values_list) — 2-tuple.
        # Numeric axes:     ax = (key, low, high)   — 3-tuple.
        axes = []
        for i, ax in enumerate(sweep_axes):
            k = ax[0]
            if k in CAT_METRIC_BY_KEY:
                axes.append(self._metric_values(k, ax[1], None, axis_counts[i]))
            else:
                axes.append(self._metric_values(k, ax[1], ax[2], axis_counts[i]))
        shape = tuple(len(ax) for ax in axes)
        total = int(np.prod(shape))

        grid = np.empty(shape, dtype=_RESULT_DTYPE)

        def _make_swept(idx):
            return {keys[d]: axes[d][idx[d]] for d in range(ndim)}

        def _progress_params(swept):
            out = {}
            for k, v in swept.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = 0.0
            return out

        # Per-cell per-qubit grids captured during the sweep so the
        # topology view does an O(1) lookup instead of paying the cold
        # compile a second time per slider move.
        per_qubit_cells: dict[tuple, dict] = {}

        def _capture(idx: tuple, full_res: dict, cell_swept: dict) -> None:
            if not keep_per_qubit_grids:
                return
            cell_cfg = (
                _resolve_cell_cold_cfg(cold_config, cell_swept)
                if cold_config is not None else None
            )
            per_qubit_cells[tuple(idx)] = _extract_per_qubit(
                full_res, cached=None, cold_cfg=cell_cfg,
            )

        if parallel and has_cold and cold_config is not None:
            def _indexed_iter():
                for idx in np.ndindex(shape):
                    yield idx, _make_swept(idx)

            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, _indexed_iter(), total,
                progress_callback, max_workers,
                keep_grids=keep_per_qubit_grids,
            )
            for idx in np.ndindex(shape):
                full_res = rmap[idx]
                grid[idx] = _result_to_row(full_res)
                _capture(idx, full_res, _make_swept(idx))
        elif has_cold and cold_config is not None:
            count = 0
            for idx in np.ndindex(shape):
                swept = _make_swept(idx)
                full_res = self._eval_point(cold_config, fixed_noise, swept)
                grid[idx] = _result_to_row(full_res)
                _capture(idx, full_res, swept)
                count += 1
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        completed=count,
                        total=total,
                        current_params=_progress_params(swept),
                    ))
        else:
            # Pure hot-path: stream indices in chunks.
            index_iter = iter(np.ndindex(shape))
            completed = 0
            while True:
                chunk_indices = list(islice(index_iter, self._HOT_BATCH_CHUNK))
                if not chunk_indices:
                    break
                noise_dicts = []
                for idx in chunk_indices:
                    noise = dict(fixed_noise)
                    for d in range(ndim):
                        noise[keys[d]] = axes[d][idx[d]]
                    noise_dicts.append(noise)

                chunk_results = self.run_hot_batch(
                    cached, noise_dicts, keep_grids=keep_per_qubit_grids,
                )
                for i, idx in enumerate(chunk_indices):
                    full_res = chunk_results[i]
                    grid[idx] = _result_to_row(full_res)
                    _capture(idx, full_res, _make_swept(idx))
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=completed,
                            total=total,
                            current_params=_progress_params(_make_swept(idx)),
                        ))
                del noise_dicts, chunk_results

        # Always snapshot the engine inputs so the topology view's
        # overlay can rebuild any cell on demand. ``cells`` (the per-
        # cell per-qubit grids) is only attached when
        # ``keep_per_qubit_grids=True``; without it, scrubbing costs a
        # cold compile per cell, but the slider scaffolding still works
        # — including after a session load where the heavy ``cells``
        # dict is stripped before serialisation.
        per_qubit_meta: dict | None = {
            "cold_config": dict(cold_config) if cold_config else None,
            "fixed_noise": dict(fixed_noise) if fixed_noise else None,
            "axis_keys": list(keys),
            "axis_values": [ax.tolist() for ax in axes],
            "shape": list(shape),
        }
        if keep_per_qubit_grids:
            per_qubit_meta["cells"] = per_qubit_cells

        return SweepResult(
            metric_keys=keys, axes=axes, grid=grid,
            per_qubit_data=per_qubit_meta,
        )

    # -- Legacy 1/2/3-D wrappers ---------------------------------------------

    def sweep_1d(
        self,
        cached: CachedMapping | None,
        metric_key: str,
        low: float,
        high: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, list]:
        """1-D sweep — wrapper around :meth:`sweep_nd` returning the
        legacy ``(xs, results_list)`` tuple."""
        sr = self.sweep_nd(
            cached=cached,
            sweep_axes=[(metric_key, low, high)],
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            progress_callback=progress_callback,
            parallel=parallel,
            max_workers=max_workers,
            max_cold=max_cold,
            max_hot=max_hot,
        )
        sd = sr.to_sweep_data()
        return np.asarray(sd["xs"]), sd["grid"]

    def sweep_2d(
        self,
        cached: CachedMapping | None,
        metric_key1: str, low1: float, high1: float,
        metric_key2: str, low2: float, high2: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """2-D sweep — wrapper around :meth:`sweep_nd`."""
        sr = self.sweep_nd(
            cached=cached,
            sweep_axes=[(metric_key1, low1, high1), (metric_key2, low2, high2)],
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            progress_callback=progress_callback,
            parallel=parallel,
            max_workers=max_workers,
            max_cold=max_cold,
            max_hot=max_hot,
        )
        sd = sr.to_sweep_data()
        return np.asarray(sd["xs"]), np.asarray(sd["ys"]), sd["grid"]

    def sweep_3d(
        self,
        cached: CachedMapping | None,
        metric_key1: str, low1: float, high1: float,
        metric_key2: str, low2: float, high2: float,
        metric_key3: str, low3: float, high3: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """3-D sweep — wrapper around :meth:`sweep_nd`."""
        sr = self.sweep_nd(
            cached=cached,
            sweep_axes=[
                (metric_key1, low1, high1),
                (metric_key2, low2, high2),
                (metric_key3, low3, high3),
            ],
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            progress_callback=progress_callback,
            parallel=parallel,
            max_workers=max_workers,
            max_cold=max_cold,
            max_hot=max_hot,
        )
        sd = sr.to_sweep_data()
        return (
            np.asarray(sd["xs"]),
            np.asarray(sd["ys"]),
            np.asarray(sd["zs"]),
            sd["grid"],
        )
