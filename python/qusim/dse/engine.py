"""
DSE computation engine.

Two-stage execution model:
  Cold path – full circuit transpilation + HQA mapping + SABRE routing.
              Required when circuit structure or topology changes (~1-10s).
  Hot path  – fidelity estimation only, reusing cached placements/swaps.
              Required when only noise parameters change (<1ms per point).
"""

# Memory module pins per-process thread pools at import time. It MUST be
# imported before numpy / qiskit / qusim so Rayon/OpenMP/BLAS pools
# initialize at 1 thread (otherwise parallel cold-path workers
# oversubscribe the CPU and the scheduler stalls).
from .memory import (
    _RESERVED_RAM_MB,
    _BYTES_PER_HOT_POINT,
    _RESERVED_RAM_MB_HOT,
    _EMPIRICAL_COLD_MB,
    _estimate_cold_mb,
    _max_hot_points_for_memory,
    _mem_budget_mb,
)

import os
import sys
import time
import multiprocessing
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import qiskit
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

import qusim
from qusim.hqa.placement import InitialPlacement

_MP_CONTEXT = multiprocessing.get_context("forkserver")

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


from .circuits import _build_circuit, _transpile_circuit
from .topology import (
    _add_intracore_edges,
    _build_topology,
    _compute_distance_matrix,
    _grid_neighbours,
    _grid_side,
    _max_B_for_layout,
    _max_K_for_layout,
    assign_core_slots,
    clamp_b_for_topology,
    clamp_k_for_topology,
    core_groups_for,
    core_slot_layout,
    inter_core_edges,
    inter_core_neighbors,
    max_data_slots,
    num_comm_groups,
    total_reserved_slots,
)
from .noise import (
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
    _result_to_row,
    _row_to_dict,
    _strip_for_grid,
    CachedMapping,
    SweepProgress,
    SweepResult,
)
from .config import (
    COLD_PATH_KEYS as _COLD_PATH_KEYS,
    INTEGER_KEYS as _INTEGER_KEYS,
    _clamp_cfg_comm_and_logical,
    _expand_qubits_alias,
    _resolve_cell_cold_cfg,
)



# ---------------------------------------------------------------------------
# DSE Engine
# ---------------------------------------------------------------------------

class DSEEngine:
    """
    Wraps qusim's map_circuit for Design Space Exploration.

    Cold path: build circuit + topology, run HQA+SABRE, cache structural results.
    Hot path:  call estimate_hardware_fidelity directly with cached data + new noise.
    """

    def __init__(self) -> None:
        self._cache: Optional[CachedMapping] = None

    # -- Cold path -----------------------------------------------------------

    def run_cold(
        self,
        circuit_type: str,
        num_qubits: int,
        num_cores: int,
        topology_type: str,
        placement_policy: str,
        seed: int,
        noise: Optional[dict] = None,
        intracore_topology: str = "all_to_all",
        routing_algorithm: str = "hqa_sabre",
        communication_qubits: int = 1,
        buffer_qubits: int = 1,
        num_logical_qubits: Optional[int] = None,
        custom_qasm: Optional[str] = None,
    ) -> CachedMapping:
        """
        Full circuit transpilation + mapping. Returns a CachedMapping that
        can be reused for many hot-path fidelity evaluations.

        ``num_qubits`` is the physical device size (drives topology /
        coupling map).  ``num_logical_qubits`` is the algorithm size used
        to build the circuit; defaults to ``num_qubits`` when omitted, and
        is clamped to ``[2, num_qubits]``.

        ``communication_qubits`` (K) carves out per-group EPR endpoints
        on each core; ``buffer_qubits`` (B) reserves additional buffer
        slots adjacent to each comm group (B ≤ K by rule).  Each core
        with ``G`` neighbours therefore reserves ``G·(K+B)`` total slots.
        """
        if circuit_type == "custom":
            if not custom_qasm:
                raise ValueError("circuit_type='custom' requires a custom_qasm string")
            from qiskit import qasm2
            _custom_circ = qasm2.loads(custom_qasm)
            num_logical_qubits = int(_custom_circ.num_qubits)
            if num_logical_qubits > int(num_qubits):
                raise ValueError(
                    f"Custom circuit uses {num_logical_qubits} logical qubits, "
                    f"which exceeds the device's {int(num_qubits)} physical qubits."
                )
        elif num_logical_qubits is None:
            num_logical_qubits = num_qubits
        num_logical_qubits = max(2, min(int(num_logical_qubits), int(num_qubits)))
        import hashlib as _hashlib
        qasm_fingerprint = (
            _hashlib.sha256(custom_qasm.encode("utf-8")).hexdigest()[:16]
            if custom_qasm else None
        )
        config_key = (
            circuit_type, num_qubits, num_cores, topology_type, intracore_topology,
            placement_policy, seed, routing_algorithm,
            int(communication_qubits or 1), int(buffer_qubits or 1),
            int(num_logical_qubits), qasm_fingerprint,
        )
        if self._cache is not None and self._cache.config_key == config_key:
            return self._cache

        if routing_algorithm == "telesabre":
            return self._run_cold_telesabre(
                circuit_type, num_qubits, num_cores, topology_type,
                intracore_topology, seed, noise, config_key,
                num_logical_qubits=num_logical_qubits,
                communication_qubits=communication_qubits,
                buffer_qubits=buffer_qubits,
                custom_qasm=custom_qasm,
            )

        t0 = time.time()

        circ = _build_circuit(circuit_type, num_logical_qubits, seed, qasm_str=custom_qasm)
        transp = _transpile_circuit(circ, seed)

        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type,
            intracore_topology=intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )

        actual_qubits = transp.num_qubits

        placement = (
            InitialPlacement.SPECTRAL_CLUSTERING
            if placement_policy == "spectral"
            else InitialPlacement.RANDOM
        )

        merged = _merge_noise(noise or {})

        result = qusim.map_circuit(
            circuit=transp,
            full_coupling_map=full_coupling_map,
            core_mapping=core_mapping,
            seed=seed,
            initial_placement=placement,
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
            dynamic_decoupling=False,
            readout_mitigation_factor=merged["readout_mitigation_factor"],
            classical_link_width=int(merged["classical_link_width"]),
            classical_clock_freq_hz=merged["classical_clock_freq_hz"],
            classical_routing_cycles=int(merged["classical_routing_cycles"]),
        )

        # Reconstruct internal gs_sparse + gate arrays as map_circuit does
        from qusim import _qiskit_circ_to_sparse_list

        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, merged)

        # Rebuild distance matrix (mirrors map_circuit logic)
        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(full_coupling_map, core_mapping, num_cores_actual)

        # SABRE-injected swaps are emitted by ``map_circuit`` and now exposed
        # on ``QusimResult.sparse_swaps``.  These represent intra-core SWAP
        # gates and MUST be passed to the hot-path noise estimator —
        # ``extract_inter_core_communications`` only sees inter-core changes
        # in the placements diff and therefore never charges intra-core
        # SWAPs on its own.  Without this, HQA's swap cost is silently
        # dropped while TeleSABRE's is correctly counted, making the two
        # algorithms structurally incomparable on the noise sweep.
        sparse_swaps = (
            result.sparse_swaps
            if getattr(result, "sparse_swaps", None) is not None
            else np.zeros((0, 3), dtype=np.int32)
        )

        cold_time = time.time() - t0

        self._cache = CachedMapping(
            gs_sparse=gs_sparse,
            placements=result.placements,
            distance_matrix=dist_mat,
            sparse_swaps=sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            gate_names=gate_names,
            total_epr_pairs=result.total_epr_pairs,
            total_swaps=result.total_swaps,
            total_teleportations=result.total_teleportations,
            total_network_distance=result.total_network_distance,
            config_key=config_key,
            cold_time_s=cold_time,
        )
        return self._cache

    # -- TeleSABRE cold path -------------------------------------------------

    def _run_cold_telesabre(
        self,
        circuit_type: str,
        num_qubits: int,
        num_cores: int,
        topology_type: str,
        intracore_topology: str,
        seed: int,
        noise: Optional[dict],
        config_key: tuple,
        num_logical_qubits: Optional[int] = None,
        communication_qubits: int = 1,
        buffer_qubits: int = 1,
        custom_qasm: Optional[str] = None,
    ) -> CachedMapping:
        """Cold path using TeleSABRE routing. Writes QASM + device JSON to
        temp files, calls telesabre_map_and_estimate, and caches the resulting
        placements/sparse_swaps for hot-path re-evaluation."""
        import json
        import tempfile
        import os
        from qiskit import qasm2
        from qusim.rust_core import telesabre_map_and_estimate

        t0 = time.time()
        merged = _merge_noise(noise or {})

        if circuit_type == "custom":
            if not custom_qasm:
                raise ValueError("circuit_type='custom' requires a custom_qasm string")
            from qiskit import qasm2
            _custom_circ = qasm2.loads(custom_qasm)
            num_logical_qubits = int(_custom_circ.num_qubits)
        elif num_logical_qubits is None:
            num_logical_qubits = num_qubits
        num_logical_qubits = max(2, min(int(num_logical_qubits), int(num_qubits)))

        circ = _build_circuit(circuit_type, num_logical_qubits, seed, qasm_str=custom_qasm)
        transp = _transpile_circuit(circ, seed)

        # Write QASM to temp file
        qasm_str = qasm2.dumps(transp)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".qasm", delete=False
        ) as f:
            f.write(qasm_str)
            qasm_path = f.name

        # Build and write TeleSABRE device JSON
        device_json = _build_telesabre_device_json(
            num_qubits, num_cores, topology_type, intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(device_json, f)
            device_path = f.name

        # Use the bundled default config
        config_path = os.path.join(
            os.path.dirname(__file__), "..",
            "tests", "fixtures", "telesabre", "configs", "default.json",
        )

        try:
            raw = telesabre_map_and_estimate(
                circuit_path=qasm_path,
                device_path=device_path,
                config_path=os.path.abspath(config_path),
                single_gate_error=merged["single_gate_error"],
                two_gate_error=merged["two_gate_error"],
                teleportation_error_per_hop=merged["teleportation_error_per_hop"],
                single_gate_time=merged["single_gate_time"],
                two_gate_time=merged["two_gate_time"],
                teleportation_time_per_hop=merged["teleportation_time_per_hop"],
                t1=merged["t1"],
                t2=merged["t2"],
                dynamic_decoupling=bool(merged.get("dynamic_decoupling", False)),
                readout_mitigation_factor=merged["readout_mitigation_factor"],
                classical_link_width=int(merged["classical_link_width"]),
                classical_clock_freq_hz=merged["classical_clock_freq_hz"],
                classical_routing_cycles=int(merged["classical_routing_cycles"]),
            )
        finally:
            os.unlink(qasm_path)
            os.unlink(device_path)

        # Use the Qiskit circuit's gs_sparse so the hot path has the correct
        # num_layers (circuit DAG depth) and gate structure for algorithmic /
        # coherence fidelity.  TeleSABRE's placements/swaps are remapped from
        # their internal gate-step space to the DAG-layer space so both grids
        # have a consistent row count.
        from qusim import _qiskit_circ_to_sparse_list
        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, merged)

        dag_layers = (
            int(gs_sparse[:, 0].max()) + 1 if len(gs_sparse) > 0 else 1
        )

        placements = _remap_to_dag_layers(raw["placements"], dag_layers)
        sparse_swaps = _remap_swaps_to_dag_layers(raw["sparse_swaps"], dag_layers)

        # Real inter-core distances so routing fidelity is non-trivial
        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type, intracore_topology=intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(full_coupling_map, core_mapping, num_cores_actual)

        cold_time = time.time() - t0
        self._cache = CachedMapping(
            gs_sparse=gs_sparse,
            placements=placements,
            distance_matrix=dist_mat,
            sparse_swaps=sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            gate_names=gate_names,
            total_epr_pairs=raw["total_epr_pairs"],
            total_swaps=raw["total_swaps"],
            total_teleportations=raw["total_teleportations"],
            total_network_distance=0,
            config_key=config_key,
            cold_time_s=cold_time,
        )
        return self._cache

    # -- Hot path ------------------------------------------------------------

    def run_hot(self, cached: CachedMapping, noise: dict) -> dict:
        """
        Fast fidelity estimation reusing cached placements and distance matrix.
        Returns a dict with overall_fidelity, algorithmic_fidelity, etc.
        """
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
        return result

    # Max noise configs per Rust batch call to bound peak memory
    _HOT_BATCH_CHUNK = 5_000

    def run_hot_batch(
        self,
        cached: CachedMapping,
        noise_dicts: list[dict],
        keep_grids: bool = False,
    ) -> list[dict]:
        """
        Batch fidelity estimation: chunked Rust calls for many noise configs.

        Structural data (tensor, placements, routing) is parsed once per chunk.
        Chunks keep peak memory bounded instead of allocating all results at once.

        ``keep_grids=True`` returns the per-qubit ``*_fidelity_grid`` ndarrays
        per result. The Rust batch entry point is scalar-only by design (for
        throughput), so this branch falls back to a per-cell ``run_hot`` loop —
        slower per cell, but the only path that actually produces the
        (num_layers, num_qubits) grids the topology overlay needs.
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
            # Strip per-qubit ``*_grid`` ndarrays before they cross the
            # process-pool pickle boundary or enter the sweep grid. The
            # batch Rust call already returns scalar-only dicts; this strip
            # is a defensive no-op there but matters when callers feed in
            # an alternative result source.
            all_results.extend(_strip_for_grid(r) for r in chunk_results)

        return all_results

    # -- Sweep methods -------------------------------------------------------

    COLD_PATH_KEYS = frozenset({
        "qubits",  # virtual alias: expands to num_qubits == num_logical_qubits
        "num_qubits", "num_cores", "communication_qubits", "buffer_qubits",
        "num_logical_qubits",
        "circuit_type", "topology_type", "intracore_topology", "routing_algorithm",
    })
    INTEGER_KEYS = frozenset({
        "qubits", "num_qubits", "num_cores", "communication_qubits",
        "buffer_qubits", "num_logical_qubits",
        "classical_link_width", "classical_routing_cycles",
    })

    def _memory_capped_max_hot(self, max_hot: int | None) -> tuple[int, int]:
        """Clamp a requested ``max_hot`` to what current RAM can hold.

        Returns ``(effective, requested)`` so callers can report when the
        cap kicked in. ``requested`` mirrors the user's configured value
        (or the registry default when ``None``); ``effective`` is the
        value the sweep should actually respect.
        """
        requested = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT
        return min(requested, _max_hot_points_for_memory()), requested

    def _metric_values(self, metric_key: str, low, high, n: int) -> np.ndarray:
        if metric_key in CAT_METRIC_BY_KEY:
            # ``low`` is the pre-selected list of option values; high/n unused.
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
        """Compute uniform points per axis for an N-D sweep within the total budget."""
        budget = MAX_TOTAL_POINTS_COLD if has_cold else MAX_TOTAL_POINTS_HOT
        # For 1-3D, use legacy values for backward compatibility
        if ndim <= 3:
            return self._sweep_points(ndim, has_cold)
        # For N >= 4, compute from budget
        n = max(MIN_POINTS_PER_AXIS, int(budget ** (1.0 / ndim)))
        # Clamp so total doesn't exceed budget
        while n ** ndim > budget and n > MIN_POINTS_PER_AXIS:
            n -= 1
        return n

    def _compute_axis_counts(
        self, sweep_axes: list, has_cold: bool,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> list[int]:
        """Compute per-axis point counts using a split budget model.

        ``sweep_axes`` entries are either:
          - ``(key, low, high)`` for numeric axes, or
          - ``(key, values_list)`` for categorical axes (2-tuple).

        Categorical axes have a fixed count (len of selected values) and are
        not subject to budget reduction.  Numeric cold-path axes (num_qubits,
        num_cores) are capped by ``max_cold``.  Hot-path axes share the
        remaining hot budget.
        """
        ndim = len(sweep_axes)
        cold_budget = max_cold if max_cold is not None else MAX_COLD_COMPILATIONS
        hot_budget = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT

        # Separate indices by kind
        cat_indices = []     # categorical: fixed count, always cold-path
        cold_indices = []    # numeric cold-path (num_qubits, num_cores)
        hot_indices = []     # numeric hot-path (noise params)
        for i, ax in enumerate(sweep_axes):
            key = ax[0]
            if key in CAT_METRIC_BY_KEY:
                cat_indices.append(i)
            elif key in self.COLD_PATH_KEYS:
                cold_indices.append(i)
            else:
                hot_indices.append(i)

        counts: list[int] = [0] * ndim

        # Categorical axes: fixed at the number of selected options
        for i in cat_indices:
            counts[i] = len(sweep_axes[i][1])

        cat_product = 1
        for i in cat_indices:
            cat_product *= counts[i]

        # Numeric cold axes: distribute remaining cold budget
        # (divide by cat_product so cold_budget applies to numeric cold combos)
        effective_cold_budget = max(1, cold_budget // max(1, cat_product))
        if cold_indices:
            n_cold = len(cold_indices)
            per_cold = max(MIN_POINTS_PER_AXIS, int(effective_cold_budget ** (1.0 / n_cold)))
            for i in cold_indices:
                key, low, high = sweep_axes[i]
                natural = int(high) - int(low) + 1
                counts[i] = max(MIN_POINTS_PER_AXIS, min(natural, per_cold))
            # Verify product is within budget, reduce if needed
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

        # Total cold multiplier (cat × numeric-cold) for hot budget division
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
            # All axes are neither cold nor hot — shouldn't happen, but fallback
            n = max(MIN_POINTS_PER_AXIS, int(hot_budget ** (1.0 / ndim)))
            counts = [n] * ndim

        return counts

    def _eval_point(
        self,
        cold_config: dict,
        noise: dict,
        swept: dict[str, float],
    ) -> dict:
        """Evaluate a single design point, re-running cold path if needed.

        Returns the scalar-only result dict destined for the sweep grid; the
        per-qubit ``*_grid`` ndarrays are dropped here (see
        ``_strip_for_grid``).
        """
        import math as _math
        cfg = dict(cold_config)
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k in self.COLD_PATH_KEYS:
                cfg[k] = int(v) if k in self.INTEGER_KEYS else v
            else:
                hot_noise[k] = v
        _expand_qubits_alias(cfg)
        cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
        # Clamp communication_qubits to the architectural cap and
        # num_logical_qubits to the remaining data-slot count for the
        # current topology / cores / num_qubits.
        _clamp_cfg_comm_and_logical(cfg)
        cached = self.run_cold(**cfg, noise=hot_noise)
        # Return the full dict (per-qubit grids included) so callers that
        # need them can capture before stripping.  ``_result_to_row`` reads
        # only the scalar keys it knows about, so feeding it a full dict is
        # equivalent to the previously-stripped one.
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
        """Run cold-path sweep points with qubit-aware parallel scheduling.

        Each unique cold config (one (num_qubits, num_cores) combo) becomes one
        group. The scheduler submits groups to a process pool so that the sum
        of their estimated peak RSS stays under the memory budget — small
        groups parallelize aggressively, large-qubit groups serialize. A single
        oversized group is always allowed to run alone to avoid deadlock.
        """
        groups: dict[tuple, list[tuple[tuple, dict]]] = {}
        for idx_key, swept in indexed_points:
            cold_vals = tuple(sorted(
                (k, int(v) if k in self.INTEGER_KEYS else v)
                for k, v in swept.items() if k in self.COLD_PATH_KEYS
            ))
            groups.setdefault(cold_vals, []).append((idx_key, swept))

        fallback_nq = int(cold_config.get("num_qubits", 16))

        def _group_nq(cv: tuple) -> int:
            # Also recognise the ``qubits`` alias (logical == physical sweep)
            # which expands to ``num_qubits`` inside the worker.  Without
            # this, memory estimation falls back to the default and large
            # qubit-count cells get scheduled in parallel and OOM.
            for k, v in cv:
                if k == "num_qubits" or k == "qubits":
                    return int(v)
            return fallback_nq

        group_cost: dict[tuple, float] = {
            cv: _estimate_cold_mb(_group_nq(cv)) for cv in groups
        }

        cpu_cap = max(1, (os.cpu_count() or 2) // 2)
        slot_cap = max_workers if max_workers else cpu_cap
        mem_budget = _mem_budget_mb()

        # Per-worker address-space cap (RLIMIT_AS). With arbitrary
        # user-supplied circuits, the cost estimate can be severely wrong
        # (e.g. dense random circuits on grid topology allocate 10x the
        # QFT-calibrated estimate).  Capping each worker to its fair share
        # of the budget ensures a runaway allocation raises MemoryError
        # inside the worker — surfacing as a "Sweep failed" banner — rather
        # than OOM-killing the whole system.
        rss_cap_bytes = (mem_budget * 1024 * 1024) // max(1, slot_cap)

        # Guard: if even the cheapest group's estimate exceeds the budget by a
        # wide margin, the run will OOM the worker.  Fail early with a
        # message the user can act on instead of a BrokenProcessPool crash.
        if group_cost:
            min_cost = min(group_cost.values())
            if min_cost > mem_budget * 1.5:
                largest_nq = max(_group_nq(cv) for cv in groups)
                raise RuntimeError(
                    f"Not enough RAM for this sweep: smallest cold-compile "
                    f"needs ~{min_cost:.0f} MB, budget is {mem_budget} MB "
                    f"(largest qubits = {largest_nq}). Reduce max qubits, "
                    f"close other apps, or lower Max cold compilations."
                )

        # Schedule largest-first so cheap groups can tuck into the remaining budget.
        pending = sorted(groups, key=lambda cv: -group_cost[cv])
        active: dict = {}  # future -> (cold_vals, cost_mb)
        results: dict[tuple, dict] = {}
        completed = 0
        cold_total = len(groups)
        cold_completed = 0

        # Recycle workers after a handful of cold compilations to flush
        # qiskit/BLAS caches that otherwise grow across reused processes
        # and drift real RSS above the per-job estimate.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=slot_cap, mp_context=_MP_CONTEXT,
            max_tasks_per_child=4,
        ) as pool:

            def _submit_fitting() -> None:
                used = sum(c for _, c in active.values())
                kept: list[tuple] = []
                for cv in pending:
                    cost = group_cost[cv]
                    if len(active) >= slot_cap:
                        kept.append(cv)
                        continue
                    # Always allow one job when the pool is idle, even if its
                    # estimate exceeds the budget (otherwise we'd deadlock on
                    # a single oversized group).
                    mem_ok = not active or used + cost <= mem_budget
                    if mem_ok:
                        swept_list = [s for _, s in groups[cv]]
                        fut = pool.submit(
                            _eval_cold_batch,
                            cold_config, noise, swept_list, rss_cap_bytes,
                            keep_grids,
                        )
                        active[fut] = (cv, cost)
                        used += cost
                    else:
                        kept.append(cv)
                pending[:] = kept

            _submit_fitting()
            while active:
                done, _still = concurrent.futures.wait(
                    list(active), return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    cv, _cost = active.pop(fut)
                    batch_results = fut.result()
                    cold_completed += 1
                    for (idx_key, swept), result in zip(groups[cv], batch_results):
                        results[idx_key] = result
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(SweepProgress(
                                completed=completed,
                                total=total,
                                current_params={k: float(v) for k, v in swept.items()},
                                cold_completed=cold_completed,
                                cold_total=cold_total,
                            ))
                _submit_fitting()

        return results

    def sweep_1d(
        self,
        cached: CachedMapping,
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
        has_cold = self._has_cold(metric_key)
        counts = self._compute_axis_counts(
            [(metric_key, low, high)], has_cold, max_cold, max_hot,
        )
        n = counts[0]
        xs = self._metric_values(metric_key, low, high, n)
        total = len(xs)

        if parallel and has_cold and cold_config is not None:
            indexed = [((i,), {metric_key: v}) for i, v in enumerate(xs)]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            return xs, [rmap[(i,)] for i in range(total)]

        results = []
        for i, v in enumerate(xs):
            if has_cold:
                results.append(self._eval_point(cold_config, fixed_noise, {metric_key: v}))
            else:
                noise = {**fixed_noise, metric_key: v}
                results.append(_strip_for_grid(self.run_hot(cached, noise)))
            if progress_callback is not None:
                progress_callback(SweepProgress(
                    completed=i + 1,
                    total=total,
                    current_params={metric_key: float(v)},
                ))
        return xs, results

    def sweep_2d(
        self,
        cached: CachedMapping,
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
        has_cold = self._has_cold(metric_key1, metric_key2)
        counts = self._compute_axis_counts(
            [(metric_key1, low1, high1), (metric_key2, low2, high2)],
            has_cold, max_cold, max_hot,
        )
        xs = self._metric_values(metric_key1, low1, high1, counts[0])
        ys = self._metric_values(metric_key2, low2, high2, counts[1])
        total = len(xs) * len(ys)

        if parallel and has_cold and cold_config is not None:
            indexed = [
                ((i, j), {metric_key1: v1, metric_key2: v2})
                for i, v1 in enumerate(xs)
                for j, v2 in enumerate(ys)
            ]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            grid = [[rmap[(i, j)] for j in range(len(ys))] for i in range(len(xs))]
            return xs, ys, grid

        grid = []
        count = 0
        for v1 in xs:
            row = []
            for v2 in ys:
                swept = {metric_key1: v1, metric_key2: v2}
                if has_cold:
                    row.append(self._eval_point(cold_config, fixed_noise, swept))
                else:
                    noise = {**fixed_noise, **swept}
                    row.append(_strip_for_grid(self.run_hot(cached, noise)))
                count += 1
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        completed=count,
                        total=total,
                        current_params={
                            metric_key1: float(v1),
                            metric_key2: float(v2),
                        },
                    ))
            grid.append(row)
        return xs, ys, grid

    def sweep_3d(
        self,
        cached: CachedMapping,
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
        has_cold = self._has_cold(metric_key1, metric_key2, metric_key3)
        counts = self._compute_axis_counts(
            [
                (metric_key1, low1, high1),
                (metric_key2, low2, high2),
                (metric_key3, low3, high3),
            ],
            has_cold, max_cold, max_hot,
        )
        xs = self._metric_values(metric_key1, low1, high1, counts[0])
        ys = self._metric_values(metric_key2, low2, high2, counts[1])
        zs = self._metric_values(metric_key3, low3, high3, counts[2])
        total = len(xs) * len(ys) * len(zs)

        if parallel and has_cold and cold_config is not None:
            indexed = [
                ((i, j, k), {metric_key1: v1, metric_key2: v2, metric_key3: v3})
                for i, v1 in enumerate(xs)
                for j, v2 in enumerate(ys)
                for k, v3 in enumerate(zs)
            ]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            grid = [
                [[rmap[(i, j, k)] for k in range(len(zs))] for j in range(len(ys))]
                for i in range(len(xs))
            ]
            return xs, ys, zs, grid

        grid = []
        count = 0
        for v1 in xs:
            plane = []
            for v2 in ys:
                row_list = []
                for v3 in zs:
                    swept = {metric_key1: v1, metric_key2: v2, metric_key3: v3}
                    if has_cold:
                        row_list.append(self._eval_point(cold_config, fixed_noise, swept))
                    else:
                        noise = {**fixed_noise, **swept}
                        row_list.append(_strip_for_grid(self.run_hot(cached, noise)))
                    count += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=count,
                            total=total,
                            current_params={
                                metric_key1: float(v1),
                                metric_key2: float(v2),
                                metric_key3: float(v3),
                            },
                        ))
                plane.append(row_list)
            grid.append(plane)
        return xs, ys, zs, grid

    # -- N-D sweep (unified) ---------------------------------------------------

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

        Parameters
        ----------
        sweep_axes : list of (metric_key, low, high) tuples
        max_cold : override for cold compilation budget
        max_hot : override for total hot-path point budget
        keep_per_qubit_grids : populate ``SweepResult.per_qubit_data`` with
            per-cell per-qubit fidelity grids + placements + effective cold
            config so the topology view can colour nodes / replot the device
            for any cell without re-running the engine.  Adds memory cost
            roughly proportional to ``num_cells × num_layers × num_qubits``.
        """
        from itertools import islice

        ndim = len(sweep_axes)
        keys = [ax[0] for ax in sweep_axes]
        has_cold = self._has_cold(*keys)

        # Clamp user-requested max_hot to what RAM can hold before computing
        # per-axis counts. Without this, an oversized budget silently
        # produces a grid the kernel OOM-kills mid-sweep.
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

        # Build per-axis value arrays.
        # Categorical axes: ax = (key, values_list) — 2-tuple.
        # Numeric axes:     ax = (key, low, high)   — 3-tuple.
        axes = []
        for i, ax in enumerate(sweep_axes):
            k = ax[0]
            if k in CAT_METRIC_BY_KEY:
                # ax[1] is the list of selected option values
                axes.append(self._metric_values(k, ax[1], None, axis_counts[i]))
            else:
                axes.append(self._metric_values(k, ax[1], ax[2], axis_counts[i]))
        shape = tuple(len(ax) for ax in axes)
        total = int(np.prod(shape))

        grid = np.empty(shape, dtype=_RESULT_DTYPE)

        def _make_swept(idx):
            """Build swept dict for one grid index, preserving value types."""
            return {keys[d]: axes[d][idx[d]] for d in range(ndim)}

        def _progress_params(swept):
            """Coerce swept values to float where possible for progress display."""
            out = {}
            for k, v in swept.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = 0.0
            return out

        # Collect per-cell per-qubit grids during the sweep when requested,
        # so the topology view can scrub cells without paying the cold
        # compile a second time.  Memory ≈ num_cells × num_layers ×
        # num_qubits × 4 grids × 8 bytes (~13 MB for 32×100×128×4).
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

        # Snapshot the engine inputs so the topology view's overlay can
        # rebuild any cell on demand via run_cold/run_hot without
        # requiring callers to keep the original config around.
        per_qubit_meta: dict | None = None
        if keep_per_qubit_grids:
            per_qubit_meta = {
                "cold_config": dict(cold_config) if cold_config else None,
                "fixed_noise": dict(fixed_noise) if fixed_noise else None,
                "axis_keys": list(keys),
                "axis_values": [ax.tolist() for ax in axes],
                "shape": list(shape),
                # Per-cell per-qubit grids captured during the sweep so the
                # topology view does an O(1) lookup instead of paying the
                # cold compile a second time per slider move.
                "cells": per_qubit_cells,
            }

        return SweepResult(
            metric_keys=keys, axes=axes, grid=grid,
            per_qubit_data=per_qubit_meta,
        )


# ---------------------------------------------------------------------------
# Process-pool worker (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _eval_cold_batch(
    cold_config: dict,
    noise: dict,
    swept_list: list[dict],
    rss_cap_bytes: int | None = None,
    keep_grids: bool = False,
) -> list[dict]:
    """Evaluate a batch of design points sharing the same cold-path config.

    Compiles the circuit once (cold path), then evaluates all noise
    variations in a single batched Rust call (no per-point Python↔Rust overhead).

    ``rss_cap_bytes`` caps the worker's address space via RLIMIT_AS so a
    runaway allocation from a pathologically large circuit raises
    MemoryError inside this process instead of thrashing the whole system.
    """
    if rss_cap_bytes and rss_cap_bytes > 0:
        try:
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS, (int(rss_cap_bytes), int(rss_cap_bytes)),
            )
        except (ImportError, ValueError, OSError):
            # RLIMIT_AS unavailable (non-Linux) or kernel refused — fall
            # back to unbounded execution; the scheduler's mem_ok check
            # still provides some protection.
            pass

    engine = DSEEngine()

    # Apply cold-path overrides from the first swept dict (all share the same cold keys)
    cfg = dict(cold_config)
    for k, v in swept_list[0].items():
        if k in DSEEngine.COLD_PATH_KEYS:
            cfg[k] = int(v)
    _expand_qubits_alias(cfg)
    cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
    _clamp_cfg_comm_and_logical(cfg)

    # Build the noise dicts for each swept point
    noise_dicts = []
    for swept in swept_list:
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k not in DSEEngine.COLD_PATH_KEYS:
                hot_noise[k] = v
        noise_dicts.append(hot_noise)

    # Single cold compilation
    cached = engine.run_cold(**cfg, noise=noise_dicts[0])

    # Single batched Rust call for all hot-path variations
    return engine.run_hot_batch(cached, noise_dicts, keep_grids=keep_grids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _remap_to_dag_layers(placements: np.ndarray, dag_layers: int) -> np.ndarray:
    """Remap TeleSABRE placements from ts-gate-step space to DAG-layer space.

    TeleSABRE's placements have shape (ts_layers+1, num_qubits) where
    ts_layers = num_teledata + num_telegate + num_swaps.  The hot path needs
    (dag_layers+1, num_qubits) to match the Qiskit gs_sparse layer count.
    """
    ts_layers = placements.shape[0] - 1
    num_qubits = placements.shape[1]
    out = np.zeros((dag_layers + 1, num_qubits), dtype=np.int32)
    for l in range(dag_layers + 1):
        ts_l = round(l * ts_layers / dag_layers) if dag_layers > 0 else 0
        out[l] = placements[min(ts_l, ts_layers)]
    return out


def _remap_swaps_to_dag_layers(sparse_swaps: np.ndarray, dag_layers: int) -> np.ndarray:
    """Remap TeleSABRE sparse_swaps layer indices to DAG-layer space."""
    if len(sparse_swaps) == 0:
        return sparse_swaps
    ts_layers = int(sparse_swaps[:, 0].max()) + 1 if len(sparse_swaps) > 0 else 1
    out = sparse_swaps.copy()
    for i in range(len(out)):
        ts_l = int(out[i, 0])
        dag_l = round(ts_l * dag_layers / ts_layers) if ts_layers > 0 else 0
        out[i, 0] = min(dag_l, dag_layers - 1)
    return out


def _build_telesabre_device_json(
    num_qubits: int,
    num_cores: int,
    topology_type: str,
    intracore_topology: str,
    communication_qubits: int = 1,
    buffer_qubits: int = 1,
) -> dict:
    """Build a TeleSABRE-format device JSON dict from DSE topology parameters.

    Two invariants that the TeleSABRE C library requires:

    1. Uniform core sizes: ``device_from_json`` computes core ownership as
       ``phys_to_core[i] = i / core_capacity`` where ``core_capacity =
       total_qubits / num_cores`` (integer division). Non-uniform sizes
       cause the last qubit to map to an out-of-bounds core index,
       corrupting the heap → double-free segfault.  Fix: always use
       ``ceil(num_qubits / num_cores) + SLACK_PER_CORE`` qubits per core.

    2. Free teleportation slots: the round-robin initial layout fills every
       physical qubit when ``total_qubits == num_qubits``.  With no free
       qubits, all nearest-free-qubit heaps are empty → Dijkstra node
       weights equal TS_INF → integer overflow → heap corruption → crash.
       Fix: ``SLACK_PER_CORE = 3`` guarantees at least 3 free slots per
       core after initial layout for mediator/target teleportation.
    """
    import math
    num_cores = max(1, min(num_cores, num_qubits))
    # TeleSABRE's routing requires free physical qubits in each core as
    # teleportation mediators and landing slots.  Without slack, round-robin
    # initial layout fills every slot and routing deadlocks with a heap
    # corruption crash (Dijkstra gets TS_INF node weights, integer arithmetic
    # overflows, and glibc detects a double-free).  Add at least 3 spare
    # qubits per core so each core always has free slots after initial layout.
    SLACK_PER_CORE = 3
    qubits_per_core = math.ceil(num_qubits / num_cores) + SLACK_PER_CORE
    total_qubits = qubits_per_core * num_cores
    offsets = [c * qubits_per_core for c in range(num_cores)]

    # Intra-core edges
    intra_edges = []
    for c in range(num_cores):
        off = offsets[c]
        size = qubits_per_core
        if size < 2:
            continue
        if intracore_topology == "all_to_all":
            for q in range(size):
                for r in range(q + 1, size):
                    intra_edges.append([off + q, off + r])
        elif intracore_topology == "ring":
            for q in range(size):
                intra_edges.append([off + q, off + (q + 1) % size])
        elif intracore_topology == "grid":
            side = math.isqrt(size)
            if side * side < size:
                side += 1
            for q in range(size):
                row, col = divmod(q, side)
                if col + 1 < side and q + 1 < size:
                    intra_edges.append([off + q, off + q + 1])
                if q + side < size:
                    intra_edges.append([off + q, off + q + side])
        else:  # linear (default)
            for q in range(size - 1):
                intra_edges.append([off + q, off + q + 1])

    # Inter-core edges: each comm qubit hosts exactly one inter-core link.
    # A core's qubits are laid out as ``[D data, group_0, group_1, ...]``
    # with each group = ``K`` comm + ``1`` buffer slots; inter_core_edges
    # pairs comm qubit ``k`` of one core's group-toward-partner with comm
    # qubit ``k`` of the partner's group-toward-this.
    inter_edges = []
    if num_cores > 1:
        groups_per_core = num_comm_groups(num_cores, topology_type)
        B_req = max(1, int(buffer_qubits or 1))
        K_caps = [
            _max_K_for_layout(qubits_per_core, groups_per_core[c],
                              b_per_group=B_req)
            for c in range(num_cores)
            if groups_per_core[c] > 0
        ]
        K_max = min(K_caps) if K_caps else 0
        K = min(int(communication_qubits or 1), max(0, K_max))
        if K >= 1:
            B_caps = [
                _max_B_for_layout(qubits_per_core, groups_per_core[c], K)
                for c in range(num_cores)
                if groups_per_core[c] > 0
            ]
            B_max = min(B_caps) if B_caps else 0
            B = min(B_req, max(1, B_max))
            per_core_layout = [
                assign_core_slots(qubits_per_core, intracore_topology,
                                  groups_per_core[c], K, b_per_group=B)
                for c in range(num_cores)
            ]
            for (a_core, a_g, a_k), (b_core, b_g, b_k) in inter_core_edges(
                num_cores, K, topology_type,
            ):
                p1 = offsets[a_core] + per_core_layout[a_core]["groups"][a_g]["comm"][a_k]
                p2 = offsets[b_core] + per_core_layout[b_core]["groups"][b_g]["comm"][b_k]
                inter_edges.append([p1, p2])

    # Simple grid node positions
    side = math.isqrt(total_qubits)
    if side * side < total_qubits:
        side += 1
    node_positions = [
        [float(q % side), -float(q // side)] for q in range(total_qubits)
    ]

    return {
        "device": {
            "name": f"dse_{total_qubits}q_{num_cores}c_{topology_type}",
            "num_cores": num_cores,
            "num_qubits": total_qubits,
            "intra_core_edges": intra_edges,
            "inter_core_edges": inter_edges,
            "node_positions": node_positions,
        }
    }
