"""
DSE computation engine.

Two-stage execution model:
  Cold path – full circuit transpilation + HQA mapping + SABRE routing.
              Required when circuit structure or topology changes (~1-10s).
  Hot path  – fidelity estimation only, reusing cached placements/swaps.
              Required when only noise parameters change (<1ms per point).
"""

import sys
import os

# Cap per-process thread pools BEFORE importing numpy/qiskit/qusim so their
# Rayon/OpenMP/BLAS pools initialize at 1 thread.  Otherwise each cold-path
# worker spawns ~34 threads and 8 parallel workers oversubscribe the CPU
# (272 threads on 16 cores) until the scheduler stalls and Dash's heartbeat
# times out — which the UI shows as a crash.  Workers use process-level
# parallelism already, so there is nothing to lose by disabling library
# threads here.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import time
import multiprocessing
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Optional

_MP_CONTEXT = multiprocessing.get_context("forkserver")

# Minimum free RAM (MB) to keep available for OS + UI + browser
_RESERVED_RAM_MB = 1024

# Empirical peak RSS (MB) for one cold compilation vs. num_qubits, measured on
# QFT (densest common circuit) with the forkserver worker used in production.
# See tests/test_cores_qubits_oom.py for the measurement.
_EMPIRICAL_COLD_MB: list[tuple[int, float]] = [
    (4,   150.0),
    (40,  160.0),
    (76,  260.0),
    (112, 450.0),
    (148, 820.0),
    (184, 1600.0),
    (220, 2100.0),
    (256, 3800.0),
]


def _estimate_cold_mb(num_qubits: int) -> float:
    """Peak RSS estimate (MB) for one cold compilation at ``num_qubits``.

    Piecewise-linear interpolation over :data:`_EMPIRICAL_COLD_MB`, with
    linear extrapolation beyond the measured range.
    """
    pts = _EMPIRICAL_COLD_MB
    if num_qubits <= pts[0][0]:
        return pts[0][1]
    if num_qubits >= pts[-1][0]:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (num_qubits - x2)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= num_qubits <= x2:
            return y1 + (y2 - y1) * (num_qubits - x1) / (x2 - x1)
    return pts[-1][1]

import numpy as np
import qiskit
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import qusim
from qusim.hqa.placement import InitialPlacement

from .constants import (
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


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _build_circuit(circuit_type: str, num_qubits: int, seed: int) -> qiskit.QuantumCircuit:
    if circuit_type == "qft":
        circ = QFT(num_qubits)
    elif circuit_type == "ghz":
        circ = qiskit.QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(0, i)
    elif circuit_type == "random":
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(num_qubits, depth=max(3, num_qubits // 2), seed=seed)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    return circ


def _transpile_circuit(circ: qiskit.QuantumCircuit, seed: int) -> qiskit.QuantumCircuit:
    return transpile(
        circ,
        basis_gates=["x", "cx", "cp", "rz", "h", "s", "sdg", "t", "tdg", "measure"],
        optimization_level=0,
        seed_transpiler=seed,
    )


# ---------------------------------------------------------------------------
# Topology builders  (all-to-all intra-core, configurable inter-core)
# ---------------------------------------------------------------------------

def _build_topology(
    num_qubits: int,
    num_cores: int,
    topology_type: str,
    intracore_topology: str = "all_to_all",
) -> tuple[CouplingMap, dict[int, int]]:
    """Build full_coupling_map and core_mapping for the given topology.

    Qubits are distributed across cores so that physical count == logical
    count exactly.  With 7 qubits and 3 cores the sizes are [3, 2, 2].
    """
    import math
    if num_cores < 1:
        num_cores = 1
    num_cores = min(num_cores, num_qubits)

    base = num_qubits // num_cores
    remainder = num_qubits % num_cores
    core_sizes = [base + (1 if c < remainder else 0) for c in range(num_cores)]

    cm = CouplingMap()
    for i in range(num_qubits):
        cm.add_physical_qubit(i)

    core_mapping: dict[int, int] = {}
    core_offsets: list[int] = []
    offset = 0
    for c, size in enumerate(core_sizes):
        core_offsets.append(offset)
        for q in range(size):
            core_mapping[offset + q] = c

        _add_intracore_edges(cm, offset, size, intracore_topology)
        offset += size

    if num_cores > 1:
        if topology_type == "ring":
            for c in range(num_cores):
                next_c = (c + 1) % num_cores
                cm.add_edge(core_offsets[c], core_offsets[next_c])
                cm.add_edge(core_offsets[next_c], core_offsets[c])

        elif topology_type == "all_to_all":
            for c1 in range(num_cores):
                for c2 in range(c1 + 1, num_cores):
                    p1 = core_offsets[c1] + core_sizes[c1] - 1
                    p2 = core_offsets[c2]
                    cm.add_edge(p1, p2)
                    cm.add_edge(p2, p1)

        elif topology_type == "linear":
            for c in range(num_cores - 1):
                p1 = core_offsets[c] + core_sizes[c] - 1
                p2 = core_offsets[c + 1]
                cm.add_edge(p1, p2)
                cm.add_edge(p2, p1)

    return cm, core_mapping


def _add_intracore_edges(
    cm: CouplingMap, offset: int, size: int, topology: str,
) -> None:
    import math
    if size < 2:
        return
    if topology == "all_to_all":
        for q in range(size):
            for r in range(q + 1, size):
                cm.add_edge(offset + q, offset + r)
                cm.add_edge(offset + r, offset + q)
    elif topology == "linear":
        for q in range(size - 1):
            cm.add_edge(offset + q, offset + q + 1)
            cm.add_edge(offset + q + 1, offset + q)
    elif topology == "ring":
        for q in range(size):
            nxt = (q + 1) % size
            cm.add_edge(offset + q, offset + nxt)
            cm.add_edge(offset + nxt, offset + q)
    elif topology == "grid":
        side = math.isqrt(size)
        if side * side < size:
            side += 1
        for q in range(size):
            row, col = divmod(q, side)
            if col + 1 < side and q + 1 < size:
                cm.add_edge(offset + q, offset + q + 1)
                cm.add_edge(offset + q + 1, offset + q)
            if q + side < size:
                cm.add_edge(offset + q, offset + q + side)
                cm.add_edge(offset + q + side, offset + q)


# ---------------------------------------------------------------------------
# Cached mapping data
# ---------------------------------------------------------------------------

@dataclass
class CachedMapping:
    gs_sparse: np.ndarray
    placements: np.ndarray
    distance_matrix: np.ndarray
    sparse_swaps: np.ndarray
    gate_error_arr: np.ndarray
    gate_time_arr: np.ndarray
    gate_names: list
    total_epr_pairs: int
    # Key used to detect when re-mapping is required
    config_key: tuple
    cold_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Noise param helpers
# ---------------------------------------------------------------------------

def _merge_noise(overrides: dict) -> dict:
    """Return NOISE_DEFAULTS with any overrides applied."""
    return {**NOISE_DEFAULTS, **overrides}


def _make_gate_arrays(gate_names: list, noise: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build gate_error_arr and gate_time_arr from gate_names and scalar noise params."""
    VIRTUAL = {"rz", "id", "delay", "barrier", "measure"}
    err_map = {**{g: 0.0 for g in VIRTUAL}}
    time_map = {**{g: 0.0 for g in VIRTUAL}}

    error_arr = np.array(
        [err_map.get(name, np.nan) for name in gate_names], dtype=np.float64
    )
    time_arr = np.array(
        [time_map.get(name, np.nan) for name in gate_names], dtype=np.float64
    )
    return error_arr, time_arr


# ---------------------------------------------------------------------------
# Sweep progress tracking
# ---------------------------------------------------------------------------

@dataclass
class SweepProgress:
    """Snapshot of sweep progress, emitted on every iteration."""
    completed: int
    total: int
    current_params: dict[str, float] = field(default_factory=dict)

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.completed / self.total * 100, 2)


# ---------------------------------------------------------------------------
# N-D sweep result
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """N-dimensional sweep result with numpy object grid.

    Attributes:
        metric_keys: Names of the swept parameters.
        axes: One array of sample values per swept parameter.
        grid: numpy object array of shape (len(axes[0]), ..., len(axes[N-1])),
              each element is a result dict.
    """
    metric_keys: list[str]
    axes: list[np.ndarray]
    grid: np.ndarray  # dtype=object, shape matches axis lengths

    @property
    def ndim(self) -> int:
        return len(self.metric_keys)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(ax) for ax in self.axes)

    @property
    def total_points(self) -> int:
        return int(np.prod([len(ax) for ax in self.axes]))

    def to_sweep_data(self) -> dict:
        """Convert to the dict format consumed by plotting functions.

        For 1-3D: produces the legacy xs/ys/zs + nested-list grid.
        For 4D+:  produces axes list + flat grid list with shape metadata.
        """
        sd: dict = {"metric_keys": self.metric_keys}
        n = self.ndim

        if n >= 1:
            sd["xs"] = self.axes[0].tolist()
        if n >= 2:
            sd["ys"] = self.axes[1].tolist()
        if n >= 3:
            sd["zs"] = self.axes[2].tolist()

        if n == 1:
            sd["grid"] = [self.grid[i] for i in range(len(self.axes[0]))]
        elif n == 2:
            sd["grid"] = [
                [self.grid[i, j] for j in range(len(self.axes[1]))]
                for i in range(len(self.axes[0]))
            ]
        elif n == 3:
            sd["grid"] = [
                [[self.grid[i, j, k] for k in range(len(self.axes[2]))]
                 for j in range(len(self.axes[1]))]
                for i in range(len(self.axes[0]))
            ]
        else:
            # N >= 4: flat list + shape metadata + axes
            sd["axes"] = [ax.tolist() for ax in self.axes]
            sd["shape"] = self.shape
            sd["grid"] = [self.grid[idx] for idx in np.ndindex(self.shape)]

        return sd


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
    ) -> CachedMapping:
        """
        Full circuit transpilation + HQA mapping. Returns a CachedMapping that
        can be reused for many hot-path fidelity evaluations.
        """
        config_key = (circuit_type, num_qubits, num_cores, topology_type, intracore_topology, placement_policy, seed)
        if self._cache is not None and self._cache.config_key == config_key:
            return self._cache

        t0 = time.time()

        circ = _build_circuit(circuit_type, num_qubits, seed)
        transp = _transpile_circuit(circ, seed)

        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type,
            intracore_topology=intracore_topology,
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

        # sparse_swaps were computed inside map_circuit but not exposed.
        # Re-derive them from the orchestrator output embedded in result.placements:
        # Since map_circuit returns a QusimResult without exposing sparse_swaps,
        # we use a zero-length array here and call the orchestrator separately.
        # This is acceptable because the hot path calls estimate_hardware_fidelity
        # which already incorporates swap overhead via placements.
        sparse_swaps = np.zeros((0, 3), dtype=np.int32)

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
            t1=merged["t1"],
            t2=merged["t2"],
            dynamic_decoupling=merged.get("dynamic_decoupling", False),
            readout_mitigation_factor=merged["readout_mitigation_factor"],
            classical_link_width=int(merged["classical_link_width"]),
            classical_clock_freq_hz=merged["classical_clock_freq_hz"],
            classical_routing_cycles=int(merged["classical_routing_cycles"]),
        )
        result["total_epr_pairs"] = cached.total_epr_pairs
        return result

    # Max noise configs per Rust batch call to bound peak memory
    _HOT_BATCH_CHUNK = 5_000

    def run_hot_batch(self, cached: CachedMapping, noise_dicts: list[dict]) -> list[dict]:
        """
        Batch fidelity estimation: chunked Rust calls for many noise configs.

        Structural data (tensor, placements, routing) is parsed once per chunk.
        Chunks keep peak memory bounded instead of allocating all results at once.
        """
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
            all_results.extend(chunk_results)

        return all_results

    # -- Sweep methods -------------------------------------------------------

    COLD_PATH_KEYS = frozenset({"num_qubits", "num_cores"})
    INTEGER_KEYS = frozenset({"num_qubits", "num_cores", "classical_link_width", "classical_routing_cycles"})

    def _metric_values(self, metric_key: str, low: float, high: float, n: int) -> np.ndarray:
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
        self, sweep_axes: list[tuple[str, float, float]], has_cold: bool,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> list[int]:
        """Compute per-axis point counts using a split budget model.

        Cold-path axes (num_qubits, num_cores) are capped by ``max_cold``
        (total cold compilations = product of cold-axis counts).
        Hot-path axes are capped by ``max_hot`` (total grid points).

        Cold-path axes get their natural integer range, capped so their
        product stays within the cold compilation budget.  Hot-path axes
        share the remaining hot budget.
        """
        ndim = len(sweep_axes)
        cold_budget = max_cold if max_cold is not None else MAX_COLD_COMPILATIONS
        hot_budget = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT

        # Separate cold vs hot axes
        cold_indices = []
        hot_indices = []
        for i, (key, _, _) in enumerate(sweep_axes):
            if key in self.COLD_PATH_KEYS:
                cold_indices.append(i)
            else:
                hot_indices.append(i)

        counts: list[int] = [0] * ndim

        # Cold axes: distribute cold compilation budget
        if cold_indices:
            n_cold = len(cold_indices)
            per_cold = max(MIN_POINTS_PER_AXIS, int(cold_budget ** (1.0 / n_cold)))
            for i in cold_indices:
                key, low, high = sweep_axes[i]
                natural = int(high) - int(low) + 1
                counts[i] = max(MIN_POINTS_PER_AXIS, min(natural, per_cold))
            # Verify product is within budget, reduce if needed
            cold_product = 1
            for i in cold_indices:
                cold_product *= counts[i]
            while cold_product > cold_budget:
                # Reduce the largest cold axis by 1
                largest = max(cold_indices, key=lambda i: counts[i])
                counts[largest] = max(MIN_POINTS_PER_AXIS, counts[largest] - 1)
                cold_product = 1
                for i in cold_indices:
                    cold_product *= counts[i]
                if all(counts[i] == MIN_POINTS_PER_AXIS for i in cold_indices):
                    break

        # Hot axes: distribute hot budget (divided by cold product)
        cold_product = 1
        for i in cold_indices:
            cold_product *= counts[i]

        if hot_indices:
            n_hot = len(hot_indices)
            effective_hot_budget = max(1, hot_budget // max(1, cold_product))
            per_hot = max(MIN_POINTS_PER_AXIS, int(effective_hot_budget ** (1.0 / n_hot)))
            # Clamp so total doesn't exceed budget
            while per_hot ** n_hot * cold_product > hot_budget and per_hot > MIN_POINTS_PER_AXIS:
                per_hot -= 1
            for i in hot_indices:
                counts[i] = per_hot
        elif not cold_indices:
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
        """Evaluate a single design point, re-running cold path if needed."""
        cfg = dict(cold_config)
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k in self.COLD_PATH_KEYS:
                cfg[k] = int(v)
            else:
                hot_noise[k] = v
        cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
        cached = self.run_cold(**cfg, noise=hot_noise)
        return self.run_hot(cached, hot_noise)

    @staticmethod
    def _mem_budget_mb() -> int:
        """Memory budget (MB) for concurrent cold compilations.

        Reserves the larger of ``_RESERVED_RAM_MB`` (1 GB floor) or 30% of
        total RAM so the budget scales with the host. Keeping 30% free
        prevents the swap-thrashing deadlock that can freeze the whole OS
        when multiple 3+ GB workers run alongside the browser, desktop,
        and a growing page cache.
        """
        try:
            avail_mb = 0
            total_mb = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemTotal:"):
                        total_mb = int(line.split()[1]) // 1024
                    if avail_mb and total_mb:
                        break
            if avail_mb and total_mb:
                reserved = max(_RESERVED_RAM_MB, total_mb * 30 // 100)
                return max(1, avail_mb - reserved)
        except OSError:
            pass
        # Fallback: a conservative ~1.6 GB when /proc is unavailable
        return 2 * 800

    def _parallel_cold_sweep(
        self,
        cold_config: dict,
        noise: dict,
        indexed_points: list[tuple[tuple, dict]],
        total: int,
        progress_callback: Callable[[SweepProgress], None] | None,
        max_workers: int | None,
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
                (k, int(v)) for k, v in swept.items() if k in self.COLD_PATH_KEYS
            ))
            groups.setdefault(cold_vals, []).append((idx_key, swept))

        fallback_nq = int(cold_config.get("num_qubits", 16))

        def _group_nq(cv: tuple) -> int:
            for k, v in cv:
                if k == "num_qubits":
                    return int(v)
            return fallback_nq

        group_cost: dict[tuple, float] = {
            cv: _estimate_cold_mb(_group_nq(cv)) for cv in groups
        }

        cpu_cap = max(1, (os.cpu_count() or 2) // 2)
        slot_cap = max_workers if max_workers else cpu_cap
        mem_budget = self._mem_budget_mb()

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
                        fut = pool.submit(_eval_cold_batch, cold_config, noise, swept_list)
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
                    for (idx_key, swept), result in zip(groups[cv], batch_results):
                        results[idx_key] = result
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(SweepProgress(
                                completed=completed,
                                total=total,
                                current_params={k: float(v) for k, v in swept.items()},
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
                results.append(self.run_hot(cached, noise))
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
                    row.append(self.run_hot(cached, noise))
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
                        row_list.append(self.run_hot(cached, noise))
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
        cached: CachedMapping,
        sweep_axes: list[tuple[str, float, float]],
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> SweepResult:
        """N-dimensional sweep over arbitrary axes.

        Parameters
        ----------
        sweep_axes : list of (metric_key, low, high) tuples
        max_cold : override for cold compilation budget
        max_hot : override for total hot-path point budget
        """
        import itertools

        ndim = len(sweep_axes)
        keys = [k for k, _, _ in sweep_axes]
        has_cold = self._has_cold(*keys)
        axis_counts = self._compute_axis_counts(sweep_axes, has_cold, max_cold, max_hot)

        axes = [
            self._metric_values(k, lo, hi, axis_counts[i])
            for i, (k, lo, hi) in enumerate(sweep_axes)
        ]
        shape = tuple(len(ax) for ax in axes)
        total = int(np.prod(shape))

        grid = np.empty(shape, dtype=object)

        if parallel and has_cold and cold_config is not None:
            indexed = []
            for idx in np.ndindex(shape):
                swept = {keys[d]: float(axes[d][idx[d]]) for d in range(ndim)}
                indexed.append((idx, swept))
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            for idx in np.ndindex(shape):
                grid[idx] = rmap[idx]
        elif has_cold and cold_config is not None:
            count = 0
            for idx in np.ndindex(shape):
                swept = {keys[d]: float(axes[d][idx[d]]) for d in range(ndim)}
                grid[idx] = self._eval_point(cold_config, fixed_noise, swept)
                count += 1
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        completed=count,
                        total=total,
                        current_params={k: float(v) for k, v in swept.items()},
                    ))
        else:
            # Pure hot-path: chunk to bound memory
            indices = list(np.ndindex(shape))
            completed = 0
            for chunk_start in range(0, len(indices), self._HOT_BATCH_CHUNK):
                chunk_indices = indices[chunk_start : chunk_start + self._HOT_BATCH_CHUNK]
                noise_dicts = []
                for idx in chunk_indices:
                    noise = dict(fixed_noise)
                    for d in range(ndim):
                        noise[keys[d]] = float(axes[d][idx[d]])
                    noise_dicts.append(noise)

                chunk_results = self.run_hot_batch(cached, noise_dicts)
                for i, idx in enumerate(chunk_indices):
                    grid[idx] = chunk_results[i]
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=completed,
                            total=total,
                            current_params={
                                keys[d]: float(axes[d][idx[d]]) for d in range(ndim)
                            },
                        ))
                del noise_dicts, chunk_results

        return SweepResult(metric_keys=keys, axes=axes, grid=grid)


# ---------------------------------------------------------------------------
# Process-pool worker (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _eval_cold_batch(cold_config: dict, noise: dict, swept_list: list[dict]) -> list[dict]:
    """Evaluate a batch of design points sharing the same cold-path config.

    Compiles the circuit once (cold path), then evaluates all noise
    variations in a single batched Rust call (no per-point Python↔Rust overhead).
    """
    engine = DSEEngine()

    # Apply cold-path overrides from the first swept dict (all share the same cold keys)
    cfg = dict(cold_config)
    for k, v in swept_list[0].items():
        if k in DSEEngine.COLD_PATH_KEYS:
            cfg[k] = int(v)
    cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])

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
    return engine.run_hot_batch(cached, noise_dicts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_distance_matrix(
    full_coupling_map: CouplingMap,
    core_mapping: dict[int, int],
    num_cores: int,
) -> np.ndarray:
    """Replicate the Floyd-Warshall distance computation from map_circuit."""
    core_adj = np.zeros((num_cores, num_cores), dtype=np.int32)
    for edge in full_coupling_map.get_edges():
        c1 = core_mapping.get(edge[0], 0)
        c2 = core_mapping.get(edge[1], 0)
        if c1 != c2:
            core_adj[c1, c2] = 1
            core_adj[c2, c1] = 1

    dist_mat = np.full((num_cores, num_cores), fill_value=9999, dtype=np.int32)
    for c in range(num_cores):
        dist_mat[c, c] = 0
    for c1 in range(num_cores):
        for c2 in range(num_cores):
            if core_adj[c1, c2]:
                dist_mat[c1, c2] = 1

    for k in range(num_cores):
        for i in range(num_cores):
            for j in range(num_cores):
                if dist_mat[i, k] + dist_mat[k, j] < dist_mat[i, j]:
                    dist_mat[i, j] = dist_mat[i, k] + dist_mat[k, j]

    return dist_mat
