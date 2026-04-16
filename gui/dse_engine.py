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
import time
import multiprocessing
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Optional

_MP_CONTEXT = multiprocessing.get_context("forkserver")

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
        )
        result["total_epr_pairs"] = cached.total_epr_pairs
        return result

    def run_hot_batch(self, cached: CachedMapping, noise_dicts: list[dict]) -> list[dict]:
        """
        Batch fidelity estimation: single Rust call for many noise configs.

        Structural data (tensor, placements, routing) is parsed once inside Rust.
        Returns scalar-only result dicts (no grids).
        """
        merged_list = [_merge_noise(n) for n in noise_dicts]
        gate_error_arr, gate_time_arr = _make_gate_arrays(cached.gate_names, merged_list[0])

        results = qusim.estimate_fidelity_from_cache_batch(
            gs_sparse=cached.gs_sparse,
            placements=cached.placements,
            distance_matrix=cached.distance_matrix,
            sparse_swaps=cached.sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            noise_dicts=merged_list,
        )
        for r in results:
            r["total_epr_pairs"] = cached.total_epr_pairs
        return results

    # -- Sweep methods -------------------------------------------------------

    COLD_PATH_KEYS = frozenset({"num_qubits", "num_cores"})

    def _metric_values(self, metric_key: str, low: float, high: float, n: int) -> np.ndarray:
        m = METRIC_BY_KEY[metric_key]
        if m.log_scale:
            return np.logspace(low, high, n)
        vals = np.linspace(low, high, n)
        if metric_key in self.COLD_PATH_KEYS:
            vals = np.unique(np.round(vals).astype(int))
        return vals

    def _has_cold(self, *keys: str) -> bool:
        return any(k in self.COLD_PATH_KEYS for k in keys)

    def _sweep_points(self, ndim: int, has_cold: bool) -> int:
        if has_cold:
            return [SWEEP_POINTS_COLD_1D, SWEEP_POINTS_COLD_2D, SWEEP_POINTS_COLD_3D][ndim - 1]
        return [SWEEP_POINTS_1D, SWEEP_POINTS_2D, SWEEP_POINTS_3D][ndim - 1]

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

    def _parallel_cold_sweep(
        self,
        cold_config: dict,
        noise: dict,
        indexed_points: list[tuple[tuple, dict]],
        total: int,
        progress_callback: Callable[[SweepProgress], None] | None,
        max_workers: int | None,
    ) -> dict[tuple, dict]:
        """Run cold-path sweep points in parallel, grouped by cold-path values.

        Each group shares the same cold-path config and is submitted as a single
        batch to a process pool.  Within a batch the first _eval_point triggers
        cold compilation; subsequent calls hit the engine's cache.

        Returns a mapping of index_key -> result_dict.
        """
        # Group by cold-path values so each batch compiles only once
        groups: dict[tuple, list[tuple[tuple, dict]]] = {}
        for idx_key, swept in indexed_points:
            cold_vals = tuple(sorted(
                (k, int(v)) for k, v in swept.items() if k in self.COLD_PATH_KEYS
            ))
            groups.setdefault(cold_vals, []).append((idx_key, swept))

        # Cap at half the CPUs so the UI thread, browser, and OS stay responsive
        cpu_cap = max(1, (os.cpu_count() or 2) // 2)
        n_workers = max_workers if max_workers else min(cpu_cap, len(groups))
        results: dict[tuple, dict] = {}
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, mp_context=_MP_CONTEXT,
        ) as pool:
            future_map = {}
            for cold_vals, group in groups.items():
                swept_list = [s for _, s in group]
                future = pool.submit(_eval_cold_batch, cold_config, noise, swept_list)
                future_map[future] = group

            for future in concurrent.futures.as_completed(future_map):
                group = future_map[future]
                batch_results = future.result()
                for (idx_key, swept), result in zip(group, batch_results):
                    results[idx_key] = result
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=completed,
                            total=total,
                            current_params={k: float(v) for k, v in swept.items()},
                        ))

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
    ) -> tuple[np.ndarray, list]:
        has_cold = self._has_cold(metric_key)
        n = self._sweep_points(1, has_cold)
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
    ) -> tuple[np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2)
        n = self._sweep_points(2, has_cold)
        xs = self._metric_values(metric_key1, low1, high1, n)
        ys = self._metric_values(metric_key2, low2, high2, n)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2, metric_key3)
        n = self._sweep_points(3, has_cold)
        xs = self._metric_values(metric_key1, low1, high1, n)
        ys = self._metric_values(metric_key2, low2, high2, n)
        zs = self._metric_values(metric_key3, low3, high3, n)
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
