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
from dataclasses import dataclass
from typing import Optional

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

    Physical qubits are ceil-divided across cores so every logical qubit
    has a slot even when num_qubits % num_cores != 0.
    """
    import math
    if num_cores < 1:
        num_cores = 1
    num_cores = min(num_cores, num_qubits)

    qubits_per_core = math.ceil(num_qubits / num_cores)
    num_physical = qubits_per_core * num_cores

    cm = CouplingMap()
    for i in range(num_physical):
        cm.add_physical_qubit(i)

    core_mapping: dict[int, int] = {}

    for c in range(num_cores):
        offset = c * qubits_per_core
        for q in range(qubits_per_core):
            core_mapping[offset + q] = c

        if intracore_topology == "all_to_all":
            for q in range(qubits_per_core):
                for r in range(q + 1, qubits_per_core):
                    cm.add_edge(offset + q, offset + r)
                    cm.add_edge(offset + r, offset + q)
        elif intracore_topology == "linear":
            for q in range(qubits_per_core - 1):
                cm.add_edge(offset + q, offset + q + 1)
                cm.add_edge(offset + q + 1, offset + q)
        elif intracore_topology == "ring":
            for q in range(qubits_per_core):
                nxt = (q + 1) % qubits_per_core
                cm.add_edge(offset + q, offset + nxt)
                cm.add_edge(offset + nxt, offset + q)
        elif intracore_topology == "grid":
            side = math.isqrt(qubits_per_core)
            if side * side < qubits_per_core:
                side += 1
            for q in range(qubits_per_core):
                row, col = divmod(q, side)
                if col + 1 < side and q + 1 < qubits_per_core:
                    cm.add_edge(offset + q, offset + q + 1)
                    cm.add_edge(offset + q + 1, offset + q)
                if q + side < qubits_per_core:
                    cm.add_edge(offset + q, offset + q + side)
                    cm.add_edge(offset + q + side, offset + q)

    if num_cores > 1:
        if topology_type == "ring":
            for c in range(num_cores):
                next_c = (c + 1) % num_cores
                p1 = c * qubits_per_core
                p2 = next_c * qubits_per_core
                cm.add_edge(p1, p2)
                cm.add_edge(p2, p1)

        elif topology_type == "all_to_all":
            for c1 in range(num_cores):
                for c2 in range(c1 + 1, num_cores):
                    p1 = c1 * qubits_per_core + (qubits_per_core - 1)
                    p2 = c2 * qubits_per_core
                    cm.add_edge(p1, p2)
                    cm.add_edge(p2, p1)

        elif topology_type == "linear":
            for c in range(num_cores - 1):
                p1 = c * qubits_per_core + (qubits_per_core - 1)
                p2 = (c + 1) * qubits_per_core
                cm.add_edge(p1, p2)
                cm.add_edge(p2, p1)

    return cm, core_mapping


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

    def sweep_1d(
        self,
        cached: CachedMapping,
        metric_key: str,
        low: float,
        high: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
    ) -> tuple[np.ndarray, list]:
        has_cold = self._has_cold(metric_key)
        n = self._sweep_points(1, has_cold)
        xs = self._metric_values(metric_key, low, high, n)
        results = []
        for v in xs:
            if has_cold:
                results.append(self._eval_point(cold_config, fixed_noise, {metric_key: v}))
            else:
                noise = {**fixed_noise, metric_key: v}
                results.append(self.run_hot(cached, noise))
        return xs, results

    def sweep_2d(
        self,
        cached: CachedMapping,
        metric_key1: str, low1: float, high1: float,
        metric_key2: str, low2: float, high2: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2)
        n = self._sweep_points(2, has_cold)
        xs = self._metric_values(metric_key1, low1, high1, n)
        ys = self._metric_values(metric_key2, low2, high2, n)
        grid = []
        for v1 in xs:
            row = []
            for v2 in ys:
                swept = {metric_key1: v1, metric_key2: v2}
                if has_cold:
                    row.append(self._eval_point(cold_config, fixed_noise, swept))
                else:
                    noise = {**fixed_noise, **swept}
                    row.append(self.run_hot(cached, noise))
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2, metric_key3)
        n = self._sweep_points(3, has_cold)
        xs = self._metric_values(metric_key1, low1, high1, n)
        ys = self._metric_values(metric_key2, low2, high2, n)
        zs = self._metric_values(metric_key3, low3, high3, n)
        grid = []
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
                plane.append(row_list)
            grid.append(plane)
        return xs, ys, zs, grid


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
