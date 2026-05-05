"""
TeleSABRE cold-path backend.

Wraps :func:`qusim.rust_core.telesabre_map_and_estimate`, which is the
PyO3 binding around the TeleSABRE C library. Two ergonomic concerns
the wrapper handles:

1. The C entry point reads QASM and device topology from JSON files,
   not in-memory objects. We marshal both via temp files.
2. TeleSABRE's placements live in *gate-step* space (one row per
   ts-layer = data + telegate + swap step); the hot-path noise
   estimator works in DAG-layer space. ``_remap_*`` resamples both
   placements and sparse swaps onto the DAG-layer count so the two
   grids align row-for-row.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from ..circuits import _build_circuit, _transpile_circuit
from ..noise import _make_gate_arrays
from ..results import CachedMapping
from ..topology import (
    _build_topology,
    _compute_distance_matrix,
    _max_B_for_layout,
    _max_K_for_layout,
    assign_core_slots,
    inter_core_edges,
    num_comm_groups,
)


# Repository root resolved relative to this file: …/python/qusim/dse/backends
# → parents[3] is the repo root. Used to find the bundled TeleSABRE config.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TELESABRE_CONFIG = (
    _REPO_ROOT / "tests" / "fixtures" / "telesabre" / "configs" / "default.json"
)


class TeleSabreBackend:
    name = "telesabre"

    def compile(
        self,
        cold_cfg: dict,
        noise: dict,
        config_key: tuple,
    ) -> CachedMapping:
        from qiskit import qasm2
        from qusim.rust_core import telesabre_map_and_estimate

        t0 = time.time()
        seed = cold_cfg["seed"]
        num_qubits = cold_cfg["num_qubits"]
        num_cores = cold_cfg["num_cores"]
        topology_type = cold_cfg["topology_type"]
        intracore_topology = cold_cfg.get("intracore_topology", "all_to_all")
        communication_qubits = cold_cfg.get("communication_qubits", 1)
        buffer_qubits = cold_cfg.get("buffer_qubits", 1)
        num_logical_qubits = cold_cfg["num_logical_qubits"]

        circ = _build_circuit(
            cold_cfg["circuit_type"], num_logical_qubits, seed,
            qasm_str=cold_cfg.get("custom_qasm"),
        )
        transp = _transpile_circuit(circ, seed)

        # Marshal QASM + device JSON via temp files (the C entry point
        # only accepts paths, not in-memory buffers).
        qasm_str = qasm2.dumps(transp)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
            f.write(qasm_str)
            qasm_path = f.name

        device_json = _build_telesabre_device_json(
            num_qubits, num_cores, topology_type, intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(device_json, f)
            device_path = f.name

        try:
            raw = telesabre_map_and_estimate(
                circuit_path=qasm_path,
                device_path=device_path,
                config_path=str(_DEFAULT_TELESABRE_CONFIG),
                single_gate_error=noise["single_gate_error"],
                two_gate_error=noise["two_gate_error"],
                teleportation_error_per_hop=noise["teleportation_error_per_hop"],
                single_gate_time=noise["single_gate_time"],
                two_gate_time=noise["two_gate_time"],
                teleportation_time_per_hop=noise["teleportation_time_per_hop"],
                t1=noise["t1"],
                t2=noise["t2"],
                dynamic_decoupling=bool(noise.get("dynamic_decoupling", False)),
                readout_mitigation_factor=noise["readout_mitigation_factor"],
                classical_link_width=int(noise["classical_link_width"]),
                classical_clock_freq_hz=noise["classical_clock_freq_hz"],
                classical_routing_cycles=int(noise["classical_routing_cycles"]),
            )
        finally:
            os.unlink(qasm_path)
            os.unlink(device_path)

        # Use Qiskit's gs_sparse so the hot path has the correct
        # num_layers (DAG depth) and gate-type structure for the
        # algorithmic / coherence fidelity model.
        from qusim import _qiskit_circ_to_sparse_list
        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, noise)

        dag_layers = (
            int(gs_sparse[:, 0].max()) + 1 if len(gs_sparse) > 0 else 1
        )
        placements = _remap_to_dag_layers(raw["placements"], dag_layers)
        sparse_swaps = _remap_swaps_to_dag_layers(raw["sparse_swaps"], dag_layers)

        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type,
            intracore_topology=intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(
            full_coupling_map, core_mapping, num_cores_actual,
        )

        return CachedMapping(
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
            cold_time_s=time.time() - t0,
        )


def _remap_to_dag_layers(placements: np.ndarray, dag_layers: int) -> np.ndarray:
    """Resample TeleSABRE placements from ts-gate-step space to DAG-layer space.

    TeleSABRE's placements have shape ``(ts_layers + 1, num_qubits)``
    where ``ts_layers = num_teledata + num_telegate + num_swaps``. The
    hot path needs ``(dag_layers + 1, num_qubits)`` to align with the
    Qiskit ``gs_sparse`` row count.
    """
    ts_layers = placements.shape[0] - 1
    num_qubits = placements.shape[1]
    out = np.zeros((dag_layers + 1, num_qubits), dtype=np.int32)
    for l in range(dag_layers + 1):
        ts_l = round(l * ts_layers / dag_layers) if dag_layers > 0 else 0
        out[l] = placements[min(ts_l, ts_layers)]
    return out


def _remap_swaps_to_dag_layers(sparse_swaps: np.ndarray, dag_layers: int) -> np.ndarray:
    """Resample TeleSABRE sparse_swaps layer indices to DAG-layer space."""
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
    """Build a TeleSABRE-format device JSON from DSE topology parameters.

    Two invariants TeleSABRE's C library requires:

    1. **Uniform core sizes** — ``device_from_json`` computes ownership as
       ``phys_to_core[i] = i / core_capacity`` (integer division).
       Non-uniform sizes map the last qubit out of bounds → heap
       corruption → double-free. Fix: pad every core to
       ``ceil(num_qubits / num_cores) + SLACK_PER_CORE``.

    2. **Free teleportation slots** — round-robin initial layout fills
       every physical qubit when ``total_qubits == num_qubits``. With no
       free qubits, all nearest-free heaps are empty → Dijkstra weights
       saturate at ``TS_INF`` → integer overflow → heap corruption.
       Fix: ``SLACK_PER_CORE = 3`` guarantees at least 3 free slots per
       core after initial layout.
    """
    num_cores = max(1, min(num_cores, num_qubits))
    SLACK_PER_CORE = 3
    qubits_per_core = math.ceil(num_qubits / num_cores) + SLACK_PER_CORE
    total_qubits = qubits_per_core * num_cores
    offsets = [c * qubits_per_core for c in range(num_cores)]

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

    inter_edges = []
    if num_cores > 1:
        groups_per_core = num_comm_groups(num_cores, topology_type)
        B_req = max(1, int(buffer_qubits or 1))
        K_caps = [
            _max_K_for_layout(qubits_per_core, groups_per_core[c], b_per_group=B_req)
            for c in range(num_cores) if groups_per_core[c] > 0
        ]
        K_max = min(K_caps) if K_caps else 0
        K = min(int(communication_qubits or 1), max(0, K_max))
        if K >= 1:
            B_caps = [
                _max_B_for_layout(qubits_per_core, groups_per_core[c], K)
                for c in range(num_cores) if groups_per_core[c] > 0
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
