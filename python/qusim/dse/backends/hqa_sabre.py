"""
HQA + SABRE cold-path backend.

Calls the high-level :func:`qusim.map_circuit` which runs HQA initial
mapping followed by SABRE swap insertion. SABRE-injected swaps come
back on ``QusimResult.sparse_swaps`` and *must* be carried into the
hot-path noise estimator — without them, intra-core SWAP cost is
silently dropped and the algorithm can't be compared to TeleSABRE on
the noise sweep.
"""

from __future__ import annotations

import time

import numpy as np

import qusim
from qusim.hqa.placement import InitialPlacement

from ..circuits import _build_circuit, _transpile_circuit
from ..noise import _make_gate_arrays
from ..results import CachedMapping
from ..topology import _build_topology, _compute_distance_matrix


class HqaSabreBackend:
    name = "hqa_sabre"

    def compile(
        self,
        cold_cfg: dict,
        noise: dict,
        config_key: tuple,
    ) -> CachedMapping:
        t0 = time.time()

        seed = cold_cfg["seed"]
        circ = _build_circuit(
            cold_cfg["circuit_type"],
            cold_cfg["num_logical_qubits"],
            seed,
            qasm_str=cold_cfg.get("custom_qasm"),
        )
        transp = _transpile_circuit(circ, seed)

        full_coupling_map, core_mapping = _build_topology(
            cold_cfg["num_qubits"],
            cold_cfg["num_cores"],
            cold_cfg["topology_type"],
            intracore_topology=cold_cfg.get("intracore_topology", "all_to_all"),
            communication_qubits=cold_cfg.get("communication_qubits", 1),
            buffer_qubits=cold_cfg.get("buffer_qubits", 1),
        )

        placement = (
            InitialPlacement.SPECTRAL_CLUSTERING
            if cold_cfg.get("placement_policy") == "spectral"
            else InitialPlacement.RANDOM
        )

        result = qusim.map_circuit(
            circuit=transp,
            full_coupling_map=full_coupling_map,
            core_mapping=core_mapping,
            seed=seed,
            initial_placement=placement,
            single_gate_error=noise["single_gate_error"],
            two_gate_error=noise["two_gate_error"],
            teleportation_error_per_hop=noise["teleportation_error_per_hop"],
            single_gate_time=noise["single_gate_time"],
            two_gate_time=noise["two_gate_time"],
            teleportation_time_per_hop=noise["teleportation_time_per_hop"],
            epr_error_per_hop=noise["epr_error_per_hop"],
            measurement_error=noise["measurement_error"],
            t1=noise["t1"],
            t2=noise["t2"],
            dynamic_decoupling=False,
            readout_mitigation_factor=noise["readout_mitigation_factor"],
            classical_link_width=int(noise["classical_link_width"]),
            classical_clock_freq_hz=noise["classical_clock_freq_hz"],
            classical_routing_cycles=int(noise["classical_routing_cycles"]),
        )

        from qusim import _qiskit_circ_to_sparse_list
        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, noise)

        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(
            full_coupling_map, core_mapping, num_cores_actual,
        )

        sparse_swaps = (
            result.sparse_swaps
            if getattr(result, "sparse_swaps", None) is not None
            else np.zeros((0, 3), dtype=np.int32)
        )

        return CachedMapping(
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
            cold_time_s=time.time() - t0,
        )
