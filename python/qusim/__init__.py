import numpy as np
from dataclasses import dataclass
from typing import List, Union, Optional
import qiskit
from qiskit.converters import circuit_to_dag, dag_to_circuit

# Import the compiled Rust extension
try:
    from qusim.rust_core import map_and_estimate
except ImportError:
    import warnings
    warnings.warn("qusim.rust_core not found. Ensure you have built the maturin extension.")

@dataclass
class QusimResult:
    """
    Results from a qusim mapping and fidelity estimation pass.
    """
    execution_success: bool
    placements: np.ndarray  # Shape: (num_layers + 1, num_qubits)
    total_teleportations: int
    total_epr_pairs: int
    total_network_distance: int
    teleportations_per_slice: List[int]
    
    operational_fidelity: float
    coherence_fidelity: float
    overall_fidelity: float
    total_circuit_time_ns: float
    
    operational_fidelity_grid: np.ndarray  # Shape: (num_layers, num_qubits)
    coherence_fidelity_grid: np.ndarray    # Shape: (num_layers, num_qubits)

    def get_qubit_fidelity_over_time(self, qubit: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the (operational, coherence) fidelity curves for a specific qubit
        across all timeslices.
        """
        op_curve = self.operational_fidelity_grid[:, qubit]
        coh_curve = self.coherence_fidelity_grid[:, qubit]
        return op_curve, coh_curve


def _qiskit_circ_to_sparse_list(circ: qiskit.QuantumCircuit) -> np.ndarray:
    """
    Parses a Qiskit circuit's DAG into layers, and extracts interaction edges.
    Returns a flat (E, 4) numpy array of [layer, q1, q2, weight].
    """
    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    
    edges = []
    
    for layer_idx, layer_as_circuit in enumerate(layers):
        for instruction in layer_as_circuit:
            if instruction.operation.num_qubits == 2:
                q0 = circ.find_bit(instruction.qubits[0]).index
                q1 = circ.find_bit(instruction.qubits[1]).index
                edges.append([float(layer_idx), float(q0), float(q1), 1.0])
                edges.append([float(layer_idx), float(q1), float(q0), 1.0])
            elif instruction.operation.num_qubits == 1:
                q0 = circ.find_bit(instruction.qubits[0]).index
                edges.append([float(layer_idx), float(q0), float(q0), 1.0])
                
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.float64)
    
    return np.array(edges, dtype=np.float64)


def _all_to_all_topology(n: int) -> np.ndarray:
    dist = np.ones((n, n), dtype=np.int32)
    np.fill_diagonal(dist, 0)
    return dist


def map_circuit(
    circuit: qiskit.QuantumCircuit,
    num_cores: int,
    qubits_per_core: Union[int, List[int]],
    distance_matrix: Optional[np.ndarray] = None,
    # Hardware defaults
    single_gate_error: float = 1e-4,
    two_gate_error: float = 1e-3,
    teleportation_error_per_hop: float = 1e-2,
    single_gate_time: float = 20.0,
    two_gate_time: float = 100.0,
    teleportation_time_per_hop: float = 1000.0,
    t1: float = 100_000.0,
    t2: float = 50_000.0,
) -> QusimResult:
    """
    Map a Qiskit quantum circuit using HQA and estimate teleportation routing and hardware fidelity.

    Args:
        circuit: A Qiskit QuantumCircuit.
        num_cores: Number of physical communication cores available.
        qubits_per_core: Number of qubits in each core (if int, assumed uniform for all cores).
        distance_matrix: Custom core-to-core network distance matrix. If None, assumes all-to-all (distance 1 for all).

    Returns:
        QusimResult dataclass containing mapping arrays and granular fidelity grids.
    """
    # 1. Parse Circuit to NumPy slices
    gs_sparse = _qiskit_circ_to_sparse_list(circuit)

    # 2. Setup inputs
    if isinstance(qubits_per_core, int):
        core_caps = np.array([qubits_per_core] * num_cores, dtype=np.uint64)
    else:
        core_caps = np.array(qubits_per_core, dtype=np.uint64)
        
    if distance_matrix is None:
        dist_mat = _all_to_all_topology(num_cores)
    else:
        dist_mat = np.array(distance_matrix, dtype=np.int32)
        
    num_virtual_qubits = circuit.num_qubits
        
    # Generate random initial partition
    # (Matches original utils.py behavior: random assignment up to core caps)
    part = []
    for c_idx, cap in enumerate(core_caps):
        part.extend([c_idx] * cap)
    part = part[:num_virtual_qubits]
    np.random.shuffle(part)
    initial_partition = np.array(part, dtype=np.int32)

    # 3. Call Rust Engine
    raw_dict = map_and_estimate(
        gs_sparse=gs_sparse,
        initial_partition=initial_partition,
        num_cores=num_cores,
        core_capacities=core_caps,
        distance_matrix=dist_mat,
        single_gate_error=single_gate_error,
        two_gate_error=two_gate_error,
        teleportation_error_per_hop=teleportation_error_per_hop,
        single_gate_time=single_gate_time,
        two_gate_time=two_gate_time,
        teleportation_time_per_hop=teleportation_time_per_hop,
        t1=t1,
        t2=t2
    )

    # 4. Wrap Results
    return QusimResult(
        execution_success=raw_dict.get("execution_success", False),
        placements=raw_dict["placements"],
        total_teleportations=raw_dict["total_teleportations"],
        total_epr_pairs=raw_dict["total_epr_pairs"],
        total_network_distance=raw_dict["total_network_distance"],
        teleportations_per_slice=raw_dict["teleportations_per_slice"],
        
        operational_fidelity=raw_dict["operational_fidelity"],
        coherence_fidelity=raw_dict["coherence_fidelity"],
        overall_fidelity=raw_dict["overall_fidelity"],
        total_circuit_time_ns=raw_dict["total_circuit_time_ns"],
        
        operational_fidelity_grid=raw_dict["operational_fidelity_grid"],
        coherence_fidelity_grid=raw_dict["coherence_fidelity_grid"],
    )
