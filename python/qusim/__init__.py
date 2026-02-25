import numpy as np
from dataclasses import dataclass
from typing import List, Union, Optional
import qiskit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qusim.orchestrator import MultiCoreOrchestrator

# Import the compiled Rust extension
try:
    from qusim.rust_core import map_and_estimate, estimate_hardware_fidelity
except ImportError:
    import warnings
    warnings.warn("qusim.rust_core not found. Ensure you have built the maturin extension.")

@dataclass
class QusimResult:
    """
    Data capsule returned by the Rust engine after completing the HQA mapping,
    routing schedule, and hardware noise estimation calculations.

    All grid arrays are natively returned via zero-copy bindings making them
    ideal for direct charting/profiling with Matplotlib or Seaborn.
    """
    execution_success: bool
    """True if mapping completed successfully bounding all core capacities."""

    placements: np.ndarray
    """Physical core indices denoting where each virtual qubit resides at every layer step of the circuit. Shape: `(num_layers + 1, num_qubits)`"""

    total_teleportations: int
    """Number of inter-core teleportations executed during the routing process."""

    total_epr_pairs: int
    """Total EPR pairs consumed to accommodate teleportation. Equivalent to teleportations executed scaled by distance distance."""

    total_network_distance: int
    """Cumulative distance of all communication tasks over the specified `distance_matrix` topology."""

    teleportations_per_slice: List[int]
    """Array denoting the varying teleportation load per timeslice."""
    
    algorithmic_fidelity: float
    """Global reliability tracking purely 1Q and 2Q native gates present in the logical user circuit."""

    routing_fidelity: float
    """Global reliability modeling overhead loss from HQA teleportations and SABRE SWAPs."""

    coherence_fidelity: float
    """Global reliability modeling solely thermal decay profiles (T1/T2 times)."""

    overall_fidelity: float
    """Total circuit execution fidelity (operational * coherence multiplied)."""

    total_circuit_time_ns: float
    """Total simulated runtime spanning latency values specified."""
    
    algorithmic_fidelity_grid: np.ndarray  
    """Floating point progression tracking logical algorithmic gate errors isolated across time. Shape: `(num_layers, num_qubits)`"""

    routing_fidelity_grid: np.ndarray  
    """Floating point progression tracking compilation-introduced physical routing errors isolated across time. Shape: `(num_layers, num_qubits)`"""

    coherence_fidelity_grid: np.ndarray    
    """Floating point exponential temporal decay matrix modeling phase and relaxation drops. Shape: `(num_layers, num_qubits)`"""

    def get_qubit_fidelity_over_time(self, qubit: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts the (algorithmic, routing, coherence) fidelity curves for a specific virtual qubit
        across all timeslices to graph hardware impacts dynamically.
        """
        algo_curve = self.algorithmic_fidelity_grid[:, qubit]
        route_curve = self.routing_fidelity_grid[:, qubit]
        coh_curve = self.coherence_fidelity_grid[:, qubit]
        return algo_curve, route_curve, coh_curve


from qusim.hqa.placement import InitialPlacement, PlacementConfig, generate_initial_placement

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
            elif instruction.operation.num_qubits == 1:
                q0 = circ.find_bit(instruction.qubits[0]).index
                edges.append([float(layer_idx), float(q0), float(q0), 1.0])
                
    if len(edges) == 0:
        return np.empty((0, 4), dtype=np.float64)
    
    return np.array(edges, dtype=np.float64)


def _all_to_all_topology(n: int) -> np.ndarray:
    """Fallback generator yielding an all-to-all topology structure of distance 1 hops."""
    dist = np.ones((n, n), dtype=np.int32)
    np.fill_diagonal(dist, 0)
    return dist


def map_circuit(
    circuit: qiskit.QuantumCircuit,
    full_coupling_map: qiskit.transpiler.CouplingMap,
    core_mapping: dict[int, int],
    seed: Optional[int] = None,
    initial_placement: InitialPlacement = InitialPlacement.RANDOM,
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

    This function calls the underlying compiled Rust engine for execution. The circuit is serialized
    into a sparse interaction representation native to Numpy and fed across the ABI zero-copy layer
    for nanosecond scalability profiling.

    Args:
        circuit (qiskit.QuantumCircuit): Qiskit QuantumCircuit comprising of your algorithms gate-suite.
        full_coupling_map (qiskit.transpiler.CouplingMap): Hardware graph of the entire multi-core system.
        core_mapping (dict[int, int]): Dictionary assigning physical qubit indices to core IDs.
        seed (Optional[int]): If provided, ensures deterministic random seeding for the HQA initial state partition.
        initial_placement (InitialPlacement): Policy for generating the layer 0 assignment heuristic.
        
        single_gate_error (float): 1Q physical fidelity loss rate.
        two_gate_error (float): 2Q local operational fidelity loss rate.
        teleportation_error_per_hop (float): Inter-core spatial fidelity degradation factor crossing distance parameters.
        single_gate_time (float): Nanosecond processing overhead per 1Q operation.
        two_gate_time (float): Nanosecond latency duration for processing local entanglements.
        teleportation_time_per_hop (float): Communication EPR networking traversal nanoseconds dictating `T1`/`T2` accumulation scaling rates.
        t1 (float): Hardware T1 characteristic exponential scaling time constant.
        t2 (float): Hardware T2 decoherence characteristic timeline.

    Returns:
        QusimResult: Structured simulation payload metrics including multidimensional matrices isolating individual qubit drop footprints across layers.
    """
    # 1. Parse Circuit to NumPy slices
    gs_sparse = _qiskit_circ_to_sparse_list(circuit)

    # 2. Setup inputs
    num_cores = max(core_mapping.values()) + 1 if core_mapping else 0
    core_caps = np.zeros(num_cores, dtype=np.uint64)
    for q, c in core_mapping.items():
        core_caps[c] += 1
        
    # Derive distance matrix from full_coupling_map
    core_adj = np.zeros((num_cores, num_cores), dtype=np.int32)
    for edge in full_coupling_map.get_edges():
        c1 = core_mapping[edge[0]]
        c2 = core_mapping[edge[1]]
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
        
    num_virtual_qubits = circuit.num_qubits
        
    config = PlacementConfig(
        policy=initial_placement,
        interaction_tensor=gs_sparse,
        num_virtual_qubits=num_virtual_qubits,
        core_caps=core_caps,
        seed=seed
    )
    initial_partition = generate_initial_placement(config)

    # 3. Call Rust Engine for HQA pass and base estimations
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

    placements = raw_dict["placements"]
    num_layers = placements.shape[0] - 1
    sparse_swaps_list = []

    # 4. Sabre Orchestration Pass
    if full_coupling_map is not None:
        
        # HQA list is indexed [timeslice][qubit] -> core_id
        hqa_list = placements.tolist()
        orchestrator = MultiCoreOrchestrator(num_cores, full_coupling_map, core_mapping, dist_mat)
        
        # Run orchestrator and grab the exact logical timestamps of where SABRE injected SWAPs!
        _, sparse_swaps_list = orchestrator.orchestrate(circuit, hqa_list)

    # 5. Fast-Path Fidelity Re-Estimation (incorporates newly resolved SWAPs natively)
    if not sparse_swaps_list:
        sparse_swaps_arr = np.zeros((0, 3), dtype=np.int32)
    else:
        sparse_swaps_arr = np.array(sparse_swaps_list, dtype=np.int32)

    fidelity_dict = estimate_hardware_fidelity(
        gs_sparse=gs_sparse,
        placements=placements,
        distance_matrix=dist_mat,
        sparse_swaps=sparse_swaps_arr,
        single_gate_error=single_gate_error,
        two_gate_error=two_gate_error,
        teleportation_error_per_hop=teleportation_error_per_hop,
        single_gate_time=single_gate_time,
        two_gate_time=two_gate_time,
        teleportation_time_per_hop=teleportation_time_per_hop,
        t1=t1,
        t2=t2
    )

    # 6. Wrap Results
    return QusimResult(
        execution_success=raw_dict.get("execution_success", False),
        placements=placements,
        total_teleportations=raw_dict["total_teleportations"],
        total_epr_pairs=raw_dict["total_epr_pairs"],
        total_network_distance=raw_dict["total_network_distance"],
        teleportations_per_slice=raw_dict["teleportations_per_slice"],
        
        algorithmic_fidelity=fidelity_dict["algorithmic_fidelity"],
        routing_fidelity=fidelity_dict["routing_fidelity"],
        coherence_fidelity=fidelity_dict["coherence_fidelity"],
        overall_fidelity=fidelity_dict["overall_fidelity"],
        total_circuit_time_ns=fidelity_dict["total_circuit_time_ns"],
        
        algorithmic_fidelity_grid=fidelity_dict["algorithmic_fidelity_grid"],
        routing_fidelity_grid=fidelity_dict["routing_fidelity_grid"],
        coherence_fidelity_grid=fidelity_dict["coherence_fidelity_grid"],
    )
