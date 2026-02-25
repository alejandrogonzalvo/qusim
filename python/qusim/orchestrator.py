import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import SabreSwap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from typing import List
import numpy as np

# =====================================================================
# Step 1: Custom Teleportation Primitives
# =====================================================================

class TeledataSend(Instruction):
    """Placeholder instructing a core to send a qubit state to a destination core."""
    def __init__(self, dst_core: int):
        super().__init__("teledata_send", 1, 0, [dst_core])

class TeledataRecv(Instruction):
    """Placeholder instructing a core to receive a qubit state from a source core."""
    def __init__(self, src_core: int):
        super().__init__("teledata_recv", 1, 0, [src_core])

class TelegateHalf(Instruction):
    """Placeholder representing one half of a non-local gate execution."""
    def __init__(self, peer_core: int, orig_op: Instruction, is_control: bool):
        self.peer_core = peer_core
        self.orig_op = orig_op
        self.is_control = is_control
        super().__init__(f"telegate_half_{orig_op.name}", 1, 0, [peer_core, int(is_control)])

# =====================================================================
# Main Orchestrator Pipeline
# =====================================================================

class MultiCoreOrchestrator:
    def __init__(self, num_cores: int, full_coupling_map: CouplingMap, core_mapping: dict[int, int], distance_matrix: np.ndarray):
        self.num_cores = num_cores
        self.full_coupling_map = full_coupling_map
        self.core_mapping = core_mapping
        self.distance_matrix = distance_matrix
        
        # Build local core topologies from the global graph
        # For SabreSwap to route a 30-qubit circuit where 10 qubits might visit a 6-qubit core,
        # we CANNOT use a 6-node CouplingMap (Sabre requires 1:1 mapping mapping for all 30 qubits simultaneously).
        # We solve this by passing a "Star-Graph Padded CouplingMap" for each core:
        # The core's internal edges remain.
        # Edges between this core and foreign cores remain (communication links).
        # All extra nodes (foreign cores) are stripped of their internal edges, acting as "parking space"
        # for inactive virtual qubits. SABRE routes them in/out of the core through valid communication links!
        self.core_topologies = []
        
        nq = max([max(edge) for edge in full_coupling_map.get_edges()]) + 1 if full_coupling_map.get_edges() else len(core_mapping)
        
        for c in range(num_cores):
            cmap = CouplingMap()
            for i in range(nq):
                cmap.add_physical_qubit(i)
                
            c_nodes = [p for p, core in core_mapping.items() if core == c]
            
            # Add all internal edges
            for edge in full_coupling_map.get_edges():
                if core_mapping[edge[0]] == c and core_mapping[edge[1]] == c:
                    cmap.add_edge(edge[0], edge[1])

            # Find boundary nodes of c in the full topology
            boundary_nodes = []
            for node in c_nodes:
                if any(core_mapping[neighbor] != c for neighbor in full_coupling_map.neighbors(node)):
                    boundary_nodes.append(node)
                    
            if not boundary_nodes:
                boundary_nodes = c_nodes # Fallback
                
            # Connect ALL foreign nodes to the boundary nodes!
            # This ensures they are in the same component and routes them through the physical boundary
            for i in range(nq):
                if i not in c_nodes:
                    for b in boundary_nodes:
                        cmap.add_edge(i, b)
                        cmap.add_edge(b, i)
                        
            self.core_topologies.append(cmap)

    def orchestrate(self, global_circuit: QuantumCircuit, hqa_mapping: List[List[int]]) -> QuantumCircuit:
        """
        Executes the two-stage Global (HQA) to Local (SABRE) compilation pass.
        """
        # Step 2: Global Circuit Partitioning
        sub_circuits = self._partition_circuit(global_circuit, hqa_mapping)
        
        # Step 3: Local Intra-Core Routing (LightSABRE)
        routed_sub_circuits = self._route_locals(sub_circuits)
        
        # Step 4: Reassembly & Stitching
        return self._stitch_circuits(routed_sub_circuits, global_circuit)

    def _partition_circuit(self, global_circuit: QuantumCircuit, hqa_mapping: List[List[int]]) -> List[QuantumCircuit]:
        """
        Splits the global DAG into C local DAGs, injecting physical I/O primitives 
        wherever the HQA mapping demands a cross-core boundary event.
        """
        from qiskit.transpiler import Layout
        self.global_layout_dict = {}
        core_phys_available = {c: sorted([p for p, core in self.core_mapping.items() if core == c]) for c in range(self.num_cores)}
        for q in range(global_circuit.num_qubits):
            c = hqa_mapping[0][q]
            if len(core_phys_available[c]) > 0:
                p = core_phys_available[c].pop(0)
                self.global_layout_dict[global_circuit.qubits[q]] = p
            else:
                # Fallback if oversubscribed (shouldn't happen with valid HQA)
                self.global_layout_dict[global_circuit.qubits[q]] = 0

        dag = circuit_to_dag(global_circuit)
        layers = list(dag.layers())
        
        # Allocate one local DAG per core. 
        # (Note: For brevity we map the entire global register into each sub-DAG. 
        # In a strict SABRE implementation, we would narrow this to only active physical qubits).
        sub_dags = [DAGCircuit() for _ in range(self.num_cores)]
        for c in range(self.num_cores):
            for qreg in global_circuit.qregs:
                sub_dags[c].add_qreg(qreg)
            for creg in global_circuit.cregs:
                sub_dags[c].add_creg(creg)
                
        for t, layer in enumerate(layers):
            current_mapping = hqa_mapping[t]
            
            # Detect inter-core movement (Teledata) between slice t-1 and t
            if t > 0:
                prev_mapping = hqa_mapping[t-1]
                for q_idx in range(global_circuit.num_qubits):
                    src_core, dst_core = prev_mapping[q_idx], current_mapping[q_idx]
                    if src_core != dst_core:
                        qubit = global_circuit.qubits[q_idx]
                        
                        # Find a magnet qubit in dst_core to pull this qubit to the boundary
                        magnets_in_dst = [i for i, c in enumerate(prev_mapping) if c == dst_core]
                        if magnets_in_dst:
                            magnet_q = global_circuit.qubits[magnets_in_dst[0]]
                            dummy_cx = Instruction("dummy_cx", 2, 0, [])
                            # Add the "pull" to src_core before it leaves
                            sub_dags[src_core].apply_operation_back(dummy_cx, [qubit, magnet_q])
                            # Add the "pull" to dst_core as it enters
                            sub_dags[dst_core].apply_operation_back(dummy_cx, [qubit, magnet_q])
                            
                        # Inject corresponding I/O primitives
                        sub_dags[src_core].apply_operation_back(TeledataSend(dst_core), [qubit])
                        sub_dags[dst_core].apply_operation_back(TeledataRecv(src_core), [qubit])
                        
            # Distribute the layer's computational gates to appropriate cores
            for node in layer['graph'].op_nodes():
                qargs = node.qargs
                cores_involved = [current_mapping[global_circuit.find_bit(q).index] for q in qargs]
                
                if len(set(cores_involved)) == 1: 
                    # Local gate: executes entirely on a single core
                    core = cores_involved[0]
                    sub_dags[core].apply_operation_back(node.op, qargs, node.cargs)
                else: 
                    # Inter-core gate: must be split into two synchronized halves
                    core_a, core_b = cores_involved[0], cores_involved[1]
                    q_a, q_b = qargs[0], qargs[1]
                    
                    sub_dags[core_a].apply_operation_back(TelegateHalf(core_b, node.op, True), [q_a])
                    sub_dags[core_b].apply_operation_back(TelegateHalf(core_a, node.op, False), [q_b])
                    
        return [dag_to_circuit(d) for d in sub_dags]

    def _route_locals(self, sub_circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
        """
        Executes Qiskit's SabreRouting pass heavily constrained by the physical topologies.
        """
        from qiskit.transpiler import Layout
        from qiskit.transpiler.passes import SetLayout
        routed_circuits = []
        for i, circ in enumerate(sub_circuits):
            if not circ.data:
                routed_circuits.append(circ)
                continue
                
            # Configure Sabre to respect the core's constrained topology.
            layout = Layout(self.global_layout_dict)
            pm = PassManager([
                SetLayout(layout),
                SabreSwap(coupling_map=self.core_topologies[i], heuristic='basic', seed=42)
            ])
            try:
                routed = pm.run(circ)
                # Strip out the dummy_cx gates we injected for routing attraction
                cleaned_routed = QuantumCircuit(*routed.qregs, *routed.cregs)
                for inst in routed.data:
                    if inst.operation.name != "dummy_cx":
                        cleaned_routed.append(inst)
                routed_circuits.append(cleaned_routed)
            except Exception as e:
                print(f"Warning: SABRE routing failed on core {i}: {e}")
                # Fallback if SABRE throws on unmapped virtual qubits (would require dynamic Layout in prod)
                routed_circuits.append(circ)
                
        return routed_circuits

    def _stitch_circuits(self, routed_sub_circuits: List[QuantumCircuit], ref_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Merges C locally routed circuits into a synchronized global DAG.
        Matches TelegateHalf and Teledata boundaries to re-emit 2Q global interactions
        without violating causality.
        """
        global_dag = DAGCircuit()
        for qreg in ref_circuit.qregs:
            global_dag.add_qreg(qreg)
        for creg in ref_circuit.cregs:
            global_dag.add_creg(creg)
            
        # Convert all routed circuits back to DAGs to allow peeking at their structural Front Layer
        sub_dags = [circuit_to_dag(c) for c in routed_sub_circuits]
        
        while any(d.op_nodes() for d in sub_dags):
            progress_made = False
            
            for c in range(self.num_cores):
                dag = sub_dags[c]
                
                # Check nodes sitting at the causal origin of this core
                for node in dag.front_layer():
                    if node.type != 'op':
                        continue
                        
                    op = node.op
                    if isinstance(op, (TeledataSend, TeledataRecv)):
                        # Look for matching peer receiver/sender on the destination core
                        peer_core = op.params[0]
                        peer_dag = sub_dags[peer_core]
                        expected_class = TeledataRecv if isinstance(op, TeledataSend) else TeledataSend
                        
                        match = next((n for n in peer_dag.front_layer() 
                                      if n.type == 'op' and isinstance(n.op, expected_class) and n.op.params[0] == c), None)
                        
                        if match:
                            # Both ends of the movement are causally ready! Insert synchronization barrier
                            barrier = Instruction("teleport_sync", 2, 0, [])
                            global_dag.apply_operation_back(barrier, [node.qargs[0], match.qargs[0]])
                            
                            # Pop both nodes from their respective core pipelines
                            dag.remove_op_node(node)
                            peer_dag.remove_op_node(match)
                            progress_made = True
                            
                    elif isinstance(op, TelegateHalf):
                        peer_core = op.peer_core
                        peer_dag = sub_dags[peer_core]
                        
                        match = next((n for n in peer_dag.front_layer() 
                                      if n.type == 'op' and isinstance(n.op, TelegateHalf) and n.op.peer_core == c), None)
                        
                        if match:
                            # Both sides of the non-local gate are ready. Reconstruct the global operation.
                            # Ensure the control and target QArgs are pushed in the correct original order!
                            orig_op = op.orig_op
                            qargs = [node.qargs[0], match.qargs[0]] if op.is_control else [match.qargs[0], node.qargs[0]]
                            
                            global_dag.apply_operation_back(orig_op, qargs)
                            
                            dag.remove_op_node(node)
                            peer_dag.remove_op_node(match)
                            progress_made = True
                            
                    else:
                        # Standard local gate (either an original gate or a SABRE-inserted SWAP)
                        # No synchronization required; safe to pop immediately
                        global_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                        dag.remove_op_node(node)
                        progress_made = True
                        
            # Deadlock prevention via graph cycle detection
            if not progress_made and any(d.op_nodes() for d in sub_dags):
                raise RuntimeError("Deadlock detected during DAG restitching. A causal dependency cycle exists between cores.")
                
        return dag_to_circuit(global_dag)
