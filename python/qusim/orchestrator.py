import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Instruction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import SabreSwap, SetLayout
from typing import List
import numpy as np
import copy

# =====================================================================
# Custom Teleportation Primitives
# =====================================================================

class TeledataSend(Instruction):
    """Placeholder instructing a core to send a qubit state to a destination core."""
    def __init__(self, dst_core: int, layer_idx: int):
        self.layer_idx = layer_idx
        super().__init__("teledata_send", 1, 0, [dst_core])

class TeledataRecv(Instruction):
    """Placeholder instructing a core to receive a qubit state from a source core."""
    def __init__(self, src_core: int, layer_idx: int):
        self.layer_idx = layer_idx
        super().__init__("teledata_recv", 1, 0, [src_core])

class TelegateHalf(Instruction):
    """Placeholder representing one half of a non-local gate execution."""
    def __init__(self, peer_core: int, orig_op: Instruction, is_control: bool, layer_idx: int):
        self.peer_core = peer_core
        self.orig_op = orig_op
        self.is_control = is_control
        self.layer_idx = layer_idx
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
        
        self.core_topologies = self._build_core_padded_topologies()
        self.global_layout_dict = {}

    def orchestrate(self, global_circuit: QuantumCircuit, hqa_mapping: List[List[int]]) -> tuple[QuantumCircuit, List[List[int]]]:
        """
        Executes the two-stage Global (HQA) to Local (SABRE) compilation pass.
        Returns None for the circuit (stitching disabled to prevent SABRE deadlocks), 
        and a sparse list of exactly when/where SABRE SWAPs occurred.
        """
        sub_circuits = self._partition_circuit(global_circuit, hqa_mapping)
        routed_sub_circuits = self._route_locals(sub_circuits)
        sparse_swaps = self._extract_sparse_swaps(global_circuit, routed_sub_circuits)
        return None, sparse_swaps

    # =====================================================================
    # Topology Building
    # =====================================================================

    def _build_core_padded_topologies(self) -> List[CouplingMap]:
        """
        Builds local core topologies from the global graph with star-graph padding 
        for foreign cores to act as parking space for inactive virtual qubits.
        """
        topologies = []
        nq = max([max(edge) for edge in self.full_coupling_map.get_edges()]) + 1 if self.full_coupling_map.get_edges() else len(self.core_mapping)
        
        for c in range(self.num_cores):
            topologies.append(self._build_single_core_topology(c, nq))
            
        return topologies
        
    def _build_single_core_topology(self, core_idx: int, nq: int) -> CouplingMap:
        cmap = CouplingMap()
        for i in range(nq):
            cmap.add_physical_qubit(i)
            
        core_nodes = [p for p, core in self.core_mapping.items() if core == core_idx]
        
        for u, v in self.full_coupling_map.get_edges():
            if self.core_mapping[u] == core_idx and self.core_mapping[v] == core_idx:
                cmap.add_edge(u, v)

        boundary_nodes = [
            node for node in core_nodes
            if any(self.core_mapping[neighbor] != core_idx for neighbor in self.full_coupling_map.neighbors(node))
        ]
        
        if not boundary_nodes:
            boundary_nodes = core_nodes
            
        for i in range(nq):
            if i in core_nodes:
                continue
            for b in boundary_nodes:
                cmap.add_edge(i, b)
                cmap.add_edge(b, i)
                
        return cmap

    # =====================================================================
    # Partitioning
    # =====================================================================

    def _partition_circuit(self, global_circuit: QuantumCircuit, hqa_mapping: List[List[int]]) -> List[QuantumCircuit]:
        """
        Splits the global DAG into C local DAGs, injecting physical I/O primitives.
        """
        self._build_global_layout(global_circuit, hqa_mapping)
        
        dag = circuit_to_dag(global_circuit)
        sub_dags = self._init_sub_dags(global_circuit)
                
        for t, layer in enumerate(dag.layers()):
            if t > 0:
                self._inject_teleportation_if_moved(sub_dags, global_circuit, hqa_mapping[t-1], hqa_mapping[t], t)
                
            self._distribute_layer_ops(sub_dags, global_circuit, layer, hqa_mapping[t], t)
                    
        return [dag_to_circuit(d) for d in sub_dags]

    def _build_global_layout(self, global_circuit: QuantumCircuit, hqa_mapping: List[List[int]]):
        self.global_layout_dict = {}
        core_phys_available = {c: sorted([p for p, core in self.core_mapping.items() if core == c]) for c in range(self.num_cores)}
        for q in range(global_circuit.num_qubits):
            c = hqa_mapping[0][q]
            if len(core_phys_available[c]) > 0:
                p = core_phys_available[c].pop(0)
                self.global_layout_dict[global_circuit.qubits[q]] = p
            else:
                self.global_layout_dict[global_circuit.qubits[q]] = 0

    def _init_sub_dags(self, global_circuit: QuantumCircuit) -> List[DAGCircuit]:
        sub_dags = [DAGCircuit() for _ in range(self.num_cores)]
        for c in range(self.num_cores):
            for qreg in global_circuit.qregs:
                sub_dags[c].add_qreg(qreg)
            for creg in global_circuit.cregs:
                sub_dags[c].add_creg(creg)
        return sub_dags

    def _inject_teleportation_if_moved(self, sub_dags: List[DAGCircuit], global_circuit: QuantumCircuit, prev_mapping: List[int], current_mapping: List[int], t: int):
        for q_idx in range(global_circuit.num_qubits):
            src_core = prev_mapping[q_idx]
            dst_core = current_mapping[q_idx]
            
            if src_core == dst_core:
                continue
                
            qubit = global_circuit.qubits[q_idx]
            self._inject_magnet_pull(sub_dags, global_circuit, prev_mapping, dst_core, src_core, qubit)
            
            sub_dags[src_core].apply_operation_back(TeledataSend(dst_core, t), [qubit])
            sub_dags[dst_core].apply_operation_back(TeledataRecv(src_core, t), [qubit])

    def _inject_magnet_pull(self, sub_dags: List[DAGCircuit], global_circuit: QuantumCircuit, prev_mapping: List[int], dst_core: int, src_core: int, qubit):
        magnets_in_dst = [i for i, c in enumerate(prev_mapping) if c == dst_core]
        if not magnets_in_dst:
            return
            
        magnet_q = global_circuit.qubits[magnets_in_dst[0]]
        dummy_cx = Instruction("dummy_cx", 2, 0, [])
        sub_dags[src_core].apply_operation_back(dummy_cx, [qubit, magnet_q])
        sub_dags[dst_core].apply_operation_back(dummy_cx, [qubit, magnet_q])

    def _distribute_layer_ops(self, sub_dags: List[DAGCircuit], global_circuit: QuantumCircuit, layer, current_mapping: List[int], t: int):
        for node in layer['graph'].op_nodes():
            qargs = node.qargs
            cores_involved = [current_mapping[global_circuit.find_bit(q).index] for q in qargs]
            
            if len(set(cores_involved)) == 1: 
                self._apply_local_gate(sub_dags, node, cores_involved[0], t)
            else: 
                self._apply_intercore_gate(sub_dags, node, cores_involved, t)

    def _apply_local_gate(self, sub_dags: List[DAGCircuit], node: DAGOpNode, core: int, t: int):
        if hasattr(node.op, 'to_mutable'):
            new_op = node.op.to_mutable()
        else:
            new_op = copy.copy(node.op)
            
        if hasattr(new_op, "label") and new_op.label:
            new_op.label = f"{new_op.label}_L{t}"
        else:
            new_op.label = f"L{t}"
        sub_dags[core].apply_operation_back(new_op, node.qargs, node.cargs)

    def _apply_intercore_gate(self, sub_dags: List[DAGCircuit], node: DAGOpNode, cores_involved: List[int], t: int):
        core_a, core_b = cores_involved[0], cores_involved[1]
        q_a, q_b = node.qargs[0], node.qargs[1]
        
        sub_dags[core_a].apply_operation_back(TelegateHalf(core_b, node.op, True, t), [q_a])
        sub_dags[core_b].apply_operation_back(TelegateHalf(core_a, node.op, False, t), [q_b])

    # =====================================================================
    # Routing and Extraction
    # =====================================================================

    def _route_locals(self, sub_circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
        """
        Executes Qiskit's SabreRouting pass heavily constrained by the physical topologies.
        """
        return [self._route_single_core(i, circ) for i, circ in enumerate(sub_circuits)]

    def _route_single_core(self, core_idx: int, circ: QuantumCircuit) -> QuantumCircuit:
        if not circ.data:
            return circ
            
        layout = Layout(self.global_layout_dict)
        pm = PassManager([
            SetLayout(layout),
            SabreSwap(coupling_map=self.core_topologies[core_idx], heuristic='basic', seed=42)
        ])
        
        try:
            routed = pm.run(circ)
            return self._strip_dummy_cx(routed)
        except Exception as e:
            print(f"Warning: SABRE routing failed on core {core_idx}: {e}")
            return circ
            
    def _strip_dummy_cx(self, routed: QuantumCircuit) -> QuantumCircuit:
        cleaned_routed = QuantumCircuit(*routed.qregs, *routed.cregs)
        for inst in routed.data:
            if inst.operation.name != "dummy_cx":
                cleaned_routed.append(inst)
        return cleaned_routed

    def _extract_sparse_swaps(self, global_circuit: QuantumCircuit, routed_sub_circuits: List[QuantumCircuit]) -> List[List[int]]:
        sparse_swaps = []
        for circ in routed_sub_circuits:
            current_layer = 0
            for inst in circ.data:
                current_layer = self._update_current_layer(inst.operation, current_layer)
                swap = self._get_sabre_swap_if_present(global_circuit, inst, current_layer)
                if swap is not None:
                    sparse_swaps.append(swap)
        return sparse_swaps

    def _update_current_layer(self, op: Instruction, current_layer: int) -> int:
        if isinstance(op, (TeledataSend, TeledataRecv, TelegateHalf)):
            return op.layer_idx
            
        if hasattr(op, 'label') and op.label and 'L' in op.label:
            try:
                return int(op.label.split('L')[-1])
            except ValueError:
                pass
                
        return current_layer

    def _get_sabre_swap_if_present(self, global_circuit: QuantumCircuit, inst, current_layer: int) -> List[int] | None:
        op = inst.operation
        if op.name == 'swap' and not (hasattr(op, 'label') and op.label and 'L' in op.label):
            q1 = global_circuit.find_bit(inst.qubits[0]).index
            q2 = global_circuit.find_bit(inst.qubits[1]).index
            return [current_layer, q1, q2]
        return None
