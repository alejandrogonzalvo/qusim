import pytest
import numpy as np
import random
import os
import sys

from qiskit.circuit.library import QFT
from qiskit import transpile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import qiskit_circ_to_slices, all_to_all_topology, ring_topology
from HQA import HQA_variation

def get_test_inputs(topology_type="all-to-all"):
    np.random.seed(42)
    random.seed(42)
    
    virtual_qubits = 25
    qubits_per_core = 5
    num_cores = 5
    
    if topology_type == "all-to-all":
        distance_matrix = all_to_all_topology(num_cores)
    else:
        distance_matrix = ring_topology(num_cores)
        
    core_capacities = [qubits_per_core] * num_cores
    
    circ = QFT(virtual_qubits)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], seed_transpiler=42)
    Gs_two, Gs_all = qiskit_circ_to_slices(transp_circ)
    Gs = Gs_all 
    
    part = [i for i in range(num_cores) for _ in range(core_capacities[i])][:virtual_qubits]
    random.shuffle(part)
    Ps = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
    Ps[0] = part
    
    return Gs, Ps, num_cores, virtual_qubits, core_capacities, distance_matrix

def test_benchmark_python_hqa_all_to_all(benchmark):
    Gs, Ps, num_cores, virtual_qubits, core_capacities, distance_matrix = get_test_inputs("all-to-all")
    result = benchmark.pedantic(HQA_variation, args=(Gs, Ps.copy(), num_cores, virtual_qubits, core_capacities.copy()), kwargs={'distance_matrix': distance_matrix}, rounds=5)
    assert result is not None

def test_benchmark_python_hqa_ring(benchmark):
    Gs, Ps, num_cores, virtual_qubits, core_capacities, distance_matrix = get_test_inputs("ring")
    result = benchmark.pedantic(HQA_variation, args=(Gs, Ps.copy(), num_cores, virtual_qubits, core_capacities.copy()), kwargs={'distance_matrix': distance_matrix}, rounds=5)
    assert result is not None
