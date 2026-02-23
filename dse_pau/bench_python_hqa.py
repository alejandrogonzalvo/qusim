import json
import sparse
import numpy as np
import pytest
from HQA import HQA_variation

def load_test_case(name):
    with open("dse_pau/test_vectors.json", "r") as f:
        data = json.load(f)
    test_case = data[name]
    num_virtual_qubits = test_case["num_virtual_qubits"]
    num_cores = test_case["num_cores"]
    input_circuit = test_case["input_circuit"]
    
    # Reconstruct Gs sparse representation
    gs_list = []
    for slice_edges in input_circuit:
        g_dense = np.zeros((num_virtual_qubits, num_virtual_qubits))
        for u, v, w in slice_edges:
            g_dense[int(u), int(v)] = w
        gs_list.append(sparse.COO.from_numpy(g_dense))
    
    gs = sparse.stack(gs_list)
        
    ps_init = np.array(test_case["input_initial_partition"])
    num_slices = len(gs)
    ps = np.zeros((num_slices + 1, num_virtual_qubits), dtype=int)
    ps[0] = ps_init
    
    core_capacities = test_case["input_core_capacities"]
    distance_matrix = np.array(test_case["input_distance_matrix"])
    
    return gs, ps, num_cores, num_virtual_qubits, core_capacities, distance_matrix

def test_python_hqa_all_to_all_25q(benchmark):
    gs, ps, num_cores, num_virtual_qubits, core_capacities, distance_matrix = load_test_case("hqa_test_qft_25_all_to_all")
    benchmark.pedantic(HQA_variation, args=(gs, ps.copy(), num_cores, num_virtual_qubits, core_capacities.copy()), kwargs={'distance_matrix': distance_matrix}, rounds=10, iterations=1)

def test_python_hqa_ring_25q(benchmark):
    gs, ps, num_cores, num_virtual_qubits, core_capacities, distance_matrix = load_test_case("hqa_test_qft_25_ring")
    benchmark.pedantic(HQA_variation, args=(gs, ps.copy(), num_cores, num_virtual_qubits, core_capacities.copy()), kwargs={'distance_matrix': distance_matrix}, rounds=10, iterations=1)

def test_python_hqa_large_cores_25q(benchmark):
    gs, ps, num_cores, num_virtual_qubits, core_capacities, distance_matrix = load_test_case("hqa_test_qft_25_large_cores")
    benchmark.pedantic(HQA_variation, args=(gs, ps.copy(), num_cores, num_virtual_qubits, core_capacities.copy()), kwargs={'distance_matrix': distance_matrix}, rounds=10, iterations=1)
