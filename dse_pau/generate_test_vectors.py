import json
import numpy as np
import random
from qiskit.circuit.library import QFT
from qiskit import transpile

import sys
import os
# Ensure we can import from dse_pau
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import qiskit_circ_to_slices, all_to_all_topology, ring_topology
from HQA import HQA_variation

def serialize_gs_flat_sparse(Gs):
    """
    Serializes the sequence of sparse matrices into a single 1D list.
    Format: [layer_idx, q1, q2, weight]
    """
    gs_serialized = []
    for layer_idx, g in enumerate(Gs):
        coords = g.coords
        data = g.data
        for i in range(len(data)):
            # We cast to float so Rust's serde automatically parses them as f64
            layer = float(layer_idx)
            u = float(coords[0][i])
            v = float(coords[1][i])
            w = float(data[i])
            gs_serialized.append([layer, u, v, w])
            
    return gs_serialized

def generate_test_vectors():
    virtual_qubits = 25
    qubits_per_core = 5
    num_cores = 5
    
    print(f"Generating slices for QFT({virtual_qubits})")
    circ = QFT(virtual_qubits)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], seed_transpiler=42)
    Gs_two, Gs_all = qiskit_circ_to_slices(transp_circ)
    
    Gs = Gs_all # Use Gs_all for mapping as in the original script
    gs_sparse = serialize_gs_flat_sparse(Gs)
    num_layers = len(Gs)

    output_dict = {}

    ## Case 1: All-to-All topology
    print("Running HQA Mapping for All-to-All...")
    np.random.seed(42)
    random.seed(42)
    
    dist_matrix_ata = all_to_all_topology(num_cores)
    core_cap_ata = [qubits_per_core] * num_cores
    part_ata = [i for i in range(num_cores) for _ in range(core_cap_ata[i])][:virtual_qubits]
    random.shuffle(part_ata)
    
    Ps_ata = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
    Ps_ata[0] = part_ata

    Ps_HQA_ata = HQA_variation(Gs, Ps_ata.copy(), num_cores, virtual_qubits, core_cap_ata.copy(), distance_matrix=dist_matrix_ata)
    
    output_dict["hqa_test_qft_25_all_to_all"] = {
        "num_virtual_qubits": virtual_qubits,
        "num_cores": num_cores,
        "num_layers": num_layers,
        "gs_sparse": gs_sparse,
        "input_initial_partition": Ps_ata[0].tolist(),
        "input_core_capacities": core_cap_ata,
        "input_distance_matrix": dist_matrix_ata,
        "expected_output": Ps_HQA_ata.tolist()
    }

    ## Case 2: Ring topology
    print("Running HQA Mapping for Ring...")
    np.random.seed(42)
    random.seed(42)
    
    dist_matrix_ring = ring_topology(num_cores)
    core_cap_ring = [qubits_per_core] * num_cores
    part_ring = [i for i in range(num_cores) for _ in range(core_cap_ring[i])][:virtual_qubits]
    random.shuffle(part_ring)
    
    Ps_ring = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
    Ps_ring[0] = part_ring

    Ps_HQA_ring = HQA_variation(Gs, Ps_ring.copy(), num_cores, virtual_qubits, core_cap_ring.copy(), distance_matrix=dist_matrix_ring)
    
    output_dict["hqa_test_qft_25_ring"] = {
        "num_virtual_qubits": virtual_qubits,
        "num_cores": num_cores,
        "num_layers": num_layers,
        "gs_sparse": gs_sparse,
        "input_initial_partition": Ps_ring[0].tolist(),
        "input_core_capacities": core_cap_ring,
        "input_distance_matrix": dist_matrix_ring,
        "expected_output": Ps_HQA_ring.tolist()
    }

    ## Case 3: Large cores (10 qubits per core)
    print("Running HQA Mapping for Large Cores (10 per core)...")
    qubits_per_core_large = 10
    np.random.seed(42)
    random.seed(42)
    
    dist_matrix_large = all_to_all_topology(num_cores)
    core_cap_large = [qubits_per_core_large] * num_cores
    # Initial partition still needs to be for 25 virtual qubits
    part_large = [(i % num_cores) for i in range(virtual_qubits)]
    random.shuffle(part_large)
    
    Ps_large = np.zeros((len(Gs)+1, virtual_qubits), dtype=int)
    Ps_large[0] = part_large

    Ps_HQA_large = HQA_variation(Gs, Ps_large.copy(), num_cores, virtual_qubits, core_cap_large.copy(), distance_matrix=dist_matrix_large)
    
    output_dict["hqa_test_qft_25_large_cores"] = {
        "num_virtual_qubits": virtual_qubits,
        "num_cores": num_cores,
        "num_layers": num_layers,
        "gs_sparse": gs_sparse,
        "input_initial_partition": Ps_large[0].tolist(),
        "input_core_capacities": core_cap_large,
        "input_distance_matrix": dist_matrix_large,
        "expected_output": Ps_HQA_large.tolist()
    }

    print(f"Writing to dse_pau/test_vectors.json")
    with open("dse_pau/test_vectors.json", "w") as f:
        json.dump(output_dict, f, indent=4)
        
    print("Test vectors generated successfully!")

if __name__ == "__main__":
    generate_test_vectors()