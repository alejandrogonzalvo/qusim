import json
import numpy as np
import random
from qiskit.circuit.library import QFT
from qiskit import transpile
import sys
import os

# Ensure we can import from dse_pau
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import qiskit_circ_to_slices, all_to_all_topology

def serialize_gs(Gs):
    gs_serialized = []
    # Gs is a sparse stack from qiskit_circ_to_slices
    for i in range(Gs.shape[0]):
        slice_g = Gs[i]
        coords = slice_g.coords
        data = slice_g.data
        edges = []
        for j in range(len(data)):
            edges.append([int(coords[0][j]), int(coords[1][j]), float(data[j])])
        gs_serialized.append(edges)
    return gs_serialized

def generate_scalability_vectors():
    core_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    qubits_per_core = 10
    
    output_dict = {}
    
    for num_cores in core_counts:
        virtual_qubits = num_cores * qubits_per_core
        print(f"Generating vectors for {num_cores} cores, {virtual_qubits} qubits...")
        
        # 1. Generate QFT circuit
        circ = QFT(virtual_qubits)
        transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], seed_transpiler=42)
        Gs_two, Gs_all = qiskit_circ_to_slices(transp_circ)
        gs_serialized = serialize_gs(Gs_all)
        
        # 2. Setup initial partition and topology
        dist_matrix = all_to_all_topology(num_cores)
        core_cap = [qubits_per_core] * num_cores
        part = [(i % num_cores) for i in range(virtual_qubits)]
        random.seed(42)
        random.shuffle(part)
        
        output_dict[str(num_cores)] = {
            "num_qubits": virtual_qubits,
            "num_cores": num_cores,
            "gs": gs_serialized,
            "ps": [part],
            "core_capacities": core_cap,
            "distance_matrix": dist_matrix
        }

    output_path = "dse_pau/scalability_vectors.json"
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_dict, f)
    print("Done!")

if __name__ == "__main__":
    generate_scalability_vectors()
