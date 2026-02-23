"""Generate a single HQA CLI input for profiling. Usage:
    python dse_pau/gen_cli_input.py <num_cores> [qubits_per_core]
    
Pipe to hqa_cli:
    python dse_pau/gen_cli_input.py 20 | samply record ./target/release/hqa_cli
"""
import json
import random
import sys
import os

from qiskit.circuit.library import QFT
from qiskit import transpile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import qiskit_circ_to_slices, all_to_all_topology

def main():
    num_cores = int(sys.argv[1])
    qubits_per_core = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    virtual_qubits = num_cores * qubits_per_core

    circ = QFT(virtual_qubits)
    transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], seed_transpiler=42)
    _, Gs_all = qiskit_circ_to_slices(transp_circ)

    gs_sparse = []
    for layer_idx in range(Gs_all.shape[0]):
        g = Gs_all[layer_idx]
        coords = g.coords
        data = g.data
        for i in range(len(data)):
            gs_sparse.append([float(layer_idx), float(coords[0][i]), float(coords[1][i]), float(data[i])])

    dist_matrix = all_to_all_topology(num_cores)
    core_cap = [qubits_per_core] * num_cores
    part = [(i % num_cores) for i in range(virtual_qubits)]
    random.seed(42)
    random.shuffle(part)

    cli_input = {
        "num_virtual_qubits": virtual_qubits,
        "num_cores": num_cores,
        "num_layers": Gs_all.shape[0],
        "gs_sparse": gs_sparse,
        "input_initial_partition": part,
        "input_core_capacities": core_cap,
        "input_distance_matrix": dist_matrix,
    }

    json.dump(cli_input, sys.stdout)

if __name__ == "__main__":
    main()
