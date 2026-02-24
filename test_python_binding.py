import sys
import numpy as np
import qiskit
from qiskit.circuit.library import QFT
from qiskit import transpile

import qusim

print("Qusim imported successfully!")

def test_qft():
    # 1. Create a 5 qubit QFT circuit
    nq = 5
    circ = QFT(nq)
    
    # Transpile to basis gates supported by our utils
    transp_circ = transpile(
        circ, 
        basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], 
        optimization_level=0,
        seed_transpiler=42
    )
    
    # 2. Setup architecture
    num_cores = 2
    qubits_per_core = [3, 2] # Exact fit for 5 qubits
    
    print("Running qusim python pipeline directly...")
    result = qusim.map_circuit(
        circuit=transp_circ,
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        distance_matrix=None, # defaults to all_to_all
    )
    
    print(f"Success: {result.execution_success}")
    print(f"Overall Fidelity: {result.overall_fidelity:.4f}")
    print(f"Coherence Fidelity: {result.coherence_fidelity:.4f}")
    print(f"Operational Fidelity: {result.operational_fidelity:.4f}")
    print(f"Total Circuit Time: {result.total_circuit_time_ns:.1f} ns")
    
    # 3. Check shape of the grids
    print("\nGrid Shapes:")
    print(f"Operational Fidelity Grid: {result.operational_fidelity_grid.shape}")
    print(f"Coherence Fidelity Grid:   {result.coherence_fidelity_grid.shape}")
    print(f"Placements Grid:           {result.placements.shape}")
    
    # 4. Show a sample qubit track
    q = 0
    op_curve, coh_curve = result.get_qubit_fidelity_over_time(q)
    print(f"\nQubit {q} curve over first 5 layers:")
    for layer in range(min(5, len(op_curve))):
        print(f"Layer {layer}: Op={op_curve[layer]:.5f}, Coh={coh_curve[layer]:.5f}")

if __name__ == "__main__":
    test_qft()
