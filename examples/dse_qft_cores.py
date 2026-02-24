import qiskit
from qiskit.circuit.library import QFT
from qiskit import transpile
import matplotlib.pyplot as plt
import time
import qusim

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def run_dse_qft():
    nq = 100
    # Core distributions to test for 100 qubits
    cores_to_test = [1, 2, 4, 5, 10, 25, 50]
    
    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)
    
    print("Transpiling to basis gates (this may take a moment for 100 qubits)...")
    start_transpile = time.time()
    transp_circ = transpile(
        circ, 
        basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], 
        optimization_level=0,
        seed_transpiler=42
    )
    print(f"Transpilation took {time.time() - start_transpile:.2f}s")
    
    fidelities = []
    teleportations = []
    
    print("\nStarting Design Space Exploration (DSE)...")
    print(f"{'Cores':<6} | {'Q/Core':<8} | {'Total EPRs':<12} | {'Time (ns)':<12} | {'Fidelity':<10}")
    print("-" * 62)
    
    for num_cores in cores_to_test:
        qubits_per_core = nq // num_cores
        
        result = qusim.map_circuit(
            circuit=transp_circ,
            num_cores=num_cores,
            qubits_per_core=qubits_per_core,
            seed=42
        )
        
        fidelities.append(result.overall_fidelity)
        teleportations.append(result.total_epr_pairs)
        
        print(f"{num_cores:<6} | {qubits_per_core:<8} | {result.total_epr_pairs:<12} | {result.total_circuit_time_ns:<12.1f} | {result.overall_fidelity:.6e}")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Cores')
    ax1.set_ylabel('Overall Fidelity', color=color)
    ax1.plot(cores_to_test, fidelities, marker='o', color=color, linewidth=2, label='Fidelity')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Use logarithmic scale if fidelity drops significantly
    if min(fidelities) > 0 and max(fidelities) / min(fidelities) > 100:
        ax1.set_yscale('log')

    # Instantiate a second axes that shares the same x-axis for teleportations
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Total EPR Pairs (Network Overhead)', color=color)  
    ax2.plot(cores_to_test, teleportations, marker='s', linestyle='--', color=color, linewidth=2, label='EPR Pairs')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'Design Space Exploration: {nq}-Qubit QFT Architecture Scaling')
    fig.tight_layout()
    
    out_file = "dse_qft_100q_cores.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nPlot saved to {out_file}")

if __name__ == "__main__":
    run_dse_qft()
