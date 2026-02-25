from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt

from plot_fidelities_utils import simulate, SimulationConfig
from qusim.hqa.placement import InitialPlacement

def plot_qft_comparison():
    nq = 30
    
    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)
    
    num_cores = 5
    qubits_per_core = 6
    
    # 1. Simulate with RANDOM placement
    config_random = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.RANDOM
    )
    result_random = simulate(circ, config_random)
    
    # 2. Simulate with SPECTRAL CLUSTERING placement
    config_spectral = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING
    )
    result_spectral = simulate(circ, config_spectral)
    
    print("\n--- RESULTS ---")
    print(f"[RANDOM]   Overall Fidelity: {result_random.overall_fidelity:.4e}")
    print(f"[SPECTRAL] Overall Fidelity: {result_spectral.overall_fidelity:.4e}")
    print("----------------\n")
    
    # Plot side by side
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True, sharey='row')
    
    # For each qubit, plot the results
    for q in range(nq):
        # Random curves
        a_r, r_r, c_r = result_random.get_qubit_fidelity_over_time(q)
        o_r = a_r * r_r * c_r
        
        # Spectral curves
        a_s, r_s, c_s = result_spectral.get_qubit_fidelity_over_time(q)
        o_s = a_s * r_s * c_s
        
        # Left column: Random
        axes[0, 0].plot(o_r, alpha=0.5, linewidth=1.0)
        axes[1, 0].plot(a_r, alpha=0.5, linewidth=1.0)
        axes[2, 0].plot(r_r, alpha=0.5, linewidth=1.0)
        axes[3, 0].plot(c_r, alpha=0.5, linewidth=1.0)
        
        # Right column: Spectral
        axes[0, 1].plot(o_s, alpha=0.5, linewidth=1.0)
        axes[1, 1].plot(a_s, alpha=0.5, linewidth=1.0)
        axes[2, 1].plot(r_s, alpha=0.5, linewidth=1.0)
        axes[3, 1].plot(c_s, alpha=0.5, linewidth=1.0)
        
    # Formatting left column
    axes[0, 0].set_title("Overall Fidelity (Random Placement)")
    axes[1, 0].set_title("Algorithmic Fidelity (Random Placement)")
    axes[2, 0].set_title("Routing Fidelity (Random Placement)")
    axes[3, 0].set_title("Coherence Fidelity (Random Placement)")
    
    # Formatting right column
    axes[0, 1].set_title("Overall Fidelity (Spectral Clustering)")
    axes[1, 1].set_title("Algorithmic Fidelity (Spectral Clustering)")
    axes[2, 1].set_title("Routing Fidelity (Spectral Clustering)")
    axes[3, 1].set_title("Coherence Fidelity (Spectral Clustering)")
    
    for i in range(4):
        axes[i, 0].set_ylabel("Fidelity")
        for j in range(2):
            axes[i, j].grid(True, linestyle='--', alpha=0.7)
            
    axes[3, 0].set_xlabel("Circuit Layer")
    axes[3, 1].set_xlabel("Circuit Layer")
    
    plt.suptitle(f"Algorithmic vs Architectural Fidelity Breakdown\nQFT {nq} on {num_cores} constrained cores (Random vs Spectral Placement)")
    plt.tight_layout()
    
    out_file = "qft30_comparison_split.png"
    plt.savefig(out_file, dpi=300)
    print(f"Side-by-side plots saved to {out_file}")

if __name__ == "__main__":
    plot_qft_comparison()
