from qiskit.circuit.library import QFT
from plot_fidelities_utils import simulate_and_plot

def plot_qft_fidelities():
    nq = 30
    
    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)
    
    num_cores = 5
    qubits_per_core = 6
    
    simulate_and_plot(
        circuit=circ,
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        title=f"Algorithmic vs Architectural Fidelity Breakdown (QFT {nq} on {num_cores} constrained cores)",
        out_file="qft30_fidelity_split.png"
    )

if __name__ == "__main__":
    plot_qft_fidelities()
