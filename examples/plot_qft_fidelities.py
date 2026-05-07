from qiskit.circuit.library import QFT
from plot_fidelities_utils import simulate_and_plot, SimulationConfig
from quadris.hqa.placement import InitialPlacement

def plot_qft_fidelities():
    nq = 30

    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)

    num_cores = 5
    qubits_per_core = 6

    config = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        title=f"Algorithmic vs Architectural Fidelity Breakdown (QFT {nq} on {num_cores} constrained cores)",
        out_file="examples/qft30_fidelity_split.png",
    )

    simulate_and_plot(circ, config)

if __name__ == "__main__":
    plot_qft_fidelities()
