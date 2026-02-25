from qiskit import QuantumCircuit
from plot_fidelities_utils import simulate_and_plot, SimulationConfig

def generate_ghz(nq: int) -> QuantumCircuit:
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(nq - 1):
        qc.cx(i, i + 1)
    return qc

from qusim.hqa.placement import InitialPlacement

def plot_ghz_fidelities():
    nq = 30
    
    print(f"Generating GHZ({nq}) circuit...")
    circ = generate_ghz(nq)
    
    num_cores = 5
    qubits_per_core = 6
    
    config = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        title=f"Algorithmic vs Architectural Fidelity Breakdown (GHZ {nq} on {num_cores} constrained cores)",
        out_file="ghz30_fidelity_split.png"
    )
    
    simulate_and_plot(circ, config)

if __name__ == "__main__":
    plot_ghz_fidelities()
