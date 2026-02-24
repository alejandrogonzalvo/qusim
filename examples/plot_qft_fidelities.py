import numpy as np
import qiskit
from qiskit.circuit.library import QFT
from qiskit import transpile
import matplotlib.pyplot as plt
from qiskit.transpiler import CouplingMap

import qusim

def plot_qft_fidelities():
    nq = 30
    
    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)
    
    print("Transpiling to basis gates...")
    transp_circ = transpile(
        circ, 
        basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], 
        optimization_level=0,
        seed_transpiler=42
    )
    
    num_cores = 5
    qubits_per_core = 6 # 5 * 6 = 30
    
    # Define a constrained core topology (e.g. 2x3 grid / heavy-hex simulation)
    # This triggers the MultiCoreOrchestrator to run SABRE and inject routing fidelity penalties!
    from qiskit.transpiler import CouplingMap
    grid_topology = CouplingMap.from_grid(2, 3)
    core_tops = [grid_topology for _ in range(num_cores)]
    
    print(f"Running qusim (Cores: {num_cores}, Qubits/Core: {qubits_per_core}, SABRE Enabled)...")
    result = qusim.map_circuit(
        circuit=transp_circ,
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        core_topologies=core_tops,
        seed=42
    )
    
    print(f"Execution Success: {result.execution_success}")
    print(f"Overall Fidelity:     {result.overall_fidelity:.4e}")
    print(f"Algorithmic Fidelity: {result.algorithmic_fidelity:.4e}")
    print(f"Routing Fidelity:     {result.routing_fidelity:.4e}")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plotting loop
    for q in range(nq):
        algo_curve, route_curve, coh_curve = result.get_qubit_fidelity_over_time(q)
        overall_curve = algo_curve * route_curve * coh_curve
        
        axes[0].plot(overall_curve, alpha=0.5, linewidth=1.0)
        axes[1].plot(algo_curve, alpha=0.5, linewidth=1.0)
        axes[2].plot(route_curve, alpha=0.5, linewidth=1.0)
        axes[3].plot(coh_curve, alpha=0.5, linewidth=1.0)
        
    # Formatting
    axes[0].set_title("Overall Fidelity")
    axes[1].set_title("Algorithmic Fidelity (Native 1Q/2Q Logic Gates)")
    axes[2].set_title("Routing Fidelity (SABRE SWAPs & Teleportation EPRs)")
    axes[3].set_title("Coherence Fidelity (T1/T2 Relaxation)")
    
    for ax in axes:
        ax.set_ylabel("Fidelity")
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[3].set_xlabel("Circuit Layer")
    
    plt.suptitle(f"Algorithmic vs Architectural Fidelity Breakdown (QFT {nq} on {num_cores} constrained cores)")
    plt.tight_layout()
    
    out_file = "qft30_fidelity_split.png"
    plt.savefig(out_file, dpi=300)
    print(f"Split plots saved to {out_file}")

if __name__ == "__main__":
    plot_qft_fidelities()
