import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.transpiler import CouplingMap

import qusim
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    num_cores: int
    qubits_per_core: int
    initial_placement: "InitialPlacement"
    title: str = ""
    out_file: str = ""
    seed: int = 42


def build_core_mapping(num_cores: int, qubits_per_core: int):
    full_coupling_map = CouplingMap()
    for i in range(num_cores * qubits_per_core):
        full_coupling_map.add_physical_qubit(i)

    core_mapping = {}

    for c in range(num_cores):
        # Create a 2x3 grid for this core
        grid = CouplingMap.from_grid(2, 3)
        offset = c * qubits_per_core
        for edge in grid.get_edges():
            full_coupling_map.add_edge(edge[0] + offset, edge[1] + offset)

        for q in range(qubits_per_core):
            core_mapping[offset + q] = c

    # Connect the cores via communication links
    # Node 5 of core c connects to node 0 of core c+1
    for c in range(num_cores - 1):
        p1 = c * qubits_per_core + 5
        p2 = (c + 1) * qubits_per_core + 0
        full_coupling_map.add_edge(p1, p2)
        full_coupling_map.add_edge(p2, p1)

    return full_coupling_map, core_mapping


from qusim.hqa.placement import InitialPlacement

def simulate(circuit, config: SimulationConfig):
    print(f"Transpiling to basis gates for {config.initial_placement.value} placement...")
    transp_circ = transpile(
        circuit,
        basis_gates=["x", "cx", "cp", "rz", "h", "s", "sdg", "t", "tdg", "measure"],
        optimization_level=0,
        seed_transpiler=config.seed,
    )

    full_coupling_map, core_mapping = build_core_mapping(
        config.num_cores, config.qubits_per_core
    )

    print(
        f"Running qusim (Cores: {config.num_cores}, Qubits/Core: {config.qubits_per_core}, SABRE Enabled, Policy: {config.initial_placement.value})..."
    )
    result = qusim.map_circuit(
        circuit=transp_circ,
        full_coupling_map=full_coupling_map,
        core_mapping=core_mapping,
        seed=config.seed,
        initial_placement=config.initial_placement
    )
    return result

def simulate_and_plot(circuit, config: SimulationConfig):
    result = simulate(circuit, config)


    print(f"Execution Success: {result.execution_success}")
    print(f"Overall Fidelity:     {result.overall_fidelity:.4e}")
    print(f"Algorithmic Fidelity: {result.algorithmic_fidelity:.4e}")
    print(f"Routing Fidelity:     {result.routing_fidelity:.4e}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    nq = circuit.num_qubits
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
        ax.grid(True, linestyle="--", alpha=0.7)

    axes[3].set_xlabel("Circuit Layer")

    plt.suptitle(config.title)
    plt.tight_layout()

    plt.savefig(config.out_file, dpi=300)
    print(f"Split plots saved to {config.out_file}")
