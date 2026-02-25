from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt

from plot_fidelities_utils import (
    simulate,
    build_core_mapping,
    build_all_to_all_core_mapping,
    build_single_core_all_to_all,
    build_single_core_grid,
    SimulationConfig,
)
from qusim.hqa.placement import InitialPlacement


def plot_qft_topology_comparison():
    nq = 30

    print(f"Generating QFT({nq}) circuit...")
    circ = QFT(nq)

    num_cores = 5
    qubits_per_core = 6

    # 1. Simulate with SPECTRAL placement on the default grid/ring topology
    grid_coupling_map, grid_core_mapping = build_core_mapping(num_cores, qubits_per_core)
    config_grid = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        coupling_map=grid_coupling_map,
        core_mapping=grid_core_mapping,
    )
    result_grid = simulate(circ, config_grid)

    # 2. Simulate with SPECTRAL placement on an all-to-all multi-core topology
    ata_coupling_map, ata_core_mapping = build_all_to_all_core_mapping(num_cores, qubits_per_core)
    config_ata = SimulationConfig(
        num_cores=num_cores,
        qubits_per_core=qubits_per_core,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        coupling_map=ata_coupling_map,
        core_mapping=ata_core_mapping,
    )
    result_ata = simulate(circ, config_ata)

    # 3. Simulate with SPECTRAL placement on a single core, all-to-all
    sc_coupling_map, sc_core_mapping = build_single_core_all_to_all(nq)
    config_sc = SimulationConfig(
        num_cores=1,
        qubits_per_core=nq,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        coupling_map=sc_coupling_map,
        core_mapping=sc_core_mapping,
    )
    result_sc = simulate(circ, config_sc)

    # 4. Simulate with SPECTRAL placement on a single core 2D grid (5x6)
    sg_coupling_map, sg_core_mapping = build_single_core_grid(rows=5, cols=6)
    config_sg = SimulationConfig(
        num_cores=1,
        qubits_per_core=nq,
        initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
        coupling_map=sg_coupling_map,
        core_mapping=sg_core_mapping,
    )
    result_sg = simulate(circ, config_sg)

    print("\n--- RESULTS ---")
    print(f"[GRID/RING]      Overall Fidelity: {result_grid.overall_fidelity:.4e}")
    print(f"[ALL-TO-ALL]     Overall Fidelity: {result_ata.overall_fidelity:.4e}")
    print(f"[SINGLE-CORE]    Overall Fidelity: {result_sc.overall_fidelity:.4e}")
    print(f"[SC-2D-GRID]     Overall Fidelity: {result_sg.overall_fidelity:.4e}")
    print("----------------\n")

    # Plot side by side (4 rows x 4 cols)
    fig, axes = plt.subplots(4, 4, figsize=(30, 16), sharex=True, sharey="row")

    for q in range(nq):
        # Grid/ring curves
        a_g, r_g, c_g = result_grid.get_qubit_fidelity_over_time(q)
        o_g = a_g * r_g * c_g

        # All-to-all multi-core curves
        a_a, r_a, c_a = result_ata.get_qubit_fidelity_over_time(q)
        o_a = a_a * r_a * c_a

        # Single-core curves
        a_s, r_s, c_s = result_sc.get_qubit_fidelity_over_time(q)
        o_s = a_s * r_s * c_s

        # Column 0: Single core all-to-all
        axes[0, 0].plot(o_s, alpha=0.5, linewidth=1.0)
        axes[1, 0].plot(a_s, alpha=0.5, linewidth=1.0)
        axes[2, 0].plot(r_s, alpha=0.5, linewidth=1.0)
        axes[3, 0].plot(c_s, alpha=0.5, linewidth=1.0)

        # Single-core 2D grid curves
        a_sg, r_sg, c_sg = result_sg.get_qubit_fidelity_over_time(q)
        o_sg = a_sg * r_sg * c_sg

        # Column 1: Single core 2D grid
        axes[0, 1].plot(o_sg, alpha=0.5, linewidth=1.0)
        axes[1, 1].plot(a_sg, alpha=0.5, linewidth=1.0)
        axes[2, 1].plot(r_sg, alpha=0.5, linewidth=1.0)
        axes[3, 1].plot(c_sg, alpha=0.5, linewidth=1.0)

        # Column 2: All-to-all multi-core
        axes[0, 2].plot(o_a, alpha=0.5, linewidth=1.0)
        axes[1, 2].plot(a_a, alpha=0.5, linewidth=1.0)
        axes[2, 2].plot(r_a, alpha=0.5, linewidth=1.0)
        axes[3, 2].plot(c_a, alpha=0.5, linewidth=1.0)

        # Column 3: Multi-core Grid+Ring
        axes[0, 3].plot(o_g, alpha=0.5, linewidth=1.0)
        axes[1, 3].plot(a_g, alpha=0.5, linewidth=1.0)
        axes[2, 3].plot(r_g, alpha=0.5, linewidth=1.0)
        axes[3, 3].plot(c_g, alpha=0.5, linewidth=1.0)

    col_labels = [
        f"Single Core All-to-All\n(1 core × {nq} qubits)",
        f"Single Core 2D Grid (5×6)\n(1 core × {nq} qubits)",
        f"Multi-core All-to-All\n({num_cores} cores × {qubits_per_core} qubits)",
        f"Multi-core Grid+Ring\n({num_cores} cores × {qubits_per_core} qubits)",
    ]
    row_labels = ["Overall Fidelity", "Algorithmic Fidelity", "Routing Fidelity", "Coherence Fidelity"]

    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=11, fontweight="bold")

    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label)
        for j in range(4):
            axes[i, j].grid(True, linestyle="--", alpha=0.7)

    for j in range(4):
        axes[3, j].set_xlabel("Circuit Layer")

    plt.suptitle(
        f"Topology Impact on Fidelity — Spectral Clustering Placement\n"
        f"QFT {nq}  |  Single Core All-to-All  →  Single Core 2D Grid  →  Multi-core All-to-All  →  Multi-core Grid+Ring",
        fontsize=13,
    )
    plt.tight_layout()

    out_file = "qft30_topology_comparison.png"
    plt.savefig(out_file, dpi=300)
    print(f"Side-by-side plots saved to {out_file}")


if __name__ == "__main__":
    plot_qft_topology_comparison()
