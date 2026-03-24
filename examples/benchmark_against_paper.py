"""
Benchmark qusim fidelity predictions against the paper's results
(arXiv:2503.06693v2) and real IBM Q hardware measurements.

Loads the paper's QASM circuits, runs them through qusim in single-core
all-to-all mode, and plots the comparison.

TODO: When IBM Q access is available, use backend.properties() to pull
exact per-qubit T1/T2 and per-gate error rates for precise reproduction.
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from qusim import map_circuit, InitialPlacement

# ---------------------------------------------------------------------------
# Configuration — typical IBM Q error rates
# ---------------------------------------------------------------------------
SINGLE_GATE_ERROR = 2.3e-4
TWO_GATE_ERROR = 8.2e-3
T1_NS = 120_000.0   # 120 μs
T2_NS = 116_000.0   # 116 μs
SINGLE_GATE_TIME = 35.0     # ns, typical IBM sx
TWO_GATE_TIME = 660.0       # ns, typical IBM cx
TELEPORTATION_ERROR = 0.0   # single core, no teleportation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'Analytic-Model-of-Fidelity-under-Depolarizing-Noise')
CIRCUITS_DIR = os.path.join(PAPER_REPO, 'circuits')
DATA_FILE = os.path.join(PAPER_REPO, 'data', 'data_1.csv')


def load_paper_data(data_file):
    """Load the paper's pre-computed fidelity results from data_1.csv."""
    with open(data_file, 'r') as f:
        data = list(csv.reader(f, delimiter=','))

    filenames = data[0]
    num_qubits = [int(q) for q in data[1]]
    num_gates = [int(g) for g in data[3]]

    fid_esp = [float(f) for f in data[6]]

    fid_min_depol = [float(eval(f)[0]) for f in data[10]]
    fid_max_depol = [float(eval(f)[1]) for f in data[10]]
    fid_avg_depol = [(fid_min_depol[i] + fid_max_depol[i]) / 2 for i in range(len(fid_min_depol))]
    depol_error = [(fid_max_depol[i] - fid_min_depol[i]) / 2 for i in range(len(fid_min_depol))]

    sr_mit_IBMQ = [float(f) for f in data[13]]

    return {
        'filenames': filenames,
        'num_qubits': num_qubits,
        'num_gates': num_gates,
        'fid_esp': fid_esp,
        'fid_avg_depol': fid_avg_depol,
        'depol_error': depol_error,
        'sr_mit_IBMQ': sr_mit_IBMQ,
    }


def build_single_core_setup(num_qubits):
    """Create a single-core all-to-all setup for benchmarking without routing overhead."""
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
            edges.append((j, i))
    coupling_map = CouplingMap(edges)
    core_mapping = {i: 0 for i in range(num_qubits)}
    return coupling_map, core_mapping


def run_qusim_on_circuit(qasm_path, num_qubits):
    """Load a QASM circuit and estimate fidelity with qusim."""
    circ = qiskit.QuantumCircuit.from_qasm_file(qasm_path)

    # Mirror circuit (circ^-1 & circ) to match the paper's benchmarking methodology
    circ_inv = circ.copy()
    circ_inv.remove_final_measurements()
    circ_inv = circ_inv.inverse()
    circ_inv.barrier()
    rev_circ = circ_inv & circ

    # Transpile to basis gates
    transp = qiskit.transpile(
        rev_circ,
        basis_gates=['x', 'cx', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'measure'],
        optimization_level=0,
    )

    n = transp.num_qubits
    coupling_map, core_mapping = build_single_core_setup(n)

    result = map_circuit(
        circuit=transp,
        full_coupling_map=coupling_map,
        core_mapping=core_mapping,
        seed=42,
        initial_placement=InitialPlacement.RANDOM,
        single_gate_error=SINGLE_GATE_ERROR,
        two_gate_error=TWO_GATE_ERROR,
        teleportation_error_per_hop=TELEPORTATION_ERROR,
        single_gate_time=SINGLE_GATE_TIME,
        two_gate_time=TWO_GATE_TIME,
        teleportation_time_per_hop=1000.0,
        t1=T1_NS,
        t2=T2_NS,
    )

    return result.overall_fidelity


def main():
    paper_data = load_paper_data(DATA_FILE)
    filenames = paper_data['filenames']
    num_gates = paper_data['num_gates']

    # Run qusim on each circuit
    qusim_fidelities = []
    valid_indices = []

    for i, fname in enumerate(filenames):
        qasm_path = os.path.join(CIRCUITS_DIR, fname)
        if not os.path.exists(qasm_path):
            print(f"  SKIP (file not found): {fname}")
            qusim_fidelities.append(None)
            continue

        try:
            fid = run_qusim_on_circuit(qasm_path, paper_data['num_qubits'][i])
            qusim_fidelities.append(fid)
            valid_indices.append(i)
            print(f"  [{i+1}/{len(filenames)}] {fname}: qusim={fid:.4f}, IBM={paper_data['sr_mit_IBMQ'][i]:.4f}")
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            qusim_fidelities.append(None)

    # Filter to valid results
    v_gates = [num_gates[i] for i in valid_indices]
    v_qusim = [qusim_fidelities[i] for i in valid_indices]
    v_esp = [paper_data['fid_esp'][i] for i in valid_indices]
    v_depol = [paper_data['fid_avg_depol'][i] for i in valid_indices]
    v_depol_err = [paper_data['depol_error'][i] for i in valid_indices]
    v_ibmq = [paper_data['sr_mit_IBMQ'][i] for i in valid_indices]
    v_nq = [paper_data['num_qubits'][i] for i in valid_indices]

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    scatter_size = 20
    ticks_size = 15
    label_size = 20
    fig, ax = plt.subplots(figsize=(20, 7))

    ax.scatter(v_gates, v_ibmq, label='IBM Q (mitigated)', color='tab:purple', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, v_esp, label='ESP', color='tab:orange', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.errorbar(v_gates, v_depol, yerr=v_depol_err, fmt='none', c='gray', alpha=0.5, zorder=-1)
    ax.scatter(v_gates, v_depol, label='Paper (depol.)', color='tab:green', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, v_qusim, label='qusim', color='tab:red', alpha=0.75,
               s=[scatter_size * q for q in v_nq], marker='x')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Gates', fontsize=label_size)
    ax.set_ylabel('Fidelity', fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.legend(fontsize=ticks_size)

    # Inset boxplot: fidelity differences vs IBM Q
    y_bottom = 0.1
    y_top = ax.get_ylim()[1] * 0.65
    axins = ax.inset_axes([13, y_bottom, 45, y_top - y_bottom], transform=ax.transData)
    axins.set_facecolor('snow')

    esp_diff = [v_esp[i] - v_ibmq[i] for i in range(len(v_ibmq))]
    depol_diff = [v_depol[i] - v_ibmq[i] for i in range(len(v_ibmq))]
    qusim_diff = [v_qusim[i] - v_ibmq[i] for i in range(len(v_ibmq))]

    colors = ['tab:orange', 'tab:green', 'tab:red']
    box = axins.boxplot(
        [esp_diff, depol_diff, qusim_diff],
        positions=[1, 2, 3],
        labels=['ESP', 'Paper', 'qusim'],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
    for median in box['medians']:
        median.set_color('black')

    axins.tick_params(axis='both', which='major', labelsize=ticks_size * 0.85)
    axins.tick_params(axis='x', which='major', labelsize=ticks_size * 0.85, rotation=45)
    axins.set_ylabel('Fidelity\nDifference', fontsize=label_size * 0.85, labelpad=-3)
    axins.axhline(y=0, color='gray', linestyle='--', zorder=-10)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_vs_paper.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
