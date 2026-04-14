"""
Benchmark qusim fidelity predictions against the paper's results
(arXiv:2503.06693v2) and real IBM Q hardware measurements.

Loads the paper's QASM circuits, runs ESP / paper's depolarizing model / qusim
all with the SAME uniform error rates, and plots against IBM Q hardware data.

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
# Configuration — typical IBM Q error rates (uniform for all qubits/gates)
# ---------------------------------------------------------------------------
SINGLE_GATE_ERROR = 2.3e-4
TWO_GATE_ERROR = 8.2e-3
T1_NS = 120_000.0   # 120 μs
T2_NS = 116_000.0   # 116 μs
SINGLE_GATE_TIME = 35.0     # ns, typical IBM sx
TWO_GATE_TIME = 660.0       # ns, typical IBM cx
TELEPORTATION_ERROR = 0.0   # single core, no teleportation

# Convert gate error rates to depolarization parameters (paper Eq. 39)
# lambda = d * (F_g - 1) / (1 - d), where F_g = 1 - error, d = 2^n_qubits
LAMBDA_1Q = 2 * ((1 - SINGLE_GATE_ERROR) - 1) / (1 - 2)  # d=2 for 1Q
LAMBDA_2Q = 4 * ((1 - TWO_GATE_ERROR) - 1) / (1 - 4)      # d=4 for 2Q

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'Analytic-Model-of-Fidelity-under-Depolarizing-Noise')
CIRCUITS_DIR = os.path.join(PAPER_REPO, 'circuits')
DATA_FILE = os.path.join(PAPER_REPO, 'data', 'data_1.csv')


def load_paper_data(data_file):
    """Load IBM Q measured fidelity from data_1.csv."""
    with open(data_file, 'r') as f:
        data = list(csv.reader(f, delimiter=','))

    filenames = data[0]
    num_qubits = [int(q) for q in data[1]]
    num_gates = [int(g) for g in data[3]]
    sr_mit_IBMQ = [float(f) for f in data[13]]

    return {
        'filenames': filenames,
        'num_qubits': num_qubits,
        'num_gates': num_gates,
        'sr_mit_IBMQ': sr_mit_IBMQ,
    }


# Gates that are virtual (zero error, zero duration) on real hardware.
VIRTUAL_GATES = frozenset({'rz', 'id', 'delay', 'barrier', 'measure'})


# ---------------------------------------------------------------------------
# ESP model with thermal (uniform error rates)
# ---------------------------------------------------------------------------
def estimate_esp_thermal(circ):
    """ESP: F = prod(1 - error_rate) per gate, then T1/T2 per measured qubit."""
    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    measured_qubits = set()

    prob = 1.0
    total_time = 0.0

    for layer in layers:
        slowest_gate = 0.0
        for gate in layer:
            if 'measure' in gate.operation.name:
                measured_qubits.add(circ.find_bit(gate.qubits[0]).index)
                continue

            if gate.operation.name in VIRTUAL_GATES:
                continue

            if gate.operation.num_qubits == 1:
                prob *= (1 - SINGLE_GATE_ERROR)
                slowest_gate = max(slowest_gate, SINGLE_GATE_TIME)
            else:
                prob *= (1 - TWO_GATE_ERROR)
                slowest_gate = max(slowest_gate, TWO_GATE_TIME)

        total_time += slowest_gate

    # T1/T2 thermal relaxation (paper Eq. 41) applied per measured qubit
    t1_s = T1_NS * 1e-9
    t2_s = T2_NS * 1e-9
    total_time_s = total_time * 1e-9
    for _ in measured_qubits:
        prob *= np.exp(-total_time_s / t1_s) * (0.5 * np.exp(-total_time_s / t2_s) + 0.5)

    return prob


# ---------------------------------------------------------------------------
# Paper's depolarizing model with thermal (uniform error rates, p_ent=0)
# ---------------------------------------------------------------------------
def estimate_depol_thermal(circ):
    """Paper's depolarizing channel model (Algorithm 1) with T1/T2, p_ent=0."""
    n = circ.num_qubits
    qubits_fidelity = [1.0] * n

    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    measured_qubits = set()

    total_time = 0.0

    for layer in layers:
        slowest_gate = 0.0
        for gate in layer:
            if 'measure' in gate.operation.name:
                measured_qubits.add(circ.find_bit(gate.qubits[0]).index)
                continue

            if gate.operation.name in VIRTUAL_GATES:
                continue

            if gate.operation.num_qubits == 1:
                q = circ.find_bit(gate.qubits[0]).index
                lam = LAMBDA_1Q
                # Paper Eq: F_q = (1-lambda)*F_q + lambda/d, with d=2, p_ent=0
                qubits_fidelity[q] = (1 - lam) * qubits_fidelity[q] + lam / 2
                slowest_gate = max(slowest_gate, SINGLE_GATE_TIME)
            else:
                q1 = circ.find_bit(gate.qubits[0]).index
                q2 = circ.find_bit(gate.qubits[1]).index
                lam = LAMBDA_2Q
                f1 = qubits_fidelity[q1]
                f2 = qubits_fidelity[q2]
                # Paper Eq. 25: eta correction for 2Q gates
                eta = 0.5 * (np.sqrt((1 - lam) * (f1 + f2)**2 + lam) - np.sqrt(1 - lam) * (f1 + f2))
                qubits_fidelity[q1] = np.sqrt(1 - lam) * f1 + eta
                qubits_fidelity[q2] = np.sqrt(1 - lam) * f2 + eta
                slowest_gate = max(slowest_gate, TWO_GATE_TIME)

        total_time += slowest_gate

        # T1/T2 thermal relaxation after each layer (paper Eq. 41)
        t1_s = T1_NS * 1e-9
        t2_s = T2_NS * 1e-9
        layer_time_s = slowest_gate * 1e-9
        for q in range(n):
            t1_decay = np.exp(-layer_time_s / t1_s)
            t2_decay = 0.5 * np.exp(-layer_time_s / t2_s) + 0.5
            qubits_fidelity[q] *= t1_decay * t2_decay

    if measured_qubits:
        return np.prod([qubits_fidelity[q] for q in measured_qubits])
    return np.prod(qubits_fidelity)


# ---------------------------------------------------------------------------
# qusim model
# ---------------------------------------------------------------------------
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


def estimate_qusim(transp):
    """Run qusim fidelity estimation on a transpiled circuit."""
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

    # Use per-qubit grids to compute fidelity over measured qubits only
    # (matching the paper's methodology)
    measured_qubits = set()
    for inst in transp:
        if 'measure' in inst.operation.name:
            measured_qubits.add(transp.find_bit(inst.qubits[0]).index)

    if measured_qubits:
        fid = 1.0
        for q in measured_qubits:
            fid *= result.algorithmic_fidelity_grid[-1, q]
            fid *= result.routing_fidelity_grid[-1, q]
            fid *= result.coherence_fidelity_grid[-1, q]
        return fid

    return result.overall_fidelity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    paper_data = load_paper_data(DATA_FILE)
    filenames = paper_data['filenames']

    results = {'qusim': [], 'esp': [], 'depol': [], 'gates': []}
    valid_indices = []

    for i, fname in enumerate(filenames):
        qasm_path = os.path.join(CIRCUITS_DIR, fname)
        if not os.path.exists(qasm_path):
            print(f"  SKIP (file not found): {fname}")
            continue

        try:
            circ = qiskit.QuantumCircuit.from_qasm_file(qasm_path)

            # Mirror circuit (circ^-1 & circ) — paper's benchmarking methodology
            circ_inv = circ.copy()
            circ_inv.remove_final_measurements()
            circ_inv = circ_inv.inverse()
            circ_inv.barrier()
            rev_circ = circ_inv & circ

            # Transpile — same circuit for all three models
            transp = qiskit.transpile(
                rev_circ,
                basis_gates=['x', 'cx', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'measure'],
                optimization_level=3,
            )

            fid_qusim = estimate_qusim(transp)
            fid_esp = estimate_esp_thermal(transp)
            fid_depol = estimate_depol_thermal(transp)

            results['qusim'].append(fid_qusim)
            results['esp'].append(fid_esp)
            results['depol'].append(fid_depol)
            results['gates'].append(len(transp.data))
            valid_indices.append(i)

            print(f"  [{i+1}/{len(filenames)}] {fname}: "
                  f"qusim={fid_qusim:.4f}, esp={fid_esp:.4f}, depol={fid_depol:.4f}, "
                  f"IBM={paper_data['sr_mit_IBMQ'][i]:.4f}")

        except Exception as e:
            print(f"  ERROR on {fname}: {e}")

    v_ibmq = [paper_data['sr_mit_IBMQ'][i] for i in valid_indices]
    v_nq = [paper_data['num_qubits'][i] for i in valid_indices]
    v_gates = results['gates']

    # ---------------------------------------------------------------------------
    # Plot — all models use same uniform error rates on same transpiled circuits
    # ---------------------------------------------------------------------------
    scatter_size = 20
    ticks_size = 15
    label_size = 20
    fig, ax = plt.subplots(figsize=(20, 7))

    ax.scatter(v_gates, v_ibmq, label='IBM Q (mitigated)', color='tab:purple', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, results['esp'], label='ESP + $T_{1,2}$ (uniform)', color='tab:orange', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, results['depol'], label='Depol. + $T_{1,2}$ (uniform)', color='tab:green', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, results['qusim'], label='qusim', color='tab:red', alpha=0.75,
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

    esp_diff = [results['esp'][i] - v_ibmq[i] for i in range(len(v_ibmq))]
    depol_diff = [results['depol'][i] - v_ibmq[i] for i in range(len(v_ibmq))]
    qusim_diff = [results['qusim'][i] - v_ibmq[i] for i in range(len(v_ibmq))]

    colors = ['tab:orange', 'tab:green', 'tab:red']
    box = axins.boxplot(
        [esp_diff, depol_diff, qusim_diff],
        positions=[1, 2, 3],
        labels=['ESP', 'Depol.', 'qusim'],
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
