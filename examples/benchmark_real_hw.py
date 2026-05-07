"""
Benchmark quadris fidelity predictions using real IBM Q calibration data.

Generates standard circuits (QFT, GHZ, random), runs the mirror-circuit
methodology, and compares four models:
  1. ESP + T1/T2 (uniform medians from calibration)
  2. Paper's depolarizing model (uniform medians)
  3. quadris (uniform medians)
  4. quadris (per-qubit/per-pair from live calibration)

Usage:
  python examples/benchmark_real_hw.py [--calibration data/calibration_ibm_marrakesh_2026-04-07.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit.circuit.library import QFTGate
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from quadris import map_circuit, InitialPlacement


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------
def load_calibration(path, basis_1q='sx', basis_2q='cz'):
    """Load a calibration snapshot JSON into quadris-compatible parameters."""
    with open(path) as f:
        cal = json.load(f)

    num_qubits = cal['num_qubits']

    # Per-qubit T1/T2 (μs → ns)
    t1_per_qubit = np.array([
        cal['qubits'][str(q)]['t1_us'] * 1e3 if cal['qubits'][str(q)]['t1_us'] else 100_000.0
        for q in range(num_qubits)
    ])
    t2_per_qubit = np.array([
        cal['qubits'][str(q)]['t2_us'] * 1e3 if cal['qubits'][str(q)]['t2_us'] else 50_000.0
        for q in range(num_qubits)
    ])

    # Per-qubit 1Q gate errors
    single_gate_error_per_qubit = np.zeros(num_qubits)
    gate_1q_data = cal['gates_1q'].get(basis_1q, {})
    for q in range(num_qubits):
        entry = gate_1q_data.get(str(q))
        if entry and entry['error'] is not None:
            single_gate_error_per_qubit[q] = entry['error']

    # Per-pair 2Q gate errors
    two_gate_error_per_pair = {}
    gate_2q_data = cal['gates_2q'].get(basis_2q, {})
    for key, entry in gate_2q_data.items():
        if entry['error'] is not None:
            q1, q2 = map(int, key.split(','))
            two_gate_error_per_pair[(q1, q2)] = entry['error']

    # Gate durations
    durations_1q = [e['duration_ns'] for e in gate_1q_data.values()
                    if e['duration_ns'] is not None]
    durations_2q = [e['duration_ns'] for e in gate_2q_data.values()
                    if e['duration_ns'] is not None]

    return {
        'backend_name': cal['backend_name'],
        'num_hw_qubits': num_qubits,
        'snapshot_time': cal['snapshot_time'],
        # Per-qubit arrays
        't1_per_qubit': t1_per_qubit,
        't2_per_qubit': t2_per_qubit,
        'single_gate_error_per_qubit': single_gate_error_per_qubit,
        'two_gate_error_per_pair': two_gate_error_per_pair,
        # Scalar medians (for uniform models and fallbacks)
        'single_gate_error': float(np.median(single_gate_error_per_qubit[single_gate_error_per_qubit > 0])),
        'two_gate_error': float(np.median([e for e in two_gate_error_per_pair.values() if e < 1.0])),
        'single_gate_time': float(np.median(durations_1q)) if durations_1q else 36.0,
        'two_gate_time': float(np.median(durations_2q)) if durations_2q else 68.0,
        't1': float(np.median(t1_per_qubit)),
        't2': float(np.median(t2_per_qubit)),
    }


# ---------------------------------------------------------------------------
# Circuit generators
# ---------------------------------------------------------------------------
def make_ghz(n):
    qc = qiskit.QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


def make_qft(n):
    qc = qiskit.QuantumCircuit(n)
    qc.append(QFTGate(n), range(n))
    qc.measure_all()
    return qc


def make_random_circuit(n, depth):
    from qiskit.circuit.random import random_circuit
    qc = random_circuit(n, depth, measure=True, seed=42)
    return qc


def make_mirror_circuit(circ):
    """Build mirror circuit: circ^-1 & circ. Ideal output is |0...0>."""
    circ_no_meas = circ.copy()
    circ_no_meas.remove_final_measurements()
    circ_inv = circ_no_meas.inverse()
    circ_inv.barrier()
    mirror = circ_inv.compose(circ_no_meas)
    mirror.measure_all()
    return mirror


def generate_benchmark_circuits():
    """Generate a suite of circuits at various sizes."""
    circuits = []
    for n in [3, 5, 8, 10, 12, 15]:
        circuits.append((f'ghz_{n}', make_ghz(n)))
    for n in [3, 5, 8, 10, 12]:
        circuits.append((f'qft_{n}', make_qft(n)))
    for n in [3, 5, 8, 10]:
        for d in [5, 10, 20]:
            circuits.append((f'rand_{n}q_{d}d', make_random_circuit(n, d)))
    return circuits


# Gates that are virtual (zero error, zero duration) on real hardware.
VIRTUAL_GATES = frozenset({'rz', 'id', 'delay', 'barrier', 'measure'})

# ---------------------------------------------------------------------------
# ESP model (uniform error rates)
# ---------------------------------------------------------------------------
def estimate_esp(circ, params):
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
                prob *= (1 - params['single_gate_error'])
                slowest_gate = max(slowest_gate, params['single_gate_time'])
            else:
                prob *= (1 - params['two_gate_error'])
                slowest_gate = max(slowest_gate, params['two_gate_time'])
        total_time += slowest_gate

    t1_s = params['t1'] * 1e-9
    t2_s = params['t2'] * 1e-9
    total_time_s = total_time * 1e-9
    for _ in measured_qubits:
        prob *= np.exp(-total_time_s / t1_s) * (0.5 * np.exp(-total_time_s / t2_s) + 0.5)

    return prob


# ---------------------------------------------------------------------------
# Paper's depolarizing model (uniform, p_ent=0)
# ---------------------------------------------------------------------------
def estimate_depol(circ, params):
    """Paper's depolarizing channel model (Algorithm 1) with T1/T2."""
    n = circ.num_qubits
    qubits_fidelity = [1.0] * n

    lam_1q = 2 * params['single_gate_error'] / 1  # d*(F_g-1)/(1-d), d=2
    lam_2q = 4 * params['two_gate_error'] / 3      # d=4

    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    measured_qubits = set()

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
                qubits_fidelity[q] = (1 - lam_1q) * qubits_fidelity[q] + lam_1q / 2
                slowest_gate = max(slowest_gate, params['single_gate_time'])
            else:
                q1 = circ.find_bit(gate.qubits[0]).index
                q2 = circ.find_bit(gate.qubits[1]).index
                f1, f2 = qubits_fidelity[q1], qubits_fidelity[q2]
                sqrt_1_lam = np.sqrt(1 - lam_2q)
                eta = 0.5 * (np.sqrt((1 - lam_2q) * (f1 + f2)**2 + lam_2q) - sqrt_1_lam * (f1 + f2))
                qubits_fidelity[q1] = sqrt_1_lam * f1 + eta
                qubits_fidelity[q2] = sqrt_1_lam * f2 + eta
                slowest_gate = max(slowest_gate, params['two_gate_time'])

        # Per-layer coherence decay
        layer_time_s = slowest_gate * 1e-9
        t1_s = params['t1'] * 1e-9
        t2_s = params['t2'] * 1e-9
        for q in range(n):
            t1_decay = np.exp(-layer_time_s / t1_s)
            t2_decay = 0.5 * np.exp(-layer_time_s / t2_s) + 0.5
            qubits_fidelity[q] *= t1_decay * t2_decay

    if measured_qubits:
        return np.prod([qubits_fidelity[q] for q in measured_qubits])
    return np.prod(qubits_fidelity)


# ---------------------------------------------------------------------------
# quadris model
# ---------------------------------------------------------------------------
def build_single_core_setup(num_qubits):
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
            edges.append((j, i))
    coupling_map = CouplingMap(edges)
    core_mapping = {i: 0 for i in range(num_qubits)}
    return coupling_map, core_mapping


def estimate_quadris(transp, params, per_qubit=False):
    """Run quadris fidelity estimation. If per_qubit=True, use calibration arrays."""
    n = transp.num_qubits
    coupling_map, core_mapping = build_single_core_setup(n)

    kwargs = dict(
        circuit=transp,
        full_coupling_map=coupling_map,
        core_mapping=core_mapping,
        seed=42,
        initial_placement=InitialPlacement.RANDOM,
        single_gate_error=params['single_gate_error'],
        two_gate_error=params['two_gate_error'],
        teleportation_error_per_hop=0.0,
        single_gate_time=params['single_gate_time'],
        two_gate_time=params['two_gate_time'],
        teleportation_time_per_hop=1000.0,
        t1=params['t1'],
        t2=params['t2'],
        dynamic_decoupling=True,
        readout_mitigation_factor=0.95,
    )

    if per_qubit:
        # Slice calibration arrays to circuit qubit count
        kwargs['single_gate_error_per_qubit'] = params['single_gate_error_per_qubit'][:n]
        kwargs['t1_per_qubit'] = params['t1_per_qubit'][:n]
        kwargs['t2_per_qubit'] = params['t2_per_qubit'][:n]
        # Filter per-pair map to pairs within circuit qubit range
        kwargs['two_gate_error_per_pair'] = {
            (q1, q2): err
            for (q1, q2), err in params['two_gate_error_per_pair'].items()
            if q1 < n and q2 < n
        }
        kwargs['readout_error_per_qubit'] = params['readout_error_per_qubit'][:n]

    result = map_circuit(**kwargs)

    # Fidelity over measured qubits only
    measured_qubits = set()
    for inst in transp:
        if 'measure' in inst.operation.name:
            measured_qubits.add(transp.find_bit(inst.qubits[0]).index)

    if measured_qubits:
        fid = 1.0
        readout_errors = kwargs.get('readout_error_per_qubit')
        mitigation = kwargs.get('readout_mitigation_factor', 0.0)
        residual = 1.0 - mitigation
        for q in measured_qubits:
            fid *= result.algorithmic_fidelity_grid[-1, q]
            fid *= result.routing_fidelity_grid[-1, q]
            fid *= result.coherence_fidelity_grid[-1, q]
            if readout_errors is not None and q < len(readout_errors):
                fid *= 1.0 - readout_errors[q] * residual
        return fid
    return result.overall_fidelity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Benchmark quadris with real IBM Q calibration')
    parser.add_argument('--calibration', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data',
                                             'calibration_ibm_marrakesh_2026-04-07.json'),
                        help='Path to calibration snapshot JSON')
    args = parser.parse_args()

    print(f"Loading calibration from: {args.calibration}")
    params = load_calibration(args.calibration)
    print(f"Backend: {params['backend_name']} ({params['num_hw_qubits']} qubits)")
    print(f"Snapshot: {params['snapshot_time']}")
    print(f"Median errors: 1Q={params['single_gate_error']:.2e}, "
          f"2Q={params['two_gate_error']:.2e}")
    print(f"Median T1={params['t1']/1e3:.0f} us, T2={params['t2']/1e3:.0f} us")
    print(f"Gate times: 1Q={params['single_gate_time']:.0f} ns, "
          f"2Q={params['two_gate_time']:.0f} ns")
    print()

    # Native basis: Heron uses {sx, x, rz, cz}
    basis_gates = ['sx', 'x', 'rz', 'cz', 'measure']

    circuits = generate_benchmark_circuits()
    results = {
        'name': [], 'num_qubits': [], 'num_gates': [],
        'esp': [], 'depol': [], 'quadris_uniform': [], 'quadris_perqubit': [],
    }

    print(f"{'Circuit':<18} {'Qubits':>6} {'Gates':>6} "
          f"{'ESP':>8} {'Depol':>8} {'quadris(U)':>9} {'quadris(PQ)':>9} {'Delta':>8}")
    print('-' * 82)

    for name, circ in circuits:
        try:
            mirror = make_mirror_circuit(circ)
            transp = qiskit.transpile(
                mirror, basis_gates=basis_gates, optimization_level=3, seed_transpiler=42,
            )
            n_gates = len(transp.data)

            fid_esp = estimate_esp(transp, params)
            fid_depol = estimate_depol(transp, params)
            fid_uniform = estimate_quadris(transp, params, per_qubit=False)
            fid_perqubit = estimate_quadris(transp, params, per_qubit=True)

            delta = fid_perqubit - fid_uniform

            results['name'].append(name)
            results['num_qubits'].append(circ.num_qubits)
            results['num_gates'].append(n_gates)
            results['esp'].append(fid_esp)
            results['depol'].append(fid_depol)
            results['quadris_uniform'].append(fid_uniform)
            results['quadris_perqubit'].append(fid_perqubit)

            print(f"  {name:<16} {circ.num_qubits:>6} {n_gates:>6} "
                  f"{fid_esp:>8.4f} {fid_depol:>8.4f} {fid_uniform:>9.4f} {fid_perqubit:>9.4f} "
                  f"{delta:>+8.4f}")

        except Exception as e:
            print(f"  {name:<16} ERROR: {e}")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    v_gates = results['num_gates']
    v_nq = results['num_qubits']
    scatter_size = 25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [3, 1]})

    # Left: Fidelity vs gates
    ax1.scatter(v_gates, results['esp'],
                label=f"ESP + $T_{{1,2}}$ (uniform)", color='tab:orange', alpha=0.75,
                s=[scatter_size * q for q in v_nq])
    ax1.scatter(v_gates, results['depol'],
                label=f"Depol. + $T_{{1,2}}$ (uniform)", color='tab:green', alpha=0.75,
                s=[scatter_size * q for q in v_nq])
    ax1.scatter(v_gates, results['quadris_uniform'],
                label='quadris (uniform)', color='tab:red', alpha=0.75,
                s=[scatter_size * q for q in v_nq], marker='x')
    ax1.scatter(v_gates, results['quadris_perqubit'],
                label=f"quadris (per-qubit, {params['backend_name']})", color='tab:blue', alpha=0.85,
                s=[scatter_size * q for q in v_nq], marker='D')

    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Gates', fontsize=18)
    ax1.set_ylabel('Fidelity', fontsize=18)
    ax1.set_title(f"Mirror-Circuit Fidelity — {params['backend_name']} calibration", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=13, loc='upper right')
    ax1.set_ylim(bottom=-0.05, top=1.05)

    # Right: boxplot of differences (per-qubit vs uniform)
    esp_diff = [results['esp'][i] - results['quadris_uniform'][i] for i in range(len(v_gates))]
    depol_diff = [results['depol'][i] - results['quadris_uniform'][i] for i in range(len(v_gates))]
    pq_diff = [results['quadris_perqubit'][i] - results['quadris_uniform'][i] for i in range(len(v_gates))]

    colors = ['tab:orange', 'tab:green', 'tab:blue']
    labels = ['ESP\nvs uniform', 'Depol\nvs uniform', 'Per-qubit\nvs uniform']
    box = ax2.boxplot(
        [esp_diff, depol_diff, pq_diff],
        positions=[1, 2, 3],
        labels=labels,
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
    for median in box['medians']:
        median.set_color('black')

    ax2.set_ylabel('Fidelity Difference vs quadris(uniform)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.axhline(y=0, color='gray', linestyle='--', zorder=-10)
    ax2.set_title('Model Comparison', fontsize=14)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_real_hw.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    # Save numerical results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_real_hw_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    plt.show()


if __name__ == '__main__':
    main()
