"""
Run mirror-circuit benchmarks on live IBM Q hardware and compare against
four fidelity models: ESP, Depol, qusim(uniform), qusim(per-qubit).

This script CONSUMES IBM Quantum runtime budget (~1-3 min of the 10 min free tier).

Usage:
  python examples/benchmark_live_hw.py [--backend ibm_marrakesh] [--shots 1000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit.circuit.library import QFTGate
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from qusim import map_circuit, InitialPlacement

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ---------------------------------------------------------------------------
# Calibration loader (same as benchmark_real_hw.py)
# ---------------------------------------------------------------------------
def load_calibration(path, basis_1q='sx', basis_2q='cz'):
    with open(path) as f:
        cal = json.load(f)

    num_qubits = cal['num_qubits']

    t1_per_qubit = np.array([
        cal['qubits'][str(q)]['t1_us'] * 1e3 if cal['qubits'][str(q)]['t1_us'] else 100_000.0
        for q in range(num_qubits)
    ])
    t2_per_qubit = np.array([
        cal['qubits'][str(q)]['t2_us'] * 1e3 if cal['qubits'][str(q)]['t2_us'] else 50_000.0
        for q in range(num_qubits)
    ])

    single_gate_error_per_qubit = np.zeros(num_qubits)
    gate_1q_data = cal['gates_1q'].get(basis_1q, {})
    for q in range(num_qubits):
        entry = gate_1q_data.get(str(q))
        if entry and entry['error'] is not None:
            single_gate_error_per_qubit[q] = entry['error']

    two_gate_error_per_pair = {}
    gate_2q_data = cal['gates_2q'].get(basis_2q, {})
    for key, entry in gate_2q_data.items():
        if entry['error'] is not None:
            q1, q2 = map(int, key.split(','))
            two_gate_error_per_pair[(q1, q2)] = entry['error']

    durations_1q = [e['duration_ns'] for e in gate_1q_data.values()
                    if e['duration_ns'] is not None]
    durations_2q = [e['duration_ns'] for e in gate_2q_data.values()
                    if e['duration_ns'] is not None]

    # Per-qubit readout errors
    readout_error_per_qubit = np.array([
        cal['qubits'][str(q)].get('readout_error', 0.0) or 0.0
        for q in range(num_qubits)
    ])

    return {
        'backend_name': cal['backend_name'],
        'num_hw_qubits': num_qubits,
        'snapshot_time': cal['snapshot_time'],
        't1_per_qubit': t1_per_qubit,
        't2_per_qubit': t2_per_qubit,
        'single_gate_error_per_qubit': single_gate_error_per_qubit,
        'two_gate_error_per_pair': two_gate_error_per_pair,
        'readout_error_per_qubit': readout_error_per_qubit,
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


def make_random_circuit(n, depth, seed=42):
    from qiskit.circuit.random import random_circuit
    return random_circuit(n, depth, measure=True, seed=seed)


def make_mirror_circuit(circ):
    circ_no_meas = circ.copy()
    circ_no_meas.remove_final_measurements()
    circ_inv = circ_no_meas.inverse()
    circ_inv.barrier()
    mirror = circ_inv.compose(circ_no_meas)
    mirror.measure_all()
    return mirror


def generate_benchmark_circuits():
    """Circuits sized for 10-min free tier budget."""
    circuits = []
    for n in [3, 5, 8, 10, 12]:
        circuits.append((f'ghz_{n}', make_ghz(n)))
    for n in [3, 5, 8, 10]:
        circuits.append((f'qft_{n}', make_qft(n)))
    for n in [3, 5, 8]:
        for d in [5, 10, 20]:
            circuits.append((f'rand_{n}q_{d}d', make_random_circuit(n, d)))
    return circuits


# Gates that are virtual (zero error, zero duration) on real hardware.
VIRTUAL_GATES = frozenset({'rz', 'id', 'delay', 'barrier', 'measure'})


# ---------------------------------------------------------------------------
# Fidelity models (ESP, Depol, qusim)
# ---------------------------------------------------------------------------
def estimate_esp(circ, params):
    dag = circuit_to_dag(circ)
    layers = [dag_to_circuit(layer['graph']) for layer in dag.layers()]
    measured_qubits = set()

    prob = 1.0
    total_time = 0.0

    for layer in layers:
        slowest_gate = 0.0
        for gate in layer:
            if gate.operation.name == 'measure':
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


def estimate_depol(circ, params):
    n = circ.num_qubits
    qubits_fidelity = [1.0] * n

    lam_1q = 2 * params['single_gate_error'] / 1
    lam_2q = 4 * params['two_gate_error'] / 3

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


def estimate_qusim(transp, params, per_qubit=False):
    n = transp.num_qubits
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edges.append((j, i))
    coupling_map = CouplingMap(edges)
    core_mapping = {i: 0 for i in range(n)}

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
    )

    if per_qubit:
        kwargs['single_gate_error_per_qubit'] = params['single_gate_error_per_qubit'][:n]
        kwargs['t1_per_qubit'] = params['t1_per_qubit'][:n]
        kwargs['t2_per_qubit'] = params['t2_per_qubit'][:n]
        kwargs['two_gate_error_per_pair'] = {
            (q1, q2): err
            for (q1, q2), err in params['two_gate_error_per_pair'].items()
            if q1 < n and q2 < n
        }

    result = map_circuit(**kwargs)

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
    parser = argparse.ArgumentParser(description='Live IBM Q benchmark for qusim')
    parser.add_argument('--backend', type=str, default='ibm_marrakesh')
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to calibration JSON (auto-detected if omitted)')
    parser.add_argument('--offline', action='store_true',
                        help='Use saved results instead of submitting to IBM Quantum')
    args = parser.parse_args()

    # --- Connect to IBM ---
    print(f"Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # --- Load calibration ---
    cal_path = args.calibration
    if cal_path is None:
        # Look for most recent calibration file
        import glob
        pattern = os.path.join(DATA_DIR, f'calibration_{args.backend}_*.json')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No calibration file found matching {pattern}")
            print("Run snapshot_calibration.py first, or pass --calibration")
            sys.exit(1)
        cal_path = files[-1]

    print(f"Loading calibration: {cal_path}")
    params = load_calibration(cal_path)
    print(f"Median errors: 1Q={params['single_gate_error']:.2e}, "
          f"2Q={params['two_gate_error']:.2e}")
    print(f"Median T1={params['t1']/1e3:.0f} us, T2={params['t2']/1e3:.0f} us")
    print()

    # --- Generate and transpile circuits ---
    raw_circuits = generate_benchmark_circuits()
    print(f"Generated {len(raw_circuits)} benchmark circuits")
    print("Building mirror circuits and transpiling to backend...")

    transpiled = []
    circuit_meta = []
    for name, circ in raw_circuits:
        try:
            mirror = make_mirror_circuit(circ)
            transp = qiskit.transpile(
                mirror, backend=backend, optimization_level=3, seed_transpiler=42,
            )
            n_gates = sum(1 for inst in transp if inst.operation.name not in ('barrier', 'measure'))
            transpiled.append(transp)
            circuit_meta.append({
                'name': name,
                'num_qubits': circ.num_qubits,
                'num_gates': n_gates,
                'num_gates_total': len(transp.data),
            })
            print(f"  {name:<16} {circ.num_qubits:>3}q  {n_gates:>5} gates")
        except Exception as e:
            print(f"  {name:<16} SKIP: {e}")

    print(f"\nSubmitting {len(transpiled)} circuits to {backend.name} "
          f"({args.shots} shots each)...")
    print("This will consume IBM Quantum runtime budget.")

    # --- Submit to hardware or load from saved ---
    t_start = time.time()
    
    if args.offline:
        print("\n[OFFLINE MODE] Loading hardware fidelities from saved results...")
        saved_results_path = os.path.join(DATA_DIR, 'benchmark_live_results.json')
        with open(saved_results_path) as f:
            saved_data = json.load(f)
        
        hw_fidelities = []
        for meta in circuit_meta:
            # Find matching circuit in saved data
            match = next(c for c in saved_data['circuits'] if c['name'] == meta['name'] and c['num_qubits'] == meta['num_qubits'])
            hw_fidelities.append(match['hw_fidelity'])
            
        t_elapsed = 0.0
        job_id = saved_data.get('job_id', 'offline')
        print(f"Loaded {len(hw_fidelities)} hardware fidelities.")
        
    else:
        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=args.shots)
        job_id = job.job_id()
        print(f"Job ID: {job_id}")
        print("Waiting for results...")
    
        result = job.result()
        t_elapsed = time.time() - t_start
        print(f"Job completed in {t_elapsed:.1f}s wall time")
    
        # --- Extract hardware fidelity ---
        hw_fidelities = []
        for i, meta in enumerate(circuit_meta):
            pub_result = result[i]
            counts = pub_result.data.meas.get_counts()
            n_meas = meta['num_qubits']
            zero_state = '0' * n_meas
            hw_fid = counts.get(zero_state, 0) / args.shots
            hw_fidelities.append(hw_fid)

    # --- Compute model predictions ---
    print("\nComputing model predictions on transpiled circuits...")
    esp_fids, depol_fids, uniform_fids, perqubit_fids = [], [], [], []

    for i, (transp, meta) in enumerate(zip(transpiled, circuit_meta)):
        fid_esp = estimate_esp(transp, params)
        fid_depol = estimate_depol(transp, params)
        fid_uniform = estimate_qusim(transp, params, per_qubit=False)
        fid_perqubit = estimate_qusim(transp, params, per_qubit=True)

        esp_fids.append(fid_esp)
        depol_fids.append(fid_depol)
        uniform_fids.append(fid_uniform)
        perqubit_fids.append(fid_perqubit)

    # --- Print results table ---
    print()
    print(f"{'Circuit':<18} {'Qubits':>6} {'Gates':>6} "
          f"{'HW':>8} {'ESP':>8} {'Depol':>8} {'qusim(U)':>9} {'qusim(PQ)':>9}")
    print('=' * 90)

    for i, meta in enumerate(circuit_meta):
        print(f"  {meta['name']:<16} {meta['num_qubits']:>6} {meta['num_gates']:>6} "
              f"{hw_fidelities[i]:>8.4f} {esp_fids[i]:>8.4f} {depol_fids[i]:>8.4f} "
              f"{uniform_fids[i]:>9.4f} {perqubit_fids[i]:>9.4f}")

    # --- Compute error metrics ---
    hw = np.array(hw_fidelities)
    models = {
        'ESP': np.array(esp_fids),
        'Depol': np.array(depol_fids),
        'qusim(uniform)': np.array(uniform_fids),
        'qusim(per-qubit)': np.array(perqubit_fids),
    }

    print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'Max Err':>8} {'R²':>8}")
    print('-' * 55)
    for name, pred in models.items():
        residuals = pred - hw
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        max_err = np.max(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hw - np.mean(hw))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"  {name:<18} {mae:>8.4f} {rmse:>8.4f} {max_err:>8.4f} {r2:>8.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # (1) Scatter: predicted vs measured
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect prediction')
    ax1.scatter(hw, esp_fids, label='ESP', color='tab:orange', alpha=0.7, s=60)
    ax1.scatter(hw, depol_fids, label='Depol', color='tab:green', alpha=0.7, s=60)
    ax1.scatter(hw, uniform_fids, label='qusim (uniform)', color='tab:red', alpha=0.7,
                s=60, marker='x')
    ax1.scatter(hw, perqubit_fids, label='qusim (per-qubit)', color='tab:blue', alpha=0.85,
                s=60, marker='D')
    ax1.set_xlabel('Hardware Fidelity (measured)', fontsize=14)
    ax1.set_ylabel('Model Prediction', fontsize=14)
    ax1.set_title('Predicted vs Measured Fidelity', fontsize=15)
    ax1.legend(fontsize=11)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')

    # (2) Fidelity vs gates (all 5 series)
    ax2 = axes[1]
    gates = [m['num_gates'] for m in circuit_meta]
    ax2.scatter(gates, hw, label='Hardware', color='black', s=80, marker='s', zorder=5)
    ax2.scatter(gates, esp_fids, label='ESP', color='tab:orange', alpha=0.7, s=50)
    ax2.scatter(gates, depol_fids, label='Depol', color='tab:green', alpha=0.7, s=50)
    ax2.scatter(gates, uniform_fids, label='qusim (uniform)', color='tab:red', alpha=0.7,
                s=50, marker='x')
    ax2.scatter(gates, perqubit_fids, label='qusim (per-qubit)', color='tab:blue', alpha=0.85,
                s=50, marker='D')
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Gates', fontsize=14)
    ax2.set_ylabel('Fidelity', fontsize=14)
    ax2.set_title(f'Fidelity vs Circuit Size — {backend.name}', fontsize=15)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)

    # (3) Residuals boxplot
    ax3 = axes[2]
    residuals_data = [m - hw for m in models.values()]
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:blue']
    labels = list(models.keys())
    box = ax3.boxplot(residuals_data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box['medians']:
        median.set_color('black')
    ax3.axhline(0, color='gray', linestyle='--', zorder=-10)
    ax3.set_ylabel('Prediction - Hardware', fontsize=14)
    ax3.set_title('Prediction Error Distribution', fontsize=15)
    ax3.tick_params(axis='x', rotation=15)

    plt.suptitle(f'qusim Live Benchmark — {backend.name} ({args.shots} shots)',
                 fontsize=17, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_live_hw.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")

    # --- Save results ---
    results_out = {
        'backend': backend.name,
        'shots': args.shots,
        'calibration': cal_path,
        'job_id': job_id,
        'wall_time_s': t_elapsed,
        'circuits': [],
    }
    for i, meta in enumerate(circuit_meta):
        results_out['circuits'].append({
            **meta,
            'hw_fidelity': hw_fidelities[i],
            'esp': esp_fids[i],
            'depol': depol_fids[i],
            'qusim_uniform': uniform_fids[i],
            'qusim_perqubit': perqubit_fids[i],
        })

    results_path = os.path.join(DATA_DIR, 'benchmark_live_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
