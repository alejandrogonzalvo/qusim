import json
import time
import subprocess
import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit.circuit.library import QFT
from qiskit import transpile
import sparse
import sys
import os

# Ensure we can import from dse_pau
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import qiskit_circ_to_slices, all_to_all_topology
from HQA import HQA_variation

def serialize_gs(Gs):
    """Serialize sparse interaction tensor to flat [layer, u, v, weight] format."""
    gs_sparse = []
    for layer_idx in range(Gs.shape[0]):
        slice_g = Gs[layer_idx]
        coords = slice_g.coords
        data = slice_g.data
        for j in range(len(data)):
            gs_sparse.append([float(layer_idx), float(coords[0][j]), float(coords[1][j]), float(data[j])])
    return gs_sparse

def run_bench():
    core_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    qubits_per_core = 10
    
    python_times = []
    rust_times = []
    speedups = []
    
    # Path to Rust binary
    rust_bin = "./target/release/hqa_cli"
    if not os.path.exists(rust_bin):
        print(f"Error: Rust binary not found at {rust_bin}. Run 'cargo build --release --bin hqa_cli' first.")
        return

    for num_cores in core_counts:
        virtual_qubits = num_cores * qubits_per_core
        print(f"Benchmarking {num_cores} cores, {virtual_qubits} qubits...")
        
        # 1. Generate QFT circuit
        circ = QFT(virtual_qubits)
        transp_circ = transpile(circ, basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], seed_transpiler=42)
        Gs_two, Gs_all = qiskit_circ_to_slices(transp_circ)
        Gs = Gs_all
        
        # 2. Setup initial partition and topology
        dist_matrix = all_to_all_topology(num_cores)
        core_cap = [qubits_per_core] * num_cores
        part = [(i % num_cores) for i in range(virtual_qubits)]
        random.seed(42)
        random.shuffle(part)
        
        Ps_init = np.zeros((Gs.shape[0] + 1, virtual_qubits), dtype=int)
        Ps_init[0] = part
        
        # 3. Python Execution
        print(f"  Running Python HQA ({Gs.shape[0]} slices)...")
        start_py = time.perf_counter()
        HQA_variation(Gs, Ps_init.copy(), num_cores, virtual_qubits, core_cap.copy(), distance_matrix=dist_matrix)
        end_py = time.perf_counter()
        py_time_ms = (end_py - start_py) * 1000.0
        python_times.append(py_time_ms)
        print(f"  Python: {py_time_ms:.2f} ms")
        
        # 4. Rust Execution (via CLI)
        print(f"  Running Rust HQA...")
        gs_sparse = serialize_gs(Gs)
        rust_input = {
            "num_virtual_qubits": virtual_qubits,
            "num_cores": num_cores,
            "num_layers": Gs.shape[0],
            "gs_sparse": gs_sparse,
            "input_initial_partition": part,
            "input_core_capacities": core_cap,
            "input_distance_matrix": dist_matrix,
        }
        
        rust_input_json = json.dumps(rust_input)
        
        process = subprocess.Popen(
            [rust_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=rust_input_json)
        
        if process.returncode != 0:
            print(f"Rust CLI failed for {num_cores} cores!")
            print(stderr)
            rust_times.append(None)
            speedups.append(None)
            continue
            
        rust_output = json.loads(stdout)
        rust_time_ms = rust_output["execution_time_ms"]
        rust_times.append(rust_time_ms)
        speedups.append(py_time_ms / rust_time_ms)
        print(f"  Rust:   {rust_time_ms:.2f} ms")

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Subplot 1: Execution Time
    ax1.plot(core_counts, python_times, 'o-', label='Python', color='red', linewidth=2)
    ax1.plot(core_counts, rust_times, 's-', label='Rust', color='blue', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Cores (10 Qubits per Core)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('HQA Execution Time: Python vs Rust')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    for i, txt in enumerate(python_times):
        ax1.annotate(f"{txt:.1f}", (core_counts[i], python_times[i]), textcoords="offset points", xytext=(0,10), ha='center')
    for i, txt in enumerate(rust_times):
        if txt:
            ax1.annotate(f"{txt:.1f}", (core_counts[i], rust_times[i]), textcoords="offset points", xytext=(0,-20), ha='center')

    # Subplot 2: Speedup
    valid_speedups = [s for s in speedups if s is not None]
    valid_cores = [c for i, c in enumerate(core_counts) if speedups[i] is not None]
    ax2.plot(valid_cores, valid_speedups, 'D-', label='Speedup (Py/Rust)', color='green', linewidth=2)
    ax2.set_xlabel('Number of Cores (10 Qubits per Core)')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_title('HQA Speedup: Rust Performance Gain')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    for i, txt in enumerate(speedups):
        if txt:
            ax2.annotate(f"{txt:.1f}x", (core_counts[i], speedups[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plot_path = "hqa_scalability_plot.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    
    # 6. Report speedups
    print("\nScalability Results Summary:")
    print(f"{'Cores':>6} | {'Qubits':>8} | {'Python (ms)':>12} | {'Rust (ms)':>10} | {'Speedup':>10}")
    print("-" * 57)
    for i, cores in enumerate(core_counts):
        qubits = cores * qubits_per_core
        if rust_times[i]:
            speedup = python_times[i] / rust_times[i]
            print(f"{cores:>6} | {qubits:>8} | {python_times[i]:>12.2f} | {rust_times[i]:>10.2f} | {speedup:>9.1f}x")

if __name__ == "__main__":
    run_bench()
