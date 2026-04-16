"""
Benchmark: 3D cold-path sweep (cores x qubits x T1).

Run with:
    python -m tests.bench_parallel_sweep

Measures wall-clock time for a 3D sweep with two cold-path axes
(num_cores, num_qubits) and one hot-path axis (t1).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.dse_engine import DSEEngine, SweepProgress


def run_benchmark(parallel=False):
    engine = DSEEngine()

    cold_config = {
        "circuit_type": "qft",
        "num_qubits": 16,
        "num_cores": 2,
        "topology_type": "ring",
        "placement_policy": "random",
        "seed": 42,
        "intracore_topology": "all_to_all",
    }
    fixed_noise = {
        "single_gate_error": 1e-4,
        "two_gate_error": 1e-3,
        "teleportation_error_per_hop": 1e-2,
        "t1": 100_000.0,
        "t2": 50_000.0,
        "single_gate_time": 20.0,
        "two_gate_time": 100.0,
        "teleportation_time_per_hop": 1_000.0,
        "readout_mitigation_factor": 0.0,
    }

    # Initial cold run to warm up imports / JIT
    cached = engine.run_cold(**cold_config, noise=fixed_noise)

    count = [0]

    def on_progress(p: SweepProgress):
        count[0] += 1
        if count[0] % 25 == 0 or p.completed == p.total:
            print(f"  [{p.percentage:5.1f}%] {p.completed}/{p.total}")

    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    print("=" * 60)
    print(f"3D Sweep ({mode}): num_cores x num_qubits x t1")
    print(f"  cores:  1 -> 8")
    print(f"  qubits: 4 -> 128")
    print(f"  t1:     1e4 -> 1e6 (log)")
    print("=" * 60)

    t0 = time.perf_counter()

    xs, ys, zs, grid = engine.sweep_3d(
        cached=cached,
        metric_key1="num_cores",
        low1=1,
        high1=8,
        metric_key2="num_qubits",
        low2=4,
        high2=128,
        metric_key3="t1",
        low3=4,
        high3=6,
        fixed_noise=fixed_noise,
        cold_config=cold_config,
        progress_callback=on_progress,
        parallel=parallel,
    )

    elapsed = time.perf_counter() - t0

    total_points = len(xs) * len(ys) * len(zs)
    print("=" * 60)
    print(f"Mode:         {mode}")
    print(f"Total points: {total_points}")
    print(f"Elapsed:      {elapsed:.2f}s")
    print(f"Per point:    {elapsed / total_points:.3f}s")
    print(f"Cores axis:   {list(xs)}")
    print(f"Qubits axis:  {list(ys)}")
    print(f"T1 axis:      {list(zs)}")
    print("=" * 60)
    return elapsed


if __name__ == "__main__":
    t_seq = run_benchmark(parallel=False)
    print()
    t_par = run_benchmark(parallel=True)
    print()
    print(f"Speedup: {t_seq / t_par:.2f}x  ({t_seq:.2f}s -> {t_par:.2f}s)")
