"""
Benchmark: QFT cold-path compilation time — TeleSABRE vs HQA+Sabre.

Run with:
    python -m tests.bench_telesabre_vs_hqa

Reports median wall-clock time per routing algorithm across several QFT
sizes, with 3 timed repetitions per (algorithm, size) pair.
"""

import sys
import os
import time
import statistics
import contextlib


@contextlib.contextmanager
def _suppress_c_stdout():
    """Redirect file-descriptor 1 (C-level stdout) to /dev/null temporarily.

    Python's sys.stdout.write goes through fd 1, so this silences both
    Python print() and C library printf() calls.  We flush the C-level
    stdio buffers before restoring fd 1 so buffered output doesn't spill
    onto the terminal after the context exits.
    """
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, 1)
        yield
    finally:
        libc.fflush(None)   # flush all C stdio streams (including TeleSABRE's)
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.dse_engine import DSEEngine

QUBIT_SIZES = [5, 10, 15, 20, 25]
NUM_CORES = 4
TOPOLOGY = "ring"
INTRACORE = "all_to_all"
SEED = 42
REPS = 3

NOISE = {
    "single_gate_error": 1e-4,
    "two_gate_error": 1e-3,
    "teleportation_error_per_hop": 1e-2,
    "t1": 100_000.0,
    "t2": 50_000.0,
    "single_gate_time": 20.0,
    "two_gate_time": 100.0,
    "teleportation_time_per_hop": 1_000.0,
    "readout_mitigation_factor": 0.0,
    "classical_link_width": 0,
    "classical_clock_freq_hz": 200e6,
    "classical_routing_cycles": 2,
}


def time_cold(algorithm: str, num_qubits: int) -> list[float]:
    """Return wall-clock times (seconds) for REPS cold compilations."""
    times = []
    for rep in range(REPS):
        engine = DSEEngine()  # fresh engine — no cache
        t0 = time.perf_counter()
        with _suppress_c_stdout():
            engine.run_cold(
                circuit_type="qft",
                num_qubits=num_qubits,
                num_cores=NUM_CORES,
                topology_type=TOPOLOGY,
                intracore_topology=INTRACORE,
                placement_policy="random",
                seed=SEED + rep,
                noise=NOISE,
                routing_algorithm=algorithm,
            )
        times.append(time.perf_counter() - t0)
    return times


def fmt(t: float) -> str:
    if t < 1:
        return f"{t*1000:.1f} ms"
    return f"{t:.2f} s"


def main():
    results: dict[str, dict[int, list[float]]] = {
        "hqa_sabre": {},
        "telesabre": {},
    }

    algorithms = ["hqa_sabre", "telesabre"]

    # Warm up imports / JIT once before timing
    print("Warming up... ", end="", flush=True)
    for algo in algorithms:
        warm_engine = DSEEngine()
        with _suppress_c_stdout():
            warm_engine.run_cold(
                circuit_type="qft",
                num_qubits=5,
                num_cores=2,
                topology_type="ring",
                intracore_topology="all_to_all",
                placement_policy="random",
                seed=0,
                noise=NOISE,
                routing_algorithm=algo,
            )
    print("done.\n")

    col_w = 14
    header = f"{'Qubits':>8}"
    for algo in algorithms:
        label = "HQA+Sabre" if algo == "hqa_sabre" else "TeleSABRE"
        header += f"  {label:>{col_w}}"
    header += f"  {'Ratio (TS/HQA)':>{col_w}}"
    print(header)
    print("-" * len(header))

    for n in QUBIT_SIZES:
        row = f"{n:>8}"
        medians = {}
        for algo in algorithms:
            label = "HQA+Sabre" if algo == "hqa_sabre" else "TeleSABRE"
            print(f"  Timing {label:10s} QFT-{n} ({REPS} reps)...", end=" ", flush=True)
            times = time_cold(algo, n)
            results[algo][n] = times
            med = statistics.median(times)
            medians[algo] = med
            print(fmt(med))
            row += f"  {fmt(med):>{col_w}}"

        if medians.get("hqa_sabre", 0) > 0:
            ratio = medians["telesabre"] / medians["hqa_sabre"]
            row += f"  {ratio:>{col_w}.2f}x"
        else:
            row += f"  {'N/A':>{col_w}}"

        print()

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for n in QUBIT_SIZES:
        row = f"{n:>8}"
        medians = {}
        for algo in algorithms:
            med = statistics.median(results[algo][n])
            medians[algo] = med
            row += f"  {fmt(med):>{col_w}}"
        if medians.get("hqa_sabre", 0) > 0:
            ratio = medians["telesabre"] / medians["hqa_sabre"]
            row += f"  {ratio:>{col_w}.2f}x"
        print(row)
    print("=" * len(header))
    print(f"\nConfig: {NUM_CORES} cores, {TOPOLOGY} inter-core, {INTRACORE} intra-core, {REPS} reps/point")


if __name__ == "__main__":
    main()
