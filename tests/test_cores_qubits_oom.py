"""Memory profile for the cores x qubits sweep scheduler.

A 2D sweep of num_cores x num_qubits runs each unique cold config as one job
in a process pool.  Prior to the qubit-aware scheduler, the cap used a fixed
per-worker estimate (800 MB) that was ~4.6x too low at 256 qubits, causing
OOM at high parallelism.  The scheduler now sizes concurrency from a
qubit-aware estimate so large-qubit groups serialize while small ones fan out.

This test:
  * measures real peak RSS for a single cold compilation at each qubit value
    along the user's sweep axis (4..256)
  * compares those measurements to the scheduler's estimate
  * simulates the scheduler's worst-case concurrent footprint for a 2D
    cores x qubits sweep and asserts it fits in available RAM
"""

import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

import pytest


# Must mirror production
_MP_CONTEXT = multiprocessing.get_context("forkserver")


def _worker(num_qubits: int, num_cores: int, out_queue) -> None:
    """Run one cold compilation, report wall time and peak RSS."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "python"))
    from gui.dse_engine import _eval_cold_batch

    cfg = dict(
        circuit_type="qft",
        num_qubits=num_qubits,
        num_cores=min(num_cores, num_qubits),
        topology_type="ring",
        placement_policy="random",
        seed=42,
        intracore_topology="all_to_all",
    )
    noise = dict(
        single_gate_error=1e-4,
        two_gate_error=1e-3,
        teleportation_error_per_hop=1e-2,
        t1=1e5,
        t2=5e4,
        single_gate_time=20.0,
        two_gate_time=100.0,
        teleportation_time_per_hop=1e3,
        readout_mitigation_factor=0.0,
    )
    swept = [{"num_qubits": num_qubits, "num_cores": min(num_cores, num_qubits)}]

    t0 = time.time()
    _eval_cold_batch(cfg, noise, swept)
    elapsed = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out_queue.put((elapsed, peak_mb))


def _measure(num_qubits: int, num_cores: int, timeout_s: int = 180) -> tuple[float, float] | None:
    """Run one cold compilation in a fresh forkserver child; return (seconds, MB) or None."""
    q = _MP_CONTEXT.Queue()
    p = _MP_CONTEXT.Process(target=_worker, args=(num_qubits, num_cores, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if p.exitcode != 0:
        return None
    try:
        return q.get(timeout=5)
    except Exception:
        return None


def _mem_available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def _mem_total_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    return 0


@pytest.mark.skipif(sys.platform != "linux", reason="Uses /proc/meminfo and RUSAGE_SELF.ru_maxrss semantics")
def test_cores_qubits_sweep_ram_hypothesis():
    from gui.dse_engine import _RESERVED_RAM_MB, _estimate_cold_mb, DSEEngine

    # Same qubit values the UI would sweep with max_cold=64 (2 cold axes -> 8 per axis)
    # and a qubits range of 4..256.  Linspace rounded to int.
    qubit_axis = [4, 40, 76, 112, 148, 184, 220, 256]
    # Measure the full axis (subset that completes in reasonable time).
    probe_values = qubit_axis

    peaks: dict[int, float] = {}
    timings: dict[int, float] = {}
    for nq in probe_values:
        result = _measure(nq, num_cores=4, timeout_s=600)
        if result is None:
            print(f"  {nq:3d} qubits: TIMEOUT or CRASH (possible OOM already)")
            break
        elapsed, mb = result
        timings[nq] = elapsed
        peaks[nq] = mb

    assert peaks, "No successful measurements (all workers crashed / timed out)"

    n_cpus = os.cpu_count() or 2
    cpu_cap = max(1, n_cpus // 2)
    avail_mb = _mem_available_mb()
    total_mb = _mem_total_mb()
    mem_budget = max(1, avail_mb - _RESERVED_RAM_MB)

    # Compare measurement vs. the scheduler's estimate.
    print()
    print("=" * 72)
    print("Cores x Qubits sweep — measurement vs. scheduler estimate")
    print("=" * 72)
    print(f"System:       {n_cpus} CPUs, {total_mb} MB total, {avail_mb} MB available")
    print(f"Sweep axis:   qubits = {qubit_axis}")
    print()
    print(f"{'qubits':>7} {'measured':>12} {'estimate':>12} {'margin':>10}")
    for nq in sorted(peaks):
        est = _estimate_cold_mb(nq)
        margin = est - peaks[nq]
        print(f"{nq:7d} {peaks[nq]:10.0f} MB {est:10.0f} MB {margin:+8.0f} MB")
    print()

    # Simulate the new scheduler: largest-first packing inside mem_budget,
    # capped by cpu_cap slots. This mirrors _parallel_cold_sweep.
    qubit_values = sorted(qubit_axis, reverse=True)
    worst_concurrent = 0
    used = 0
    slots_used = 0
    admitted: list[int] = []
    for nq in qubit_values:
        cost = _estimate_cold_mb(nq)
        if slots_used < cpu_cap and (not admitted or used + cost <= mem_budget):
            admitted.append(nq)
            used += cost
            slots_used += 1
            worst_concurrent = max(worst_concurrent, used)

    print(f"Scheduler worst-case first wave (largest-first packing):")
    print(f"  admitted qubits:     {admitted}")
    print(f"  concurrent estimate: {used:.0f} MB")
    print(f"  mem budget:          {mem_budget} MB")
    print(f"  cpu cap:             {cpu_cap} slots")
    print()

    # Also rough-check using measured peaks: the largest admitted job's real
    # peak should still leave headroom below avail_mb when combined with other
    # admitted jobs' real peaks.
    real_concurrent = 0.0
    for nq in admitted:
        # interpolate measured peak for nq if not in peaks
        if nq in peaks:
            real_concurrent += peaks[nq]
        else:
            xs = sorted(peaks)
            ys = [peaks[x] for x in xs]
            if nq <= xs[0]:
                real_concurrent += ys[0]
            elif nq >= xs[-1]:
                real_concurrent += ys[-1]
            else:
                for a, b in zip(xs, xs[1:]):
                    if a <= nq <= b:
                        ya, yb = peaks[a], peaks[b]
                        real_concurrent += ya + (yb - ya) * (nq - a) / (b - a)
                        break

    print(f"Real peak of the admitted first wave: {real_concurrent:.0f} MB")
    print(f"Leaves {avail_mb - real_concurrent:.0f} MB of free RAM at the worst moment")
    print("=" * 72)

    # Scheduler must keep the first-wave real peak under available RAM with
    # the reserved headroom.
    assert real_concurrent + _RESERVED_RAM_MB <= avail_mb, (
        f"Scheduler first wave would exceed available RAM: "
        f"{real_concurrent:.0f} MB + {_RESERVED_RAM_MB} MB reserve "
        f"> {avail_mb} MB available"
    )
