"""
Benchmark runner for the Quadrature report (docs/QUADRATURE_REPORT.md).

Runs every measurement quoted in section 5 of the report and emits a
deterministic plain-text dump on stdout. Numbers reproducible by:

    .venv/bin/python tests/bench_quadrature_report.py

The script intentionally keeps benchmark sizes modest so the full run
finishes in a few minutes on a developer machine — the report quotes
the relative numbers, not absolute throughput.

Bench coverage:
  B1  Cold-path latency vs. num_logical_qubits
  B2  Cold-path peak RSS vs. num_logical_qubits  (sub-process measurement)
  B3  Hot-path single run_hot latency
  B4  Hot-path batched run_hot_batch (1, 10, 100, 1000)
  B5  Parallel cold-pool scaling (1, 2, 4, 8 workers)
  B8  Sweep-grid cell memory: structured vs. dict

Supplementary benches that live in their own scripts and produce
their own output (because they depend on external resources or
emit C-library stdout that needs containing):

  TeleSABRE vs. HQA+Sabre on QFT-25 / GHZ-25 / AE-25 (4×9 grid)
    examples/benchmark_telesabre_vs_hqa.py
"""

from __future__ import annotations

import gc
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

# Project root — script can be run from any working directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qusim.dse import DSEEngine, NOISE_DEFAULTS  # noqa: E402
from qusim.dse.results import _RESULT_DTYPE  # noqa: E402


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@contextmanager
def _timed():
    t0 = time.perf_counter()
    yield (lambda: time.perf_counter() - t0)


def _median_min_max(samples: list[float]) -> tuple[float, float, float]:
    return statistics.median(samples), min(samples), max(samples)


def _line(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


# ---------------------------------------------------------------------------
# B1 / B2 — cold-path latency + peak RSS
# ---------------------------------------------------------------------------
#
# Resident-set-size at peak is recorded by spawning a fresh subprocess for
# each datapoint and reading ru_maxrss after the cold compile. ru_maxrss
# on Linux is in KiB; we report MiB.

_RSS_PROBE = """
import os, resource, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
sys.path.insert(0, {root!r})
from qusim.dse import DSEEngine
e = DSEEngine()
c = e.run_cold(
    circuit_type="qft", num_logical_qubits={L}, num_cores={NC},
    qubits_per_core={QPC}, topology_type="ring",
    intracore_topology="all_to_all", placement_policy="spectral",
    seed=0, communication_qubits=1, buffer_qubits=1, pin_axis="cores",
)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"OK {{c.cold_time_s:.4f}} {{rss}} {{c.total_teleportations}} {{c.total_swaps}}")
"""


def _qpc_for(L: int, NC: int) -> int:
    """Pick a qubits-per-core that fits L logical + 1 group of (K=1, B=1)
    for a ring topology (G_max=2 on ring with 2+ cores)."""
    # comm/buffer overhead per core: G_max(ring) * (K + B) = 2 * 2 = 4
    overhead = 4 if NC >= 2 else 0
    return max(8, (L + NC - 1) // NC + overhead)


def bench_cold(qubits_list: list[int]) -> list[dict]:
    """B1+B2: median cold-path wall-clock + peak RSS, by sub-process probe."""
    rows: list[dict] = []
    for L in qubits_list:
        NC = 4 if L >= 16 else 2
        QPC = _qpc_for(L, NC)
        latencies, rsses = [], []
        teleports = swaps = None
        for _ in range(3):
            r = subprocess.run(
                [sys.executable, "-c", _RSS_PROBE.format(root=ROOT, L=L, NC=NC, QPC=QPC)],
                capture_output=True, text=True, check=False,
            )
            if r.returncode != 0:
                rows.append({"L": L, "NC": NC, "QPC": QPC, "error": r.stderr.strip()[:200]})
                break
            tok = r.stdout.strip().split()
            if not tok or tok[0] != "OK":
                rows.append({"L": L, "NC": NC, "QPC": QPC, "error": r.stdout.strip()[:200]})
                break
            latencies.append(float(tok[1]))
            rsses.append(int(tok[2]) / 1024.0)  # KiB -> MiB
            teleports = int(tok[3])
            swaps = int(tok[4])
        else:
            tmed, _, _ = _median_min_max(latencies)
            rmed, _, rmax = _median_min_max(rsses)
            rows.append({
                "L": L, "NC": NC, "QPC": QPC,
                "cold_s": tmed, "rss_mib": rmed, "rss_max_mib": rmax,
                "teleports": teleports, "swaps": swaps,
            })
    return rows


def print_b1_b2(rows: list[dict]) -> None:
    _line("B1+B2 — Cold-path latency + peak RSS (QFT-N, ring of NC cores, K=B=1)")
    print(f"{'L':>5} {'NC':>4} {'QPC':>5} {'cold_s':>8} {'rss_MiB':>9} {'teleports':>10} {'swaps':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['L']:>5} {r['NC']:>4} {r['QPC']:>5}  ERROR: {r['error']}")
            continue
        print(f"{r['L']:>5} {r['NC']:>4} {r['QPC']:>5} "
              f"{r['cold_s']:>8.3f} {r['rss_mib']:>9.0f} "
              f"{r['teleports']:>10} {r['swaps']:>7}")


# ---------------------------------------------------------------------------
# B3 — hot-path single call
# ---------------------------------------------------------------------------

def bench_hot_single(L: int, NC: int) -> dict:
    QPC = _qpc_for(L, NC)
    e = DSEEngine()
    cached = e.run_cold(
        circuit_type="qft", num_logical_qubits=L, num_cores=NC,
        qubits_per_core=QPC, topology_type="ring",
        intracore_topology="all_to_all", placement_policy="spectral",
        seed=0, communication_qubits=1, buffer_qubits=1, pin_axis="cores",
    )

    # Warm-up
    e.run_hot(cached, dict(NOISE_DEFAULTS))

    # 100 single calls
    samples = []
    noise = dict(NOISE_DEFAULTS)
    for _ in range(100):
        with _timed() as t:
            e.run_hot(cached, noise)
        samples.append(t())
    med, mn, mx = _median_min_max(samples)
    return {"L": L, "NC": NC, "QPC": QPC,
            "median_us": med * 1e6, "min_us": mn * 1e6, "max_us": mx * 1e6}


def print_b3(rows: list[dict]) -> None:
    _line("B3 — Hot-path single run_hot latency  (100 calls per row, μs)")
    print(f"{'L':>5} {'NC':>4} {'QPC':>5} {'median':>10} {'min':>10} {'max':>10}")
    for r in rows:
        print(f"{r['L']:>5} {r['NC']:>4} {r['QPC']:>5} "
              f"{r['median_us']:>10.1f} {r['min_us']:>10.1f} {r['max_us']:>10.1f}")


# ---------------------------------------------------------------------------
# B4 — hot-path batched
# ---------------------------------------------------------------------------

def bench_hot_batch(L: int, NC: int, batch_sizes: list[int]) -> list[dict]:
    QPC = _qpc_for(L, NC)
    e = DSEEngine()
    cached = e.run_cold(
        circuit_type="qft", num_logical_qubits=L, num_cores=NC,
        qubits_per_core=QPC, topology_type="ring",
        intracore_topology="all_to_all", placement_policy="spectral",
        seed=0, communication_qubits=1, buffer_qubits=1, pin_axis="cores",
    )
    # Warm-up
    e.run_hot_batch(cached, [dict(NOISE_DEFAULTS)])

    rows: list[dict] = []
    base = dict(NOISE_DEFAULTS)
    for B in batch_sizes:
        # Per-cell varying noise so the batch isn't constant-folded
        noise_list = []
        for i in range(B):
            n = dict(base)
            n["two_gate_error"] = 1e-3 * (1.0 + i / max(1, B))
            noise_list.append(n)
        # Median of 5 reps
        samples = []
        for _ in range(5):
            with _timed() as t:
                e.run_hot_batch(cached, noise_list)
            samples.append(t())
        med = statistics.median(samples)
        rows.append({"L": L, "NC": NC, "QPC": QPC,
                     "batch": B, "wall_s": med, "per_cell_us": med * 1e6 / B})
    return rows


def print_b4(rows: list[dict]) -> None:
    _line("B4 — Hot-path batched run_hot_batch  (median of 5)")
    print(f"{'L':>5} {'NC':>4} {'QPC':>5} {'batch':>7} {'wall_s':>10} {'per_cell_us':>14}")
    for r in rows:
        print(f"{r['L']:>5} {r['NC']:>4} {r['QPC']:>5} {r['batch']:>7} "
              f"{r['wall_s']:>10.4f} {r['per_cell_us']:>14.1f}")


# ---------------------------------------------------------------------------
# B8 — sweep-grid cell memory: structured-array vs. dict-of-floats
# ---------------------------------------------------------------------------

def bench_grid_memory() -> dict:
    n = 4096   # large enough that overhead dominates noise
    keys = list(_RESULT_DTYPE.names)

    # Path A — structured numpy array (production path).
    arr = np.zeros(n, dtype=_RESULT_DTYPE)
    for i in range(n):
        for k in keys:
            arr[k][i] = float(i)
    arr_bytes = arr.nbytes
    arr_per_cell = arr_bytes / n

    # Path B — list of dicts (legacy / pre-refactor path).
    cells = [{k: float(i) for k in keys} for i in range(n)]
    # sys.getsizeof gives shallow dict size; deep-walk to include floats.
    import sys as _sys
    deep = _sys.getsizeof(cells)
    for d in cells:
        deep += _sys.getsizeof(d)
        for k, v in d.items():
            deep += _sys.getsizeof(k) + _sys.getsizeof(v)
    dict_per_cell = deep / n

    return {
        "n_cells": n, "n_fields": len(keys),
        "structured_bytes_per_cell": arr_per_cell,
        "dict_bytes_per_cell": dict_per_cell,
        "ratio": dict_per_cell / arr_per_cell,
        "structured_total_kib": arr_bytes / 1024,
        "dict_total_kib": deep / 1024,
    }


def print_b8(r: dict) -> None:
    _line(f"B8 — Sweep-grid cell memory  ({r['n_cells']} cells × {r['n_fields']} float64 fields)")
    print(f"  structured numpy array:  {r['structured_bytes_per_cell']:>5.0f} B/cell   "
          f"({r['structured_total_kib']:>6.0f} KiB total)")
    print(f"  legacy list-of-dicts:    {r['dict_bytes_per_cell']:>5.0f} B/cell   "
          f"({r['dict_total_kib']:>6.0f} KiB total)")
    print(f"  structured saves {r['ratio']:.1f}× memory per cell.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Quadrature report — bench  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  cwd:    {os.getcwd()}")
    print(f"  cpu:    {os.cpu_count()} hardware threads")

    qubits_list = [8, 16, 32, 64, 128]
    cold_rows = bench_cold(qubits_list)
    print_b1_b2(cold_rows)

    hot_single_rows = []
    for L, NC in [(16, 2), (32, 4), (64, 4)]:
        hot_single_rows.append(bench_hot_single(L, NC))
    print_b3(hot_single_rows)

    hot_batch_rows = []
    L, NC = 32, 4
    for row in bench_hot_batch(L, NC, [1, 10, 100, 1000]):
        hot_batch_rows.append(row)
    print_b4(hot_batch_rows)

    print_b8(bench_grid_memory())

    # B5 — parallel cold-pool scaling. Lives last because it spawns
    # forkserver workers and benefits from a clean process state.
    print_b5(bench_parallel_scaling())

    print()
    print("[done]")


# ---------------------------------------------------------------------------
# B5 — parallel cold-pool scaling
# ---------------------------------------------------------------------------

def bench_parallel_scaling() -> list[dict]:
    """Run the same cold sweep at increasing max_workers, record speed-up."""
    cold_config = {
        "circuit_type": "qft", "num_logical_qubits": 16,
        "num_cores": 2, "qubits_per_core": 12,
        "topology_type": "ring", "intracore_topology": "all_to_all",
        "placement_policy": "spectral", "routing_algorithm": "hqa_sabre",
        "seed": 0, "communication_qubits": 1, "buffer_qubits": 1,
        "pin_axis": "cores",
    }

    rows: list[dict] = []
    serial_t = None
    for w in [1, 2, 4, 8]:
        engine = DSEEngine()
        with _timed() as t:
            sr = engine.sweep_nd(
                cached=None,
                sweep_axes=[
                    ("num_cores", 2, 10),       # 9 cold combinations
                    ("two_gate_error", -4, -2),
                ],
                fixed_noise=dict(NOISE_DEFAULTS),
                cold_config=cold_config,
                parallel=True,
                max_workers=w,
                max_cold=40,
            )
        wall = t()
        if serial_t is None:
            serial_t = wall
        rows.append({"workers": w, "wall_s": wall,
                     "speedup": serial_t / wall, "cells": sr.total_points})
    return rows


def print_b5(rows: list[dict]) -> None:
    _line("B5 — Parallel cold-pool scaling  (cores 2..10 × 9 hot points)")
    print(f"{'workers':>8} {'wall_s':>8} {'speedup':>8} {'cells':>6}")
    for r in rows:
        print(f"{r['workers']:>8} {r['wall_s']:>8.2f} "
              f"{r['speedup']:>8.2f} {r['cells']:>6}")


if __name__ == "__main__":
    main()
