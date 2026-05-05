"""
Memory budgeting for DSE sweeps.

Two budgets:

* **Cold path** — each compilation is a separate forkserver worker that
  allocates Qiskit/Sabre data structures. Peak RSS scales roughly linearly
  with ``num_qubits`` once the topology is dense; we keep an empirical
  table (calibrated on QFT, the densest common circuit) and interpolate.

* **Hot path** — the sweep grid lives in the *main* process for the
  lifetime of the plot. Reserve more headroom because we can't recover
  if the kernel OOM-kills it.

Pure stdlib + ``/proc/meminfo`` (Linux). Falls back to generous defaults
where unavailable.
"""

from __future__ import annotations


# Minimum free RAM (MB) to keep available for OS + UI + browser.
_RESERVED_RAM_MB = 1024

# Reserved headroom for OS + browser + UI when computing the hot-path
# ceiling. Larger than ``_RESERVED_RAM_MB`` (cold path) because the main
# process — not a short-lived worker — has to hold the full result grid
# for the lifetime of the plot.
_RESERVED_RAM_MB_HOT = 3072

# Peak memory per hot-path grid cell. The sweep grid is a structured
# numpy array of ``_RESULT_DTYPE``: 7 × float64 = 56 B per cell, flat in
# memory (no per-cell Python overhead). 128 B keeps a ~2x safety margin
# for the transient noise-dict / chunk buffers the sweep holds alongside
# the grid.
_BYTES_PER_HOT_POINT = 128

# Empirical peak RSS (MB) for one cold compilation vs. num_qubits,
# measured on QFT (densest common circuit) with the forkserver worker
# used in production. See tests/test_cores_qubits_oom.py for the
# measurement methodology.
_EMPIRICAL_COLD_MB: list[tuple[int, float]] = [
    (4,   150.0),
    (40,  160.0),
    (76,  260.0),
    (112, 450.0),
    (148, 820.0),
    (184, 1600.0),
    (220, 2100.0),
    (256, 3800.0),
]


def _max_hot_points_for_memory() -> int:
    """Maximum hot-path grid size that fits in currently available RAM.

    Sweep results live in a numpy structured grid (or — on legacy paths
    — a list of dicts) until the plot is rendered. Exceeding
    ``MemAvailable`` gets the process OOM-killed by the kernel, which
    surfaces in the UI as a "crash". Returning a conservative ceiling
    lets ``DSEEngine.sweep_nd`` clamp a user-requested ``max_hot`` below
    the danger zone — the sweep then either fits, or fails loudly with
    the usual "hot budget too tight" guard. Either way, the process
    stays alive.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
                    break
            else:
                return 1_000_000
    except OSError:
        return 1_000_000
    budget_bytes = max(0, avail_mb - _RESERVED_RAM_MB_HOT) * 1024 * 1024
    return max(10_000, budget_bytes // _BYTES_PER_HOT_POINT)


def _estimate_cold_mb(num_qubits: int) -> float:
    """Peak RSS estimate (MB) for one cold compilation at ``num_qubits``.

    Piecewise-linear interpolation over :data:`_EMPIRICAL_COLD_MB`, with
    linear extrapolation beyond the measured range.
    """
    pts = _EMPIRICAL_COLD_MB
    if num_qubits <= pts[0][0]:
        return pts[0][1]
    if num_qubits >= pts[-1][0]:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (num_qubits - x2)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= num_qubits <= x2:
            return y1 + (y2 - y1) * (num_qubits - x1) / (x2 - x1)
    return pts[-1][1]


def _mem_budget_mb() -> int:
    """Memory budget (MB) for concurrent cold compilations.

    Reserves the larger of ``_RESERVED_RAM_MB`` (1 GB floor) or 30% of
    total RAM so the budget scales with the host. Keeping 30% free
    prevents the swap-thrashing deadlock that can freeze the whole OS
    when multiple 3+ GB workers run alongside the browser, desktop, and
    a growing page cache.
    """
    try:
        avail_mb = 0
        total_mb = 0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
                elif line.startswith("MemTotal:"):
                    total_mb = int(line.split()[1]) // 1024
                if avail_mb and total_mb:
                    break
        if avail_mb and total_mb:
            reserved = max(_RESERVED_RAM_MB, total_mb * 30 // 100)
            return max(1, avail_mb - reserved)
    except OSError:
        pass
    # Fallback: a conservative ~1.6 GB when /proc is unavailable
    return 2 * 800


def cap_thread_pools() -> None:
    """Pin per-process thread pools to 1 thread.

    Must be called BEFORE importing numpy / qiskit / qusim so their
    Rayon / OpenMP / BLAS pools initialize at 1 thread. Otherwise each
    cold-path worker spawns ~34 threads and 8 parallel workers
    oversubscribe the CPU (272 threads on 16 cores) until the scheduler
    stalls. Workers use process-level parallelism already, so there is
    nothing to lose by disabling library threads here.
    """
    import os
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")


# Apply the thread cap at import time so any code path importing this
# module before numpy/qiskit picks it up.
cap_thread_pools()
