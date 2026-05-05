"""
Sweep orchestration: shared cold-path helper, parallel pool scheduler,
and the worker-side cold-batch entry point used by the multiprocessing
:class:`ProcessPoolExecutor`.

Pulled out of :mod:`qusim.dse.engine` so the same ``_compile_one`` is
called by ``DSEEngine.run_cold`` (foreground) and ``_eval_cold_batch``
(worker process). Without this, every drift in the cold-path body had
to be mirrored twice — and silently desynchronised on miss.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from typing import Callable, Iterable

from .backends import get_backend
from .config import (
    COLD_PATH_KEYS,
    INTEGER_KEYS,
    _clamp_cfg_comm_and_logical,
    _expand_qubits_alias,
)
from .memory import (
    _estimate_cold_mb,
    _max_hot_points_for_memory,
    _mem_budget_mb,
)
from .noise import _merge_noise
from .results import CachedMapping, SweepProgress


_MP_CONTEXT = multiprocessing.get_context("forkserver")


# ---------------------------------------------------------------------------
# Shared cold-path entry point
# ---------------------------------------------------------------------------

def _compile_one(
    cold_cfg: dict,
    noise: dict | None,
    config_key: tuple,
) -> CachedMapping:
    """Run the cold path for one (cold_cfg, noise) combination.

    The single source of truth shared by ``DSEEngine.run_cold`` and the
    multiprocessing worker (:func:`_eval_cold_batch`). ``cold_cfg`` is
    expected to already be alias-expanded and clamped — callers either
    do that themselves (sweep paths) or rely on
    :meth:`DSEEngine.run_cold` which does it for them.

    Routing-algorithm dispatch happens here via the backend registry —
    add a new backend to :mod:`qusim.dse.backends` and it lights up
    everywhere this function is called.
    """
    backend = get_backend(cold_cfg.get("routing_algorithm", "hqa_sabre"))
    merged = _merge_noise(noise or {})
    return backend.compile(cold_cfg, merged, config_key)


# ---------------------------------------------------------------------------
# Memory-aware budget helpers
# ---------------------------------------------------------------------------

def _memory_capped_max_hot(max_hot: int | None) -> tuple[int, int]:
    """Return ``(effective_max_hot, requested_max_hot)``.

    Clamps a user-requested ``max_hot`` to whatever the host RAM can
    actually accommodate as a result grid, so an oversized budget can't
    silently produce a sweep the kernel will OOM-kill.
    """
    cap = _max_hot_points_for_memory()
    if max_hot is None:
        from .axes import MAX_TOTAL_POINTS_HOT
        requested = MAX_TOTAL_POINTS_HOT
    else:
        requested = max_hot
    return min(cap, requested), requested


# ---------------------------------------------------------------------------
# Worker entry point — must be top-level for pickling.
# ---------------------------------------------------------------------------

def _eval_cold_batch(
    cold_config: dict,
    noise: dict,
    swept_list: list[dict],
    rss_cap_bytes: int | None = None,
    keep_grids: bool = False,
) -> list[dict]:
    """Evaluate a batch of design points sharing the same cold-path config.

    Compiles the circuit once via :func:`_compile_one`, then evaluates
    every noise variation in a single batched Rust call. ``rss_cap_bytes``
    caps the worker's address space via ``RLIMIT_AS`` so a runaway
    allocation raises ``MemoryError`` inside this process instead of
    OOM-killing the whole system.
    """
    if rss_cap_bytes and rss_cap_bytes > 0:
        try:
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS, (int(rss_cap_bytes), int(rss_cap_bytes)),
            )
        except (ImportError, ValueError, OSError):
            # RLIMIT_AS unavailable (non-Linux) or kernel refused — fall
            # back to unbounded execution; the scheduler's mem_ok check
            # still provides some protection.
            pass

    # Apply cold-path overrides from the first swept dict (all share the
    # same cold keys by construction in _parallel_cold_sweep).
    cfg = dict(cold_config)
    for k, v in swept_list[0].items():
        if k in COLD_PATH_KEYS:
            cfg[k] = int(v) if k in INTEGER_KEYS else v
    _expand_qubits_alias(cfg)
    cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
    _clamp_cfg_comm_and_logical(cfg)

    # Build per-point hot-noise dicts.
    noise_dicts = []
    for swept in swept_list:
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k not in COLD_PATH_KEYS:
                hot_noise[k] = v
        noise_dicts.append(hot_noise)

    # Single cold compilation. Routing through DSEEngine.run_cold (rather
    # than directly through _compile_one) lets it apply the same num_logical_qubits
    # default + custom-circuit validation it would in the foreground path,
    # so a sweep cell sees identical pre-conditions whether it runs in
    # this worker or in the main process.
    from .engine import DSEEngine  # local import to avoid module cycle
    engine = DSEEngine()
    cached = engine.run_cold(**cfg, noise=noise_dicts[0])

    # Single batched Rust call for all hot-path variations.
    return engine.run_hot_batch(cached, noise_dicts, keep_grids=keep_grids)


def _build_config_key(cfg: dict) -> tuple:
    """Mirror of the cache key DSEEngine.run_cold builds.

    Kept here so :func:`_eval_cold_batch` can construct an identical
    key without going through DSEEngine. The fingerprint of any
    ``custom_qasm`` is a 16-char SHA-256 prefix — enough to detect
    distinct circuits without bloating the key.
    """
    import hashlib
    qasm = cfg.get("custom_qasm")
    fp = hashlib.sha256(qasm.encode("utf-8")).hexdigest()[:16] if qasm else None
    return (
        cfg["circuit_type"], cfg["num_qubits"], cfg["num_cores"],
        cfg["topology_type"], cfg.get("intracore_topology", "all_to_all"),
        cfg.get("placement_policy", "random"),
        cfg.get("seed", 0),
        cfg.get("routing_algorithm", "hqa_sabre"),
        int(cfg.get("communication_qubits", 1) or 1),
        int(cfg.get("buffer_qubits", 1) or 1),
        int(cfg["num_logical_qubits"]),
        fp,
    )


# ---------------------------------------------------------------------------
# Parallel scheduler
# ---------------------------------------------------------------------------

def _parallel_cold_sweep(
    cold_config: dict,
    noise: dict,
    indexed_points: Iterable[tuple[tuple, dict]],
    total: int,
    progress_callback: Callable[[SweepProgress], None] | None,
    max_workers: int | None,
    keep_grids: bool = False,
) -> dict[tuple, dict]:
    """Run cold-path sweep points with qubit-aware parallel scheduling.

    Each unique cold config (one (num_qubits, num_cores) combo) becomes
    one group. The scheduler submits groups to a process pool so that
    the sum of their estimated peak RSS stays under the memory budget —
    small groups parallelize aggressively, large-qubit groups serialize.
    A single oversized group is always allowed to run alone to avoid
    deadlock.
    """
    groups: dict[tuple, list[tuple[tuple, dict]]] = {}
    for idx_key, swept in indexed_points:
        cold_vals = tuple(sorted(
            (k, int(v) if k in INTEGER_KEYS else v)
            for k, v in swept.items() if k in COLD_PATH_KEYS
        ))
        groups.setdefault(cold_vals, []).append((idx_key, swept))

    fallback_nq = int(cold_config.get("num_qubits", 16))

    def _group_nq(cv: tuple) -> int:
        # Recognise the ``qubits`` alias (logical == physical sweep)
        # which expands to ``num_qubits`` inside the worker.
        for k, v in cv:
            if k == "num_qubits" or k == "qubits":
                return int(v)
        return fallback_nq

    group_cost: dict[tuple, float] = {
        cv: _estimate_cold_mb(_group_nq(cv)) for cv in groups
    }

    cpu_cap = max(1, (os.cpu_count() or 2) // 2)
    slot_cap = max_workers if max_workers else cpu_cap
    mem_budget = _mem_budget_mb()

    # Per-worker address-space cap (RLIMIT_AS). Capping each worker to
    # its fair share of the budget ensures a runaway allocation raises
    # MemoryError inside the worker rather than OOM-killing the system.
    rss_cap_bytes = (mem_budget * 1024 * 1024) // max(1, slot_cap)

    if group_cost:
        min_cost = min(group_cost.values())
        if min_cost > mem_budget * 1.5:
            largest_nq = max(_group_nq(cv) for cv in groups)
            raise RuntimeError(
                f"Not enough RAM for this sweep: smallest cold-compile "
                f"needs ~{min_cost:.0f} MB, budget is {mem_budget} MB "
                f"(largest qubits = {largest_nq}). Reduce max qubits, "
                f"close other apps, or lower Max cold compilations."
            )

    # Schedule largest-first so cheap groups can tuck into the remaining budget.
    pending = sorted(groups, key=lambda cv: -group_cost[cv])
    active: dict = {}  # future -> (cold_vals, cost_mb)
    results: dict[tuple, dict] = {}
    completed = 0
    cold_total = len(groups)
    cold_completed = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=slot_cap, mp_context=_MP_CONTEXT,
        max_tasks_per_child=4,
    ) as pool:

        def _submit_fitting() -> None:
            used = sum(c for _, c in active.values())
            kept: list[tuple] = []
            for cv in pending:
                cost = group_cost[cv]
                if len(active) >= slot_cap:
                    kept.append(cv)
                    continue
                # Always allow one job when the pool is idle, even if its
                # estimate exceeds the budget (otherwise we'd deadlock on
                # a single oversized group).
                mem_ok = not active or used + cost <= mem_budget
                if mem_ok:
                    swept_list = [s for _, s in groups[cv]]
                    fut = pool.submit(
                        _eval_cold_batch,
                        cold_config, noise, swept_list, rss_cap_bytes,
                        keep_grids,
                    )
                    active[fut] = (cv, cost)
                    used += cost
                else:
                    kept.append(cv)
            pending[:] = kept

        _submit_fitting()
        while active:
            done, _still = concurrent.futures.wait(
                list(active), return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                cv, _cost = active.pop(fut)
                batch_results = fut.result()
                cold_completed += 1
                for (idx_key, swept), result in zip(groups[cv], batch_results):
                    results[idx_key] = result
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=completed,
                            total=total,
                            current_params={k: float(v) for k, v in swept.items()},
                            cold_completed=cold_completed,
                            cold_total=cold_total,
                        ))
            _submit_fitting()

    return results
