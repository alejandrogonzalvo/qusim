"""Tests for the qubit-aware parallel cold-sweep scheduler.

Covers the memory estimator and `_parallel_cold_sweep` behavior with an
in-process thread-pool shim that replaces ProcessPoolExecutor — this keeps
the scheduling logic under test without spawning real cold compilations.
"""

import concurrent.futures
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from gui import dse_engine
from gui.dse_engine import DSEEngine, SweepProgress, _estimate_cold_mb


# ---------------------------------------------------------------------------
# Thread-pool shim: stands in for ProcessPoolExecutor during tests.
# Accepts and ignores `mp_context`; preserves `max_workers` semantics.
# ---------------------------------------------------------------------------
class _ThreadPoolShim(concurrent.futures.ThreadPoolExecutor):
    """Drop-in stand-in for ProcessPoolExecutor in tests.

    Accepts (and ignores) keyword args that only make sense for
    ProcessPoolExecutor: ``mp_context``, ``max_tasks_per_child``, ``initializer``.
    """

    _PROCESS_ONLY_KW = {"mp_context", "max_tasks_per_child", "initializer", "initargs"}

    def __init__(self, max_workers=None, **kw):
        for k in self._PROCESS_ONLY_KW:
            kw.pop(k, None)
        super().__init__(max_workers=max_workers, **kw)


# ---------------------------------------------------------------------------
# Memory estimator
# ---------------------------------------------------------------------------

class TestThreadCaps:
    """Importing dse_engine must cap per-process thread pools.

    Each cold-path worker used to spawn ~34 threads (qiskit/Rayon/BLAS/OpenMP).
    With 8 parallel workers that oversubscribed 16 CPUs by >17x and stalled
    the scheduler.  dse_engine now sets OMP/MKL/Rayon/etc. NUM_THREADS=1
    before importing numpy/qiskit so each worker stays at ~4 threads.
    """

    @pytest.mark.parametrize("var", [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ])
    def test_env_var_set_to_one(self, var):
        assert os.environ.get(var) == "1", (
            f"{var} must be '1' after importing gui.dse_engine"
        )


class TestEstimateColdMb:
    def test_small_qubit_baseline(self):
        assert 100 < _estimate_cold_mb(4) < 200

    def test_grows_monotonically(self):
        prev = _estimate_cold_mb(4)
        for nq in (40, 76, 112, 148, 184, 220, 256):
            cur = _estimate_cold_mb(nq)
            assert cur >= prev, f"{nq} dropped below {nq - 36}"
            prev = cur

    def test_extrapolates_beyond_range(self):
        assert _estimate_cold_mb(300) > _estimate_cold_mb(256)

    def test_fits_measurements_within_2x(self):
        for nq, measured in dse_engine._EMPIRICAL_COLD_MB:
            est = _estimate_cold_mb(nq)
            assert 0.5 * measured <= est <= 2.0 * measured, (
                f"{nq} qubits: estimate {est:.0f} MB vs measured {measured:.0f} MB"
            )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _fake_eval_cold_batch(cold_config, noise, swept_list, rss_cap_bytes=None, keep_grids=False):
    """Emulate a cold compilation with a short sleep, record timing."""
    nq = int(swept_list[0].get("num_qubits", cold_config.get("num_qubits", 0)))
    start = time.monotonic()
    time.sleep(0.05 + nq * 0.0005)
    end = time.monotonic()
    _TIMINGS.append((nq, start, end))
    return [{"overall_fidelity": 1.0} for _ in swept_list]


_TIMINGS: list[tuple[int, float, float]] = []


def _max_concurrency(rows: list[tuple[int, float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for _, s, e in rows:
        events.append((s, +1))
        events.append((e, -1))
    events.sort()
    current = 0
    peak = 0
    for _, d in events:
        current += d
        peak = max(peak, current)
    return peak


@pytest.fixture
def cold_config():
    return {
        "circuit_type": "ghz",
        "num_qubits": 8,
        "num_cores": 2,
        "topology_type": "ring",
        "placement_policy": "random",
        "seed": 42,
        "intracore_topology": "all_to_all",
    }


@pytest.fixture(autouse=True)
def _reset_timings():
    _TIMINGS.clear()
    yield
    _TIMINGS.clear()


class TestScheduler:
    def test_small_qubits_run_in_parallel(self, cold_config):
        engine = DSEEngine()
        indexed = [((i,), {"num_qubits": 8, "num_cores": c})
                   for i, c in enumerate([1, 2, 3, 4])]
        with patch.object(dse_engine, "_eval_cold_batch", _fake_eval_cold_batch), \
             patch.object(dse_engine.concurrent.futures, "ProcessPoolExecutor", _ThreadPoolShim):
            engine._parallel_cold_sweep(
                cold_config, {}, indexed, total=4,
                progress_callback=None, max_workers=4,
            )
        assert len(_TIMINGS) == 4
        assert _max_concurrency(_TIMINGS) >= 2

    def test_large_qubits_serialize_under_tight_budget(self, cold_config):
        engine = DSEEngine()
        # Distinct (cores, qubits) → 3 separate groups that each cost ~3800 MB.
        indexed = [((i,), {"num_qubits": 256, "num_cores": c})
                   for i, c in enumerate([2, 3, 4])]
        tight_mb = 4000
        with patch.object(dse_engine, "_eval_cold_batch", _fake_eval_cold_batch), \
             patch.object(dse_engine.concurrent.futures, "ProcessPoolExecutor", _ThreadPoolShim), \
             patch.object(DSEEngine, "_mem_budget_mb", staticmethod(lambda: tight_mb)):
            engine._parallel_cold_sweep(
                cold_config, {}, indexed, total=3,
                progress_callback=None, max_workers=4,
            )
        assert len(_TIMINGS) == 3
        assert _max_concurrency(_TIMINGS) == 1, \
            "256-qubit groups must serialize under a 4 GB budget"

    def test_mixed_workload_packs_within_budget(self, cold_config):
        engine = DSEEngine()
        # Distinct (cores, qubits) per group.
        # One 256-qubit (~3800 MB) + four 8-qubit jobs (~150 MB each).
        indexed = [((0,), {"num_qubits": 256, "num_cores": 4})] + [
            ((i,), {"num_qubits": 8, "num_cores": c})
            for i, c in enumerate([1, 2, 3, 4], start=1)
        ]
        generous_mb = 5000
        with patch.object(dse_engine, "_eval_cold_batch", _fake_eval_cold_batch), \
             patch.object(dse_engine.concurrent.futures, "ProcessPoolExecutor", _ThreadPoolShim), \
             patch.object(DSEEngine, "_mem_budget_mb", staticmethod(lambda: generous_mb)):
            engine._parallel_cold_sweep(
                cold_config, {}, indexed, total=5,
                progress_callback=None, max_workers=4,
            )
        assert len(_TIMINGS) == 5
        # The 8-qubit jobs should overlap with the 256-qubit one.
        assert _max_concurrency(_TIMINGS) >= 2

    def test_raises_when_budget_too_small(self, cold_config):
        engine = DSEEngine()
        indexed = [((0,), {"num_qubits": 256, "num_cores": 4})]
        with patch.object(DSEEngine, "_mem_budget_mb", staticmethod(lambda: 50)):
            with pytest.raises(RuntimeError, match="Not enough RAM"):
                engine._parallel_cold_sweep(
                    cold_config, {}, indexed, total=1,
                    progress_callback=None, max_workers=4,
                )

    def test_progress_callback_fires_for_every_point(self, cold_config):
        engine = DSEEngine()
        indexed = [((i,), {"num_qubits": 8, "num_cores": c})
                   for i, c in enumerate([1, 2, 3])]
        seen: list[int] = []

        def on_progress(p: SweepProgress) -> None:
            seen.append(p.completed)

        with patch.object(dse_engine, "_eval_cold_batch", _fake_eval_cold_batch), \
             patch.object(dse_engine.concurrent.futures, "ProcessPoolExecutor", _ThreadPoolShim):
            engine._parallel_cold_sweep(
                cold_config, {}, indexed, total=3,
                progress_callback=on_progress, max_workers=3,
            )
        assert sorted(seen) == [1, 2, 3]
