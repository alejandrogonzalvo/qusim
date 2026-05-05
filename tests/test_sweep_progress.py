"""Tests for DSE sweep progress tracking.

Covers:
  - SweepProgress dataclass stores correct state
  - Progress callback is invoked on each iteration
  - Percentage calculation is correct
  - Parameter columns reflect current sweep values
  - Iteration ordering: innermost param changes fastest
"""

import pytest
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SweepProgress data structure
# ---------------------------------------------------------------------------

class TestSweepProgress:
    def test_progress_fields(self):
        from gui.dse_engine import SweepProgress
        p = SweepProgress(
            completed=5,
            total=100,
            current_params={"t1": 1000.0, "t2": 500.0},
        )
        assert p.completed == 5
        assert p.total == 100
        assert p.current_params == {"t1": 1000.0, "t2": 500.0}

    def test_percentage(self):
        from gui.dse_engine import SweepProgress
        p = SweepProgress(completed=25, total=100, current_params={})
        assert p.percentage == 25.0

    def test_percentage_zero_total(self):
        from gui.dse_engine import SweepProgress
        p = SweepProgress(completed=0, total=0, current_params={})
        assert p.percentage == 0.0

    def test_percentage_complete(self):
        from gui.dse_engine import SweepProgress
        p = SweepProgress(completed=60, total=60, current_params={})
        assert p.percentage == 100.0

    def test_percentage_partial(self):
        from gui.dse_engine import SweepProgress
        p = SweepProgress(completed=1, total=3, current_params={})
        assert abs(p.percentage - 33.33) < 0.01


# ---------------------------------------------------------------------------
# Progress callback invocation during sweeps
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Verify that sweep methods call the progress_callback with correct data."""

    def test_1d_sweep_calls_progress(self):
        """1D sweep should call progress n times with incrementing completed."""
        from gui.dse_engine import SweepProgress
        calls: list[SweepProgress] = []

        def on_progress(p: SweepProgress):
            calls.append(p)

        # Use a mock engine that doesn't need real qusim
        from gui.dse_engine import DSEEngine
        engine = DSEEngine()

        # Patch run_hot to avoid real computation
        engine.run_hot = lambda cached, noise: {"overall_fidelity": 0.9}
        engine.run_hot_batch = lambda cached, noises, **kw: [{"overall_fidelity": 0.9} for _ in noises]
        engine._cache = _make_fake_cached()

        xs, results = engine.sweep_1d(
            cached=engine._cache,
            metric_key="t1",
            low=3, high=6,
            fixed_noise={"t1": 1e4, "t2": 5e4},
            progress_callback=on_progress,
        )

        assert len(calls) == len(xs)
        assert calls[0].completed == 1
        assert calls[-1].completed == len(xs)
        assert calls[-1].total == len(xs)
        # Each call should have the current t1 value
        for i, c in enumerate(calls):
            assert "t1" in c.current_params

    def test_2d_sweep_calls_progress(self):
        """2D sweep should call progress n*m times."""
        from gui.dse_engine import SweepProgress
        calls: list[SweepProgress] = []

        def on_progress(p: SweepProgress):
            calls.append(p)

        from gui.dse_engine import DSEEngine
        engine = DSEEngine()
        engine.run_hot = lambda cached, noise: {"overall_fidelity": 0.9}
        engine.run_hot_batch = lambda cached, noises, **kw: [{"overall_fidelity": 0.9} for _ in noises]
        engine._cache = _make_fake_cached()

        xs, ys, grid = engine.sweep_2d(
            cached=engine._cache,
            metric_key1="t1", low1=3, high1=6,
            metric_key2="t2", low2=3, high2=6,
            fixed_noise={"t1": 1e4, "t2": 5e4},
            progress_callback=on_progress,
        )

        total = len(xs) * len(ys)
        assert len(calls) == total
        assert calls[-1].completed == total
        assert calls[-1].total == total

    def test_3d_sweep_calls_progress(self):
        """3D sweep should call progress n*m*k times."""
        from gui.dse_engine import SweepProgress
        calls: list[SweepProgress] = []

        def on_progress(p: SweepProgress):
            calls.append(p)

        from gui.dse_engine import DSEEngine
        engine = DSEEngine()
        engine.run_hot = lambda cached, noise: {"overall_fidelity": 0.9}
        engine.run_hot_batch = lambda cached, noises, **kw: [{"overall_fidelity": 0.9} for _ in noises]
        engine._cache = _make_fake_cached()

        xs, ys, zs, grid = engine.sweep_3d(
            cached=engine._cache,
            metric_key1="t1", low1=3, high1=6,
            metric_key2="t2", low2=3, high2=6,
            metric_key3="two_gate_time", low3=1, high3=3,
            fixed_noise={"t1": 1e4, "t2": 5e4, "two_gate_time": 100.0},
            progress_callback=on_progress,
        )

        total = len(xs) * len(ys) * len(zs)
        assert len(calls) == total
        assert calls[-1].completed == total

    def test_3d_inner_param_changes_fastest(self):
        """In 3D sweep, the 3rd metric should change on every iteration."""
        from gui.dse_engine import SweepProgress
        calls: list[SweepProgress] = []

        from gui.dse_engine import DSEEngine
        engine = DSEEngine()
        engine.run_hot = lambda cached, noise: {"overall_fidelity": 0.9}
        engine.run_hot_batch = lambda cached, noises, **kw: [{"overall_fidelity": 0.9} for _ in noises]
        engine._cache = _make_fake_cached()

        xs, ys, zs, grid = engine.sweep_3d(
            cached=engine._cache,
            metric_key1="t1", low1=3, high1=6,
            metric_key2="t2", low2=3, high2=6,
            metric_key3="two_gate_time", low3=1, high3=3,
            fixed_noise={"t1": 1e4, "t2": 5e4, "two_gate_time": 100.0},
            progress_callback=lambda p: calls.append(p),
        )

        n_z = len(zs)
        if n_z > 1:
            # First n_z calls should have same t1 and t2 but different two_gate_time
            first_block = calls[:n_z]
            t1_vals = [c.current_params["t1"] for c in first_block]
            z_vals = [c.current_params["two_gate_time"] for c in first_block]
            # t1 should be constant in the first block
            assert len(set(t1_vals)) == 1
            # two_gate_time should vary
            assert len(set(z_vals)) == n_z

    def test_no_callback_does_not_error(self):
        """Sweep without progress_callback should work normally."""
        from gui.dse_engine import DSEEngine
        engine = DSEEngine()
        engine.run_hot = lambda cached, noise: {"overall_fidelity": 0.9}
        engine.run_hot_batch = lambda cached, noises, **kw: [{"overall_fidelity": 0.9} for _ in noises]
        engine._cache = _make_fake_cached()

        xs, results = engine.sweep_1d(
            cached=engine._cache,
            metric_key="t1",
            low=3, high=6,
            fixed_noise={"t1": 1e4, "t2": 5e4},
        )
        assert len(results) == len(xs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cached():
    """Create a minimal fake CachedMapping for testing without real qusim."""
    from gui.dse_engine import CachedMapping
    import numpy as np
    return CachedMapping(
        gs_sparse=np.zeros((1, 1)),
        placements=np.zeros((1,)),
        distance_matrix=np.zeros((1, 1)),
        sparse_swaps=np.zeros((1,)),
        gate_error_arr=np.array([0.001]),
        gate_time_arr=np.array([20.0]),
        gate_names=["cx"],
        total_epr_pairs=0,
        total_swaps=0,
        total_teleportations=0,
        total_network_distance=0,
        config_key=("qft", 4, 1, "ring", "random", 42, "all_to_all"),
        cold_time_s=0.01,
    )
