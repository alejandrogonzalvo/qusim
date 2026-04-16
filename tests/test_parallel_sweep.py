"""Tests for parallel cold-path sweep evaluation.

Covers:
  - Parallel 3D sweep produces identical results to sequential
  - Parallel 1D/2D cold sweeps produce identical results to sequential
  - Progress callback is still invoked for every point
  - max_workers parameter is respected
  - Single-worker parallel degrades to sequential behavior
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.dse_engine import DSEEngine, SweepProgress


_SCALAR_KEYS = {
    "overall_fidelity", "algorithmic_fidelity", "routing_fidelity",
    "coherence_fidelity", "readout_fidelity", "total_circuit_time_ns",
    "total_epr_pairs",
}


def _assert_results_equal(a: dict, b: dict):
    """Compare scalar metric keys shared between full and batch result dicts."""
    common = _SCALAR_KEYS & set(a.keys()) & set(b.keys())
    assert len(common) > 0, "No common scalar keys to compare"
    for key in common:
        va, vb = a[key], b[key]
        if isinstance(va, (float, int, np.floating, np.integer)):
            assert abs(float(va) - float(vb)) < 1e-10, (
                f"Mismatch [{key}]: {va} vs {vb}"
            )
        else:
            assert va == vb, f"Mismatch [{key}]: {va} vs {vb}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return DSEEngine()


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


@pytest.fixture
def fixed_noise():
    return {
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


# ---------------------------------------------------------------------------
# 3D parallel sweep
# ---------------------------------------------------------------------------

class TestParallelSweep3D:
    """Parallel 3D cold sweep must produce the same grid as sequential."""

    def test_parallel_matches_sequential(self, engine, cold_config, fixed_noise):
        """Run the same 3D cold sweep sequentially and in parallel, compare grids."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        # Sequential (current behavior)
        xs_seq, ys_seq, zs_seq, grid_seq = engine.sweep_3d(
            cached=cached,
            metric_key1="num_cores", low1=1, high1=4,
            metric_key2="num_qubits", low2=4, high2=16,
            metric_key3="t1", low3=4, high3=6,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
        )

        # Parallel
        xs_par, ys_par, zs_par, grid_par = engine.sweep_3d(
            cached=cached,
            metric_key1="num_cores", low1=1, high1=4,
            metric_key2="num_qubits", low2=4, high2=16,
            metric_key3="t1", low3=4, high3=6,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
        )

        # Axes must match exactly
        np.testing.assert_array_equal(xs_seq, xs_par)
        np.testing.assert_array_equal(ys_seq, ys_par)
        np.testing.assert_array_equal(zs_seq, zs_par)

        # Every fidelity value in the grid must match
        for i in range(len(xs_seq)):
            for j in range(len(ys_seq)):
                for k in range(len(zs_seq)):
                    _assert_results_equal(grid_seq[i][j][k], grid_par[i][j][k])

    def test_parallel_progress_callback(self, engine, cold_config, fixed_noise):
        """Parallel sweep must still invoke progress for every point."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)
        calls = []

        xs, ys, zs, grid = engine.sweep_3d(
            cached=cached,
            metric_key1="num_cores", low1=1, high1=4,
            metric_key2="num_qubits", low2=4, high2=16,
            metric_key3="t1", low3=4, high3=6,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
            progress_callback=lambda p: calls.append(p),
        )

        total = len(xs) * len(ys) * len(zs)
        assert len(calls) == total
        assert calls[-1].completed == total
        assert calls[-1].percentage == 100.0


# ---------------------------------------------------------------------------
# 2D parallel sweep
# ---------------------------------------------------------------------------

class TestParallelSweep2D:
    def test_parallel_matches_sequential(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_seq, ys_seq, grid_seq = engine.sweep_2d(
            cached=cached,
            metric_key1="num_cores", low1=1, high1=4,
            metric_key2="num_qubits", low2=4, high2=16,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
        )

        xs_par, ys_par, grid_par = engine.sweep_2d(
            cached=cached,
            metric_key1="num_cores", low1=1, high1=4,
            metric_key2="num_qubits", low2=4, high2=16,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
        )

        np.testing.assert_array_equal(xs_seq, xs_par)
        np.testing.assert_array_equal(ys_seq, ys_par)

        for i in range(len(xs_seq)):
            for j in range(len(ys_seq)):
                _assert_results_equal(grid_seq[i][j], grid_par[i][j])


# ---------------------------------------------------------------------------
# 1D parallel sweep
# ---------------------------------------------------------------------------

class TestParallelSweep1D:
    def test_parallel_matches_sequential(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_seq, results_seq = engine.sweep_1d(
            cached=cached,
            metric_key="num_cores",
            low=1, high=4,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
        )

        xs_par, results_par = engine.sweep_1d(
            cached=cached,
            metric_key="num_cores",
            low=1, high=4,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
        )

        np.testing.assert_array_equal(xs_seq, xs_par)
        for i in range(len(xs_seq)):
            _assert_results_equal(results_seq[i], results_par[i])


# ---------------------------------------------------------------------------
# max_workers control
# ---------------------------------------------------------------------------

class TestMaxWorkers:
    def test_single_worker_matches_sequential(self, engine, cold_config, fixed_noise):
        """parallel=True with max_workers=1 should still produce correct results."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_seq, results_seq = engine.sweep_1d(
            cached=cached,
            metric_key="num_qubits",
            low=4, high=16,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
        )

        xs_par, results_par = engine.sweep_1d(
            cached=cached,
            metric_key="num_qubits",
            low=4, high=16,
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
            max_workers=1,
        )

        np.testing.assert_array_equal(xs_seq, xs_par)
        for i in range(len(xs_seq)):
            _assert_results_equal(results_seq[i], results_par[i])


# ---------------------------------------------------------------------------
# Hot-path sweeps should be unaffected by parallel flag
# ---------------------------------------------------------------------------

class TestHotPathUnaffected:
    def test_hot_only_sweep_ignores_parallel(self, engine, cold_config, fixed_noise):
        """A pure hot-path sweep should work identically with parallel=True."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        # Hot-path only sweep (t1 is hot)
        xs_seq, results_seq = engine.sweep_1d(
            cached=cached,
            metric_key="t1",
            low=4, high=6,
            fixed_noise=fixed_noise,
        )

        xs_par, results_par = engine.sweep_1d(
            cached=cached,
            metric_key="t1",
            low=4, high=6,
            fixed_noise=fixed_noise,
            parallel=True,
        )

        np.testing.assert_array_equal(xs_seq, xs_par)
        for i in range(len(xs_seq)):
            _assert_results_equal(results_seq[i], results_par[i])
