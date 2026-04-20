"""Tests for N-dimensional DSE sweep support.

Covers:
  - SweepResult dataclass construction and properties
  - sweep_nd() produces same results as sweep_1d/2d/3d for N=1,2,3
  - sweep_nd() works for N=4,5 (higher dimensions)
  - Grid point budget calculation
  - _flatten_sweep_to_table handles N-D grids
  - View resolution for N >= 4 (analysis-only views)
  - Generalized frozen slicing for N-D
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.dse_engine import DSEEngine, SweepResult, SweepProgress
from gui.constants import (
    METRIC_BY_KEY,
    MAX_TOTAL_POINTS_HOT,
    MAX_TOTAL_POINTS_COLD,
    VIEW_TABS,
    ANALYSIS_TABS,
)
from gui.plotting import _flatten_sweep_to_table, build_figure, plot_empty


_SCALAR_KEYS = {
    "overall_fidelity", "algorithmic_fidelity", "routing_fidelity",
    "coherence_fidelity", "readout_fidelity", "total_circuit_time_ns",
    "total_epr_pairs",
}


def _assert_results_close(a: dict, b: dict, tol: float = 1e-10):
    """Compare scalar metric keys shared between two result dicts."""
    common = _SCALAR_KEYS & set(a.keys()) & set(b.keys())
    assert len(common) > 0, "No common scalar keys to compare"
    for key in common:
        va, vb = a[key], b[key]
        if isinstance(va, (float, int, np.floating, np.integer)):
            assert abs(float(va) - float(vb)) < tol, (
                f"Mismatch [{key}]: {va} vs {vb}"
            )


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
# SweepResult dataclass
# ---------------------------------------------------------------------------

class TestSweepResult:
    """SweepResult stores N-D sweep data in a flat numpy array with shape."""

    def test_construction_1d(self):
        result = SweepResult(
            metric_keys=["t1"],
            axes=[np.array([1e4, 1e5, 1e6])],
            grid=np.empty(3, dtype=object),
        )
        assert result.ndim == 1
        assert result.shape == (3,)
        assert result.total_points == 3

    def test_construction_2d(self):
        result = SweepResult(
            metric_keys=["t1", "t2"],
            axes=[np.array([1e4, 1e5]), np.array([1e3, 1e4, 1e5])],
            grid=np.empty((2, 3), dtype=object),
        )
        assert result.ndim == 2
        assert result.shape == (2, 3)
        assert result.total_points == 6

    def test_construction_5d(self):
        axes = [np.linspace(0, 1, n) for n in [3, 4, 5, 3, 2]]
        result = SweepResult(
            metric_keys=["t1", "t2", "single_gate_error", "two_gate_error", "single_gate_time"],
            axes=axes,
            grid=np.empty((3, 4, 5, 3, 2), dtype=object),
        )
        assert result.ndim == 5
        assert result.shape == (3, 4, 5, 3, 2)
        assert result.total_points == 360

    def test_to_sweep_data_1d(self):
        """to_sweep_data() must return the dict format expected by plotting."""
        grid = np.array([{"overall_fidelity": 0.9}, {"overall_fidelity": 0.8}], dtype=object)
        result = SweepResult(
            metric_keys=["t1"],
            axes=[np.array([1e4, 1e5])],
            grid=grid,
        )
        sd = result.to_sweep_data()
        assert sd["metric_keys"] == ["t1"]
        assert len(sd["xs"]) == 2
        assert "grid" in sd

    def test_to_sweep_data_3d(self):
        """3D to_sweep_data includes xs, ys, zs and nested grid."""
        shape = (2, 3, 4)
        grid = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            grid[idx] = {"overall_fidelity": 0.5}
        result = SweepResult(
            metric_keys=["t1", "t2", "single_gate_error"],
            axes=[np.linspace(0, 1, s) for s in shape],
            grid=grid,
        )
        sd = result.to_sweep_data()
        assert len(sd["xs"]) == 2
        assert len(sd["ys"]) == 3
        assert len(sd["zs"]) == 4
        assert len(sd["grid"]) == 2
        assert len(sd["grid"][0]) == 3
        assert len(sd["grid"][0][0]) == 4

    def test_to_sweep_data_5d(self):
        """5D to_sweep_data includes all axes and flat grid."""
        shape = (2, 2, 2, 2, 2)
        grid = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            grid[idx] = {"overall_fidelity": 0.5}
        result = SweepResult(
            metric_keys=["t1", "t2", "single_gate_error", "two_gate_error", "single_gate_time"],
            axes=[np.linspace(0, 1, s) for s in shape],
            grid=grid,
        )
        sd = result.to_sweep_data()
        assert sd["metric_keys"] == ["t1", "t2", "single_gate_error", "two_gate_error", "single_gate_time"]
        assert "axes" in sd
        assert len(sd["axes"]) == 5


# ---------------------------------------------------------------------------
# Grid point budget
# ---------------------------------------------------------------------------

class TestGridPointBudget:
    """Points per axis must stay within the total budget."""

    def test_points_per_axis_1d(self, engine):
        n = engine._points_per_axis(1, has_cold=False)
        assert n <= MAX_TOTAL_POINTS_HOT
        assert n >= 10

    def test_points_per_axis_3d(self, engine):
        n = engine._points_per_axis(3, has_cold=False)
        total = n ** 3
        assert total <= MAX_TOTAL_POINTS_HOT

    def test_points_per_axis_5d(self, engine):
        n = engine._points_per_axis(5, has_cold=False)
        total = n ** 5
        assert total <= MAX_TOTAL_POINTS_HOT
        assert n >= 3  # At least 3 points per axis

    def test_cold_budget_smaller(self, engine):
        hot_n = engine._points_per_axis(3, has_cold=False)
        cold_n = engine._points_per_axis(3, has_cold=True)
        assert cold_n <= hot_n

    def test_backward_compat_1d_hot(self, engine):
        """1D hot sweep should still produce ~60 points."""
        n = engine._points_per_axis(1, has_cold=False)
        assert n >= 50

    def test_split_budget_cold_gets_natural_range(self, engine):
        """Cold-path axes should get their natural integer range."""
        from gui.constants import MAX_COLD_COMPILATIONS
        axes = [
            ("num_cores", 1, 8),
            ("t1", 4, 6),
            ("t2", 4, 6),
            ("single_gate_error", -5, -3),
        ]
        counts = engine._compute_axis_counts(axes, has_cold=True)
        # num_cores 1-8 = 8 natural points, should get all of them
        assert counts[0] == 8
        # Cold compilations should be within budget
        assert counts[0] <= MAX_COLD_COMPILATIONS

    def test_split_budget_hot_axes_get_more(self, engine):
        """With only 1 cold axis, hot axes should get generous counts."""
        from gui.constants import MIN_POINTS_PER_AXIS as MIN_PTS
        axes = [
            ("num_cores", 1, 4),
            ("t1", 4, 6),
            ("t2", 4, 6),
        ]
        counts = engine._compute_axis_counts(axes, has_cold=True)
        # Hot axes should get more than MIN_POINTS_PER_AXIS
        assert counts[1] > MIN_PTS
        assert counts[2] > MIN_PTS

    def test_raises_when_min_points_exceed_hot_budget(self, engine):
        """High-D with tight max_hot must raise instead of silently blowing past it.

        9 hot axes + 2 cold axes (cores 1-8, qubits 4-256) at max_cold=512 force
        3^9 * 176 = 3,464,208 points — 35x over max_hot=100,000. Should raise.
        """
        axes = [
            ("num_cores", 1, 8),
            ("num_qubits", 4, 256),
            ("t1", 4, 6),
            ("t2", 4, 6),
            ("single_gate_error", -5, -3),
            ("two_gate_error", -4, -2),
            ("teleportation_error_per_hop", -3, -1),
            ("single_gate_time", 1, 3),
            ("teleportation_time_per_hop", 1, 5),
            ("readout_mitigation_factor", 0, 1),
            ("two_gate_time", 1, 4),
        ]
        with pytest.raises(RuntimeError, match="Hot budget too tight"):
            engine._compute_axis_counts(
                axes, has_cold=True, max_cold=512, max_hot=100_000,
            )

    def test_generous_hot_budget_does_not_raise(self, engine):
        """Same sweep with a big-enough hot budget should succeed."""
        axes = [
            ("num_cores", 1, 8),
            ("num_qubits", 4, 256),
            ("t1", 4, 6),
            ("t2", 4, 6),
            ("single_gate_error", -5, -3),
            ("two_gate_error", -4, -2),
            ("teleportation_error_per_hop", -3, -1),
            ("single_gate_time", 1, 3),
            ("teleportation_time_per_hop", 1, 5),
            ("readout_mitigation_factor", 0, 1),
            ("two_gate_time", 1, 4),
        ]
        # Big enough hot budget: 3^9 * 176 ≈ 3.46M, round up.
        counts = engine._compute_axis_counts(
            axes, has_cold=True, max_cold=512, max_hot=4_000_000,
        )
        assert len(counts) == len(axes)
        assert all(c >= 3 for c in counts)


# ---------------------------------------------------------------------------
# sweep_nd: backward compatibility with 1D/2D/3D
# ---------------------------------------------------------------------------

class TestSweepNdBackwardCompat:
    """sweep_nd must produce the same results as sweep_1d/2d/3d."""

    def test_1d_matches(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_old, results_old = engine.sweep_1d(
            cached, "t1", 4, 6, fixed_noise,
        )

        result_nd = engine.sweep_nd(
            cached=cached,
            sweep_axes=[("t1", 4, 6)],
            fixed_noise=fixed_noise,
        )

        np.testing.assert_array_equal(xs_old, result_nd.axes[0])
        assert result_nd.ndim == 1
        for i in range(len(xs_old)):
            _assert_results_close(results_old[i], result_nd.grid[i])

    def test_2d_matches(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_old, ys_old, grid_old = engine.sweep_2d(
            cached, "t1", 4, 6, "t2", 4, 6, fixed_noise,
        )

        result_nd = engine.sweep_nd(
            cached=cached,
            sweep_axes=[("t1", 4, 6), ("t2", 4, 6)],
            fixed_noise=fixed_noise,
        )

        np.testing.assert_array_equal(xs_old, result_nd.axes[0])
        np.testing.assert_array_equal(ys_old, result_nd.axes[1])
        assert result_nd.ndim == 2
        for i in range(len(xs_old)):
            for j in range(len(ys_old)):
                _assert_results_close(grid_old[i][j], result_nd.grid[i, j])

    def test_3d_matches(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        xs_old, ys_old, zs_old, grid_old = engine.sweep_3d(
            cached, "t1", 4, 6, "t2", 4, 6,
            "single_gate_error", -5, -3, fixed_noise,
        )

        result_nd = engine.sweep_nd(
            cached=cached,
            sweep_axes=[("t1", 4, 6), ("t2", 4, 6), ("single_gate_error", -5, -3)],
            fixed_noise=fixed_noise,
        )

        np.testing.assert_array_equal(xs_old, result_nd.axes[0])
        np.testing.assert_array_equal(ys_old, result_nd.axes[1])
        np.testing.assert_array_equal(zs_old, result_nd.axes[2])
        assert result_nd.ndim == 3
        for i in range(len(xs_old)):
            for j in range(len(ys_old)):
                for k in range(len(zs_old)):
                    _assert_results_close(grid_old[i][j][k], result_nd.grid[i, j, k])


# ---------------------------------------------------------------------------
# sweep_nd: higher dimensions
# ---------------------------------------------------------------------------

class TestSweepNdHigherDimensions:
    """sweep_nd should work for 4D and 5D sweeps."""

    def test_4d_hot_sweep(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        result = engine.sweep_nd(
            cached=cached,
            sweep_axes=[
                ("t1", 4, 6),
                ("t2", 4, 6),
                ("single_gate_error", -5, -3),
                ("two_gate_error", -4, -2),
            ],
            fixed_noise=fixed_noise,
        )

        assert result.ndim == 4
        assert all(len(ax) >= 3 for ax in result.axes)
        assert result.total_points == np.prod([len(ax) for ax in result.axes])
        # Every point should have a valid fidelity
        for idx in np.ndindex(result.shape):
            assert "overall_fidelity" in result.grid[idx]
            assert 0.0 <= result.grid[idx]["overall_fidelity"] <= 1.0

    def test_5d_hot_sweep(self, engine, cold_config, fixed_noise):
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        result = engine.sweep_nd(
            cached=cached,
            sweep_axes=[
                ("t1", 4, 6),
                ("t2", 4, 6),
                ("single_gate_error", -5, -3),
                ("two_gate_error", -4, -2),
                ("single_gate_time", 1, 2),
            ],
            fixed_noise=fixed_noise,
        )

        assert result.ndim == 5
        assert result.total_points <= MAX_TOTAL_POINTS_HOT
        for idx in np.ndindex(result.shape):
            assert "overall_fidelity" in result.grid[idx]

    def test_4d_with_cold_axis(self, engine, cold_config, fixed_noise):
        """4D sweep with num_cores as one cold axis."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)

        result = engine.sweep_nd(
            cached=cached,
            sweep_axes=[
                ("num_cores", 1, 4),
                ("t1", 4, 6),
                ("t2", 4, 6),
                ("single_gate_error", -5, -3),
            ],
            fixed_noise=fixed_noise,
            cold_config=cold_config,
            parallel=True,
        )

        assert result.ndim == 4
        # Total points may exceed cold budget — only cold compilations
        # (unique num_cores values) are capped, not the total grid.
        assert result.total_points <= MAX_TOTAL_POINTS_HOT
        # Cold axis (num_cores) count should be within cold budget
        from gui.constants import MAX_COLD_COMPILATIONS
        assert len(result.axes[0]) <= MAX_COLD_COMPILATIONS
        for idx in np.ndindex(result.shape):
            assert "overall_fidelity" in result.grid[idx]

    def test_progress_callback_nd(self, engine, cold_config, fixed_noise):
        """Progress callback fires for every point in N-D sweep."""
        cached = engine.run_cold(**cold_config, noise=fixed_noise)
        calls = []

        result = engine.sweep_nd(
            cached=cached,
            sweep_axes=[
                ("t1", 4, 6),
                ("t2", 4, 6),
                ("single_gate_error", -5, -3),
                ("two_gate_error", -4, -2),
            ],
            fixed_noise=fixed_noise,
            progress_callback=lambda p: calls.append(p),
        )

        assert len(calls) == result.total_points
        assert calls[-1].percentage == 100.0


# ---------------------------------------------------------------------------
# _flatten_sweep_to_table: N-D support
# ---------------------------------------------------------------------------

class TestFlattenNd:
    """_flatten_sweep_to_table must handle N-D sweep_data dicts."""

    def _make_sweep_data(self, ndim: int) -> dict:
        """Create a minimal sweep_data dict with ndim axes."""
        keys = ["t1", "t2", "single_gate_error", "two_gate_error",
                "single_gate_time", "two_gate_time"][:ndim]
        n = 3  # points per axis
        axes = [np.linspace(1, 10, n).tolist() for _ in range(ndim)]
        shape = tuple([n] * ndim)

        # Build nested grid for 1-3D, flat for 4D+
        if ndim == 1:
            grid = [{"overall_fidelity": 0.5 + i * 0.1, "total_epr_pairs": 10}
                    for i in range(n)]
        elif ndim == 2:
            grid = [[{"overall_fidelity": 0.5, "total_epr_pairs": 10}
                     for _ in range(n)] for _ in range(n)]
        elif ndim == 3:
            grid = [[[{"overall_fidelity": 0.5, "total_epr_pairs": 10}
                      for _ in range(n)] for _ in range(n)] for _ in range(n)]
        else:
            # For N >= 4, use the axes + flat_grid format
            flat = [{"overall_fidelity": 0.5, "total_epr_pairs": 10}
                    for _ in range(n ** ndim)]
            grid = flat

        sd = {"metric_keys": keys, "xs": axes[0]}
        if ndim >= 2:
            sd["ys"] = axes[1]
        if ndim >= 3:
            sd["zs"] = axes[2]
        if ndim >= 4:
            sd["axes"] = axes
            sd["grid"] = grid
            sd["shape"] = shape
        else:
            sd["grid"] = grid

        return sd

    def test_1d_flatten(self):
        sd = self._make_sweep_data(1)
        keys, outputs, rows = _flatten_sweep_to_table(sd)
        assert len(rows) == 3
        assert len(rows[0]) == len(keys) + len(outputs)

    def test_2d_flatten(self):
        sd = self._make_sweep_data(2)
        keys, outputs, rows = _flatten_sweep_to_table(sd)
        assert len(rows) == 9  # 3x3

    def test_3d_flatten(self):
        sd = self._make_sweep_data(3)
        keys, outputs, rows = _flatten_sweep_to_table(sd)
        assert len(rows) == 27  # 3x3x3

    def test_4d_flatten(self):
        sd = self._make_sweep_data(4)
        keys, outputs, rows = _flatten_sweep_to_table(sd)
        assert len(rows) == 81  # 3^4
        # Each row should have 4 param cols + output cols
        assert len(rows[0]) >= 4

    def test_5d_flatten(self):
        sd = self._make_sweep_data(5)
        keys, outputs, rows = _flatten_sweep_to_table(sd)
        assert len(rows) == 243  # 3^5


# ---------------------------------------------------------------------------
# View resolution for N >= 4
# ---------------------------------------------------------------------------

class TestViewResolutionNd:

    def test_4d_only_analysis_views(self):
        """For 4D+, only analysis tabs should be available."""
        from gui.app import resolve_view_type
        view = resolve_view_type(None, 4)
        analysis_keys = {t["value"] for t in ANALYSIS_TABS}
        assert view in analysis_keys

    def test_4d_preserves_analysis_view(self):
        from gui.app import resolve_view_type
        view = resolve_view_type("parallel", 4)
        assert view == "parallel"

    def test_4d_rejects_3d_view(self):
        """scatter3d is not valid for 4D."""
        from gui.app import resolve_view_type
        view = resolve_view_type("scatter3d", 4)
        assert view != "scatter3d"

    def test_build_figure_4d_parallel(self):
        """build_figure should produce a valid figure for 4D parallel coords."""
        sd = TestFlattenNd()._make_sweep_data(4)
        fig = build_figure(4, sd, "overall_fidelity", view_type="parallel")
        assert fig is not None
        assert len(fig.data) > 0

    def test_build_figure_4d_importance(self):
        sd = TestFlattenNd()._make_sweep_data(4)
        fig = build_figure(4, sd, "overall_fidelity", view_type="importance")
        assert fig is not None

    def test_build_figure_4d_default(self):
        """4D with no view_type should fall back to analysis view, not error."""
        sd = TestFlattenNd()._make_sweep_data(4)
        fig = build_figure(4, sd, "overall_fidelity")
        assert fig is not None


# ---------------------------------------------------------------------------
# Frozen slicing generalized
# ---------------------------------------------------------------------------

class TestFrozenSliceNd:
    """Frozen slicing should work for N >= 3 by freezing N-2 axes."""

    def test_frozen_slider_config_4d(self):
        from gui.interpolation import frozen_slider_config_nd
        sd = {
            "metric_keys": ["t1", "t2", "single_gate_error", "two_gate_error"],
            "axes": [
                [1e4, 1e5, 1e6],
                [1e3, 1e4, 1e5],
                [1e-5, 1e-4, 1e-3],
                [1e-4, 1e-3, 1e-2],
            ],
        }
        configs = frozen_slider_config_nd(sd, free_axes=[0, 1])
        # Should have 2 frozen axis configs (axes 2 and 3)
        assert len(configs) == 2
        assert configs[0]["metric_key"] == "single_gate_error"
        assert configs[1]["metric_key"] == "two_gate_error"

    def test_frozen_slider_config_3d_compat(self):
        """For 3D, frozen_slider_config_nd with free_axes=[0,1] gives 1 slider."""
        from gui.interpolation import frozen_slider_config_nd
        sd = {
            "metric_keys": ["t1", "t2", "single_gate_error"],
            "zs": [1e-5, 1e-4, 1e-3],
            "axes": [
                [1e4, 1e5, 1e6],
                [1e3, 1e4, 1e5],
                [1e-5, 1e-4, 1e-3],
            ],
        }
        configs = frozen_slider_config_nd(sd, free_axes=[0, 1])
        assert len(configs) == 1
        assert configs[0]["metric_key"] == "single_gate_error"
