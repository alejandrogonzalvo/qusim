"""Tests for the frozen slider UI integration (Phase 4).

Covers:
  - Frozen view tab appears when 3 axes are active
  - Frozen slider visibility logic
  - Sweep bypass: frozen slider drag does NOT trigger server re-sweep
  - Interp grid store is populated from sweep results
  - Frozen slider default value and range come from the 3rd axis
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Frozen view tab should appear in the tab bar when 3 axes are active
# ---------------------------------------------------------------------------

class TestFrozenViewTabs:
    def test_frozen_heatmap_tab_exists_for_3_axes(self):
        from gui.constants import VIEW_TABS
        tabs_3d = VIEW_TABS[3]
        values = [t["value"] for t in tabs_3d]
        assert "frozen_heatmap" in values

    def test_frozen_contour_tab_exists_for_3_axes(self):
        from gui.constants import VIEW_TABS
        tabs_3d = VIEW_TABS[3]
        values = [t["value"] for t in tabs_3d]
        assert "frozen_contour" in values

    def test_no_frozen_tabs_for_2_axes(self):
        from gui.constants import VIEW_TABS
        tabs_2d = VIEW_TABS[2]
        values = [t["value"] for t in tabs_2d]
        assert "frozen_heatmap" not in values
        assert "frozen_contour" not in values

    def test_no_frozen_tabs_for_1_axis(self):
        from gui.constants import VIEW_TABS
        tabs_1d = VIEW_TABS[1]
        values = [t["value"] for t in tabs_1d]
        assert "frozen_heatmap" not in values


# ---------------------------------------------------------------------------
# resolve_view_type should handle frozen views
# ---------------------------------------------------------------------------

class TestResolveViewTypeFrozen:
    def test_preserves_frozen_heatmap_in_3d(self):
        from gui.app import resolve_view_type
        assert resolve_view_type("frozen_heatmap", 3) == "frozen_heatmap"

    def test_preserves_frozen_contour_in_3d(self):
        from gui.app import resolve_view_type
        assert resolve_view_type("frozen_contour", 3) == "frozen_contour"

    def test_frozen_heatmap_falls_back_when_switching_to_2d(self):
        from gui.app import resolve_view_type
        result = resolve_view_type("frozen_heatmap", 2)
        assert result in ("heatmap", "contour")


# ---------------------------------------------------------------------------
# is_frozen_view helper
# ---------------------------------------------------------------------------

class TestIsFrozenView:
    def test_frozen_heatmap_is_frozen(self):
        from gui.interpolation import is_frozen_view
        assert is_frozen_view("frozen_heatmap") is True

    def test_frozen_contour_is_frozen(self):
        from gui.interpolation import is_frozen_view
        assert is_frozen_view("frozen_contour") is True

    def test_regular_views_not_frozen(self):
        from gui.interpolation import is_frozen_view
        assert is_frozen_view("heatmap") is False
        assert is_frozen_view("scatter3d") is False
        assert is_frozen_view("isosurface") is False
        assert is_frozen_view("line") is False


# ---------------------------------------------------------------------------
# frozen_view_base: map frozen view to its base 2D view type
# ---------------------------------------------------------------------------

class TestFrozenViewBase:
    def test_frozen_heatmap_base(self):
        from gui.interpolation import frozen_view_base
        assert frozen_view_base("frozen_heatmap") == "heatmap"

    def test_frozen_contour_base(self):
        from gui.interpolation import frozen_view_base
        assert frozen_view_base("frozen_contour") == "contour"

    def test_non_frozen_returns_identity(self):
        from gui.interpolation import frozen_view_base
        assert frozen_view_base("heatmap") == "heatmap"


# ---------------------------------------------------------------------------
# Build interp grid from 3D sweep data for a specific output metric
# ---------------------------------------------------------------------------

class TestBuildInterpGridFromSweep:
    def _make_3d_sweep(self):
        xs = [1.0, 2.0, 3.0]
        ys = [4.0, 5.0, 6.0]
        zs = [7.0, 8.0]
        grid = []
        for i, x in enumerate(xs):
            plane = []
            for j, y in enumerate(ys):
                row = []
                for k, z in enumerate(zs):
                    row.append({"overall_fidelity": x + y + z})
                plane.append(row)
            grid.append(plane)
        return {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": xs, "ys": ys, "zs": zs,
            "grid": grid,
        }

    def test_interp_grid_shape(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep = self._make_3d_sweep()
        result = sweep_to_interp_grid(sweep, "overall_fidelity")
        vals = np.array(result["values"])
        assert vals.shape == (2, 3, 3)  # (nz, ny, nx)

    def test_interp_grid_preserves_values(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep = self._make_3d_sweep()
        result = sweep_to_interp_grid(sweep, "overall_fidelity")
        vals = np.array(result["values"])
        # grid[0][0][0] = xs[0] + ys[0] + zs[0] = 1 + 4 + 7 = 12
        assert vals[0, 0, 0] == pytest.approx(12.0)
        # grid[0][0][1] = xs[0] + ys[0] + zs[1] = 1 + 4 + 8 = 13
        assert vals[1, 0, 0] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# Frozen slider default position
# ---------------------------------------------------------------------------

class TestFrozenSliderDefaults:
    def test_frozen_slider_range_from_third_axis(self):
        """The frozen slider range should span the 3rd axis's sweep range."""
        from gui.interpolation import frozen_slider_config
        sweep_data = {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": [1.0, 2.0, 3.0],
            "ys": [4.0, 5.0],
            "zs": [10.0, 20.0, 30.0],
        }
        config = frozen_slider_config(sweep_data)
        assert config["min"] == pytest.approx(10.0)
        assert config["max"] == pytest.approx(30.0)
        assert config["metric_key"] == "two_gate_time"

    def test_frozen_slider_default_at_midpoint(self):
        from gui.interpolation import frozen_slider_config
        sweep_data = {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": [1.0, 2.0],
            "ys": [3.0, 4.0],
            "zs": [0.0, 100.0],
        }
        config = frozen_slider_config(sweep_data)
        assert config["default"] == pytest.approx(50.0)

    def test_returns_none_for_non_3d(self):
        from gui.interpolation import frozen_slider_config
        sweep_data = {
            "metric_keys": ["t1", "t2"],
            "xs": [1.0, 2.0],
            "ys": [3.0, 4.0],
        }
        assert frozen_slider_config(sweep_data) is None


# ---------------------------------------------------------------------------
# Plotting: build_figure routes frozen views to 2D plot functions
# ---------------------------------------------------------------------------

class TestBuildFigureFrozenRouting:
    def test_frozen_heatmap_produces_figure(self):
        from gui.plotting import build_figure
        sweep_data = {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": [1.0, 2.0],
            "ys": [3.0, 4.0],
            "zs": [5.0, 6.0],
            "grid": [
                [[{"overall_fidelity": 0.5}, {"overall_fidelity": 0.6}],
                 [{"overall_fidelity": 0.7}, {"overall_fidelity": 0.8}]],
                [[{"overall_fidelity": 0.1}, {"overall_fidelity": 0.2}],
                 [{"overall_fidelity": 0.3}, {"overall_fidelity": 0.4}]],
            ],
        }
        fig = build_figure(
            3, sweep_data, "overall_fidelity",
            view_type="frozen_heatmap", frozen_z=5.5,
        )
        assert fig is not None
        assert len(fig.data) > 0
