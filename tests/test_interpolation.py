"""Tests for client-side interpolation utilities (Phases 3+4 of hot-reload plan).

Covers:
  - 1D linear interpolation (lerp)
  - 2D bilinear interpolation (bilerp)
  - 3D trilinear interpolation (trilerp)
  - Grid format conversion (sweep_data → compact interp format)
  - Frozen slice extraction (3D grid → 2D slice at frozen value)
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1-D linear interpolation
# ---------------------------------------------------------------------------

class TestLerp:
    def test_exact_grid_point(self):
        from gui.interpolation import lerp
        xs = np.array([1.0, 2.0, 3.0])
        vs = np.array([10.0, 20.0, 30.0])
        assert lerp(xs, vs, 2.0) == pytest.approx(20.0)

    def test_midpoint(self):
        from gui.interpolation import lerp
        xs = np.array([0.0, 1.0])
        vs = np.array([0.0, 10.0])
        assert lerp(xs, vs, 0.5) == pytest.approx(5.0)

    def test_clamps_below_range(self):
        from gui.interpolation import lerp
        xs = np.array([1.0, 2.0, 3.0])
        vs = np.array([10.0, 20.0, 30.0])
        assert lerp(xs, vs, 0.0) == pytest.approx(10.0)

    def test_clamps_above_range(self):
        from gui.interpolation import lerp
        xs = np.array([1.0, 2.0, 3.0])
        vs = np.array([10.0, 20.0, 30.0])
        assert lerp(xs, vs, 5.0) == pytest.approx(30.0)

    def test_quarter_point(self):
        from gui.interpolation import lerp
        xs = np.array([0.0, 4.0])
        vs = np.array([0.0, 100.0])
        assert lerp(xs, vs, 1.0) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 2-D bilinear interpolation
# ---------------------------------------------------------------------------

class TestBilerp:
    def test_exact_corner(self):
        from gui.interpolation import bilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert bilerp(grid, xs, ys, 0.0, 0.0) == pytest.approx(1.0)
        assert bilerp(grid, xs, ys, 1.0, 0.0) == pytest.approx(2.0)
        assert bilerp(grid, xs, ys, 0.0, 1.0) == pytest.approx(3.0)
        assert bilerp(grid, xs, ys, 1.0, 1.0) == pytest.approx(4.0)

    def test_center_bilinear(self):
        from gui.interpolation import bilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        grid = np.array([[1.0, 3.0], [3.0, 5.0]])
        assert bilerp(grid, xs, ys, 0.5, 0.5) == pytest.approx(3.0)

    def test_3x3_grid(self):
        from gui.interpolation import bilerp
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        grid = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ])
        assert bilerp(grid, xs, ys, 0.5, 0.5) == pytest.approx(1.0)
        assert bilerp(grid, xs, ys, 1.5, 1.5) == pytest.approx(3.0)

    def test_clamps_out_of_bounds(self):
        from gui.interpolation import bilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        grid = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert bilerp(grid, xs, ys, -1.0, 0.5) == pytest.approx(
            bilerp(grid, xs, ys, 0.0, 0.5)
        )

    def test_vectorised_output(self):
        """bilerp over a mesh of query points returns a 2D array."""
        from gui.interpolation import bilerp_mesh
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        grid = np.array([[0.0, 10.0], [10.0, 20.0]])
        qx = np.array([0.0, 0.5, 1.0])
        qy = np.array([0.0, 0.5, 1.0])
        result = bilerp_mesh(grid, xs, ys, qx, qy)
        assert result.shape == (3, 3)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(10.0)
        assert result[2, 2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 3-D trilinear interpolation
# ---------------------------------------------------------------------------

class TestTrilerp:
    def test_exact_corner(self):
        from gui.interpolation import trilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        # shape: (nz, ny, nx) = (2, 2, 2)
        grid = np.array([
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ])
        assert trilerp(grid, xs, ys, zs, 0.0, 0.0, 0.0) == pytest.approx(0.0)
        assert trilerp(grid, xs, ys, zs, 1.0, 1.0, 1.0) == pytest.approx(7.0)

    def test_center(self):
        from gui.interpolation import trilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        grid = np.ones((2, 2, 2)) * 8.0
        assert trilerp(grid, xs, ys, zs, 0.5, 0.5, 0.5) == pytest.approx(8.0)

    def test_linear_ramp(self):
        from gui.interpolation import trilerp
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        grid = np.array([
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ])
        assert trilerp(grid, xs, ys, zs, 0.25, 0.0, 0.0) == pytest.approx(0.25)
        assert trilerp(grid, xs, ys, zs, 0.75, 0.0, 0.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Frozen slice: extract 2D slice from 3D grid at a frozen z value
# ---------------------------------------------------------------------------

class TestFrozenSlice:
    def test_slice_at_exact_z(self):
        """Slicing at an exact z grid value returns that z-plane."""
        from gui.interpolation import frozen_slice
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        grid = np.array([
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ])
        result = frozen_slice(grid, xs, ys, zs, z_value=0.0)
        expected = np.array([[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_slice_at_midpoint_z(self):
        """Slicing between two z-planes interpolates linearly."""
        from gui.interpolation import frozen_slice
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([0.0, 1.0])
        grid = np.array([
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ])
        result = frozen_slice(grid, xs, ys, zs, z_value=0.5)
        expected = np.full((2, 2), 5.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_3x3x3_grid(self):
        from gui.interpolation import frozen_slice
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        zs = np.array([0.0, 5.0, 10.0])
        # grid[z][y][x] = z_value (so slicing at z=2.5 should give all 2.5)
        grid = np.zeros((3, 3, 3))
        for zi in range(3):
            grid[zi, :, :] = zs[zi]
        result = frozen_slice(grid, xs, ys, zs, z_value=2.5)
        np.testing.assert_array_almost_equal(result, np.full((3, 3), 2.5))

    def test_clamps_z_below(self):
        from gui.interpolation import frozen_slice
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        zs = np.array([1.0, 2.0])
        grid = np.array([
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ])
        result = frozen_slice(grid, xs, ys, zs, z_value=0.0)
        expected = np.array([[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Grid format conversion: sweep_data → compact interpolation format
# ---------------------------------------------------------------------------

class TestSweepToInterpGrid:
    def test_2d_sweep_data(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep_data = {
            "metric_keys": ["t1", "t2"],
            "xs": [1.0, 2.0],
            "ys": [3.0, 4.0],
            "grid": [
                [{"overall_fidelity": 0.9, "coherence_fidelity": 0.8},
                 {"overall_fidelity": 0.7, "coherence_fidelity": 0.6}],
                [{"overall_fidelity": 0.5, "coherence_fidelity": 0.4},
                 {"overall_fidelity": 0.3, "coherence_fidelity": 0.2}],
            ],
        }
        result = sweep_to_interp_grid(sweep_data, "overall_fidelity")
        assert result["xs"] == [1.0, 2.0]
        assert result["ys"] == [3.0, 4.0]
        assert result["ndim"] == 2
        # grid shape: (ny, nx) = (2, 2)
        expected_z = [[0.9, 0.5], [0.7, 0.3]]
        np.testing.assert_array_almost_equal(result["values"], expected_z)

    def test_3d_sweep_data(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep_data = {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": [1.0, 2.0],
            "ys": [3.0, 4.0],
            "zs": [5.0, 6.0],
            "grid": [
                [[{"overall_fidelity": 0.1}, {"overall_fidelity": 0.2}],
                 [{"overall_fidelity": 0.3}, {"overall_fidelity": 0.4}]],
                [[{"overall_fidelity": 0.5}, {"overall_fidelity": 0.6}],
                 [{"overall_fidelity": 0.7}, {"overall_fidelity": 0.8}]],
            ],
        }
        result = sweep_to_interp_grid(sweep_data, "overall_fidelity")
        assert result["ndim"] == 3
        assert result["zs"] == [5.0, 6.0]
        # grid shape: (nz, ny, nx) = (2, 2, 2)
        vals = np.array(result["values"])
        assert vals.shape == (2, 2, 2)

    def test_1d_sweep_data(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep_data = {
            "metric_keys": ["t1"],
            "xs": [1.0, 2.0, 3.0],
            "grid": [
                {"overall_fidelity": 0.9},
                {"overall_fidelity": 0.5},
                {"overall_fidelity": 0.1},
            ],
        }
        result = sweep_to_interp_grid(sweep_data, "overall_fidelity")
        assert result["ndim"] == 1
        assert result["values"] == [0.9, 0.5, 0.1]

    def test_missing_output_key_returns_zeros(self):
        from gui.interpolation import sweep_to_interp_grid
        sweep_data = {
            "metric_keys": ["t1"],
            "xs": [1.0, 2.0],
            "grid": [
                {"overall_fidelity": 0.9},
                {"overall_fidelity": 0.5},
            ],
        }
        result = sweep_to_interp_grid(sweep_data, "nonexistent_metric")
        assert result["values"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Frozen slider detection: which axis to freeze when 3D + 2D view
# ---------------------------------------------------------------------------

class TestFrozenAxisSelection:
    def test_returns_third_axis_index(self):
        """When 3 axes are active and view is 2D, freeze the 3rd axis."""
        from gui.interpolation import pick_frozen_axis
        assert pick_frozen_axis(num_axes=3, view_type="heatmap") == 2
        assert pick_frozen_axis(num_axes=3, view_type="contour") == 2

    def test_returns_none_for_3d_view(self):
        """No frozen axis when displaying all 3 in a native 3D view."""
        from gui.interpolation import pick_frozen_axis
        assert pick_frozen_axis(num_axes=3, view_type="scatter3d") is None
        assert pick_frozen_axis(num_axes=3, view_type="isosurface") is None

    def test_returns_none_for_2d_data(self):
        """No frozen axis when only 2 axes are active."""
        from gui.interpolation import pick_frozen_axis
        assert pick_frozen_axis(num_axes=2, view_type="heatmap") is None

    def test_returns_none_for_1d_data(self):
        from gui.interpolation import pick_frozen_axis
        assert pick_frozen_axis(num_axes=1, view_type="line") is None
