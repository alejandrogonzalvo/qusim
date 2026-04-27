"""Tests for the Merit-tab plot variants (Heatmap and Pareto).

The Merit tab evaluates a user-defined Figure of Merit (numerator /
denominator) over the sweep grid and renders it either as a 2-D heatmap
across two selected sweep axes (with the others pinned via frozen sliders)
or as a numerator-vs-denominator scatter with Pareto-front highlighting and
iso-FoM guide lines.

These tests focus on the *data-shaping* logic — Plotly figures are heavy and
visual output is verified manually; here we assert that the right traces are
produced and that the helpers (Pareto front, frozen-axis snapping) behave.
"""

import numpy as np
import pytest

from gui.fom import FomConfig
from gui.plotting import (
    _frozen_mask,
    _pareto_front_mask,
    _snap_to_grid,
    _surface_iso_segments,
    plot_merit,
    plot_merit_heatmap,
    plot_merit_pareto,
    plot_merit_surface,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_3d_sweep(t1_vals, t2_vals, cores_vals):
    """Build a tiny 3-axis structured-numpy sweep — the production grid path
    for ndim ≥ 4 also handles ndim 3, so this exercises the fast path.
    """
    shape = (len(t1_vals), len(t2_vals), len(cores_vals))
    dtype = np.dtype([("overall_fidelity", "f8"),
                      ("total_circuit_time_ns", "f8")])
    grid = np.empty(shape, dtype=dtype)
    T1, T2, C = np.meshgrid(t1_vals, t2_vals, cores_vals, indexing="ij")
    grid["overall_fidelity"] = 0.5 + 0.4 * (T1 / T1.max()) * (T2 / T2.max())
    grid["total_circuit_time_ns"] = 1000.0 / C
    return {
        "metric_keys": ["t1", "t2", "cores"],
        "axes": [list(t1_vals), list(t2_vals), list(cores_vals)],
        "grid": grid,
        "shape": shape,
    }


@pytest.fixture
def fom_fid_per_time():
    return FomConfig(
        name="Fid/Time",
        numerator="overall_fidelity",
        denominator="max(total_circuit_time_ns, 1)",
    )


@pytest.fixture
def sweep_3d():
    return _structured_3d_sweep(
        np.array([10.0, 20.0, 30.0]),
        np.array([5.0, 10.0]),
        np.array([2.0, 4.0, 8.0, 16.0]),
    )


# ---------------------------------------------------------------------------
# Pareto-front helper
# ---------------------------------------------------------------------------


class TestParetoFrontMask:
    def test_empty_input_returns_empty_mask(self):
        mask = _pareto_front_mask(np.array([]), np.array([]))
        assert mask.shape == (0,)

    def test_single_point_is_on_front(self):
        mask = _pareto_front_mask(np.array([1.0]), np.array([1.0]))
        assert mask.tolist() == [True]

    def test_dominated_point_excluded(self):
        # idx 0 dominated by idx 1 (same denominator, lower numerator).
        num = np.array([1.0, 2.0])
        den = np.array([1.0, 1.0])
        mask = _pareto_front_mask(num, den)
        assert mask.tolist() == [False, True]

    def test_tradeoff_keeps_both_endpoints(self):
        # Two non-dominated points + one strictly worse.
        num = np.array([1.0, 5.0, 0.5])
        den = np.array([1.0, 10.0, 0.2])
        mask = _pareto_front_mask(num, den)
        # idx 0 (n=1, d=1): not dominated — n=5 has d=10 (worse), n=0.5 has n<. → FRONT
        # idx 1 (n=5, d=10): no point has n>=5 AND d<=10 strictly — FRONT
        # idx 2 (n=0.5, d=0.2): no point has n>=0.5 AND d<=0.2 strictly — FRONT
        assert mask.tolist() == [True, True, True]

    def test_strictly_dominated_chain(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([3.0, 2.0, 1.0])  # idx 2 dominates everything
        mask = _pareto_front_mask(num, den)
        assert mask.tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# Frozen-axis snapping
# ---------------------------------------------------------------------------


class TestSnapToGrid:
    def test_snaps_to_nearest_grid_point(self):
        grid = np.array([1.0, 2.0, 5.0, 10.0])
        assert _snap_to_grid(3.0, grid) == 2.0
        assert _snap_to_grid(4.0, grid) == 5.0
        assert _snap_to_grid(0.0, grid) == 1.0
        assert _snap_to_grid(100.0, grid) == 10.0

    def test_empty_grid_returns_input(self):
        assert _snap_to_grid(7.5, np.array([])) == 7.5


class TestFrozenMask:
    def test_no_frozen_axes_keeps_everything(self):
        primitives = {"a": np.array([1.0, 2.0, 3.0])}
        mask, snapped = _frozen_mask(primitives, {}, 3)
        assert mask.all() and snapped == {}

    def test_snaps_then_masks_rows(self):
        # Two-axis sweep flattened: axis "a" repeats, "b" varies.
        a = np.array([1.0, 1.0, 2.0, 2.0])
        b = np.array([10.0, 20.0, 10.0, 20.0])
        mask, snapped = _frozen_mask({"a": a, "b": b}, {"a": 1.4}, 4)
        # 1.4 → snaps to 1.0; only first two rows keep.
        assert mask.tolist() == [True, True, False, False]
        assert snapped == {"a": 1.0}

    def test_unknown_axis_is_ignored(self):
        a = np.array([1.0, 2.0])
        mask, snapped = _frozen_mask({"a": a}, {"missing": 99.0}, 2)
        assert mask.all() and snapped == {}


# ---------------------------------------------------------------------------
# Heatmap mode
# ---------------------------------------------------------------------------


class TestPlotMeritHeatmap:
    def test_empty_axes_shows_message(self, fom_fid_per_time):
        sweep = {"metric_keys": [], "axes": [], "grid": [], "shape": ()}
        fig = plot_merit_heatmap(sweep, fom_fid_per_time)
        assert len(fig.data) == 0
        ann_texts = [a.text for a in (fig.layout.annotations or [])]
        assert any("Add at least one sweep axis" in t for t in ann_texts)

    def test_one_axis_falls_back_to_line_plot(self, fom_fid_per_time):
        sweep = {
            "metric_keys": ["t1"],
            "axes": [[10.0, 20.0, 30.0]],
            "grid": [
                {"overall_fidelity": 0.6, "total_circuit_time_ns": 100.0},
                {"overall_fidelity": 0.7, "total_circuit_time_ns": 110.0},
                {"overall_fidelity": 0.8, "total_circuit_time_ns": 120.0},
            ],
            "shape": (3,),
        }
        fig = plot_merit_heatmap(sweep, fom_fid_per_time)
        assert len(fig.data) == 1
        # 1-D fallback uses a Scatter trace with lines+markers, not a Heatmap.
        assert fig.data[0].type == "scatter"
        assert "lines" in (fig.data[0].mode or "")

    def test_two_axis_produces_heatmap(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit_heatmap(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2",
            frozen_values={"cores": 4.0},
        )
        # First trace is the Heatmap; with frozen axis, no contour overlay.
        assert fig.data[0].type == "heatmap"
        z = np.asarray(fig.data[0].z)
        assert z.shape == (2, 3)  # (len(t2), len(t1))

    def test_threshold_overlays_drawn_within_data_range(
        self, sweep_3d, fom_fid_per_time,
    ):
        fig = plot_merit_heatmap(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2",
            frozen_values={"cores": 4.0},
            thresholds=[0.0025, 0.0030, 999.0],  # third threshold out of range
            threshold_colors=["#d73027", "#fc8d59", "#fee08b"],
        )
        contours = [t for t in fig.data if t.type == "contour"]
        # Two in-range thresholds → two contour overlays.
        assert len(contours) == 2

    def test_invalid_xy_falls_back_to_first_axes(
        self, sweep_3d, fom_fid_per_time,
    ):
        # Stale axis selection should not crash — should fall back gracefully.
        fig = plot_merit_heatmap(
            sweep_3d, fom_fid_per_time,
            x_axis="bogus_axis", y_axis="bogus_axis_2",
            frozen_values={},
        )
        assert any(t.type == "heatmap" for t in fig.data)

    def test_frozen_axis_value_snaps_to_grid(self, sweep_3d, fom_fid_per_time):
        # cores grid is [2, 4, 8, 16] — 5.0 snaps to 4.
        fig = plot_merit_heatmap(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2",
            frozen_values={"cores": 5.0},
        )
        # Customdata for the heatmap stores the snapped frozen value at
        # column 2 (after num and den).
        cd = np.asarray(fig.data[0].customdata)
        # Frozen value column should be uniform = 4.0 (snapped).
        assert np.allclose(cd[:, :, 2], 4.0)


# ---------------------------------------------------------------------------
# Pareto mode
# ---------------------------------------------------------------------------


class TestPlotMeritPareto:
    def test_renders_dominated_and_front_traces(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit_pareto(sweep_3d, fom_fid_per_time)
        # At minimum: ≥1 marker trace (dominated and/or front) + a Pareto
        # connecting line, plus iso-FoM guide lines.
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(scatter_traces) >= 2

    def test_threshold_iso_lines_use_picked_colors(
        self, sweep_3d, fom_fid_per_time,
    ):
        fig = plot_merit_pareto(
            sweep_3d, fom_fid_per_time,
            thresholds=[0.0001, 0.0005],
            threshold_colors=["#d73027", "#fc8d59"],
        )
        # Iso-FoM lines are dashed Scatter traces with the user-picked colors.
        line_colors = [
            t.line.color for t in fig.data
            if t.type == "scatter" and t.mode and "lines" in t.mode
            and t.line and t.line.dash == "dash"
        ]
        assert "#d73027" in line_colors
        assert "#fc8d59" in line_colors

    def test_color_by_axis_uses_axis_label(self, sweep_3d, fom_fid_per_time):
        # Should not crash and should pick up the colour-by axis.
        fig = plot_merit_pareto(sweep_3d, fom_fid_per_time, color_by="t1")
        # The Pareto-front marker trace carries the colorbar; ensure it has
        # the expected continuous colour mapping.
        marker_traces = [
            t for t in fig.data
            if t.type == "scatter" and t.mode == "markers"
            and getattr(t.marker, "colorbar", None) is not None
        ]
        assert marker_traces

    def test_color_by_none_uses_neutral_color(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit_pareto(sweep_3d, fom_fid_per_time, color_by="none")
        # No marker trace should declare a colorscale.
        for t in fig.data:
            if t.type == "scatter" and t.mode == "markers":
                assert t.marker.colorscale is None


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


class TestPlotMeritDispatcher:
    def test_default_is_heatmap(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2", frozen_values={"cores": 4.0},
        )
        assert fig.data[0].type == "heatmap"

    def test_pareto_mode_routes_to_scatter(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit(sweep_3d, fom_fid_per_time, mode="pareto")
        assert any(t.type == "scatter" for t in fig.data)
        assert not any(t.type == "heatmap" for t in fig.data)

    def test_no_sweep_returns_empty_state(self, fom_fid_per_time):
        fig = plot_merit(None, fom_fid_per_time)
        ann = [a.text for a in (fig.layout.annotations or [])]
        assert any("No sweep loaded" in t for t in ann)

    def test_3d_mode_routes_to_surface(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit(
            sweep_3d, fom_fid_per_time, mode="3d",
            x_axis="t1", y_axis="t2", frozen_values={"cores": 4.0},
        )
        assert any(t.type == "surface" for t in fig.data)


# ---------------------------------------------------------------------------
# 3D Surface mode + iso-segment helper
# ---------------------------------------------------------------------------


class TestSurfaceIsoSegments:
    def test_no_crossings_returns_empty(self):
        # All cells flat at z=0 — no iso-line at z=1 inside the grid.
        z = np.zeros((3, 3))
        out = _surface_iso_segments(np.arange(3.0), np.arange(3.0), z, 1.0)
        assert out.shape == (0, 3)

    def test_simple_diagonal_ramp_produces_segments(self):
        # z = x + y → iso-line at level=2.0 crosses several cells.
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.add.outer(y, x)  # shape (3, 3): z[j,i] = y[j]+x[i]
        out = _surface_iso_segments(x, y, z, 2.0)
        assert out.shape[0] > 0
        # All non-NaN points must sit at z=level.
        finite = out[np.isfinite(out[:, 2])]
        assert np.allclose(finite[:, 2], 2.0)

    def test_skips_nan_cells(self):
        z = np.zeros((3, 3))
        z[1, 1] = np.nan
        # Without NaN handling this would crash; we just want a clean result.
        out = _surface_iso_segments(np.arange(3.0), np.arange(3.0), z, 0.5)
        assert out.dtype == float


class TestPlotMeritSurface:
    def test_empty_axes_shows_message(self, fom_fid_per_time):
        sweep = {"metric_keys": [], "axes": [], "grid": [], "shape": ()}
        fig = plot_merit_surface(sweep, fom_fid_per_time)
        assert len(fig.data) == 0
        ann = [a.text for a in (fig.layout.annotations or [])]
        assert any("Add at least one sweep axis" in t for t in ann)

    def test_two_axis_produces_surface_trace(self, sweep_3d, fom_fid_per_time):
        fig = plot_merit_surface(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2", frozen_values={"cores": 4.0},
        )
        assert fig.data[0].type == "surface"
        # Surface uses the FoM as both height and colour: surfacecolor must
        # match the z grid, not e.g. the input-axis value.
        z = np.asarray(fig.data[0].z)
        sc = np.asarray(fig.data[0].surfacecolor)
        assert z.shape == sc.shape
        assert np.allclose(z[np.isfinite(z)], sc[np.isfinite(sc)])

    def test_threshold_iso_lines_added_as_scatter3d(
        self, sweep_3d, fom_fid_per_time,
    ):
        fig = plot_merit_surface(
            sweep_3d, fom_fid_per_time,
            x_axis="t1", y_axis="t2", frozen_values={"cores": 4.0},
            thresholds=[0.0025, 0.003, 999.0],  # third out of range
            threshold_colors=["#d73027", "#fc8d59", "#fee08b"],
        )
        rings = [t for t in fig.data if t.type == "scatter3d"]
        assert len(rings) == 2

    def test_one_axis_falls_back_to_line(self, fom_fid_per_time):
        sweep = {
            "metric_keys": ["t1"],
            "axes": [[10.0, 20.0, 30.0]],
            "grid": [
                {"overall_fidelity": 0.6, "total_circuit_time_ns": 100.0},
                {"overall_fidelity": 0.7, "total_circuit_time_ns": 110.0},
                {"overall_fidelity": 0.8, "total_circuit_time_ns": 120.0},
            ],
            "shape": (3,),
        }
        fig = plot_merit_surface(sweep, fom_fid_per_time)
        # Degrades to a 2-D line (no surface trace possible with 1 axis).
        assert all(t.type != "surface" for t in fig.data)
        assert any(t.type == "scatter" for t in fig.data)
