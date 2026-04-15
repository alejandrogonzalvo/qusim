"""Tests for DSE GUI plotting views."""

import pytest
import numpy as np
import plotly.graph_objects as go

from gui.plotting import (
    plot_1d,
    plot_2d,
    plot_2d_contour,
    plot_3d,
    plot_3d_isosurface,
    plot_parallel_coordinates,
    plot_slice,
    plot_importance,
    plot_pareto,
    plot_correlation,
    build_figure,
    plot_empty,
    sweep_to_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_1d_data():
    xs = np.linspace(1e-5, 1e-1, 10)
    results = [{"overall_fidelity": 1.0 - i * 0.05, "total_epr_pairs": float(i)}
               for i in range(10)]
    return xs, results


@pytest.fixture
def sweep_2d_data():
    xs = np.linspace(1e-5, 1e-1, 5)
    ys = np.linspace(1e-4, 1e-2, 4)
    grid = []
    for i in range(len(xs)):
        row = []
        for j in range(len(ys)):
            row.append({
                "overall_fidelity": 1.0 - (i + j) * 0.02,
                "total_epr_pairs": float(i + j),
            })
        grid.append(row)
    return xs, ys, grid


@pytest.fixture
def sweep_data_store_1d(sweep_1d_data):
    xs, results = sweep_1d_data
    return {
        "metric_keys": ["single_gate_error"],
        "xs": xs.tolist(),
        "grid": results,
    }


@pytest.fixture
def sweep_data_store_2d(sweep_2d_data):
    xs, ys, grid = sweep_2d_data
    return {
        "metric_keys": ["single_gate_error", "two_gate_error"],
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Tests: plot_2d_contour
# ---------------------------------------------------------------------------

class TestPlot2dContour:
    def test_returns_figure(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Heatmap" in trace_types

    def test_has_contour_trace(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Contour" in trace_types

    def test_contour_shows_lines(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        contour_traces = [t for t in fig.data if isinstance(t, go.Contour)]
        assert len(contour_traces) >= 1
        contour = contour_traces[0]
        assert contour.contours.showlabels is True

    def test_z_shape_matches_grid(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        assert z.shape == (len(ys), len(xs))

    def test_non_fidelity_metric(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "total_epr_pairs")
        assert isinstance(fig, go.Figure)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.zmin != 0.0 or heatmap.zmax != 1.0

    def test_colorbar_present(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorbar is not None


# ---------------------------------------------------------------------------
# Tests: build_figure dispatches by view_type
# ---------------------------------------------------------------------------

class TestBuildFigureViewRouting:
    def test_default_2d_returns_heatmap(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Heatmap" in trace_types

    def test_contour_view_returns_contour(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity", view_type="contour")
        assert isinstance(fig, go.Figure)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Contour" in trace_types

    def test_heatmap_view_returns_plain_heatmap(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity", view_type="heatmap")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Heatmap" in trace_types
        assert "Contour" not in trace_types

    def test_1d_ignores_view_type(self, sweep_data_store_1d):
        fig = build_figure(1, sweep_data_store_1d, "overall_fidelity", view_type="contour")
        assert isinstance(fig, go.Figure)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter" in trace_types

    def test_none_sweep_data_returns_empty(self):
        fig = build_figure(1, None, "overall_fidelity")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_default_view_type_is_none(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: existing plot functions still work
# ---------------------------------------------------------------------------

class TestExistingPlots:
    def test_plot_1d_returns_figure(self, sweep_1d_data):
        xs, results = sweep_1d_data
        fig = plot_1d(xs, results, "single_gate_error", "overall_fidelity")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_plot_2d_returns_figure(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d(xs, ys, grid, "single_gate_error", "two_gate_error", "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_plot_3d_returns_figure(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d(xs, ys, zs, grid,
                      "single_gate_error", "two_gate_error", "t1", "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_plot_empty_returns_figure(self):
        fig = plot_empty("test message")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Fixtures: 3D sweep data
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_3d_data():
    xs = np.linspace(1e-5, 1e-1, 4)
    ys = np.linspace(1e-4, 1e-2, 3)
    zs = np.linspace(1e4, 1e6, 3)
    grid = []
    for i in range(len(xs)):
        plane = []
        for j in range(len(ys)):
            row = []
            for k in range(len(zs)):
                row.append({
                    "overall_fidelity": max(0.0, 1.0 - (i + j) * 0.05 + k * 0.02),
                    "total_epr_pairs": float(i + j + k),
                })
            plane.append(row)
        grid.append(plane)
    return xs, ys, zs, grid


@pytest.fixture
def sweep_3d_data_small():
    """A very sparse 2x2x2 grid — should trigger scatter3d fallback."""
    xs = np.array([1e-5, 1e-1])
    ys = np.array([1e-4, 1e-2])
    zs = np.array([1e4, 1e6])
    grid = []
    for i in range(2):
        plane = []
        for j in range(2):
            row = []
            for k in range(2):
                row.append({"overall_fidelity": 0.5 + 0.1 * (i + j + k),
                            "total_epr_pairs": float(i + j + k)})
            plane.append(row)
        grid.append(plane)
    return xs, ys, zs, grid


@pytest.fixture
def sweep_data_store_3d(sweep_3d_data):
    xs, ys, zs, grid = sweep_3d_data
    return {
        "metric_keys": ["single_gate_error", "two_gate_error", "t1"],
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "zs": zs.tolist(),
        "grid": grid,
    }


@pytest.fixture
def sweep_data_store_3d_small(sweep_3d_data_small):
    xs, ys, zs, grid = sweep_3d_data_small
    return {
        "metric_keys": ["single_gate_error", "two_gate_error", "t1"],
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "zs": zs.tolist(),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Tests: plot_3d_isosurface
# ---------------------------------------------------------------------------

class TestPlot3dIsosurface:
    def test_returns_figure(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_isosurface_trace(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Isosurface" in trace_types

    def test_has_3d_scene(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        assert fig.layout.scene is not None
        assert fig.layout.scene.xaxis.title.text is not None

    def test_colorbar_present(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        iso = [t for t in fig.data if isinstance(t, go.Isosurface)][0]
        assert iso.colorbar is not None

    def test_fallback_to_scatter3d_on_sparse_grid(self, sweep_3d_data_small):
        xs, ys, zs, grid = sweep_3d_data_small
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter3d" in trace_types

    def test_non_fidelity_metric(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "total_epr_pairs")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: build_figure 3D routing
# ---------------------------------------------------------------------------

class TestBuildFigure3dRouting:
    def test_default_3d_returns_scatter(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity")
        assert isinstance(fig, go.Figure)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter3d" in trace_types

    def test_isosurface_view_type(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity",
                           view_type="isosurface")
        assert isinstance(fig, go.Figure)
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Isosurface" in trace_types

    def test_scatter3d_view_type(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity",
                           view_type="scatter3d")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter3d" in trace_types

    def test_isosurface_fallback_on_sparse(self, sweep_data_store_3d_small):
        fig = build_figure(3, sweep_data_store_3d_small, "overall_fidelity",
                           view_type="isosurface")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter3d" in trace_types


# ---------------------------------------------------------------------------
# Tests: plot_parallel_coordinates
# ---------------------------------------------------------------------------

class TestPlotParallelCoordinates:
    def test_returns_figure_from_1d(self, sweep_data_store_1d):
        fig = plot_parallel_coordinates(sweep_data_store_1d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_2d(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_3d(self, sweep_data_store_3d):
        fig = plot_parallel_coordinates(sweep_data_store_3d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_parcoords_trace(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Parcoords" in trace_types

    def test_one_dimension_per_metric_plus_output(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "overall_fidelity")
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)][0]
        num_metrics = len(sweep_data_store_2d["metric_keys"])
        assert len(parcoords.dimensions) >= num_metrics + 1

    def test_lines_colored_by_output(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "overall_fidelity")
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)][0]
        assert parcoords.line.color is not None
        assert len(parcoords.line.color) > 0

    def test_non_fidelity_output(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "total_epr_pairs")
        assert isinstance(fig, go.Figure)
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)][0]
        assert parcoords.line.color is not None

    def test_all_output_metrics_as_dimensions(self, sweep_data_store_2d):
        fig = plot_parallel_coordinates(sweep_data_store_2d, "overall_fidelity")
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)][0]
        dim_labels = [d.label for d in parcoords.dimensions]
        assert any("fidelity" in l.lower() for l in dim_labels)


# ---------------------------------------------------------------------------
# Tests: build_figure parallel routing
# ---------------------------------------------------------------------------

class TestBuildFigureParallelRouting:
    def test_parallel_view_from_1d(self, sweep_data_store_1d):
        fig = build_figure(1, sweep_data_store_1d, "overall_fidelity",
                           view_type="parallel")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Parcoords" in trace_types

    def test_parallel_view_from_2d(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity",
                           view_type="parallel")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Parcoords" in trace_types

    def test_parallel_view_from_3d(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity",
                           view_type="parallel")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Parcoords" in trace_types


# ---------------------------------------------------------------------------
# Tests: plot_slice (marginal effects — Step 2.2)
# ---------------------------------------------------------------------------

class TestPlotSlice:
    def test_returns_figure_from_1d(self, sweep_data_store_1d):
        fig = plot_slice(sweep_data_store_1d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_2d(self, sweep_data_store_2d):
        fig = plot_slice(sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_3d(self, sweep_data_store_3d):
        fig = plot_slice(sweep_data_store_3d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_one_subplot_per_parameter(self, sweep_data_store_2d):
        fig = plot_slice(sweep_data_store_2d, "overall_fidelity")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        num_params = len(sweep_data_store_2d["metric_keys"])
        assert len(scatter_traces) == num_params

    def test_scatter_traces_have_data(self, sweep_data_store_2d):
        fig = plot_slice(sweep_data_store_2d, "overall_fidelity")
        for t in fig.data:
            if isinstance(t, go.Scatter):
                assert len(t.x) > 0
                assert len(t.y) > 0

    def test_shared_y_range_for_fidelity(self, sweep_data_store_2d):
        fig = plot_slice(sweep_data_store_2d, "overall_fidelity")
        yaxis = fig.layout.yaxis
        assert yaxis.range is not None or "fidelity" in str(yaxis.title)

    def test_non_fidelity_metric(self, sweep_data_store_2d):
        fig = plot_slice(sweep_data_store_2d, "total_epr_pairs")
        assert isinstance(fig, go.Figure)


class TestBuildFigureSliceRouting:
    def test_slice_view_from_1d(self, sweep_data_store_1d):
        fig = build_figure(1, sweep_data_store_1d, "overall_fidelity",
                           view_type="slices")
        assert isinstance(fig, go.Figure)

    def test_slice_view_from_2d(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity",
                           view_type="slices")
        assert isinstance(fig, go.Figure)

    def test_slice_view_from_3d(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity",
                           view_type="slices")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: plot_importance (parameter sensitivity — Step 2.3)
# ---------------------------------------------------------------------------

class TestPlotImportance:
    def test_returns_figure_from_1d(self, sweep_data_store_1d):
        fig = plot_importance(sweep_data_store_1d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_2d(self, sweep_data_store_2d):
        fig = plot_importance(sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_3d(self, sweep_data_store_3d):
        fig = plot_importance(sweep_data_store_3d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self, sweep_data_store_2d):
        fig = plot_importance(sweep_data_store_2d, "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Bar" in trace_types

    def test_one_bar_per_parameter(self, sweep_data_store_2d):
        fig = plot_importance(sweep_data_store_2d, "overall_fidelity")
        bar = [t for t in fig.data if isinstance(t, go.Bar)][0]
        num_params = len(sweep_data_store_2d["metric_keys"])
        assert len(bar.y) == num_params

    def test_bars_are_horizontal(self, sweep_data_store_2d):
        fig = plot_importance(sweep_data_store_2d, "overall_fidelity")
        bar = [t for t in fig.data if isinstance(t, go.Bar)][0]
        assert bar.orientation == "h"

    def test_bars_sorted_descending(self, sweep_data_store_2d):
        fig = plot_importance(sweep_data_store_2d, "overall_fidelity")
        bar = [t for t in fig.data if isinstance(t, go.Bar)][0]
        values = list(bar.x)
        assert values == sorted(values)


class TestBuildFigureImportanceRouting:
    def test_importance_view_from_2d(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity",
                           view_type="importance")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Bar" in trace_types


# ---------------------------------------------------------------------------
# Tests: plot_pareto (multi-objective front — Step 2.4)
# ---------------------------------------------------------------------------

class TestPlotPareto:
    def test_returns_figure_from_1d(self, sweep_data_store_1d):
        fig = plot_pareto(sweep_data_store_1d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_2d(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_scatter_traces(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Scatter" in trace_types

    def test_dominated_and_pareto_traces(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity")
        assert len(fig.data) >= 2

    def test_pareto_trace_is_highlighted(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity")
        names = [t.name for t in fig.data if hasattr(t, "name")]
        assert any("pareto" in (n or "").lower() for n in names)

    def test_axes_labeled(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity")
        assert fig.layout.xaxis.title is not None
        assert fig.layout.yaxis.title is not None


class TestBuildFigureParetoRouting:
    def test_pareto_view_from_2d(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity",
                           view_type="pareto")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: plot_correlation (Spearman correlation matrix — Step 2.5)
# ---------------------------------------------------------------------------

class TestPlotCorrelation:
    def test_returns_figure_from_1d(self, sweep_data_store_1d):
        fig = plot_correlation(sweep_data_store_1d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_from_2d(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Heatmap" in trace_types

    def test_square_matrix(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        hm = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(hm.z)
        assert z.shape[0] == z.shape[1]

    def test_diagonal_is_one(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        hm = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(hm.z)
        diag = np.diag(z)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_diverging_colorscale(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        hm = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert hm.zmin == -1.0
        assert hm.zmax == 1.0

    def test_annotations_present(self, sweep_data_store_2d):
        fig = plot_correlation(sweep_data_store_2d, "overall_fidelity")
        assert len(fig.layout.annotations) > 0


class TestBuildFigureCorrelationRouting:
    def test_correlation_view_from_2d(self, sweep_data_store_2d):
        fig = build_figure(2, sweep_data_store_2d, "overall_fidelity",
                           view_type="correlation")
        trace_types = [type(t).__name__ for t in fig.data]
        assert "Heatmap" in trace_types


# ---------------------------------------------------------------------------
# Tests: threshold overlay (cross-cutting feature)
# ---------------------------------------------------------------------------

class TestThresholdOverlay:
    def test_1d_no_threshold_no_extra_shapes(self, sweep_1d_data):
        xs, results = sweep_1d_data
        fig = plot_1d(xs, results, "single_gate_error", "overall_fidelity")
        shapes = fig.layout.shapes or ()
        assert len(shapes) == 0

    def test_1d_threshold_adds_hline(self, sweep_1d_data):
        xs, results = sweep_1d_data
        fig = plot_1d(xs, results, "single_gate_error", "overall_fidelity",
                      threshold=0.9)
        shapes = list(fig.layout.shapes)
        hlines = [s for s in shapes if s.type == "line" and s.y0 == s.y1 == 0.9]
        assert len(hlines) == 1

    def test_1d_threshold_adds_shaded_region(self, sweep_1d_data):
        xs, results = sweep_1d_data
        fig = plot_1d(xs, results, "single_gate_error", "overall_fidelity",
                      threshold=0.9)
        shapes = list(fig.layout.shapes)
        rects = [s for s in shapes if s.type == "rect"]
        assert len(rects) == 1

    def test_1d_threshold_none_no_shapes(self, sweep_1d_data):
        xs, results = sweep_1d_data
        fig = plot_1d(xs, results, "single_gate_error", "overall_fidelity",
                      threshold=None)
        shapes = fig.layout.shapes or ()
        assert len(shapes) == 0

    def test_2d_contour_threshold_adds_bold_contour(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error",
                              "overall_fidelity", threshold=0.9)
        contour_traces = [t for t in fig.data if isinstance(t, go.Contour)]
        assert len(contour_traces) >= 2

    def test_2d_contour_no_threshold_normal(self, sweep_2d_data):
        xs, ys, grid = sweep_2d_data
        fig = plot_2d_contour(xs, ys, grid, "single_gate_error", "two_gate_error",
                              "overall_fidelity")
        contour_traces = [t for t in fig.data if isinstance(t, go.Contour)]
        assert len(contour_traces) == 1

    def test_pareto_threshold_adds_hline(self, sweep_data_store_2d):
        fig = plot_pareto(sweep_data_store_2d, "overall_fidelity", threshold=0.9)
        shapes = list(fig.layout.shapes)
        hlines = [s for s in shapes if s.type == "line" and s.y0 == s.y1 == 0.9]
        assert len(hlines) == 1

    def test_build_figure_passes_threshold(self, sweep_data_store_1d):
        fig = build_figure(1, sweep_data_store_1d, "overall_fidelity",
                           threshold=0.9)
        shapes = list(fig.layout.shapes)
        assert len(shapes) >= 1

    # --- 3D scatter threshold ---

    def test_scatter3d_no_threshold_single_trace(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d(xs, ys, zs, grid,
                      "single_gate_error", "two_gate_error", "t1",
                      "overall_fidelity")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        assert len(scatter_traces) == 1

    def test_scatter3d_threshold_splits_into_two_traces(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d(xs, ys, zs, grid,
                      "single_gate_error", "two_gate_error", "t1",
                      "overall_fidelity", threshold=0.7)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        assert len(scatter_traces) == 2

    def test_scatter3d_threshold_dimmed_trace_has_low_opacity(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d(xs, ys, zs, grid,
                      "single_gate_error", "two_gate_error", "t1",
                      "overall_fidelity", threshold=0.7)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        dimmed = [t for t in scatter_traces if "below" in (t.name or "").lower()]
        assert len(dimmed) == 1
        assert dimmed[0].marker.opacity <= 0.3

    def test_scatter3d_threshold_none_no_split(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d(xs, ys, zs, grid,
                      "single_gate_error", "two_gate_error", "t1",
                      "overall_fidelity", threshold=None)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
        assert len(scatter_traces) == 1

    # --- 3D isosurface threshold ---

    def test_isosurface_threshold_adds_extra_surface(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity", threshold=0.7)
        iso_traces = [t for t in fig.data if isinstance(t, go.Isosurface)]
        assert len(iso_traces) == 2

    def test_isosurface_threshold_surface_is_red(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity", threshold=0.7)
        iso_traces = [t for t in fig.data if isinstance(t, go.Isosurface)]
        threshold_iso = [t for t in iso_traces if "threshold" in (t.name or "").lower()]
        assert len(threshold_iso) == 1

    def test_isosurface_no_threshold_single_iso(self, sweep_3d_data):
        xs, ys, zs, grid = sweep_3d_data
        fig = plot_3d_isosurface(xs, ys, zs, grid,
                                 "single_gate_error", "two_gate_error", "t1",
                                 "overall_fidelity")
        iso_traces = [t for t in fig.data if isinstance(t, go.Isosurface)]
        assert len(iso_traces) == 1

    def test_build_figure_3d_passes_threshold(self, sweep_data_store_3d):
        fig = build_figure(3, sweep_data_store_3d, "overall_fidelity",
                           threshold=0.7)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: CSV export (sweep_to_csv)
# ---------------------------------------------------------------------------

class TestSweepToCsv:
    def test_returns_string(self, sweep_data_store_1d):
        csv = sweep_to_csv(sweep_data_store_1d)
        assert isinstance(csv, str)
        assert len(csv) > 0

    def test_has_header_row(self, sweep_data_store_1d):
        csv = sweep_to_csv(sweep_data_store_1d)
        header = csv.split("\n")[0]
        assert "single_gate_error" in header or "1Q Gate Error" in header

    def test_has_data_rows(self, sweep_data_store_1d):
        csv = sweep_to_csv(sweep_data_store_1d)
        lines = [l for l in csv.strip().split("\n") if l]
        assert len(lines) >= 2

    def test_2d_has_all_points(self, sweep_data_store_2d):
        csv = sweep_to_csv(sweep_data_store_2d)
        lines = [l for l in csv.strip().split("\n") if l]
        xs = sweep_data_store_2d["xs"]
        ys = sweep_data_store_2d["ys"]
        assert len(lines) == 1 + len(xs) * len(ys)

    def test_3d_has_all_points(self, sweep_data_store_3d):
        csv = sweep_to_csv(sweep_data_store_3d)
        lines = [l for l in csv.strip().split("\n") if l]
        xs = sweep_data_store_3d["xs"]
        ys = sweep_data_store_3d["ys"]
        zs = sweep_data_store_3d["zs"]
        assert len(lines) == 1 + len(xs) * len(ys) * len(zs)
