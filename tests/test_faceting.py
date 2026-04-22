"""Tests for categorical faceting logic."""

import pytest
import numpy as np
import plotly.graph_objects as go

from gui.constants import CATEGORICAL_METRICS, CAT_METRIC_BY_KEY
from gui.plotting import build_figure, sweep_to_csv


# ---------------------------------------------------------------------------
# Fixtures for faceted plotting
# ---------------------------------------------------------------------------


def _make_1d_facet(label_dict, n=10):
    """Build a single-facet 1D sweep_data dict."""
    xs = np.linspace(1e-5, 1e-1, n).tolist()
    grid = [{"overall_fidelity": 1.0 - i * 0.05, "total_epr_pairs": float(i)}
            for i in range(n)]
    return {"label": label_dict, "metric_keys": ["single_gate_error"], "xs": xs, "grid": grid}


def _make_4d_facet(label_dict, n=3):
    """Build a single-facet 4D sweep_data dict (structured numpy grid)."""
    axes = [np.linspace(0, 1, n).tolist() for _ in range(4)]
    shape = (n, n, n, n)
    total = n ** 4
    dt = np.dtype([("overall_fidelity", "f8"), ("total_epr_pairs", "f8")])
    grid = np.empty(shape, dtype=dt)
    for idx in np.ndindex(shape):
        grid[idx] = (1.0 - sum(idx) * 0.01, float(sum(idx)))
    return {
        "label": label_dict,
        "metric_keys": ["t1", "t2", "two_gate_time", "single_gate_error"],
        "axes": axes,
        "shape": list(shape),
        "grid": grid,
    }


def _make_2d_facet(label_dict, nx=5, ny=4):
    """Build a single-facet 2D sweep_data dict."""
    xs = np.linspace(1e-5, 1e-1, nx).tolist()
    ys = np.linspace(1e-4, 1e-2, ny).tolist()
    grid = []
    for i in range(nx):
        row = []
        for j in range(ny):
            row.append({"overall_fidelity": 1.0 - (i + j) * 0.02, "total_epr_pairs": float(i + j)})
        grid.append(row)
    return {"label": label_dict, "metric_keys": ["single_gate_error", "two_gate_error"], "xs": xs, "ys": ys, "grid": grid}


@pytest.fixture
def faceted_1d_data():
    """Faceted 1D sweep: 2 routing algorithms."""
    return {
        "metric_keys": ["single_gate_error"],
        "facet_keys": ["routing_algorithm"],
        "facets": [
            _make_1d_facet({"routing_algorithm": "HQA + Sabre"}),
            _make_1d_facet({"routing_algorithm": "TeleSABRE"}),
        ],
    }


@pytest.fixture
def faceted_4d_data():
    """Faceted 4D sweep: 2 placements (structured numpy grid)."""
    return {
        "metric_keys": ["t1", "t2", "two_gate_time", "single_gate_error"],
        "facet_keys": ["placement"],
        "facets": [
            _make_4d_facet({"placement": "Random"}),
            _make_4d_facet({"placement": "Spectral Clustering"}),
        ],
    }


@pytest.fixture
def faceted_2d_data():
    """Faceted 2D sweep: 2 routing x 3 topology."""
    facets = []
    for routing in ["HQA + Sabre", "TeleSABRE"]:
        for topo in ["Ring", "All-to-All", "Linear"]:
            facets.append(_make_2d_facet({"routing_algorithm": routing, "topology_type": topo}))
    return {
        "metric_keys": ["single_gate_error", "two_gate_error"],
        "facet_keys": ["routing_algorithm", "topology_type"],
        "facets": facets,
    }


# ---------------------------------------------------------------------------
# Tests: faceted build_figure
# ---------------------------------------------------------------------------


class TestFacetedBuildFigure:

    def test_1d_faceted_returns_figure(self, faceted_1d_data):
        fig = build_figure(1, faceted_1d_data, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_1d_faceted_has_traces_per_facet(self, faceted_1d_data):
        fig = build_figure(1, faceted_1d_data, "overall_fidelity")
        # Each 1D panel adds at least 1 trace
        assert len(fig.data) >= 2

    def test_2d_faceted_returns_figure(self, faceted_2d_data):
        fig = build_figure(2, faceted_2d_data, "overall_fidelity", view_type="contour")
        assert isinstance(fig, go.Figure)

    def test_2d_faceted_has_traces(self, faceted_2d_data):
        fig = build_figure(2, faceted_2d_data, "overall_fidelity", view_type="contour")
        # 6 facets x (heatmap + contour) = 12 traces minimum
        assert len(fig.data) >= 6

    def test_unfaceted_still_works(self):
        """Build_figure with no 'facets' key works as before."""
        xs = np.linspace(1e-5, 1e-1, 10).tolist()
        grid = [{"overall_fidelity": 1.0 - i * 0.05} for i in range(10)]
        data = {"metric_keys": ["single_gate_error"], "xs": xs, "grid": grid}
        fig = build_figure(1, data, "overall_fidelity")
        assert isinstance(fig, go.Figure)

    def test_subplot_titles_present(self, faceted_1d_data):
        fig = build_figure(1, faceted_1d_data, "overall_fidelity")
        annotations = [a for a in fig.layout.annotations if a.text]
        titles = [a.text for a in annotations]
        assert "HQA + Sabre" in titles
        assert "TeleSABRE" in titles

    def test_2d_shared_colorscale(self, faceted_2d_data):
        """All heatmap traces in a faceted 2D figure should share zmin/zmax."""
        fig = build_figure(2, faceted_2d_data, "overall_fidelity", view_type="heatmap")
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        if len(heatmaps) >= 2:
            zmins = {t.zmin for t in heatmaps if t.zmin is not None}
            zmaxs = {t.zmax for t in heatmaps if t.zmax is not None}
            assert len(zmins) == 1, f"Expected shared zmin, got {zmins}"
            assert len(zmaxs) == 1, f"Expected shared zmax, got {zmaxs}"


# ---------------------------------------------------------------------------
# Tests: faceted CSV export
# ---------------------------------------------------------------------------


class TestFacetedCsv:

    def test_csv_has_facet_key_columns(self, faceted_1d_data):
        csv = sweep_to_csv(faceted_1d_data)
        header = csv.split("\n")[0]
        assert "routing_algorithm" in header

    def test_csv_has_all_data_rows(self, faceted_1d_data):
        csv = sweep_to_csv(faceted_1d_data)
        lines = csv.strip().split("\n")
        # 1 header + 2 facets x 10 points = 21 lines
        assert len(lines) == 21

    def test_csv_2d_has_two_facet_columns(self, faceted_2d_data):
        csv = sweep_to_csv(faceted_2d_data)
        header = csv.split("\n")[0]
        assert "routing_algorithm" in header
        assert "topology_type" in header

    def test_unfaceted_csv_unchanged(self):
        """Non-faceted sweep_to_csv still works."""
        xs = np.linspace(1e-5, 1e-1, 5).tolist()
        grid = [{"overall_fidelity": 0.9 - i * 0.1} for i in range(5)]
        data = {"metric_keys": ["single_gate_error"], "xs": xs, "grid": grid}
        csv = sweep_to_csv(data)
        lines = csv.strip().split("\n")
        assert len(lines) == 6  # 1 header + 5 data
        assert "routing_algorithm" not in lines[0]


# ---------------------------------------------------------------------------
# Tests: faceted analysis views (parallel, slices, importance, pareto, corr.)
# ---------------------------------------------------------------------------


class TestFacetedAnalysisViews:
    """Analysis views must work with faceted data without crashing."""

    @pytest.mark.parametrize("view_type", [
        "parallel", "slices", "importance", "pareto", "correlation",
    ])
    def test_1d_faceted_analysis_views(self, faceted_1d_data, view_type):
        fig = build_figure(1, faceted_1d_data, "overall_fidelity", view_type=view_type)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    @pytest.mark.parametrize("view_type", [
        "parallel", "slices", "importance", "pareto", "correlation",
    ])
    def test_2d_faceted_analysis_views(self, faceted_2d_data, view_type):
        fig = build_figure(2, faceted_2d_data, "overall_fidelity", view_type=view_type)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_parallel_has_categorical_dimension(self, faceted_1d_data):
        """Parallel coords should include the categorical as a dimension."""
        fig = build_figure(1, faceted_1d_data, "overall_fidelity", view_type="parallel")
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)]
        assert len(parcoords) == 1
        dim_labels = [d["label"] for d in parcoords[0].dimensions]
        assert "Routing Algorithm" in dim_labels

    def test_parallel_categorical_has_tick_labels(self, faceted_1d_data):
        """Categorical dimension should show option names, not 0/1."""
        fig = build_figure(1, faceted_1d_data, "overall_fidelity", view_type="parallel")
        parcoords = [t for t in fig.data if isinstance(t, go.Parcoords)]
        cat_dim = [d for d in parcoords[0].dimensions if d["label"] == "Routing Algorithm"][0]
        assert "ticktext" in cat_dim
        assert "HQA + Sabre" in cat_dim["ticktext"]
        assert "TeleSABRE" in cat_dim["ticktext"]

    @pytest.mark.parametrize("view_type", [
        "parallel", "slices", "importance", "pareto", "correlation",
    ])
    def test_4d_faceted_analysis_views(self, faceted_4d_data, view_type):
        """4D + categorical (structured numpy grid) must not crash."""
        fig = build_figure(4, faceted_4d_data, "overall_fidelity", view_type=view_type)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_slices_categorical_has_tick_labels(self, faceted_1d_data):
        """Slice plot x-axis for categorical should show option names."""
        fig = build_figure(1, faceted_1d_data, "overall_fidelity", view_type="slices")
        # The subplot for the categorical axis should have ticktext set
        found_cat_labels = False
        for ax_key in dir(fig.layout):
            if ax_key.startswith("xaxis"):
                ax = getattr(fig.layout, ax_key)
                if ax.ticktext is not None and "HQA + Sabre" in list(ax.ticktext):
                    found_cat_labels = True
                    break
        assert found_cat_labels, "Expected categorical tick labels on slice x-axis"


# ---------------------------------------------------------------------------
# Tests: categorical on axes — toggle slider/checklist
# ---------------------------------------------------------------------------


class TestCategoricalOnAxes:

    def test_categorical_keys_in_registry(self):
        """All categoricals have entries in CAT_METRIC_BY_KEY."""
        for cat in CATEGORICAL_METRICS:
            assert cat.key in CAT_METRIC_BY_KEY

    def test_cold_config_key_defaults_to_key(self):
        """CatMetricDef.cold_config_key defaults to self.key."""
        for cat in CATEGORICAL_METRICS:
            if cat.key != "placement":
                assert cat.cold_config_key == cat.key

    def test_placement_maps_to_placement_policy(self):
        """Placement key maps to placement_policy for the engine."""
        cat = CAT_METRIC_BY_KEY["placement"]
        assert cat.cold_config_key == "placement_policy"

    def test_each_categorical_has_options(self):
        """Every categorical has at least 2 options."""
        for cat in CATEGORICAL_METRICS:
            assert len(cat.options) >= 2
