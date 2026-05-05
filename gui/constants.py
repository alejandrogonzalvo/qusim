"""
GUI-only presentation knobs for the DSE app.

The pure-data parts of the parameter registry (``MetricDef``,
``SWEEPABLE_METRICS``, ``NOISE_DEFAULTS``, etc.) live in
:mod:`qusim.dse.axes` so Python users can consume them without pulling
Dash. This module re-exports them and adds the bits that only matter
to the Dash callbacks (view-tab choices, view-mode dropdowns, ...).
"""

from qusim.dse.axes import (  # noqa: F401  — re-export for back-compat
    CAT_METRIC_BY_KEY,
    CATEGORICAL_METRICS,
    CIRCUIT_TYPES,
    CatMetricDef,
    DEFAULT_SWEEP_AXES,
    FIDELITY_METRICS,
    INTRACORE_TOPOLOGY_TYPES,
    MAX_COLD_COMPILATIONS,
    MAX_SWEEP_AXES,
    MAX_TOTAL_POINTS_COLD,
    MAX_TOTAL_POINTS_HOT,
    MAX_WORKERS_DEFAULT,
    METRIC_BY_KEY,
    MIN_POINTS_PER_AXIS,
    MetricDef,
    NOISE_DEFAULTS,
    OUTPUT_METRIC_LABEL,
    OUTPUT_METRICS,
    PARETO_METRIC_ORIENTATION,
    PLACEMENT_OPTIONS,
    ROUTING_ALGORITHM_OPTIONS,
    SWEEP_POINTS_1D,
    SWEEP_POINTS_2D,
    SWEEP_POINTS_3D,
    SWEEP_POINTS_COLD_1D,
    SWEEP_POINTS_COLD_2D,
    SWEEP_POINTS_COLD_3D,
    SWEEPABLE_METRICS,
    TOPOLOGY_TYPES,
)


# ---------------------------------------------------------------------------
# GUI-only: view-tab definitions and view-mode dropdown.
# These shape what the Dash UI exposes — they have no meaning to a Python
# script consuming the library directly.
# ---------------------------------------------------------------------------

VIEW_TABS: dict[int, list[dict]] = {
    1: [
        {"value": "line", "label": "Line"},
    ],
    2: [
        {"value": "heatmap", "label": "Heatmap"},
    ],
    3: [
        {"value": "scatter3d", "label": "Scatter"},
        {"value": "isosurface", "label": "Isosurface"},
        {"value": "frozen_heatmap", "label": "Frozen Heat"},
    ],
}

VIEW_TAB_DEFAULTS: dict[int, str] = {
    1: "line",
    2: "heatmap",
    3: "isosurface",
}

VIEW_TAB_DEFAULT_ND = "parallel"

ANALYSIS_TABS: list[dict] = [
    {"value": "parallel", "label": "Parallel"},
    {"value": "slices", "label": "Slices"},
    {"value": "importance", "label": "Importance"},
    {"value": "pareto", "label": "Pareto"},
    {"value": "correlation", "label": "Corr."},
    {"value": "elasticity", "label": "Elasticity"},
    {"value": "merit", "label": "Merit"},
    {"value": "topology", "label": "Topology"},
]


# View-mode toggle: transforms the underlying scalar field on every
# dimensional view (Line / Heatmap / Isosurface / Frozen Heat).
VIEW_MODES: list[dict] = [
    {"value": "absolute", "label": "Absolute", "dims": None},
    {"value": "gradient_magnitude", "label": "|∇F|  (gradient magnitude)", "dims": None},
    {"value": "elasticity", "label": "Elasticity", "dims": (1,)},
    {"value": "second_derivative", "label": "d²F/dx²  (marks inflection)", "dims": (1,)},
    {"value": "mixed_partial", "label": "∂²F/∂x∂y  (interaction map)", "dims": (2,)},
]
DEFAULT_VIEW_MODE = "absolute"


def view_modes_for_dim(num_metrics: int) -> list[dict]:
    """Return the VIEW_MODES entries (sans the ``dims`` tag) that apply at
    the given sweep dimensionality. Always includes ``Absolute`` so the
    dropdown is never empty."""
    out = []
    for m in VIEW_MODES:
        dims = m.get("dims")
        if dims is None or num_metrics in dims:
            out.append({"value": m["value"], "label": m["label"]})
    return out or [{"value": "absolute", "label": "Absolute"}]
