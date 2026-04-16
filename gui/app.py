"""
qusim DSE GUI — main Dash application.

Launch:
    python gui/app.py   (from project root)

Layout
------
┌──────────────────────────────────────────────────────────┐
│  Left sidebar        │  Centre (plot)  │  Right panel    │
│  (sweep metrics)     │                 │  (config)       │
└──────────────────────────────────────────────────────────┘
"""

import os
import sys
import threading
import time
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, ctx, dcc, html

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.components import (
    COLORS,
    _linear_marks,
    _log_marks,
    make_add_metric_button,
    make_fixed_config_panel,
    make_metric_selector,
    make_view_tab_bar,
)
from gui.constants import (
    ANALYSIS_TABS,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    OUTPUT_METRICS,
    SWEEPABLE_METRICS,
    VIEW_TAB_DEFAULTS,
    VIEW_TABS,
)
from gui.dse_engine import DSEEngine, SweepProgress
from gui.interpolation import (
    frozen_slider_config,
    is_frozen_view,
    sweep_to_interp_grid,
)
from gui.plotting import build_figure, plot_empty, sweep_to_csv

_ANALYSIS_VIEW_KEYS = {t["value"] for t in ANALYSIS_TABS}
_SWEEP_VIEW_KEYS: dict[int, set[str]] = {
    dim: {t["value"] for t in tabs} for dim, tabs in VIEW_TABS.items()
}


def resolve_view_type(current_view: str | None, num_active: int) -> str:
    """Return the view type to use for a sweep with *num_active* dimensions.

    Preserves *current_view* when it is valid for *num_active* (either a
    sweep-specific tab or a dimensionality-agnostic analysis tab).  Falls
    back to the default only on first run or when the dimensionality changed
    to one where the previous view doesn't exist.
    """
    if current_view is not None:
        valid = _SWEEP_VIEW_KEYS.get(num_active, set()) | _ANALYSIS_VIEW_KEYS
        if current_view in valid:
            return current_view
    return VIEW_TAB_DEFAULTS.get(num_active, "line")


def should_skip_poll(
    triggered_ids: list[str],
    hot_reload: list[str],
    dirty: int,
    processed: int,
) -> bool:
    """Decide whether the sweep-poll tick should be skipped.

    Returns False (do NOT skip) when auto-run-trigger or run-btn is among
    the triggers, even if sweep-poll also fired in the same render cycle.
    """
    ids = set(triggered_ids)
    if ids - {"sweep-poll"}:
        return False
    if "sweep-poll" in ids:
        hot_reload_on = hot_reload and "on" in hot_reload
        if not hot_reload_on or dirty == processed:
            return True
    return False


def sweep_gate(dirty: int, processed: int) -> int | None:
    """Return the trigger value when a sweep is needed, None otherwise.

    Python mirror of the clientside JS gate callback.  The JS version
    returns ``window.dash_clientside.no_update`` instead of None.
    """
    if dirty > processed:
        return dirty
    return None


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    title="qusim DSE",
    update_title=None,
    suppress_callback_exceptions=True,
)
server = app.server

_engine = DSEEngine()
sweep_lock = threading.Lock()
MAX_METRICS = 3

# Shared progress state — written by sweep callback, read by Flask route.
# Simple dict is safe under the GIL for atomic reads/writes.
_sweep_progress: dict = {"running": False}


def _update_progress(p: SweepProgress) -> None:
    """Progress callback passed into sweep methods."""
    global _sweep_progress
    _sweep_progress = {
        "running": True,
        "completed": p.completed,
        "total": p.total,
        "percentage": p.percentage,
        "current_params": {
            METRIC_BY_KEY[k].label if k in METRIC_BY_KEY else k: v
            for k, v in p.current_params.items()
        },
    }


@server.route("/api/progress")
def _api_progress():
    import json
    return json.dumps(_sweep_progress), 200, {"Content-Type": "application/json"}


# Global CSS is in gui/assets/style.css — Dash auto-loads it.

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _topbar() -> html.Div:
    return html.Div(
        style={
            "background": COLORS["surface"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "0 20px",
            "height": "52px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Span(
                        "qusim",
                        style={
                            "fontSize": "18px",
                            "fontWeight": "700",
                            "color": COLORS["accent"],
                        },
                    ),
                    html.Span(
                        "DSE Explorer",
                        style={"fontSize": "14px", "color": COLORS["text_muted"]},
                    ),
                ],
            ),
            html.Div(
                id="status-bar",
                children="Ready",
                style={
                    "fontSize": "12px",
                    "color": COLORS["text_muted"],
                    "flex": "1",
                    "textAlign": "center",
                },
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    dcc.Checklist(
                        id="hot-reload-toggle",
                        options=[{"label": " Hot reload", "value": "on"}],
                        value=["on"],
                        style={"fontSize": "12px", "color": COLORS["text_muted"]},
                        inputStyle={"marginRight": "4px"},
                    ),
                    html.Button(
                        "▶  Run",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            "background": COLORS["accent"],
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "6px 20px",
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "display": "none",
                        },
                    ),
                ],
            ),
        ],
    )


def _left_sidebar() -> html.Div:
    return html.Div(
        className="sidebar-scroll",
        style={
            "width": "270px",
            "minWidth": "250px",
            "background": COLORS["bg"],
            "borderRight": f"1px solid {COLORS['border']}",
            "padding": "14px 12px",
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Div(
                "Sweep Axes",
                style={
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": COLORS["accent"],
                    "marginBottom": "8px",
                    "paddingBottom": "6px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                },
            ),
            # Always render all 3 rows; show/hide via 'display'
            html.Div(id="metric-row-wrap-0", children=[make_metric_selector(0)]),
            html.Div(id="metric-row-wrap-1", children=[make_metric_selector(1)]),
            html.Div(id="metric-row-wrap-2", children=[make_metric_selector(2)]),
            # Add / remove buttons
            html.Div(
                style={"display": "flex", "gap": "8px", "marginTop": "4px"},
                children=[
                    html.Button(
                        "+ Add axis",
                        id="add-metric-btn",
                        n_clicks=0,
                        style={
                            "flex": "1",
                            "background": "transparent",
                            "border": f"1px dashed {COLORS['border']}",
                            "color": COLORS["text_muted"],
                            "borderRadius": "6px",
                            "padding": "6px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                        },
                    ),
                    html.Button(
                        "− Remove",
                        id="remove-metric-btn",
                        n_clicks=0,
                        style={
                            "flex": "1",
                            "background": "transparent",
                            "border": f"1px solid {COLORS['border']}",
                            "color": COLORS["text_muted"],
                            "borderRadius": "6px",
                            "padding": "6px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                        },
                    ),
                ],
            ),
        ],
    )


def _center_panel() -> html.Div:
    return html.Div(
        style={
            "flex": "1",
            "minWidth": "0",
            "padding": "10px",
            "display": "flex",
            "flexDirection": "column",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                },
                children=[
                    html.Div(
                        id="view-tab-container",
                        children=[
                            make_view_tab_bar(num_metrics=3, active="isosurface")
                        ],
                    ),
                    html.Button(
                        "CSV",
                        id="export-csv-btn",
                        n_clicks=0,
                        style={
                            "background": "transparent",
                            "border": f"1px solid {COLORS['border']}",
                            "color": COLORS["text_muted"],
                            "borderRadius": "4px",
                            "padding": "4px 12px",
                            "fontSize": "12px",
                            "cursor": "pointer",
                            "marginLeft": "8px",
                        },
                    ),
                ],
            ),
            html.Div(
                id="plot-container",
                style={
                    "flex": "1",
                    "minHeight": "0",
                    "display": "flex",
                    "flexDirection": "column",
                    "position": "relative",
                },
                children=[
                    dcc.Graph(
                        id="main-plot",
                        figure=plot_empty(),
                        style={"flex": "1", "minHeight": "0", "height": "100%"},
                        responsive=True,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                            "toImageButtonOptions": {
                                "format": "png",
                                "width": 1200,
                                "height": 800,
                                "scale": 2,
                            },
                        },
                    ),
                    html.Div(
                        id="sweep-progress-overlay",
                        style={"display": "none"},
                    ),
                ],
            ),
            html.Div(
                id="frozen-slider-container",
                style={"display": "none", "padding": "4px 16px 8px"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "8px"},
                        children=[
                            html.Span(
                                id="frozen-slider-label",
                                children="Frozen axis",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "600",
                                    "color": COLORS["text_muted"],
                                    "whiteSpace": "nowrap",
                                },
                            ),
                            dcc.Slider(
                                id="frozen-slider",
                                min=0, max=1, step=0.01, value=0.5,
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True},
                                updatemode="drag",
                            ),
                            html.Span(
                                id="frozen-slider-value",
                                style={
                                    "fontSize": "11px",
                                    "color": COLORS["accent"],
                                    "minWidth": "60px",
                                    "textAlign": "right",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Download(id="csv-download"),
        ],
    )


def _right_panel() -> html.Div:
    return html.Div(
        style={
            "width": "270px",
            "minWidth": "250px",
            "background": COLORS["bg"],
            "borderLeft": f"1px solid {COLORS['border']}",
            "padding": "14px 12px 0",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                "Configuration",
                style={
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": COLORS["accent"],
                    "marginBottom": "8px",
                    "paddingBottom": "6px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                },
            ),
            html.Div(
                id="fixed-config-container",
                className="config-scroll",
                style={"flex": "1", "overflow": "auto", "paddingBottom": "12px"},
                children=[make_fixed_config_panel()],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    style={"height": "100vh", "overflow": "hidden", "background": COLORS["bg"]},
    children=[
        _topbar(),
        html.Div(
            style={
                "display": "flex",
                "height": "calc(100vh - 52px)",
                "overflow": "hidden",
            },
            children=[_left_sidebar(), _center_panel(), _right_panel()],
        ),
        # State stores
        dcc.Store(id="num-metrics-store", data=3, storage_type="memory"),
        dcc.Store(id="sweep-result-store", data=None, storage_type="memory"),
        dcc.Store(id="view-type-store", data="isosurface", storage_type="memory"),
        dcc.Store(id="num-thresholds-store", data=3, storage_type="memory"),
        dcc.Store(id="sweep-dirty", data=1, storage_type="memory"),
        dcc.Store(id="sweep-processed", data=0, storage_type="memory"),
        dcc.Store(id="sweep-trigger", data=0, storage_type="memory"),
        dcc.Store(id="interp-grid-store", data=None, storage_type="memory"),
        dcc.Interval(id="sweep-check", interval=16, n_intervals=0),
    ],
)


# ---------------------------------------------------------------------------
# Clientside gate: 60fps JS check, zero HTTP cost when idle
# ---------------------------------------------------------------------------
# sweep-processed is a State (not Input) so no dependency cycle.
# A pending flag ensures exactly one server trigger per needed sweep.

app.clientside_callback(
    """function(n, dirty, processed) {
        if (processed !== (window._lastProcessed || 0)) {
            window._lastProcessed = processed;
            window._sweepPending = false;
        }
        if (dirty > processed && !window._sweepPending) {
            window._sweepPending = true;
            return n;
        }
        return window.dash_clientside.no_update;
    }""",
    Output("sweep-trigger", "data"),
    Input("sweep-check", "n_intervals"),
    State("sweep-dirty", "data"),
    State("sweep-processed", "data"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Callback: add / remove metric rows (show/hide)
# ---------------------------------------------------------------------------


@app.callback(
    Output("metric-row-wrap-0", "style"),
    Output("metric-row-wrap-1", "style"),
    Output("metric-row-wrap-2", "style"),
    Output("remove-metric-btn", "style"),
    Output("num-metrics-store", "data"),
    Input("add-metric-btn", "n_clicks"),
    Input("remove-metric-btn", "n_clicks"),
    State("num-metrics-store", "data"),
    prevent_initial_call=True,
)
def toggle_metric_rows(add_clicks, remove_clicks, num_metrics):
    triggered = ctx.triggered_id
    if triggered == "add-metric-btn":
        num_metrics = min(MAX_METRICS, num_metrics + 1)
    elif triggered == "remove-metric-btn":
        num_metrics = max(1, num_metrics - 1)

    def _show():
        return {}

    def _hide():
        return {"display": "none"}

    row_styles = [
        _show(),
        _show() if num_metrics >= 2 else _hide(),
        _show() if num_metrics >= 3 else _hide(),
    ]

    remove_style = (
        {
            "flex": "1",
            "background": "transparent",
            "border": f"1px solid {COLORS['border']}",
            "color": COLORS["text_muted"],
            "borderRadius": "6px",
            "padding": "6px",
            "cursor": "pointer",
            "fontSize": "12px",
        }
        if num_metrics > 1
        else {"display": "none"}
    )

    return *row_styles, remove_style, num_metrics


# ---------------------------------------------------------------------------
# Callback: add / remove threshold rows
# ---------------------------------------------------------------------------


@app.callback(
    *[Output(f"threshold-row-{i}", "style") for i in range(5)],
    Output("remove-threshold-btn", "style"),
    Output("num-thresholds-store", "data"),
    Input("add-threshold-btn", "n_clicks"),
    Input("remove-threshold-btn", "n_clicks"),
    State("num-thresholds-store", "data"),
    prevent_initial_call=True,
)
def toggle_threshold_rows(add_clicks, remove_clicks, num_thresholds):
    triggered = ctx.triggered_id
    if triggered == "add-threshold-btn":
        num_thresholds = min(5, num_thresholds + 1)
    elif triggered == "remove-threshold-btn":
        num_thresholds = max(1, num_thresholds - 1)

    row_styles = [{} if i < num_thresholds else {"display": "none"} for i in range(5)]

    remove_style = (
        {
            "background": "transparent",
            "border": f"1px solid {COLORS['border']}",
            "color": COLORS["text_muted"],
            "borderRadius": "4px",
            "width": "28px",
            "height": "28px",
            "cursor": "pointer",
            "fontSize": "14px",
        }
        if num_thresholds > 1
        else {"display": "none"}
    )

    return *row_styles, remove_style, num_thresholds


# ---------------------------------------------------------------------------
# Callback: show/hide noise rows when swept metric dropdowns change
# ---------------------------------------------------------------------------
# All noise-row-{key} divs are always in the DOM; we toggle display only.

from gui.constants import SWEEPABLE_METRICS as _SM


@app.callback(
    [Output(f"noise-row-{m.key}", "style") for m in _SM]
    + [Output("cfg-row-num-qubits", "style"), Output("cfg-row-num-cores", "style")],
    Input("metric-dropdown-0", "value"),
    Input("metric-dropdown-1", "value"),
    Input("metric-dropdown-2", "value"),
    State("num-metrics-store", "data"),
    prevent_initial_call=False,
)
def toggle_noise_rows(m0, m1, m2, num_metrics):
    swept = set()
    for i, val in enumerate([m0, m1, m2]):
        if i < (num_metrics or 1) and val:
            swept.add(val)
    noise_styles = [
        {"display": "none"} if m.is_cold_path or m.key in swept else {} for m in _SM
    ]
    qubits_style = {"display": "none"} if "num_qubits" in swept else {}
    cores_style = {"display": "none"} if "num_cores" in swept else {}
    return noise_styles + [qubits_style, cores_style]


# ---------------------------------------------------------------------------
# Helper: convert result dict/object to plain JSON-safe dict
# ---------------------------------------------------------------------------


def _result_to_dict(r: Any) -> dict:
    if isinstance(r, dict):
        return {
            k: float(v)
            for k, v in r.items()
            if isinstance(v, (int, float, np.floating))
        }
    return {
        "overall_fidelity": float(r.overall_fidelity),
        "algorithmic_fidelity": float(r.algorithmic_fidelity),
        "routing_fidelity": float(r.routing_fidelity),
        "coherence_fidelity": float(r.coherence_fidelity),
        "total_circuit_time_ns": float(r.total_circuit_time_ns),
        "total_epr_pairs": float(getattr(r, "total_epr_pairs", 0)),
    }


def _slider_to_value(slider_pos: float, log_scale: bool) -> float:
    return 10.0**slider_pos if log_scale else slider_pos


# ---------------------------------------------------------------------------
# Callback: run sweep
# ---------------------------------------------------------------------------

_NOISE_SLIDER_STATES = [State(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS]

_NUM_SWEEP_OUTPUTS = 11
_NO_UPDATE_SWEEP = (dash.no_update,) * _NUM_SWEEP_OUTPUTS

# Clientside: any simulation input change → increment dirty counter
_SIM_INPUTS = [
    Input("metric-dropdown-0", "value"),
    Input("metric-slider-0", "value"),
    Input("metric-dropdown-1", "value"),
    Input("metric-slider-1", "value"),
    Input("metric-dropdown-2", "value"),
    Input("metric-slider-2", "value"),
    Input("num-metrics-store", "data"),
    Input("cfg-circuit-type", "value"),
    Input("cfg-num-qubits", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-topology", "value"),
    Input("cfg-intracore-topology", "value"),
    Input("cfg-placement", "value"),
    Input("cfg-seed", "value"),
    Input("cfg-dynamic-decoupling", "value"),
    *[Input(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS],
]

app.clientside_callback(
    """function() {
        window._sweepDirty = (window._sweepDirty || 0) + 1;
        return window._sweepDirty;
    }""",
    Output("sweep-dirty", "data"),
    *_SIM_INPUTS,
    prevent_initial_call=True,
)


@app.callback(
    Output("main-plot", "figure"),
    Output("sweep-result-store", "data"),
    Output("status-bar", "children"),
    Output("view-type-store", "data"),
    Output("view-tab-container", "children"),
    Output("sweep-processed", "data"),
    Output("interp-grid-store", "data"),
    Output("frozen-slider-container", "style"),
    Output("frozen-slider", "min"),
    Output("frozen-slider", "max"),
    Output("frozen-slider-label", "children"),
    Input("sweep-trigger", "data"),
    Input("run-btn", "n_clicks"),
    State("sweep-dirty", "data"),
    State("hot-reload-toggle", "value"),
    State("metric-dropdown-0", "value"),
    State("metric-slider-0", "value"),
    State("metric-dropdown-1", "value"),
    State("metric-slider-1", "value"),
    State("metric-dropdown-2", "value"),
    State("metric-slider-2", "value"),
    State("num-metrics-store", "data"),
    State("cfg-circuit-type", "value"),
    State("cfg-num-qubits", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-topology", "value"),
    State("cfg-intracore-topology", "value"),
    State("cfg-placement", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-output-metric", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    State("view-type-store", "data"),
    *_NOISE_SLIDER_STATES,
    prevent_initial_call=True,
)
def run_sweep(
    trigger,
    n_clicks,
    dirty,
    hot_reload,
    m0_key,
    m0_range,
    m1_key,
    m1_range,
    m2_key,
    m2_range,
    num_metrics,
    circuit_type,
    num_qubits,
    num_cores,
    topology,
    intracore_topology,
    placement,
    seed,
    dynamic_decoupling,
    output_key,
    threshold_enable,
    t0,
    t1,
    t2,
    t3,
    t4,
    tc0,
    tc1,
    tc2,
    tc3,
    tc4,
    num_thresholds,
    current_view,
    *noise_slider_vals,
):
    is_run_btn = ctx.triggered_id == "run-btn"
    hot_reload_on = hot_reload and "on" in hot_reload
    if not is_run_btn and not hot_reload_on:
        return _NO_UPDATE_SWEEP

    sweep_lock.acquire()

    try:
        global _sweep_progress
        _sweep_progress = {"running": True, "completed": 0, "total": 0, "percentage": 0, "current_params": {}}
        t_start = time.time()

        # Build fixed noise dict from right-panel sliders
        fixed_noise: dict = {}
        for i, m in enumerate(SWEEPABLE_METRICS):
            if m.is_cold_path:
                continue
            val = noise_slider_vals[i]
            if val is not None:
                fixed_noise[m.key] = _slider_to_value(val, m.log_scale)
            else:
                fixed_noise[m.key] = NOISE_DEFAULTS[m.key]
        fixed_noise["dynamic_decoupling"] = bool(dynamic_decoupling)

        # Active metrics
        all_inputs = [(m0_key, m0_range), (m1_key, m1_range), (m2_key, m2_range)]
        active = []
        seen: set = set()
        for i, (k, r) in enumerate(all_inputs[: int(num_metrics or 1)]):
            if k and r and k not in seen:
                seen.add(k)
                active.append((k, r))

        if not active:
            return (
                plot_empty("Add at least one metric axis and click Run"),
                None,
                "No metrics configured",
                dash.no_update,
                dash.no_update,
                dirty,
                None,
                {"display": "none"},
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        cold_config = {
            "circuit_type": circuit_type or "qft",
            "num_qubits": int(num_qubits or 16),
            "num_cores": int(num_cores or 4),
            "topology_type": topology or "ring",
            "placement_policy": placement or "random",
            "seed": int(seed or 42),
            "intracore_topology": intracore_topology or "all_to_all",
        }

        cached = _engine.run_cold(**cold_config, noise=fixed_noise)
        cold_elapsed = time.time() - t_start

        sweep_data: dict = {"metric_keys": [k for k, _ in active]}

        if len(active) == 1:
            k0, r0 = active[0]
            xs, results = _engine.sweep_1d(
                cached, k0, r0[0], r0[1], fixed_noise, cold_config=cold_config,
                progress_callback=_update_progress,
            )
            sweep_data["xs"] = xs.tolist()
            sweep_data["grid"] = [_result_to_dict(r) for r in results]

        elif len(active) == 2:
            k0, r0 = active[0]
            k1, r1 = active[1]
            xs, ys, grid = _engine.sweep_2d(
                cached,
                k0, r0[0], r0[1],
                k1, r1[0], r1[1],
                fixed_noise,
                cold_config=cold_config,
                progress_callback=_update_progress,
            )
            sweep_data["xs"] = xs.tolist()
            sweep_data["ys"] = ys.tolist()
            sweep_data["grid"] = [[_result_to_dict(r) for r in row] for row in grid]

        elif len(active) == 3:
            k0, r0 = active[0]
            k1, r1 = active[1]
            k2, r2 = active[2]
            xs, ys, zs, grid = _engine.sweep_3d(
                cached,
                k0, r0[0], r0[1],
                k1, r1[0], r1[1],
                k2, r2[0], r2[1],
                fixed_noise,
                cold_config=cold_config,
                progress_callback=_update_progress,
            )
            sweep_data["xs"] = xs.tolist()
            sweep_data["ys"] = ys.tolist()
            sweep_data["zs"] = zs.tolist()
            sweep_data["grid"] = [
                [[_result_to_dict(r) for r in row] for row in plane] for plane in grid
            ]

        sweep_elapsed = time.time() - t_start - cold_elapsed
        total_elapsed = time.time() - t_start
        cache_indicator = (
            "cached ✓" if cached.cold_time_s < 0.1 else f"mapped in {cold_elapsed:.1f}s"
        )
        status = (
            f"Cold path: {cache_indicator}  |  "
            f"Sweep ({len(active)}D, {_count_points(sweep_data)} pts): {sweep_elapsed:.2f}s  |  "
            f"Total: {total_elapsed:.1f}s"
        )

        view = resolve_view_type(current_view, len(active))
        n_t = int(num_thresholds or 3)
        all_t = [t0, t1, t2, t3, t4][:n_t]
        all_c = [tc0, tc1, tc2, tc3, tc4][:n_t]
        thresh_vals = [v for v in all_t if v is not None]
        thresh_colors = [all_c[i] for i, v in enumerate(all_t) if v is not None]
        thresh = thresh_vals if threshold_enable and "yes" in threshold_enable else None
        if view in ("isosurface", "scatter3d"):
            thresh = thresh_vals or None

        out_key = output_key or "overall_fidelity"
        interp_grid = None
        frozen_style = {"display": "none"}
        frozen_min = dash.no_update
        frozen_max = dash.no_update
        frozen_label = dash.no_update

        if len(active) == 3:
            interp_grid = sweep_to_interp_grid(sweep_data, out_key)
            fs_cfg = frozen_slider_config(sweep_data)
            if fs_cfg and is_frozen_view(view):
                frozen_style = {"padding": "4px 16px 8px"}
                frozen_min = fs_cfg["min"]
                frozen_max = fs_cfg["max"]
                frozen_label = f"{METRIC_BY_KEY[fs_cfg['metric_key']].label}"

        fig = build_figure(
            len(active),
            sweep_data,
            out_key,
            view_type=view,
            thresholds=thresh,
            threshold_colors=thresh_colors or None,
        )
        return (
            fig,
            sweep_data,
            status,
            view,
            make_view_tab_bar(len(active), view),
            dirty,
            interp_grid,
            frozen_style,
            frozen_min,
            frozen_max,
            frozen_label,
        )

    except Exception as exc:
        return (
            plot_empty(f"Error: {exc}"),
            None,
            f"Error: {exc}",
            dash.no_update,
            dash.no_update,
            dirty,
            None,
            {"display": "none"},
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )
    finally:
        _sweep_progress = {"running": False}
        sweep_lock.release()


def _count_points(sweep_data: dict) -> int:
    xs = sweep_data.get("xs", [])
    ys = sweep_data.get("ys", [xs])
    zs = sweep_data.get("zs", [xs])
    return len(xs) * len(ys) * len(zs)


# ---------------------------------------------------------------------------
# Callback: re-plot when output metric changes (no re-sweep needed)
# ---------------------------------------------------------------------------


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("cfg-output-metric", "value"),
    Input("cfg-threshold-enable", "value"),
    *[Input(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[Input(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("num-thresholds-store", "data"),
    prevent_initial_call=True,
)
def replot_on_output_change(
    output_key,
    threshold_enable,
    t0,
    t1,
    t2,
    t3,
    t4,
    tc0,
    tc1,
    tc2,
    tc3,
    tc4,
    sweep_data,
    view_type,
    num_thresholds,
):
    if sweep_data is None:
        return dash.no_update
    num_metrics = len(sweep_data.get("metric_keys", []))
    n_t = int(num_thresholds or 3)
    all_t = [t0, t1, t2, t3, t4][:n_t]
    all_c = [tc0, tc1, tc2, tc3, tc4][:n_t]
    thresh_vals = [v for v in all_t if v is not None]
    thresh_colors = [all_c[i] for i, v in enumerate(all_t) if v is not None]
    thresh = thresh_vals if threshold_enable and "yes" in threshold_enable else None
    if view_type in ("isosurface", "scatter3d"):
        thresh = thresh_vals or None
    return build_figure(
        num_metrics,
        sweep_data,
        output_key or "overall_fidelity",
        view_type=view_type,
        thresholds=thresh,
        threshold_colors=thresh_colors or None,
    )


# ---------------------------------------------------------------------------
# Callbacks: update range labels and reconfigure sliders when dropdown changes
# ---------------------------------------------------------------------------

for _idx in range(MAX_METRICS):

    @app.callback(
        Output(f"metric-range-label-{_idx}", "children"),
        Input(f"metric-slider-{_idx}", "value"),
        State(f"metric-dropdown-{_idx}", "value"),
        prevent_initial_call=False,
    )
    def _range_label(slider_val, metric_key, _i=_idx):
        if not slider_val or not metric_key:
            return []
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return []
        lo = _slider_to_value(slider_val[0], m.log_scale)
        hi = _slider_to_value(slider_val[1], m.log_scale)

        def fmt(v: float) -> str:
            if abs(v) < 1e-3 or abs(v) >= 1e5:
                return f"{v:.2e}"
            if abs(v) < 10:
                return f"{v:.4f}"
            return f"{v:.1f}"

        unit = f" {m.unit}" if m.unit else ""
        return [
            html.Span(fmt(lo) + unit, style={"color": COLORS["accent"]}),
            html.Span(" → ", style={"color": COLORS["text_muted"]}),
            html.Span(fmt(hi) + unit, style={"color": COLORS["accent2"]}),
        ]

    @app.callback(
        Output(f"metric-slider-{_idx}", "min"),
        Output(f"metric-slider-{_idx}", "max"),
        Output(f"metric-slider-{_idx}", "step"),
        Output(f"metric-slider-{_idx}", "marks"),
        Output(f"metric-slider-{_idx}", "value"),
        Input(f"metric-dropdown-{_idx}", "value"),
        prevent_initial_call=True,
    )
    def _reconfigure_slider(metric_key, _i=_idx):
        if not metric_key:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return (
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
        marks = (
            _log_marks(m.slider_min, m.slider_max)
            if m.log_scale
            else _linear_marks(m.slider_min, m.slider_max)
        )
        if m.is_cold_path:
            step = 2 if m.key == "num_qubits" else 1
        else:
            step = (m.slider_max - m.slider_min) / 200
        return (
            m.slider_min,
            m.slider_max,
            step,
            marks,
            [m.slider_default_low, m.slider_default_high],
        )


# ---------------------------------------------------------------------------
# Callback: filter dropdown options so a metric can't be selected twice
# ---------------------------------------------------------------------------

_ALL_METRIC_OPTIONS = [{"label": m.label, "value": m.key} for m in SWEEPABLE_METRICS]


@app.callback(
    Output("metric-dropdown-0", "options"),
    Output("metric-dropdown-1", "options"),
    Output("metric-dropdown-2", "options"),
    Input("metric-dropdown-0", "value"),
    Input("metric-dropdown-1", "value"),
    Input("metric-dropdown-2", "value"),
    prevent_initial_call=True,
)
def _filter_dropdown_options(v0, v1, v2):
    values = [v0, v1, v2]
    results = []
    for i in range(3):
        taken = {values[j] for j in range(3) if j != i and values[j]}
        results.append([
            {**opt, "disabled": opt["value"] in taken}
            for opt in _ALL_METRIC_OPTIONS
        ])
    return results


# ---------------------------------------------------------------------------
# Callback: view tab click — switch plot type without re-sweep
# ---------------------------------------------------------------------------


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Output("view-type-store", "data", allow_duplicate=True),
    Output("view-tab-container", "children", allow_duplicate=True),
    Input({"type": "view-tab-btn", "index": ALL}, "n_clicks"),
    State("sweep-result-store", "data"),
    State("num-metrics-store", "data"),
    State("cfg-output-metric", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    prevent_initial_call=True,
)
def on_view_tab_click(
    n_clicks_list,
    sweep_data,
    num_metrics,
    output_key,
    threshold_enable,
    t0,
    t1,
    t2,
    t3,
    t4,
    tc0,
    tc1,
    tc2,
    tc3,
    tc4,
    num_thresholds,
):
    if not ctx.triggered_id or not any(n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update

    view_type = ctx.triggered_id["index"]

    if sweep_data is None:
        return dash.no_update, view_type, make_view_tab_bar(num_metrics or 1, view_type)

    actual_metrics = len(sweep_data.get("metric_keys", []))
    n_t = int(num_thresholds or 3)
    all_t = [t0, t1, t2, t3, t4][:n_t]
    all_c = [tc0, tc1, tc2, tc3, tc4][:n_t]
    thresh_vals = [v for v in all_t if v is not None]
    thresh_colors = [all_c[i] for i, v in enumerate(all_t) if v is not None]
    thresh = thresh_vals if threshold_enable and "yes" in threshold_enable else None
    if view_type in ("isosurface", "scatter3d"):
        thresh = thresh_vals or None
    fig = build_figure(
        actual_metrics,
        sweep_data,
        output_key or "overall_fidelity",
        view_type=view_type,
        thresholds=thresh,
        threshold_colors=thresh_colors or None,
    )
    return fig, view_type, make_view_tab_bar(actual_metrics, view_type)


# ---------------------------------------------------------------------------
# Callback: CSV export
# ---------------------------------------------------------------------------


@app.callback(
    Output("csv-download", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("sweep-result-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, sweep_data):
    if not n_clicks or sweep_data is None:
        return dash.no_update
    csv_str = sweep_to_csv(sweep_data)
    return dict(content=csv_str, filename="dse_sweep.csv", type="text/csv")


# ---------------------------------------------------------------------------
# Clientside callbacks: sync threshold color swatches
# ---------------------------------------------------------------------------

for _ci in range(5):
    app.clientside_callback(
        """function(color) {
            return {
                "width": "24px", "height": "24px",
                "borderRadius": "4px",
                "border": "1px solid #D4D4D4",
                "flexShrink": "0",
                "background": color || "#ccc"
            };
        }""",
        Output(f"cfg-threshold-swatch-{_ci}", "style"),
        Input(f"cfg-threshold-color-{_ci}", "value"),
        prevent_initial_call=False,
    )


# ---------------------------------------------------------------------------
# Callback: toggle Run button visibility based on hot-reload state
# ---------------------------------------------------------------------------

app.clientside_callback(
    """function(hotReload) {
        var on = hotReload && hotReload.indexOf("on") !== -1;
        return {"background": on ? "transparent" : "#5B8DEF",
                "color": on ? "transparent" : "#fff",
                "border": "none",
                "borderRadius": "6px",
                "padding": "6px 20px",
                "fontWeight": "600",
                "fontSize": "13px",
                "cursor": on ? "default" : "pointer",
                "display": on ? "none" : "inline-block"};
    }""",
    Output("run-btn", "style"),
    Input("hot-reload-toggle", "value"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Clientside callback: frozen slider drag → interpolate 2D slice, update plot
# ---------------------------------------------------------------------------
# This runs entirely in the browser. No server round-trip.
# The interp-grid-store holds the precomputed 3D grid; frozenSlice extracts
# a 2D plane at the slider value, then Plotly.react diffs the data in-place.

app.clientside_callback(
    """function(frozenVal, interpGrid, viewType) {
        if (!interpGrid || !interpGrid.values || interpGrid.ndim !== 3) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        if (viewType !== "frozen_heatmap" && viewType !== "frozen_contour") {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var qi = window.qusimInterp;
        if (!qi) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }

        var slice2d = qi.frozenSlice(
            interpGrid.values, interpGrid.xs, interpGrid.ys,
            interpGrid.zs, frozenVal
        );

        var plotDiv = document.getElementById("main-plot");
        if (plotDiv && plotDiv.data && plotDiv.data.length > 0) {
            var trace = plotDiv.data[0];
            trace.z = slice2d;
            plotDiv.layout.datarevision = (plotDiv.layout.datarevision || 0) + 1;
            Plotly.react(plotDiv, plotDiv.data, plotDiv.layout);
        }

        var v = frozenVal;
        var label = (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)
            ? v.toExponential(2) : (Math.abs(v) < 10 ? v.toFixed(4) : v.toFixed(1));

        return [window.dash_clientside.no_update, label];
    }""",
    Output("main-plot", "figure", allow_duplicate=True),
    Output("frozen-slider-value", "children"),
    Input("frozen-slider", "value"),
    State("interp-grid-store", "data"),
    State("view-type-store", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Callback: show/hide frozen slider when view tab changes
# ---------------------------------------------------------------------------

@app.callback(
    Output("frozen-slider-container", "style", allow_duplicate=True),
    Input("view-type-store", "data"),
    State("sweep-result-store", "data"),
    prevent_initial_call=True,
)
def toggle_frozen_slider_visibility(view_type, sweep_data):
    if sweep_data is None:
        return {"display": "none"}
    num_metrics = len(sweep_data.get("metric_keys", []))
    if num_metrics == 3 and is_frozen_view(view_type):
        return {"padding": "4px 16px 8px"}
    return {"display": "none"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("qusim DSE GUI starting at http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
