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

import sys
import os
import time
from typing import Any

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.constants import (
    SWEEPABLE_METRICS,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    OUTPUT_METRICS,
)
from gui.components import (
    COLORS,
    make_metric_selector,
    make_add_metric_button,
    make_fixed_config_panel,
    _log_marks,
    _linear_marks,
)
from gui.plotting import build_figure, plot_empty
from gui.dse_engine import DSEEngine

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
MAX_METRICS = 3


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
                    html.Span("qusim", style={"fontSize": "18px", "fontWeight": "700",
                                              "color": COLORS["accent"]}),
                    html.Span("DSE Explorer", style={"fontSize": "14px",
                                                     "color": COLORS["text_muted"]}),
                ],
            ),
            html.Div(id="status-bar", children="Ready — click Run to start a sweep",
                     style={"fontSize": "12px", "color": COLORS["text_muted"],
                            "flex": "1", "textAlign": "center"}),
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
                },
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
            html.Div("Sweep Axes", style={
                "fontSize": "10px", "fontWeight": "700",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
                "color": COLORS["text_muted"], "marginBottom": "10px",
            }),
            # Always render all 3 rows; show/hide via 'display'
            html.Div(id="metric-row-wrap-0", children=[make_metric_selector(0)]),
            html.Div(id="metric-row-wrap-1", children=[make_metric_selector(1)],
                     style={"display": "none"}),
            html.Div(id="metric-row-wrap-2", children=[make_metric_selector(2)],
                     style={"display": "none"}),

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
                            "display": "none",
                        },
                    ),
                ],
            ),
        ],
    )


def _center_panel() -> html.Div:
    return html.Div(
        style={"flex": "1", "minWidth": "0", "padding": "10px"},
        children=[
            dcc.Loading(
                type="circle",
                color=COLORS["accent"],
                children=dcc.Graph(
                    id="main-plot",
                    figure=plot_empty(),
                    style={"height": "calc(100vh - 80px)"},
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "toImageButtonOptions": {
                            "format": "png", "width": 1200, "height": 800, "scale": 2,
                        },
                    },
                ),
            ),
        ],
    )


def _right_panel() -> html.Div:
    return html.Div(
        className="config-scroll",
        style={
            "width": "270px",
            "minWidth": "250px",
            "background": COLORS["bg"],
            "borderLeft": f"1px solid {COLORS['border']}",
            "padding": "14px 12px",
        },
        children=[
            html.Div("Configuration", style={
                "fontSize": "10px", "fontWeight": "700",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
                "color": COLORS["text_muted"], "marginBottom": "10px",
            }),
            html.Div(id="fixed-config-container", children=[make_fixed_config_panel()]),
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
            style={"display": "flex", "height": "calc(100vh - 52px)", "overflow": "hidden"},
            children=[_left_sidebar(), _center_panel(), _right_panel()],
        ),
        # State stores
        dcc.Store(id="num-metrics-store", data=1, storage_type="memory"),
        dcc.Store(id="sweep-result-store", data=None, storage_type="memory"),
    ],
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

    def _show(): return {}
    def _hide(): return {"display": "none"}

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
# Callback: show/hide noise rows when swept metric dropdowns change
# ---------------------------------------------------------------------------
# All noise-row-{key} divs are always in the DOM; we toggle display only.

from gui.constants import SWEEPABLE_METRICS as _SM

@app.callback(
    [Output(f"noise-row-{m.key}", "style") for m in _SM],
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
    return [{"display": "none"} if m.key in swept else {} for m in _SM]


# ---------------------------------------------------------------------------
# Helper: convert result dict/object to plain JSON-safe dict
# ---------------------------------------------------------------------------

def _result_to_dict(r: Any) -> dict:
    if isinstance(r, dict):
        return {k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))}
    return {
        "overall_fidelity": float(r.overall_fidelity),
        "algorithmic_fidelity": float(r.algorithmic_fidelity),
        "routing_fidelity": float(r.routing_fidelity),
        "coherence_fidelity": float(r.coherence_fidelity),
        "total_circuit_time_ns": float(r.total_circuit_time_ns),
        "total_epr_pairs": float(getattr(r, "total_epr_pairs", 0)),
    }


def _slider_to_value(slider_pos: float, log_scale: bool) -> float:
    return 10.0 ** slider_pos if log_scale else slider_pos


# ---------------------------------------------------------------------------
# Callback: run sweep
# ---------------------------------------------------------------------------

_NOISE_SLIDER_STATES = [State(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS]


@app.callback(
    Output("main-plot", "figure"),
    Output("sweep-result-store", "data"),
    Output("status-bar", "children"),
    Input("run-btn", "n_clicks"),
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
    State("cfg-placement", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-output-metric", "value"),
    *_NOISE_SLIDER_STATES,
    prevent_initial_call=True,
)
def run_sweep(
    n_clicks,
    m0_key, m0_range,
    m1_key, m1_range,
    m2_key, m2_range,
    num_metrics,
    circuit_type, num_qubits, num_cores, topology, placement, seed,
    dynamic_decoupling,
    output_key,
    *noise_slider_vals,
):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update

    t_start = time.time()

    # Build fixed noise dict from right-panel sliders
    fixed_noise: dict = {}
    for i, m in enumerate(SWEEPABLE_METRICS):
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
    for i, (k, r) in enumerate(all_inputs[:int(num_metrics or 1)]):
        if k and r and k not in seen:
            seen.add(k)
            active.append((k, r))

    if not active:
        return plot_empty("Add at least one metric axis and click Run"), None, "No metrics configured"

    try:
        cached = _engine.run_cold(
            circuit_type=circuit_type or "qft",
            num_qubits=int(num_qubits or 16),
            num_cores=int(num_cores or 4),
            topology_type=topology or "ring",
            placement_policy=placement or "random",
            seed=int(seed or 42),
            noise=fixed_noise,
        )
        cold_elapsed = time.time() - t_start

        sweep_data: dict = {"metric_keys": [k for k, _ in active]}

        if len(active) == 1:
            k0, r0 = active[0]
            xs, results = _engine.sweep_1d(cached, k0, r0[0], r0[1], fixed_noise)
            sweep_data["xs"] = xs.tolist()
            sweep_data["grid"] = [_result_to_dict(r) for r in results]

        elif len(active) == 2:
            k0, r0 = active[0]
            k1, r1 = active[1]
            xs, ys, grid = _engine.sweep_2d(
                cached, k0, r0[0], r0[1], k1, r1[0], r1[1], fixed_noise,
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
            )
            sweep_data["xs"] = xs.tolist()
            sweep_data["ys"] = ys.tolist()
            sweep_data["zs"] = zs.tolist()
            sweep_data["grid"] = [
                [[_result_to_dict(r) for r in row] for row in plane]
                for plane in grid
            ]

        sweep_elapsed = time.time() - t_start - cold_elapsed
        total_elapsed = time.time() - t_start
        cache_indicator = "cached ✓" if cached.cold_time_s < 0.1 else f"mapped in {cold_elapsed:.1f}s"
        status = (
            f"Cold path: {cache_indicator}  |  "
            f"Sweep ({len(active)}D, {_count_points(sweep_data)} pts): {sweep_elapsed:.2f}s  |  "
            f"Total: {total_elapsed:.1f}s"
        )

        fig = build_figure(len(active), sweep_data, output_key or "overall_fidelity")
        return fig, sweep_data, status

    except Exception as exc:
        return plot_empty(f"Error: {exc}"), None, f"Error: {exc}"


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
    State("sweep-result-store", "data"),
    prevent_initial_call=True,
)
def replot_on_output_change(output_key, sweep_data):
    if sweep_data is None:
        return dash.no_update
    num_metrics = len(sweep_data.get("metric_keys", []))
    return build_figure(num_metrics, sweep_data, output_key or "overall_fidelity")


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
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        marks = (
            _log_marks(m.slider_min, m.slider_max)
            if m.log_scale
            else _linear_marks(m.slider_min, m.slider_max)
        )
        step = (m.slider_max - m.slider_min) / 200
        return m.slider_min, m.slider_max, step, marks, [m.slider_default_low, m.slider_default_high]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("qusim DSE GUI starting at http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
