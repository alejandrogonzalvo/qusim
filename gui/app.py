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
    _tooltip_cfg,
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
    VIEW_TAB_DEFAULT_ND,
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

    For N >= 4, only analysis views are valid (no dimension-specific plots).
    """
    if current_view is not None:
        valid = _SWEEP_VIEW_KEYS.get(num_active, set()) | _ANALYSIS_VIEW_KEYS
        if current_view in valid:
            return current_view
    return VIEW_TAB_DEFAULTS.get(num_active, VIEW_TAB_DEFAULT_ND)


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
MAX_METRICS = 11  # All sweepable metrics can be swept simultaneously

# ---------------------------------------------------------------------------
# Server-side sweep cache
#
# The browser store (``sweep-result-store``) used to carry the full grid
# (~12 MB for a 50k-point 3D sweep). Every downstream callback re-parsed that
# blob on the browser main thread. Instead we keep the full grid server-side
# and only send the browser a small token + axes metadata; callbacks that
# need the grid fetch it back via ``_get_sweep``.
# ---------------------------------------------------------------------------

from collections import OrderedDict

_SWEEP_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SWEEP_CACHE_MAX = 3
_sweep_token_counter = 0
_sweep_cache_lock = threading.Lock()


def _store_sweep(data: dict) -> str:
    """Stash a full sweep_data dict server-side, return a short token."""
    global _sweep_token_counter
    with _sweep_cache_lock:
        _sweep_token_counter += 1
        token = f"sweep-{_sweep_token_counter}"
        _SWEEP_CACHE[token] = data
        while len(_SWEEP_CACHE) > _SWEEP_CACHE_MAX:
            _SWEEP_CACHE.popitem(last=False)
    return token


def _get_sweep(browser_store: dict | None) -> dict | None:
    """Look up the full sweep_data for a browser-side slim record."""
    if not browser_store:
        return None
    token = browser_store.get("token") if isinstance(browser_store, dict) else None
    if not token:
        return None
    with _sweep_cache_lock:
        return _SWEEP_CACHE.get(token)


def _slim_sweep_for_browser(data: dict) -> dict:
    """Strip the grid, keep axes + metric_keys + token for browser state.

    The axes are small (few hundred floats at most) and several callbacks
    read them to render UI knobs (frozen slider, shape hints) so we keep
    them inline.
    """
    slim: dict = {"token": _store_sweep(data), "metric_keys": data.get("metric_keys", [])}
    for k in ("xs", "ys", "zs", "axes", "shape"):
        if k in data:
            slim[k] = data[k]
    return slim

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
        "cold_completed": p.cold_completed,
        "cold_total": p.cold_total,
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
            # Always render all MAX_METRICS rows; show/hide via 'display'
            *[
                html.Div(
                    id=f"metric-row-wrap-{i}",
                    children=[make_metric_selector(i)],
                    style={} if i < 3 else {"display": "none"},
                )
                for i in range(MAX_METRICS)
            ],
            # Add / remove buttons
            html.Div(
                style={"display": "flex", "gap": "8px", "marginTop": "4px"},
                children=[
                    html.Button(
                        "+ Add axis",
                        id="add-metric-btn",
                        className="axis-btn axis-btn--add",
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
                    html.Button(
                        "− Remove",
                        id="remove-metric-btn",
                        className="axis-btn axis-btn--remove",
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
                ],
            ),
            # Estimated point count display
            html.Div(
                id="estimated-points",
                style={
                    "fontSize": "11px",
                    "color": COLORS["text_muted"],
                    "textAlign": "center",
                    "marginTop": "6px",
                },
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
                id="error-banner",
                style={"display": "none"},
                children=[],
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
    *[Output(f"metric-row-wrap-{i}", "style") for i in range(MAX_METRICS)],
    *[Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
    Output("add-metric-btn", "style"),
    Output("remove-metric-btn", "style"),
    Output("num-metrics-store", "data"),
    Input("add-metric-btn", "n_clicks"),
    Input("remove-metric-btn", "n_clicks"),
    State("num-metrics-store", "data"),
    *[State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    prevent_initial_call=True,
)
def toggle_metric_rows(add_clicks, remove_clicks, num_metrics, *dropdown_vals):
    triggered = ctx.triggered_id
    old_num = num_metrics
    if triggered == "add-metric-btn":
        num_metrics = min(MAX_METRICS, num_metrics + 1)
    elif triggered == "remove-metric-btn":
        num_metrics = max(1, num_metrics - 1)

    row_styles = [
        {} if i < num_metrics else {"display": "none"}
        for i in range(MAX_METRICS)
    ]

    # Build list of taken metric keys from currently visible rows
    taken = {dropdown_vals[i] for i in range(old_num) if dropdown_vals[i]}
    all_keys = [m.key for m in SWEEPABLE_METRICS]

    # For newly revealed rows, assign first available untaken metric
    new_values = list(dropdown_vals)
    for i in range(old_num, num_metrics):
        current = new_values[i]
        if current in taken:
            available = [k for k in all_keys if k not in taken]
            new_values[i] = available[0] if available else current
        taken.add(new_values[i])

    _btn_base = {
        "flex": "1",
        "background": "transparent",
        "borderRadius": "6px",
        "padding": "6px",
        "cursor": "pointer",
        "fontSize": "12px",
    }

    add_style = (
        {**_btn_base, "border": f"1px solid {COLORS['border']}", "color": COLORS["text_muted"]}
        if num_metrics < MAX_METRICS
        else {"display": "none"}
    )

    remove_style = (
        {**_btn_base, "border": f"1px dashed {COLORS['border']}", "color": COLORS["text_muted"]}
        if num_metrics > 1
        else {"display": "none"}
    )

    return *row_styles, *new_values, add_style, remove_style, num_metrics


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
    *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    Input("num-metrics-store", "data"),
    prevent_initial_call=False,
)
def toggle_noise_rows(*args):
    dropdown_vals = args[:MAX_METRICS]
    num_metrics = args[MAX_METRICS] or 1
    swept = set()
    for i, val in enumerate(dropdown_vals):
        if i < num_metrics and val:
            swept.add(val)
    noise_styles = [
        {"display": "none"} if m.is_cold_path or m.key in swept else {} for m in _SM
    ]
    qubits_style = {"display": "none"} if "num_qubits" in swept else {}
    cores_style = {"display": "none"} if "num_cores" in swept else {}
    return noise_styles + [qubits_style, cores_style]


# ---------------------------------------------------------------------------
# Callback: live budget warning
#
# Re-runs `_compute_axis_counts` whenever the user changes budgets or axes
# so they see the budget error BEFORE clicking Run.
# ---------------------------------------------------------------------------

_BUDGET_WARNING_STYLE = {
    "background": "#FDECEC",
    "border": "1px solid #E57373",
    "color": "#8B1A1A",
    "borderRadius": "6px",
    "padding": "8px 10px",
    "marginTop": "-4px",
    "marginBottom": "10px",
    "fontSize": "12px",
    "lineHeight": "1.4",
}


@app.callback(
    Output("sweep-budget-warning", "children"),
    Output("sweep-budget-warning", "style"),
    Input("cfg-max-cold", "value"),
    Input("cfg-max-hot", "value"),
    Input("num-metrics-store", "data"),
    *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    *[Input(f"metric-slider-{i}", "value") for i in range(MAX_METRICS)],
    prevent_initial_call=False,
)
def update_budget_warning(max_cold, max_hot, num_metrics, *dynamic_args):
    dropdown_vals = dynamic_args[:MAX_METRICS]
    slider_vals = dynamic_args[MAX_METRICS:]
    n = int(num_metrics or 1)

    active: list[tuple[str, float, float]] = []
    seen: set = set()
    for i in range(n):
        k = dropdown_vals[i]
        r = slider_vals[i]
        if k and r and k not in seen:
            seen.add(k)
            active.append((k, float(r[0]), float(r[1])))

    if not active:
        return [], {"display": "none"}

    has_cold = any(k in _engine.COLD_PATH_KEYS for k, _, _ in active)
    try:
        _engine._compute_axis_counts(
            active, has_cold,
            max_cold=int(max_cold) if max_cold else None,
            max_hot=int(max_hot) if max_hot else None,
        )
    except RuntimeError as exc:
        return (
            [html.Span("⚠ ", style={"fontWeight": "700"}), str(exc)],
            _BUDGET_WARNING_STYLE,
        )
    return [], {"display": "none"}


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

_NUM_SWEEP_OUTPUTS = 13
_NO_UPDATE_SWEEP = (dash.no_update,) * _NUM_SWEEP_OUTPUTS


_ERROR_BANNER_HIDDEN_STYLE = {"display": "none"}


def _error_banner_visible_style() -> dict:
    return {
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "10px",
        "background": "#FDECEC",
        "border": "1px solid #E57373",
        "color": "#8B1A1A",
        "borderRadius": "6px",
        "padding": "10px 12px",
        "margin": "6px 0",
        "fontSize": "13px",
        "lineHeight": "1.4",
        "fontFamily": "Inter, system-ui, sans-serif",
    }


def _build_error_banner_children(title: str, message: str) -> list:
    return [
        html.Span("⚠", style={"fontSize": "16px", "lineHeight": "1.2"}),
        html.Div(
            style={"flex": "1"},
            children=[
                html.Div(title, style={"fontWeight": "600", "marginBottom": "2px"}),
                html.Div(message, style={"whiteSpace": "pre-wrap"}),
            ],
        ),
        html.Button(
            "×",
            id="error-banner-dismiss",
            n_clicks=0,
            style={
                "background": "transparent",
                "border": "none",
                "color": "#8B1A1A",
                "fontSize": "18px",
                "lineHeight": "1",
                "cursor": "pointer",
                "padding": "0 4px",
            },
        ),
    ]

# Clientside: any simulation input change → increment dirty counter
_SIM_INPUTS = [
    *[inp for i in range(MAX_METRICS) for inp in [
        Input(f"metric-dropdown-{i}", "value"),
        Input(f"metric-slider-{i}", "value"),
    ]],
    Input("num-metrics-store", "data"),
    Input("cfg-circuit-type", "value"),
    Input("cfg-num-qubits", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-topology", "value"),
    Input("cfg-intracore-topology", "value"),
    Input("cfg-placement", "value"),
    Input("cfg-seed", "value"),
    Input("cfg-dynamic-decoupling", "value"),
    Input("cfg-max-cold", "value"),
    Input("cfg-max-hot", "value"),
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


_METRIC_DROPDOWN_STATES = [State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)]
_METRIC_SLIDER_STATES = [State(f"metric-slider-{i}", "value") for i in range(MAX_METRICS)]


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
    Output("error-banner", "children"),
    Output("error-banner", "style"),
    Input("sweep-trigger", "data"),
    Input("run-btn", "n_clicks"),
    State("sweep-dirty", "data"),
    State("hot-reload-toggle", "value"),
    *_METRIC_DROPDOWN_STATES,
    *_METRIC_SLIDER_STATES,
    State("num-metrics-store", "data"),
    State("cfg-circuit-type", "value"),
    State("cfg-num-qubits", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-topology", "value"),
    State("cfg-intracore-topology", "value"),
    State("cfg-placement", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-max-cold", "value"),
    State("cfg-max-hot", "value"),
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
    *all_args,
):
    is_run_btn = ctx.triggered_id == "run-btn"
    hot_reload_on = hot_reload and "on" in hot_reload
    if not is_run_btn and not hot_reload_on:
        return _NO_UPDATE_SWEEP

    # Unpack dynamic args
    idx = 0
    dropdown_vals = all_args[idx:idx + MAX_METRICS]; idx += MAX_METRICS
    slider_vals = all_args[idx:idx + MAX_METRICS]; idx += MAX_METRICS
    num_metrics = all_args[idx]; idx += 1
    circuit_type = all_args[idx]; idx += 1
    num_qubits = all_args[idx]; idx += 1
    num_cores = all_args[idx]; idx += 1
    topology = all_args[idx]; idx += 1
    intracore_topology = all_args[idx]; idx += 1
    placement = all_args[idx]; idx += 1
    seed = all_args[idx]; idx += 1
    dynamic_decoupling = all_args[idx]; idx += 1
    max_cold = all_args[idx]; idx += 1
    max_hot = all_args[idx]; idx += 1
    output_key = all_args[idx]; idx += 1
    threshold_enable = all_args[idx]; idx += 1
    t_vals = all_args[idx:idx + 5]; idx += 5
    tc_vals = all_args[idx:idx + 5]; idx += 5
    num_thresholds = all_args[idx]; idx += 1
    current_view = all_args[idx]; idx += 1
    noise_slider_vals = all_args[idx:]

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

        # Active metrics — collect from all N dropdowns
        active = []
        seen: set = set()
        for i in range(int(num_metrics or 1)):
            k = dropdown_vals[i]
            r = slider_vals[i]
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
                [],
                _ERROR_BANNER_HIDDEN_STYLE,
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

        # Use sweep_nd for N >= 4, legacy methods for 1-3D
        if len(active) >= 4:
            sweep_axes = [(k, r[0], r[1]) for k, r in active]
            result = _engine.sweep_nd(
                cached=cached,
                sweep_axes=sweep_axes,
                fixed_noise=fixed_noise,
                cold_config=cold_config,
                progress_callback=_update_progress,
                parallel=True,
                max_cold=int(max_cold) if max_cold else None,
                max_hot=int(max_hot) if max_hot else None,
            )
            sweep_data = result.to_sweep_data()
            # Convert result dicts to JSON-safe format
            if "grid" in sweep_data:
                if isinstance(sweep_data["grid"], list):
                    sweep_data["grid"] = [
                        _result_to_dict(r) if isinstance(r, dict) else r
                        for r in sweep_data["grid"]
                    ]
        else:
            sweep_data = {"metric_keys": [k for k, _ in active]}
            if len(active) == 1:
                k0, r0 = active[0]
                xs, results = _engine.sweep_1d(
                    cached, k0, r0[0], r0[1], fixed_noise, cold_config=cold_config,
                    progress_callback=_update_progress, parallel=True,
                    max_cold=int(max_cold) if max_cold else None,
                    max_hot=int(max_hot) if max_hot else None,
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
                    progress_callback=_update_progress, parallel=True,
                    max_cold=int(max_cold) if max_cold else None,
                    max_hot=int(max_hot) if max_hot else None,
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
                    progress_callback=_update_progress, parallel=True,
                    max_cold=int(max_cold) if max_cold else None,
                    max_hot=int(max_hot) if max_hot else None,
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
        all_t = list(t_vals[:n_t])
        all_c = list(tc_vals[:n_t])
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
        # Send only a small token + axes metadata to the browser; the full
        # grid lives in ``_SWEEP_CACHE`` and is fetched by downstream callbacks.
        slim_store = _slim_sweep_for_browser(sweep_data)
        return (
            fig,
            slim_store,
            status,
            view,
            make_view_tab_bar(len(active), view),
            dirty,
            interp_grid,
            frozen_style,
            frozen_min,
            frozen_max,
            frozen_label,
            [],
            _ERROR_BANNER_HIDDEN_STYLE,
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Distinguish expected, user-actionable failures (RuntimeError raised
        # by the scheduler — e.g. insufficient RAM) from unexpected crashes.
        if isinstance(exc, RuntimeError):
            banner_title = "Sweep cancelled"
        else:
            banner_title = "Sweep failed"
        banner_children = _build_error_banner_children(banner_title, str(exc))
        return (
            plot_empty(banner_title),
            None,
            banner_title,
            dash.no_update,
            dash.no_update,
            dirty,
            None,
            {"display": "none"},
            dash.no_update,
            dash.no_update,
            dash.no_update,
            banner_children,
            _error_banner_visible_style(),
        )
    finally:
        _sweep_progress = {"running": False}
        sweep_lock.release()


def _count_points(sweep_data: dict) -> int:
    if "shape" in sweep_data:
        result = 1
        for s in sweep_data["shape"]:
            result *= s
        return result
    if "axes" in sweep_data:
        result = 1
        for ax in sweep_data["axes"]:
            result *= len(ax)
        return result
    xs = sweep_data.get("xs", [])
    ys = sweep_data.get("ys", [xs])
    zs = sweep_data.get("zs", [xs])
    return len(xs) * len(ys) * len(zs)


# ---------------------------------------------------------------------------
# Callback: dismiss error banner
# ---------------------------------------------------------------------------

app.clientside_callback(
    """function(n_clicks) {
        if (!n_clicks) return window.dash_clientside.no_update;
        return [[], {display: "none"}];
    }""",
    [
        Output("error-banner", "children", allow_duplicate=True),
        Output("error-banner", "style", allow_duplicate=True),
    ],
    Input("error-banner-dismiss", "n_clicks"),
    prevent_initial_call=True,
)


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
    sweep_store,
    view_type,
    num_thresholds,
):
    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update
    num_metrics = len(full.get("metric_keys", []))
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
        full,
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
        Output(f"metric-slider-{_idx}", "tooltip"),
        Input(f"metric-dropdown-{_idx}", "value"),
        prevent_initial_call=True,
    )
    def _reconfigure_slider(metric_key, _i=_idx):
        no = dash.no_update
        if not metric_key:
            return (no, no, no, no, no, no)
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return (no, no, no, no, no, no)
        marks = (
            _log_marks(m.slider_min, m.slider_max, m.unit)
            if m.log_scale
            else _linear_marks(m.slider_min, m.slider_max, unit=m.unit)
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
            _tooltip_cfg(m.log_scale, m.unit, always_visible=True),
        )

# Register default metrics for rows 3+ (rows 0-2 already have defaults in make_metric_selector)
# These will use the DEFAULT_SWEEP_AXES list + cycle through remaining metrics


# ---------------------------------------------------------------------------
# Callback: filter dropdown options so a metric can't be selected twice
# ---------------------------------------------------------------------------

_ALL_METRIC_OPTIONS = [{"label": m.label, "value": m.key} for m in SWEEPABLE_METRICS]


@app.callback(
    *[Output(f"metric-dropdown-{i}", "options") for i in range(MAX_METRICS)],
    *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    Input("num-metrics-store", "data"),
    prevent_initial_call=True,
)
def _filter_dropdown_options(*args):
    values = args[:MAX_METRICS]
    num_metrics = args[MAX_METRICS] or 1
    # Only consider dropdowns from visible (active) rows as "taken"
    results = []
    for i in range(MAX_METRICS):
        taken = {
            values[j]
            for j in range(num_metrics)
            if j != i and values[j]
        }
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
    sweep_store,
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

    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update, view_type, make_view_tab_bar(num_metrics or 1, view_type)

    actual_metrics = len(full.get("metric_keys", []))
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
        full,
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
def export_csv(n_clicks, sweep_store):
    if not n_clicks:
        return dash.no_update
    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update
    csv_str = sweep_to_csv(full)
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
