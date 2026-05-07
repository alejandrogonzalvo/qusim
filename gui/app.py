"""
quadris DSE GUI — main Dash application.

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
import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.components import (
    COLORS,
    FEEDBACK_COLORS,
    build_topology_elements,
    make_view_tab_bar,
)
from gui.constants import (
    ANALYSIS_TABS,
    CAT_METRIC_BY_KEY,
    MAX_SWEEP_AXES,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    SWEEPABLE_METRICS,
    VIEW_TAB_DEFAULTS,
    VIEW_TAB_DEFAULT_ND,
    VIEW_TABS,
    view_modes_for_dim,
)
from gui.dse_engine import DSEEngine, SweepProgress
from gui.examples import example_path as _example_path
from gui.interpolation import (
    frozen_slider_config,
    is_frozen_view,
    permute_sweep_for_frozen,
    sweep_to_interp_grid,
)
from gui.fom import DEFAULT_FOM, PRESETS, FomConfig, compute_for_sweep
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
    title="Quadris · DSE Explorer",
    update_title=None,
    suppress_callback_exceptions=True,
)
server = app.server

from gui.server_state import (
    _engine,
    _get_sweep,
    _next_session_hw,
    _per_cell_cache_get,
    _per_cell_cache_key,
    _per_cell_cache_put,
    _progress_label,
    _set_progress,
    _slim_sweep_for_browser,
    _sweep_progress_tls,
    _update_progress,
    register_routes as _register_state_routes,
    sweep_lock,
)

_register_state_routes(server)

from gui.utils import (
    _axis_dropdown_options,
    _count_points,
    _facet_options,
    _facet_per_qubit_data,
    _fmt_fid,
    _frozen_values_from_sliders,
    _human_axis_value,
    _parse_intermediates,
    _resolve_thresholds,
    _slider_marks,
    _slider_to_value,
    _value_to_slider,
)

MAX_METRICS = MAX_SWEEP_AXES  # Alias for the centralised cap


def _compute_per_qubit_for_cell(
    sweep_data: dict, cell_idx: tuple[int, ...],
    facet_idx: int | None = None,
) -> dict | None:
    """Build per-qubit fidelity grids + placements for a single sweep cell.

    Reads the cold-config / fixed-noise snapshot from the sweep result and
    re-runs the engine for the requested cell.  ``cold`` axes hit the
    engine's cache after the first hit; ``hot`` axes only re-evaluate the
    Rust hot path (~ms).
    """
    import math
    meta = _facet_per_qubit_data(sweep_data, facet_idx)
    if not meta or "cold_config" not in meta or "fixed_noise" not in meta:
        return None
    cold_config = meta["cold_config"]
    fixed_noise = meta["fixed_noise"]
    axis_keys = meta["axis_keys"]
    axis_values = meta["axis_values"]
    shape = meta.get("shape", [len(v) for v in axis_values])

    # Fast path: the sweep retained per-cell per-qubit grids when run with
    # ``keep_per_qubit_grids=True``.  Look them up directly so scrubbing
    # the topology slider never re-enters run_cold/run_hot.
    cells = meta.get("cells")
    if cells:
        # Clamp the index using the same logic as the slow path so a cell
        # past the axis tail still resolves to the last cell.
        safe_lookup = []
        for d, sz in enumerate(shape):
            i = int(cell_idx[d]) if d < len(cell_idx) else 0
            safe_lookup.append(max(0, min(i, max(0, sz - 1))))
        cached_cell = cells.get(tuple(safe_lookup))
        if cached_cell is not None:
            out = dict(cached_cell)
            out.setdefault(
                "num_logical_qubits",
                int(out.get("num_physical", cold_config.get("num_qubits", 0))),
            )
            out["cell_idx"] = tuple(safe_lookup)
            return out

    # Clamp cell_idx to valid range per axis.
    safe_idx = []
    for d, sz in enumerate(shape):
        i = int(cell_idx[d]) if d < len(cell_idx) else 0
        safe_idx.append(max(0, min(i, max(0, sz - 1))))

    # Apply swept overrides to cold cfg / hot noise.
    cfg = dict(cold_config)
    hot_noise = dict(fixed_noise)
    for d, key in enumerate(axis_keys):
        v = axis_values[d][safe_idx[d]]
        if key in DSEEngine.COLD_PATH_KEYS:
            cfg[key] = int(v) if key in DSEEngine.INTEGER_KEYS else v
        else:
            hot_noise[key] = v
    from gui.dse_engine import _resolve_architecture
    feasibility = _resolve_architecture(cfg)
    if not feasibility["feasible"]:
        return {
            "infeasible": True,
            "infeasible_reason": feasibility["reason"],
            "cell_idx": tuple(safe_idx),
        }
    # run_cold rejects bookkeeping keys the resolver writes; pass only
    # what its signature accepts.
    accepted = {
        "circuit_type", "num_logical_qubits", "num_cores", "qubits_per_core",
        "topology_type", "intracore_topology", "placement_policy", "seed",
        "routing_algorithm", "communication_qubits", "buffer_qubits",
        "pin_axis", "custom_qasm",
    }
    run_kwargs = {k: cfg[k] for k in accepted if k in cfg}
    cached = _engine.run_cold(**run_kwargs, noise=hot_noise)
    full = _engine.run_hot(cached, hot_noise)
    return {
        "algorithmic_fidelity_grid": full.get("algorithmic_fidelity_grid"),
        "routing_fidelity_grid": full.get("routing_fidelity_grid"),
        "coherence_fidelity_grid": full.get("coherence_fidelity_grid"),
        "placements": cached.placements,
        "num_physical": int(cfg["num_qubits"]),
        "num_cores": int(cfg["num_cores"]),
        "num_logical_qubits": int(cfg.get("num_logical_qubits", cfg["num_qubits"])),
        "communication_qubits": int(cfg.get("communication_qubits", 1)),
        "buffer_qubits": int(cfg.get("buffer_qubits", 1)),
        "topology_type": cfg.get("topology_type", "ring"),
        "intracore_topology": cfg.get("intracore_topology", "all_to_all"),
        "cell_idx": tuple(safe_idx),
    }


_BASE_FID_KEYS = (
    "algorithmic_fidelity",
    "routing_fidelity",
    "coherence_fidelity",
)


def _per_logical_fidelity(
    cell_data: dict, metric_key: str = "routing_fidelity",
) -> np.ndarray:
    """Geomean fidelity per logical qubit across all DAG layers.

    Indexed by logical qubit so the result is independent of which physical
    layout the routing algorithm chose (HQA's placements live in the
    num_qubits coupling-map space, TeleSABRE's in the slack-expanded device
    space — both have the same logical qubit count).  The topology view
    projects this onto the per-core "data qubit" slots in id order.

    ``metric_key="overall_fidelity"`` returns the elementwise product of
    the algorithmic / routing / coherence grids before the geomean,
    matching the scalar overall fidelity definition.
    """
    n_log_default = max(1, int(cell_data.get("num_logical_qubits", 1)))
    if metric_key == "overall_fidelity":
        grids = [cell_data.get(f"{k}_grid") for k in _BASE_FID_KEYS]
        grids = [g for g in grids if g is not None]
        if not grids:
            return np.ones(n_log_default)
        arr = np.ones_like(np.asarray(grids[0], dtype=np.float64))
        for g in grids:
            arr = arr * np.asarray(g, dtype=np.float64)
    else:
        grid = cell_data.get(f"{metric_key}_grid")
        if grid is None:
            return np.ones(n_log_default)
        arr = np.asarray(grid, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.ones(n_log_default)
    EPS = 1e-12
    arr = np.clip(arr, EPS, 1.0)
    return np.exp(np.log(arr).mean(axis=0))


# Global CSS is in gui/assets/style.css — Dash auto-loads it.

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

from gui.layout import build_layout

app.layout = build_layout()


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
        // Load-cascade grace period: for ~1.5s after a session load,
        // ignore any dirty bumps that come from the load's downstream
        // callbacks (sliders being clobbered, dropdowns settling,
        // architecture summary recomputing).  Without this, a load
        // immediately fires a phantom recompile that overwrites the
        // pre-baked sweep result with one computed against half-
        // settled cfg state.
        var loadAt = window._loadCompleteAt || 0;
        if (loadAt > 0 && (Date.now() - loadAt) < 1500) {
            window._sweepDirty = window._lastProcessed;
            return window.dash_clientside.no_update;
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
# Pattern: Stores are the single source of truth for UI state. Mutators
# (button clicks, session loads) write Stores; derived UI (visibility,
# styles, computed labels) lives in dedicated Input-driven callbacks
# that listen to those Stores. This guarantees every mutation path —
# manual buttons OR session load — produces the same on-screen result,
# and lets new derived elements (an "axis count" badge, an empty-state
# placeholder, etc.) plug in by adding a new Input(<store>) callback
# without touching the mutators.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: convert result dict/object to plain JSON-safe dict
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Callback: run sweep
# ---------------------------------------------------------------------------

_NOISE_SLIDER_STATES = [State(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS]

_NUM_SWEEP_OUTPUTS = 15
_NO_UPDATE_SWEEP = (dash.no_update,) * _NUM_SWEEP_OUTPUTS


_ERROR_BANNER_HIDDEN_STYLE = {"display": "none"}


def _error_banner_visible_style() -> dict:
    palette = FEEDBACK_COLORS["error"]
    return {
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "10px",
        "background": palette["bg"],
        "border": f"1px solid {palette['border']}",
        "color": palette["text"],
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
                "color": FEEDBACK_COLORS["error"]["text"],
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
    *[Input(f"metric-checklist-{i}", "value") for i in range(MAX_METRICS)],
    Input("cfg-circuit-type", "value"),
    Input("cfg-topology", "value"),
    Input("cfg-intracore-topology", "value"),
    Input("cfg-placement", "value"),
    Input("cfg-routing-algorithm", "value"),
    Input("cfg-qubits-per-core", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-communication-qubits", "value"),
    Input("cfg-buffer-qubits", "value"),
    Input("cfg-num-logical-qubits", "value"),
    Input("cfg-pin-axis", "data"),
    Input("cfg-seed", "value"),
    Input("cfg-dynamic-decoupling", "value"),
    Input("cfg-max-cold", "value"),
    Input("cfg-max-hot", "value"),
    Input("cfg-max-workers", "value"),
    *[Input(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS],
]

app.clientside_callback(
    """function() {
        // Don't bump dirty during the load grace period — the cascading
        // callbacks (architecture summary, sweep-axis dropdowns settling)
        // would otherwise trigger a phantom recompile right after a load.
        var loadAt = window._loadCompleteAt || 0;
        if (loadAt > 0 && (Date.now() - loadAt) < 1500) {
            return window.dash_clientside.no_update;
        }
        window._sweepDirty = (window._sweepDirty || 0) + 1;
        return window._sweepDirty;
    }""",
    Output("sweep-dirty", "data"),
    *_SIM_INPUTS,
    prevent_initial_call=True,
)


# Generate (or reuse) a per-tab session id so the server can route sweep
# progress back to the user who started it. Persists across reloads via
# sessionStorage (same tab) and is exposed on window so progress.js can
# read it for /api/progress polling.
app.clientside_callback(
    """function(_n, existing) {
        var sid = existing;
        if (!sid) {
            try { sid = sessionStorage.getItem('quadris_sid') || ''; } catch (e) { sid = ''; }
        }
        if (!sid) {
            if (window.crypto && window.crypto.randomUUID) {
                sid = window.crypto.randomUUID();
            } else {
                sid = 'sid-' + Math.random().toString(36).slice(2) + '-' + Date.now().toString(36);
            }
        }
        try { sessionStorage.setItem('quadris_sid', sid); } catch (e) {}
        window._userSid = sid;
        return sid;
    }""",
    Output("user-sid", "data"),
    Input("sid-init", "n_intervals"),
    State("user-sid", "data"),
)


_METRIC_DROPDOWN_STATES = [State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)]
_METRIC_SLIDER_STATES = [State(f"metric-slider-{i}", "value") for i in range(MAX_METRICS)]
_METRIC_CHECKLIST_STATES = [State(f"metric-checklist-{i}", "value") for i in range(MAX_METRICS)]


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
    Output("frozen-slider", "value", allow_duplicate=True),
    Output("frozen-axis-dropdown", "options"),
    Output("frozen-axis-dropdown", "value"),
    Output("error-banner", "children"),
    Output("error-banner", "style"),
    Input("sweep-trigger", "data"),
    Input("run-btn", "n_clicks"),
    State("sweep-dirty", "data"),
    State("hot-reload-toggle", "value"),
    State("frozen-axis-store", "data"),
    *_METRIC_DROPDOWN_STATES,
    *_METRIC_SLIDER_STATES,
    State("num-metrics-store", "data"),
    *_METRIC_CHECKLIST_STATES,
    State("cfg-circuit-type", "value"),
    State("cfg-qubits-per-core", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-communication-qubits", "value"),
    State("cfg-buffer-qubits", "value"),
    State("cfg-num-logical-qubits", "value"),
    State("cfg-pin-axis", "data"),
    State("cfg-topology", "value"),
    State("cfg-intracore-topology", "value"),
    State("cfg-placement", "value"),
    State("cfg-routing-algorithm", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-max-cold", "value"),
    State("cfg-max-hot", "value"),
    State("cfg-max-workers", "value"),
    State("cfg-output-metric", "value"),
    State("cfg-view-mode", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    State("view-type-store", "data"),
    State("pareto-x-axis-dropdown", "value"),
    State("pareto-y-axis-dropdown", "value"),
    State("fom-config-store", "data"),
    State("user-sid", "data"),
    State("custom-qasm-store", "data"),
    *_NOISE_SLIDER_STATES,
    prevent_initial_call=True,
)
def run_sweep(
    trigger,
    n_clicks,
    dirty,
    hot_reload,
    frozen_axis_idx,
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
    checklist_vals = all_args[idx:idx + MAX_METRICS]; idx += MAX_METRICS
    circuit_type = all_args[idx]; idx += 1
    qubits_per_core = all_args[idx]; idx += 1
    num_cores = all_args[idx]; idx += 1
    communication_qubits = all_args[idx]; idx += 1
    buffer_qubits = all_args[idx]; idx += 1
    num_logical_qubits = all_args[idx]; idx += 1
    pin_axis = all_args[idx]; idx += 1
    topology = all_args[idx]; idx += 1
    intracore_topology = all_args[idx]; idx += 1
    placement = all_args[idx]; idx += 1
    routing_algorithm = all_args[idx]; idx += 1
    seed = all_args[idx]; idx += 1
    dynamic_decoupling = all_args[idx]; idx += 1
    max_cold = all_args[idx]; idx += 1
    max_hot = all_args[idx]; idx += 1
    max_workers = all_args[idx]; idx += 1
    output_key = all_args[idx]; idx += 1
    view_mode = all_args[idx]; idx += 1
    threshold_enable = all_args[idx]; idx += 1
    t_vals = all_args[idx:idx + 5]; idx += 5
    tc_vals = all_args[idx:idx + 5]; idx += 5
    num_thresholds = all_args[idx]; idx += 1
    current_view = all_args[idx]; idx += 1
    pareto_x = all_args[idx]; idx += 1
    pareto_y = all_args[idx]; idx += 1
    fom_config = all_args[idx]; idx += 1
    user_sid = all_args[idx]; idx += 1
    custom_qasm_data = all_args[idx] or {}; idx += 1
    noise_slider_vals = all_args[idx:]
    custom_qasm_str = custom_qasm_data.get("qasm")
    custom_qasm_nq = custom_qasm_data.get("num_qubits")
    if custom_qasm_str:
        circuit_type = "custom"
        if custom_qasm_nq:
            num_logical_qubits = int(custom_qasm_nq)

    sweep_lock.acquire()

    try:
        # Route progress for this sweep to the user who started it.
        # Empty sid (e.g. clientside callback hasn't fired yet) falls back to a
        # neutral key so the sweep still runs but no overlay is shown anywhere.
        sid = user_sid or "_no_sid"
        _sweep_progress_tls.sid = sid
        _set_progress(sid, {"running": True, "completed": 0, "total": 0, "percentage": 0, "current_params": {}, "phase": "compiling"})
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

        # Sweep axes from left sidebar dropdowns.
        active_numeric = []
        active_categorical = []
        seen: set = set()
        # Single source of truth for which metrics are sweepable in the
        # current GUI state (pin axis, custom QASM, …). The dropdown
        # filter and auto-assign already grey/skip these; here we
        # additionally refuse to start a sweep when one of them is
        # *selected and active* — surfacing the rule's reason instead of
        # producing a degenerate sweep where the engine silently
        # overrides the swept value.
        from gui.callbacks.sidebar import (
            AvailabilityContext as _AvailCtx,
            axis_availability as _axis_availability,
        )
        _avail_ctx = _AvailCtx(
            pin_axis=pin_axis or "cores",
            custom_qasm_active=bool(custom_qasm_str),
        )
        _disabled_axis_reasons = _axis_availability(_avail_ctx)
        _violations: list[tuple[int, str, str]] = []  # (axis_idx, key, reason)
        for i in range(int(num_metrics or 1)):
            k = dropdown_vals[i]
            if not k or k in seen:
                continue
            if k in _disabled_axis_reasons:
                _violations.append((i + 1, k, _disabled_axis_reasons[k]))
                continue
            # When a custom QASM circuit is loaded, circuit-shape axes are
            # meaningless — silently drop them so the sweep can still run.
            if custom_qasm_str and k in _CUSTOM_QASM_DISABLED_KEYS:
                continue
            seen.add(k)
            cat = CAT_METRIC_BY_KEY.get(k)
            if cat:
                checked = checklist_vals[i]
                if checked:
                    active_categorical.append((cat.key, list(checked)))
            else:
                r = slider_vals[i]
                if r:
                    active_numeric.append((k, r))

        if _violations:
            lines = "; ".join(
                f"Axis {idx} ({k}): {reason}"
                for idx, k, reason in _violations
            )
            banner = _build_error_banner_children(
                "Sweep configuration invalid",
                f"{lines}. Either change the metric on those axes, or "
                f"flip the pin axis / remove the custom QASM upload to "
                f"make them compatible.",
            )
            return (
                plot_empty("Sweep blocked: incompatible axis selection"),
                dash.no_update, "Sweep blocked",
                dash.no_update, dash.no_update,
                dirty, None,
                {"display": "none"},
                dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update,
                banner, _error_banner_visible_style(),
            )

        active = active_numeric + active_categorical
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
                dash.no_update,
                dash.no_update,
                [],
                _ERROR_BANNER_HIDDEN_STYLE,
            )

        _logi = max(2, int(num_logical_qubits or 16))
        cold_config = {
            "circuit_type": circuit_type or "qft",
            "num_logical_qubits": _logi,
            "num_cores": int(num_cores or 1),
            "qubits_per_core": int(qubits_per_core or 16),
            "pin_axis": pin_axis or "cores",
            "communication_qubits": int(communication_qubits or 1),
            "buffer_qubits": int(buffer_qubits or 1),
            "topology_type": topology or "ring",
            "placement_policy": placement or "random",
            "seed": int(seed or 42),
            "intracore_topology": intracore_topology or "all_to_all",
            "routing_algorithm": routing_algorithm or "hqa_sabre",
            "custom_qasm": custom_qasm_str,
        }

        # Build numeric sweep_axes from slider ranges.
        sweep_axes = [(k, r[0], r[1]) for k, r in active_numeric]

        if not active_categorical:
            # Pure numeric sweep.  Skip the main-thread ``run_cold`` when the
            # inner sweep takes the parallel cold path (which recompiles
            # inside workers anyway); otherwise we hide one cold compile
            # behind the "Compiling circuit…" spinner before the progress
            # callback ever fires.
            _inner_has_cold = _engine._has_cold(
                *[ax[0] for ax in sweep_axes]
            )
            # Flip phase to "sweeping" immediately so the loading bar shows
            # rather than the spinner.
            _update_progress(SweepProgress(completed=0, total=0))
            cached = (
                None if _inner_has_cold
                else _engine.run_cold(**cold_config, noise=fixed_noise)
            )
            cold_elapsed = time.time() - t_start
            result = _engine.sweep_nd(
                cached=cached,
                sweep_axes=sweep_axes,
                fixed_noise=fixed_noise,
                cold_config=cold_config,
                progress_callback=_update_progress,
                parallel=True,
                keep_per_qubit_grids=True,
                max_workers=int(max_workers) if max_workers else None,
                max_cold=int(max_cold) if max_cold else None,
                max_hot=int(max_hot) if max_hot else None,
            )
            sweep_data = result.to_sweep_data()
        else:
            # Categorical axes present — faceted sweep.
            # Build cartesian product of selected categorical values.
            from itertools import product as _product
            cat_keys = [k for k, _ in active_categorical]
            cat_val_lists = [vals for _, vals in active_categorical]
            cat_combos = list(_product(*cat_val_lists))
            n_combos = len(cat_combos)

            # Compute per-facet grid and cold-group counts analytically so
            # the first facet's cold compilations show up in the loading bar
            # instead of being hidden behind a silent "Compiling circuit…"
            # dry run.
            _inner_has_cold = _engine._has_cold(*[ax[0] for ax in sweep_axes])
            _eff_hot, _ = _engine._memory_capped_max_hot(
                int(max_hot) if max_hot else None,
            )
            _axis_counts = _engine._compute_axis_counts(
                sweep_axes, _inner_has_cold,
                int(max_cold) if max_cold else None,
                _eff_hot,
            )
            per_facet_pts = int(np.prod(_axis_counts)) if _axis_counts else 1
            _cold_axis_counts = [
                _axis_counts[i] for i, ax in enumerate(sweep_axes)
                if ax[0] in _engine.COLD_PATH_KEYS
            ]
            groups_per_facet = (
                int(np.prod(_cold_axis_counts)) if _cold_axis_counts else 1
            )
            total_points = per_facet_pts * n_combos
            total_cold = max(1, groups_per_facet * n_combos)

            # Unified progress: track completed points and cold groups
            # across all facets so the loading bar advances monotonically.
            # ``current_facet_labels`` is refreshed per categorical combo and
            # merged into the inner progress's ``current_params`` so the
            # overlay shows the active qualitative variable (e.g. the
            # circuit type) alongside the numeric axes.
            facet_progress = {"completed_offset": 0, "cold_done_offset": 0}
            current_facet_labels: dict[str, str] = {}

            def _faceted_progress(p: SweepProgress) -> None:
                merged_params = {**current_facet_labels, **p.current_params}
                _update_progress(SweepProgress(
                    completed=facet_progress["completed_offset"] + p.completed,
                    total=total_points,
                    current_params=merged_params,
                    cold_completed=facet_progress["cold_done_offset"] + p.cold_completed,
                    cold_total=total_cold,
                ))

            # Flip the phase to "sweeping" before any cold work so the UI
            # shows the real progress bar (0/N cold) instead of the
            # indefinite "Compiling circuit…" spinner. Seed with the first
            # facet's labels so the overlay isn't blank while compiling.
            first_labels: dict[str, str] = {}
            if cat_combos:
                for cat_key, cat_value in zip(cat_keys, cat_combos[0]):
                    cat_def = CAT_METRIC_BY_KEY[cat_key]
                    first_labels[cat_key] = next(
                        (o["label"] for o in cat_def.options if o["value"] == cat_value),
                        cat_value,
                    )
            _update_progress(SweepProgress(
                completed=0,
                total=total_points,
                current_params=first_labels,
                cold_completed=0,
                cold_total=total_cold,
            ))

            facet_results = []
            for combo in cat_combos:
                fc = {**cold_config}
                label_dict = {}
                for cat_key, cat_value in zip(cat_keys, combo):
                    cat_def = CAT_METRIC_BY_KEY[cat_key]
                    fc[cat_def.cold_config_key] = cat_value
                    label_dict[cat_key] = next(
                        (o["label"] for o in cat_def.options if o["value"] == cat_value),
                        cat_value,
                    )
                current_facet_labels.clear()
                current_facet_labels.update(label_dict)
                # ``_parallel_cold_sweep`` recompiles inside workers, so
                # only compile on the main thread when the inner sweep
                # will take the pure hot-path (no cold axes).
                cached = (
                    None if _inner_has_cold
                    else _engine.run_cold(**fc, noise=fixed_noise)
                )
                res = _engine.sweep_nd(
                    cached=cached,
                    sweep_axes=sweep_axes,
                    fixed_noise=fixed_noise,
                    cold_config=fc,
                    progress_callback=_faceted_progress,
                    parallel=True,
                    keep_per_qubit_grids=True,
                    max_workers=int(max_workers) if max_workers else None,
                    max_cold=int(max_cold) if max_cold else None,
                    max_hot=int(max_hot) if max_hot else None,
                )
                sd = res.to_sweep_data()
                facet_results.append({"label": label_dict, **sd})
                facet_progress["completed_offset"] += res.grid.size
                facet_progress["cold_done_offset"] += groups_per_facet

            # For a single categorical combo, unwrap to plain sweep_data.
            if len(facet_results) == 1:
                sweep_data = facet_results[0]
            else:
                sweep_data = {
                    "metric_keys": facet_results[0].get("metric_keys", []),
                    "facets": facet_results,
                    "facet_keys": cat_keys,
                }
            cold_elapsed = time.time() - t_start

        sweep_elapsed = time.time() - t_start - cold_elapsed
        total_elapsed = time.time() - t_start
        n_facets = len(cat_combos) if active_categorical else 1
        pts = _count_points(sweep_data)
        facet_tag = f" x {n_facets} facets" if n_facets > 1 else ""
        status = (
            f"Cold: {cold_elapsed:.1f}s  |  "
            f"Sweep ({len(active_numeric)}D, {pts} pts{facet_tag}): {sweep_elapsed:.2f}s  |  "
            f"Total: {total_elapsed:.1f}s"
        )

        ndim = len(active_numeric) or 1
        view = resolve_view_type(current_view, ndim)
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
        frozen_value = dash.no_update

        frozen_dropdown_options = dash.no_update
        frozen_dropdown_value = dash.no_update

        if ndim == 3 and "facets" not in sweep_data:
            metric_keys_3d = sweep_data.get("metric_keys", [])
            f_idx = frozen_axis_idx if frozen_axis_idx in (0, 1, 2) else 2

            permuted = permute_sweep_for_frozen(sweep_data, f_idx)
            interp_grid = sweep_to_interp_grid(permuted, out_key)
            fs_cfg = frozen_slider_config(permuted)
            # Always seed slider min/max/value so it matches the frozen axis even
            # when the user is not currently on a frozen view tab — they may
            # switch later, and the JS slice expects values inside zs's range.
            if fs_cfg:
                frozen_min = fs_cfg["min"]
                frozen_max = fs_cfg["max"]
                frozen_value = fs_cfg["default"]
                if is_frozen_view(view):
                    frozen_style = {"padding": "4px 16px 8px"}

            frozen_dropdown_options = [
                {
                    "label": (METRIC_BY_KEY.get(mk).label if METRIC_BY_KEY.get(mk) else mk),
                    "value": i,
                }
                for i, mk in enumerate(metric_keys_3d)
            ]
            frozen_dropdown_value = f_idx
            # Use the permuted sweep_data so build_figure and the slim store
            # see the rearranged axes.
            sweep_data = permuted

        fig = build_figure(
            ndim,
            sweep_data,
            out_key,
            view_type=view,
            thresholds=thresh,
            threshold_colors=thresh_colors or None,
            pareto_x=pareto_x,
            pareto_y=pareto_y,
            fom_config=fom_config,
            view_mode=view_mode or "absolute",
        )
        # Send only a small token + axes metadata to the browser; the full
        # grid lives in ``_SWEEP_CACHE`` and is fetched by downstream callbacks.
        slim_store = _slim_sweep_for_browser(sweep_data)
        return (
            fig,
            slim_store,
            status,
            view,
            make_view_tab_bar(ndim, view),
            dirty,
            interp_grid,
            frozen_style,
            frozen_min,
            frozen_max,
            frozen_value,
            frozen_dropdown_options,
            frozen_dropdown_value,
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
            dash.no_update,
            dash.no_update,
            banner_children,
            _error_banner_visible_style(),
        )
    finally:
        sid = getattr(_sweep_progress_tls, "sid", None)
        if sid:
            _set_progress(sid, {"running": False})
        _sweep_progress_tls.sid = None
        sweep_lock.release()


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
    Input("cfg-view-mode", "value"),
    Input("cfg-threshold-enable", "value"),
    *[Input(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[Input(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    Input("pareto-x-axis-dropdown", "value"),
    Input("pareto-y-axis-dropdown", "value"),
    Input("elasticity-trajectory-dropdown", "value"),
    Input("importance-mode-dropdown", "value"),
    Input("correlation-mode-dropdown", "value"),
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("num-thresholds-store", "data"),
    State("fom-config-store", "data"),
    State("merit-mode-store", "data"),
    State("merit-x-axis-dropdown", "value"),
    State("merit-y-axis-dropdown", "value"),
    State({"type": "merit-frozen-slider", "index": ALL}, "value"),
    State("merit-color-by-dropdown", "value"),
    prevent_initial_call=True,
)
def replot_on_output_change(
    output_key,
    view_mode,
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
    pareto_x,
    pareto_y,
    elasticity_trajectory,
    importance_mode,
    correlation_mode,
    sweep_store,
    view_type,
    num_thresholds,
    fom_config,
    merit_mode,
    merit_x_axis,
    merit_y_axis,
    merit_frozen_slider_values,
    merit_color_by,
):
    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update
    # Each per-view mode dropdown only affects its own view — skip rebuilds
    # elsewhere so we don't re-render expensive 3-D figures on every toggle.
    if ctx.triggered_id in ("pareto-x-axis-dropdown", "pareto-y-axis-dropdown") \
            and view_type != "pareto":
        return dash.no_update
    if ctx.triggered_id == "elasticity-trajectory-dropdown" and view_type != "elasticity":
        return dash.no_update
    if ctx.triggered_id == "importance-mode-dropdown" and view_type != "importance":
        return dash.no_update
    if ctx.triggered_id == "correlation-mode-dropdown" and view_type != "correlation":
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
    merit_frozen = _frozen_values_from_sliders(
        list(full.get("metric_keys", [])), merit_frozen_slider_values,
    )
    return build_figure(
        num_metrics,
        full,
        output_key or "overall_fidelity",
        view_type=view_type,
        thresholds=thresh,
        threshold_colors=thresh_colors or None,
        pareto_x=pareto_x,
        pareto_y=pareto_y,
        fom_config=fom_config,
        merit_mode=merit_mode or "heatmap",
        merit_x_axis=merit_x_axis,
        merit_y_axis=merit_y_axis,
        merit_frozen_values=merit_frozen,
        merit_color_by=merit_color_by,
        view_mode=view_mode or "absolute",
        elasticity_trajectory=elasticity_trajectory,
        importance_mode=importance_mode or "range",
        correlation_mode=correlation_mode or "spearman",
    )


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
    State("cfg-view-mode", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    State("pareto-x-axis-dropdown", "value"),
    State("pareto-y-axis-dropdown", "value"),
    State("elasticity-trajectory-dropdown", "value"),
    State("importance-mode-dropdown", "value"),
    State("correlation-mode-dropdown", "value"),
    State("fom-config-store", "data"),
    prevent_initial_call=True,
)
def on_view_tab_click(
    n_clicks_list,
    sweep_store,
    num_metrics,
    output_key,
    view_mode,
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
    pareto_x,
    pareto_y,
    elasticity_trajectory,
    importance_mode,
    correlation_mode,
    fom_config,
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
        pareto_x=pareto_x,
        pareto_y=pareto_y,
        fom_config=fom_config,
        view_mode=view_mode or "absolute",
        elasticity_trajectory=elasticity_trajectory,
        importance_mode=importance_mode or "range",
        correlation_mode=correlation_mode or "spearman",
    )
    return fig, view_type, make_view_tab_bar(actual_metrics, view_type)


# ---------------------------------------------------------------------------
# Callback: save session → gzipped JSON download
# ---------------------------------------------------------------------------


@app.callback(
    Output("session-download", "data"),
    Input("save-btn", "n_clicks"),
    *_METRIC_DROPDOWN_STATES,
    *_METRIC_SLIDER_STATES,
    State("num-metrics-store", "data"),
    *_METRIC_CHECKLIST_STATES,
    State("cfg-circuit-type", "value"),
    State("cfg-qubits-per-core", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-communication-qubits", "value"),
    State("cfg-buffer-qubits", "value"),
    State("cfg-num-logical-qubits", "value"),
    State("cfg-pin-axis", "data"),
    State("cfg-topology", "value"),
    State("cfg-intracore-topology", "value"),
    State("cfg-placement", "value"),
    State("cfg-routing-algorithm", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-max-cold", "value"),
    State("cfg-max-hot", "value"),
    State("cfg-max-workers", "value"),
    State("cfg-output-metric", "value"),
    State("cfg-view-mode", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    State("view-type-store", "data"),
    State("frozen-axis-store", "data"),
    State("frozen-slider", "value"),
    State("hot-reload-toggle", "value"),
    State("sweep-result-store", "data"),
    State("session-name", "value"),
    State("fom-config-store", "data"),
    *_NOISE_SLIDER_STATES,
    prevent_initial_call=True,
)
def on_save_session(n_clicks, *all_args):
    if not n_clicks:
        return dash.no_update

    from gui.session import (
        build_controls_dict, build_view_dict, collect_session, dump,
        sanitize_filename,
    )
    import time as _time

    idx = 0
    dropdown_vals = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    slider_vals   = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    num_metrics   = all_args[idx]; idx += 1
    checklist_vals = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    cfg_circuit_type = all_args[idx]; idx += 1
    cfg_qubits_per_core = all_args[idx]; idx += 1
    cfg_num_cores = all_args[idx]; idx += 1
    cfg_communication_qubits = all_args[idx]; idx += 1
    cfg_buffer_qubits = all_args[idx]; idx += 1
    cfg_num_logical_qubits = all_args[idx]; idx += 1
    cfg_pin_axis = all_args[idx]; idx += 1
    cfg_topology = all_args[idx]; idx += 1
    cfg_intracore_topology = all_args[idx]; idx += 1
    cfg_placement = all_args[idx]; idx += 1
    cfg_routing_algorithm = all_args[idx]; idx += 1
    cfg_seed = all_args[idx]; idx += 1
    cfg_dd = all_args[idx]; idx += 1
    cfg_max_cold = all_args[idx]; idx += 1
    cfg_max_hot = all_args[idx]; idx += 1
    cfg_max_workers = all_args[idx]; idx += 1
    cfg_output_metric = all_args[idx]; idx += 1
    cfg_view_mode = all_args[idx]; idx += 1
    cfg_threshold_enable = all_args[idx]; idx += 1
    t_vals = list(all_args[idx:idx + 5]); idx += 5
    tc_vals = list(all_args[idx:idx + 5]); idx += 5
    num_thresholds = all_args[idx]; idx += 1
    view_type = all_args[idx]; idx += 1
    frozen_axis = all_args[idx]; idx += 1
    frozen_slider_value = all_args[idx]; idx += 1
    hot_reload = all_args[idx]; idx += 1
    sweep_store = all_args[idx]; idx += 1
    session_name = all_args[idx]; idx += 1
    fom_config = all_args[idx]; idx += 1
    noise_slider_vals = list(all_args[idx:])

    noise_values: dict = {}
    for i, m in enumerate(SWEEPABLE_METRICS):
        v = noise_slider_vals[i]
        if v is None:
            continue
        noise_values[m.key] = _slider_to_value(v, m.log_scale)

    controls = build_controls_dict(
        num_metrics=num_metrics,
        dropdown_vals=dropdown_vals,
        slider_vals=slider_vals,
        checklist_vals=checklist_vals,
        cfg_circuit_type=cfg_circuit_type,
        cfg_qubits_per_core=cfg_qubits_per_core,
        cfg_num_cores=cfg_num_cores,
        cfg_communication_qubits=cfg_communication_qubits,
        cfg_buffer_qubits=cfg_buffer_qubits,
        cfg_num_logical_qubits=cfg_num_logical_qubits,
        cfg_pin_axis=cfg_pin_axis or "cores",
        cfg_topology=cfg_topology,
        cfg_intracore_topology=cfg_intracore_topology,
        cfg_placement=cfg_placement,
        cfg_routing_algorithm=cfg_routing_algorithm,
        cfg_seed=cfg_seed,
        cfg_dynamic_decoupling=cfg_dd,
        cfg_max_cold=cfg_max_cold,
        cfg_max_hot=cfg_max_hot,
        cfg_max_workers=cfg_max_workers,
        cfg_output_metric=cfg_output_metric,
        cfg_view_mode=cfg_view_mode,
        cfg_threshold_enable=cfg_threshold_enable,
        num_thresholds=num_thresholds,
        threshold_values=t_vals,
        threshold_colors=tc_vals,
        noise_values=noise_values,
        hot_reload=hot_reload,
        fom_config=fom_config,
    )
    view = build_view_dict(view_type, frozen_axis, frozen_slider_value)
    sweep_data = _get_sweep(sweep_store)

    session = collect_session(controls, view, sweep_data, name=session_name or "")
    raw = dump(session)

    stem = sanitize_filename(session_name or "")
    fname = _time.strftime(f"{stem}-%Y%m%d-%H%M%S.quadris.json.gz")
    import base64
    return dict(
        content=base64.b64encode(raw).decode("ascii"),
        filename=fname,
        base64=True,
        type="application/gzip",
    )


# ---------------------------------------------------------------------------
# Callback: load session → rehydrate controls, view, and sweep result
# ---------------------------------------------------------------------------

_AXIS_OUTPUTS = (
    [Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
    + [Output(f"metric-slider-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
    + [Output(f"metric-checklist-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
)

_NOISE_OUTPUTS = [Output(f"noise-{m.key}", "value", allow_duplicate=True) for m in SWEEPABLE_METRICS]

_CFG_OUTPUTS = [
    Output("cfg-circuit-type", "value", allow_duplicate=True),
    Output("cfg-qubits-per-core", "value", allow_duplicate=True),
    Output("cfg-num-cores", "value", allow_duplicate=True),
    Output("cfg-communication-qubits", "value", allow_duplicate=True),
    Output("cfg-buffer-qubits", "value", allow_duplicate=True),
    Output("cfg-num-logical-qubits", "value", allow_duplicate=True),
    Output("cfg-topology", "value", allow_duplicate=True),
    Output("cfg-intracore-topology", "value", allow_duplicate=True),
    Output("cfg-placement", "value", allow_duplicate=True),
    Output("cfg-routing-algorithm", "value", allow_duplicate=True),
    Output("cfg-seed", "value", allow_duplicate=True),
    Output("cfg-dynamic-decoupling", "value", allow_duplicate=True),
    Output("cfg-max-cold", "value", allow_duplicate=True),
    Output("cfg-max-hot", "value", allow_duplicate=True),
    Output("cfg-max-workers", "value", allow_duplicate=True),
    Output("cfg-output-metric", "value", allow_duplicate=True),
    Output("cfg-view-mode", "value", allow_duplicate=True),
    Output("cfg-threshold-enable", "value", allow_duplicate=True),
    Output("cfg-pin-axis", "data", allow_duplicate=True),
]

_THRESH_OUTPUTS = (
    [Output(f"cfg-threshold-{i}", "value", allow_duplicate=True) for i in range(5)]
    + [Output(f"cfg-threshold-color-{i}", "value", allow_duplicate=True) for i in range(5)]
)


@app.callback(
    *_AXIS_OUTPUTS,
    *_CFG_OUTPUTS,
    *_THRESH_OUTPUTS,
    Output("num-metrics-store", "data", allow_duplicate=True),
    Output("num-thresholds-store", "data", allow_duplicate=True),
    Output("hot-reload-toggle", "value", allow_duplicate=True),
    Output("view-type-store", "data", allow_duplicate=True),
    Output("frozen-axis-store", "data", allow_duplicate=True),
    *_NOISE_OUTPUTS,
    Output("main-plot", "figure", allow_duplicate=True),
    Output("sweep-result-store", "data", allow_duplicate=True),
    Output("interp-grid-store", "data", allow_duplicate=True),
    Output("view-tab-container", "children", allow_duplicate=True),
    Output("frozen-slider-container", "style", allow_duplicate=True),
    Output("frozen-slider", "min", allow_duplicate=True),
    Output("frozen-slider", "max", allow_duplicate=True),
    Output("frozen-slider", "value", allow_duplicate=True),
    Output("sweep-dirty", "data", allow_duplicate=True),
    Output("sweep-processed", "data", allow_duplicate=True),
    Output("session-loaded-tick", "data", allow_duplicate=True),
    Output("suppress-cascade", "data", allow_duplicate=True),
    Output("session-name", "value", allow_duplicate=True),
    Output("status-bar", "children", allow_duplicate=True),
    Output("error-banner", "children", allow_duplicate=True),
    Output("error-banner", "style", allow_duplicate=True),
    Output("fom-config-store", "data", allow_duplicate=True),
    Output("fom-name", "value", allow_duplicate=True),
    Output("fom-numerator", "value", allow_duplicate=True),
    Output("fom-denominator", "value", allow_duplicate=True),
    Output("fom-intermediates", "value", allow_duplicate=True),
    Output("pareto-x-axis-dropdown", "value", allow_duplicate=True),
    Output("pareto-y-axis-dropdown", "value", allow_duplicate=True),
    Output("examples-dropdown", "value", allow_duplicate=True),
    Input("session-upload", "contents"),
    Input("examples-dropdown", "value"),
    State("session-upload", "filename"),
    prevent_initial_call=True,
)
def on_load_session(contents, example_id, filename):
    import base64
    from gui.session import load as session_load, apply_session, SessionError

    triggered = ctx.triggered_id
    if triggered == "examples-dropdown":
        if not example_id:
            raise dash.exceptions.PreventUpdate
        try:
            raw = _example_path(example_id).read_bytes()
        except OSError as exc:
            banner = _build_error_banner_children(
                "Failed to load example",
                f"{exc}.  Run `python scripts/generate_example_sessions.py` "
                f"to (re)build the example bundles.",
            )
            return _load_error_return(banner)
        filename = f"example: {example_id}"
    elif triggered == "session-upload":
        if contents is None:
            raise dash.exceptions.PreventUpdate
        # ``contents`` has shape 'data:<mime>;base64,<payload>'.
        try:
            _, b64 = contents.split(",", 1)
            raw = base64.b64decode(b64)
        except (ValueError, OSError) as exc:
            banner = _build_error_banner_children("Failed to load session", str(exc))
            return _load_error_return(banner)
    else:
        raise dash.exceptions.PreventUpdate

    try:
        session = session_load(raw)
        result = apply_session(session)
    except (SessionError, ValueError, OSError) as exc:
        banner = _build_error_banner_children("Failed to load session", str(exc))
        return _load_error_return(banner)

    ctrls = result.controls
    view = result.view

    # Axes → MAX_METRICS slots. Pad unused slots with dash.no_update.
    dropdown_out = [dash.no_update] * MAX_METRICS
    slider_out   = [dash.no_update] * MAX_METRICS
    checklist_out = [dash.no_update] * MAX_METRICS
    for i, ax in enumerate(ctrls["axes"][:MAX_METRICS]):
        dropdown_out[i] = ax["key"]
        slider_out[i] = ax["slider"] if ax["slider"] is not None else dash.no_update
        checklist_out[i] = ax["checklist"] if ax["checklist"] is not None else []

    circuit = ctrls["circuit"]
    _qpc_default = int(circuit.get("qubits_per_core", 16) or 16)
    cfg_out = [
        circuit["circuit_type"],
        _qpc_default,
        circuit.get("num_cores", 1),
        int(circuit.get("communication_qubits", 1) or 1),
        int(circuit.get("buffer_qubits", 1) or 1),
        int(circuit.get("num_logical_qubits", _qpc_default) or _qpc_default),
        circuit["topology_type"],
        circuit["intracore_topology"],
        circuit["placement"],
        circuit["routing_algorithm"],
        circuit["seed"],
        ["yes"] if circuit["dynamic_decoupling"] else [],
        ctrls["performance"]["max_cold"],
        ctrls["performance"]["max_hot"],
        ctrls["performance"]["max_workers"],
        ctrls["thresholds"]["output_metric"],
        ctrls["thresholds"].get("view_mode", "absolute") or "absolute",
        # Iso-levels always-on: the user-facing toggle was removed, so we
        # ignore the saved value and force the gate open regardless.
        ["yes"],
        circuit.get("pin_axis", "cores") or "cores",
    ]

    t_vals = list(ctrls["thresholds"]["values"]) + [None] * 5
    tc_vals = list(ctrls["thresholds"]["colors"]) + [None] * 5
    thresh_out = t_vals[:5] + tc_vals[:5]

    noise = ctrls["noise"]
    noise_out = [
        _value_to_slider(noise.get(m.key), m.log_scale)
        if noise.get(m.key) is not None
        else dash.no_update
        for m in SWEEPABLE_METRICS
    ]

    fom_dict = ctrls.get("fom") or DEFAULT_FOM.to_dict()
    fom_cfg_obj = FomConfig.from_dict(fom_dict)
    fom_name_out = fom_cfg_obj.name
    fom_num_out = fom_cfg_obj.numerator
    fom_den_out = fom_cfg_obj.denominator
    fom_inter_out = "\n".join(f"{n} = {e}" for n, e in fom_cfg_obj.intermediates)

    msg = f"Loaded {filename or 'session'}"
    if result.warnings:
        msg += f"  ({len(result.warnings)} warning(s))"
    banner_children = []
    banner_style = _ERROR_BANNER_HIDDEN_STYLE
    if result.warnings:
        banner_children = _build_error_banner_children(
            "Session loaded with warnings",
            "\n".join(result.warnings),
        )
        banner_style = _error_banner_visible_style()

    # --- Rehydrate the sweep result (if present) -------------------------
    fig_out = dash.no_update
    sweep_store_out = dash.no_update
    interp_out = dash.no_update
    view_tab_out = dash.no_update
    frozen_style = {"display": "none"}
    frozen_min = dash.no_update
    frozen_max = dash.no_update
    frozen_val = dash.no_update

    sweep_data = result.sweep_data
    if sweep_data is not None:
        from gui.interpolation import permute_sweep_for_frozen
        ndim = len(sweep_data.get("metric_keys", []))
        out_key = ctrls["thresholds"]["output_metric"] or "overall_fidelity"

        # Permute so axis 2 is the frozen axis; downstream (build_figure,
        # sweep_to_interp_grid, frozen_slice) all assume axis 2 is frozen.
        if ndim == 3 and "facets" not in sweep_data:
            sweep_data = permute_sweep_for_frozen(sweep_data, view["frozen_axis"] or 2)
            interp_out = sweep_to_interp_grid(sweep_data, out_key)
            fs_cfg = frozen_slider_config(sweep_data)
            if fs_cfg and is_frozen_view(view["view_type"]):
                frozen_style = {"padding": "4px 16px 8px"}
                frozen_min = fs_cfg["min"]
                frozen_max = fs_cfg["max"]
                frozen_val = view["frozen_slider_value"] if view["frozen_slider_value"] is not None else fs_cfg["default"]

        thresh_vals = [v for v in ctrls["thresholds"]["values"] if v is not None]
        thresh_colors = [
            c for i, c in enumerate(ctrls["thresholds"]["colors"])
            if ctrls["thresholds"]["values"][i] is not None
        ]
        thresh = thresh_vals if ctrls["thresholds"]["enable"] else None
        if view["view_type"] in ("isosurface", "scatter3d"):
            thresh = thresh_vals or None

        fig_out = build_figure(
            ndim, sweep_data, out_key,
            view_type=view["view_type"],
            thresholds=thresh,
            threshold_colors=thresh_colors or None,
            view_mode=ctrls["thresholds"].get("view_mode", "absolute") or "absolute",
        )
        sweep_store_out = _slim_sweep_for_browser(sweep_data)
        view_tab_out = make_view_tab_bar(ndim, view["view_type"])

    # Suppress auto-sweep: advance both counters to a fresh high-water mark.
    hw = _next_session_hw(0)

    pareto_x_out = view.get("pareto_x") or dash.no_update
    pareto_y_out = view.get("pareto_y") or dash.no_update

    return (
        *dropdown_out,
        *slider_out,
        *checklist_out,
        *cfg_out,
        *thresh_out,
        ctrls["num_metrics"],
        ctrls["thresholds"]["num_thresholds"],
        ["on"] if ctrls["hot_reload"] else [],
        view["view_type"],
        view["frozen_axis"],
        *noise_out,
        # --- new sweep-rehydration outputs ---
        fig_out, sweep_store_out, interp_out, view_tab_out,
        frozen_style, frozen_min, frozen_max, frozen_val,
        hw, hw,  # sweep-dirty, sweep-processed
        hw,      # session-loaded-tick (reuse hw for monotonicity)
        True,    # suppress-cascade: keep loaded slider/checklist values
        result.name,  # session-name input
        # ---
        msg,
        banner_children,
        banner_style,
        fom_dict, fom_name_out, fom_num_out, fom_den_out, fom_inter_out,
        pareto_x_out, pareto_y_out,
        None,  # examples-dropdown reset so the next pick re-fires
    )


# Count of named scalar Outputs between thresholds and noise in the main
# decorator — see on_load_session: num-metrics, num-thresholds, hot-reload,
# view-type, frozen-axis.
_LOAD_SCALAR_OUTPUTS = 5
# Count of trailing Outputs: status-bar, banner.children, banner.style,
# fom-config-store, fom-name, fom-numerator, fom-denominator, fom-intermediates,
# pareto-x, pareto-y, examples-dropdown.
_LOAD_TRAILING_OUTPUTS = 11
_LOAD_SWEEP_OUTPUTS = 13  # figure, sweep-store, interp, view-tabs, frozen-style, frozen-min/max/val, sweep-dirty, sweep-processed, session-loaded-tick, suppress-cascade, session-name


def _load_error_return(banner_children):
    """Return tuple for on_load_session error path — everything else no-op."""
    outputs_total = (
        len(_AXIS_OUTPUTS)
        + len(_CFG_OUTPUTS)
        + len(_THRESH_OUTPUTS)
        + _LOAD_SCALAR_OUTPUTS
        + len(_NOISE_OUTPUTS)
        + _LOAD_SWEEP_OUTPUTS
        + _LOAD_TRAILING_OUTPUTS
    )
    stub = [dash.no_update] * outputs_total
    # Trailing order (last 11): status-bar, banner.children, banner.style,
    # 5 FoM outputs, pareto-x, pareto-y, examples-dropdown.
    stub[-11] = "Load failed"
    stub[-10] = banner_children
    stub[-9] = _error_banner_visible_style()
    stub[-1] = None  # reset the dropdown so the user can retry
    return tuple(stub)


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
        return {"display": on ? "none" : "inline-flex"};
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

# Server-side companion: when the user drags the frozen slider in a
# derivative view mode, the clientside re-slicer skips (its interp-grid
# is the absolute field). Rebuild the figure on the server so the
# heatmap actually tracks the slider in |∇F| / mixed-partial modes too.
@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("frozen-slider", "value"),
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("cfg-view-mode", "value"),
    State("cfg-output-metric", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    prevent_initial_call=True,
)
def replot_frozen_slider_in_derivative_mode(
    frozen_val,
    sweep_store, view_type, view_mode, output_key, threshold_enable,
    t0, t1, t2, t3, t4, tc0, tc1, tc2, tc3, tc4, num_thresholds,
):
    if view_type not in ("frozen_heatmap", "frozen_contour"):
        return dash.no_update
    if not view_mode or view_mode == "absolute":
        return dash.no_update  # clientside handler already updated the heatmap
    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update
    n_t = int(num_thresholds or 3)
    all_t = [t0, t1, t2, t3, t4][:n_t]
    all_c = [tc0, tc1, tc2, tc3, tc4][:n_t]
    thresh_vals = [v for v in all_t if v is not None]
    thresh_colors = [all_c[i] for i, v in enumerate(all_t) if v is not None]
    thresh = thresh_vals if threshold_enable and "yes" in threshold_enable else None
    return build_figure(
        len(full.get("metric_keys", [])),
        full,
        output_key or "overall_fidelity",
        view_type=view_type,
        thresholds=thresh,
        threshold_colors=thresh_colors or None,
        view_mode=view_mode,
        frozen_z=frozen_val,
    )


app.clientside_callback(
    """function(frozenVal, interpGrid, viewType, viewMode) {
        if (!interpGrid || !interpGrid.values || interpGrid.ndim !== 3) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        if (viewType !== "frozen_heatmap" && viewType !== "frozen_contour") {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var v = frozenVal;
        var label = (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)
            ? v.toExponential(2) : (Math.abs(v) < 10 ? v.toFixed(4) : v.toFixed(1));

        // The cached interp-grid stores absolute fidelity values, so
        // re-slicing it in derivative mode would silently revert the
        // heatmap to absolute even though the colorbar still says |∇F|.
        // In derivative mode we skip the in-place restyle and let the
        // server-side replot handler re-render on the next user input;
        // the slider label still updates so the drag feels responsive.
        if (viewMode && viewMode !== "absolute") {
            return [window.dash_clientside.no_update, label];
        }

        var qi = window.quadrisInterp;
        if (!qi) {
            return [window.dash_clientside.no_update, label];
        }

        var slice2d = qi.frozenSlice(
            interpGrid.values, interpGrid.xs, interpGrid.ys,
            interpGrid.zs, frozenVal
        );

        // dcc.Graph wraps the plotly div in an outer container — only the
        // inner .js-plotly-plot element carries the live `.data` array.
        var outer = document.getElementById("main-plot");
        var plotDiv = outer ? outer.querySelector(".js-plotly-plot") : null;
        if (plotDiv && plotDiv.data && plotDiv.data.length > 0) {
            Plotly.restyle(plotDiv, {z: [slice2d]}, [0]);
        }

        return [window.dash_clientside.no_update, label];
    }""",
    Output("main-plot", "figure", allow_duplicate=True),
    Output("frozen-slider-value", "children"),
    Input("frozen-slider", "value"),
    State("interp-grid-store", "data"),
    State("view-type-store", "data"),
    State("cfg-view-mode", "value"),
    prevent_initial_call=True,
)


@app.callback(
    Output("fom-name", "value", allow_duplicate=True),
    Output("fom-numerator", "value", allow_duplicate=True),
    Output("fom-denominator", "value", allow_duplicate=True),
    Output("fom-intermediates", "value", allow_duplicate=True),
    Input("fom-preset", "value"),
    prevent_initial_call=True,
)
def on_fom_preset_change(preset_key):
    preset = PRESETS.get(preset_key or "")
    if not preset:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    intermediates = preset.get("intermediates") or []
    text = "\n".join(f"{n} = {e}" for n, e in intermediates)
    return preset["name"], preset["numerator"], preset["denominator"], text


@app.callback(
    Output("fom-config-store", "data"),
    Output("fom-status", "children"),
    Output("fom-status", "style"),
    Input("fom-name", "value"),
    Input("fom-numerator", "value"),
    Input("fom-denominator", "value"),
    Input("fom-intermediates", "value"),
    State("sweep-result-store", "data"),
    prevent_initial_call=False,
)
def on_fom_formula_change(name, numerator, denominator, intermediates_text, sweep_store):
    config = FomConfig(
        name=(name or "Figure of Merit").strip() or "Figure of Merit",
        numerator=(numerator or "").strip() or "1",
        denominator=(denominator or "").strip() or "1",
        intermediates=tuple(_parse_intermediates(intermediates_text or "")),
    )
    base_style = {
        "fontSize": "11px",
        "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
        "marginTop": "4px",
        "whiteSpace": "nowrap",
        "overflow": "hidden",
        "textOverflow": "ellipsis",
    }

    full = _get_sweep(sweep_store) if sweep_store is not None else None
    if full is None:
        status = "No sweep loaded — formula will apply once a sweep runs."
        style = {**base_style, "color": COLORS["text_muted"]}
        return config.to_dict(), status, style

    result = compute_for_sweep(full, config)
    if result.error:
        style = {**base_style, "color": FEEDBACK_COLORS["error"]["text"]}
        primitives_hint = ", ".join(result.primitives[:6])
        more = "..." if len(result.primitives) > 6 else ""
        status = f"✗ {result.error}  |  vars: {primitives_hint}{more}"
        return config.to_dict(), status, style

    values = result.values
    assert values is not None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        style = {**base_style, "color": FEEDBACK_COLORS["warning"]["text"]}
        status = "⚠ All FoM values were non-finite (check for divide-by-zero)."
        return config.to_dict(), status, style

    vmin, vmax, vmean = float(finite.min()), float(finite.max()), float(finite.mean())
    style = {**base_style, "color": COLORS["accent"]}
    status = (
        f"✓ FoM over sweep: min {vmin:.4g}  mean {vmean:.4g}  max {vmax:.4g}"
        f"  ({finite.size}/{values.size} finite)"
    )
    return config.to_dict(), status, style


def _build_merit_figure(
    sweep_store, fom_config, threshold_enable, threshold_vals, threshold_cols,
    mode, x_axis, y_axis, frozen_slider_values, color_by,
):
    """Resolve every merit-tab input into a ``build_figure`` call.

    Returns ``dash.no_update`` if the sweep cache is empty so callers don't
    need to repeat the guard.
    """
    full = _get_sweep(sweep_store)
    if full is None:
        return dash.no_update
    metric_keys = list(full.get("metric_keys", []))
    thresh, thresh_colors = _resolve_thresholds(
        threshold_enable, threshold_vals, threshold_cols,
    )
    frozen = _frozen_values_from_sliders(metric_keys, frozen_slider_values)
    return build_figure(
        len(metric_keys), full, "overall_fidelity",
        view_type="merit",
        thresholds=thresh, threshold_colors=thresh_colors,
        fom_config=fom_config,
        merit_mode=mode or "heatmap",
        merit_x_axis=x_axis,
        merit_y_axis=y_axis,
        merit_frozen_values=frozen,
        merit_color_by=color_by,
    )


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("fom-config-store", "data"),
    State("view-type-store", "data"),
    State("sweep-result-store", "data"),
    State("merit-mode-store", "data"),
    State("merit-x-axis-dropdown", "value"),
    State("merit-y-axis-dropdown", "value"),
    State({"type": "merit-frozen-slider", "index": ALL}, "value"),
    State("merit-color-by-dropdown", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    prevent_initial_call=True,
)
def replot_on_fom_change(
    fom_config, view_type, sweep_store, mode, x_axis, y_axis,
    frozen_slider_values, color_by, threshold_enable,
    t0, t1, t2, t3, t4, tc0, tc1, tc2, tc3, tc4,
):
    if view_type != "merit":
        return dash.no_update
    return _build_merit_figure(
        sweep_store, fom_config, threshold_enable,
        [t0, t1, t2, t3, t4], [tc0, tc1, tc2, tc3, tc4],
        mode, x_axis, y_axis, frozen_slider_values, color_by,
    )


@app.callback(
    Output("merit-mode-store", "data"),
    Output("merit-heatmap-controls", "style"),
    Output("merit-pareto-controls", "style"),
    Output({"type": "merit-mode-btn", "index": ALL}, "style"),
    Input({"type": "merit-mode-btn", "index": ALL}, "n_clicks"),
    State({"type": "merit-mode-btn", "index": ALL}, "id"),
    State("merit-mode-store", "data"),
    prevent_initial_call=False,
)
def on_merit_mode_change(_n_clicks, ids, current_mode):
    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "merit-mode-btn":
        new_mode = triggered.get("index", current_mode or "heatmap")
    else:
        new_mode = current_mode or "heatmap"

    button_styles = []
    for entry in ids:
        is_active = entry.get("index") == new_mode
        button_styles.append({
            "background": COLORS["accent"] if is_active else "transparent",
            "color": "#fff" if is_active else COLORS["text_muted"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "4px",
            "padding": "3px 12px",
            "fontSize": "11px",
            "fontWeight": "600" if is_active else "400",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
        })

    # 3D mode reuses the same XY/frozen-slider controls as Heatmap — only
    # the Pareto colour-by lives under its own panel.
    heatmap_style = ({"display": "block"} if new_mode in ("heatmap", "3d")
                     else {"display": "none"})
    pareto_style = {"display": "block"} if new_mode == "pareto" else {"display": "none"}
    return new_mode, heatmap_style, pareto_style, button_styles


@app.callback(
    Output("merit-x-axis-dropdown", "options"),
    Output("merit-x-axis-dropdown", "value"),
    Output("merit-y-axis-dropdown", "options"),
    Output("merit-y-axis-dropdown", "value"),
    Output("merit-color-by-dropdown", "options"),
    Output("merit-color-by-dropdown", "value"),
    Output({"type": "merit-frozen-slider", "index": ALL}, "min"),
    Output({"type": "merit-frozen-slider", "index": ALL}, "max"),
    Output({"type": "merit-frozen-slider", "index": ALL}, "step"),
    Output({"type": "merit-frozen-slider", "index": ALL}, "value"),
    Output({"type": "merit-frozen-slider", "index": ALL}, "marks"),
    Output({"type": "merit-frozen-slider-label", "index": ALL}, "children"),
    Input("sweep-result-store", "data"),
    State("merit-x-axis-dropdown", "value"),
    State("merit-y-axis-dropdown", "value"),
    State("merit-color-by-dropdown", "value"),
    State({"type": "merit-frozen-slider", "index": ALL}, "id"),
    prevent_initial_call=False,
)
def populate_merit_controls_on_sweep(sweep_store, prev_x, prev_y, prev_color, slider_ids):
    n_sliders = len(slider_ids)

    metric_keys: list[str] = []
    axes_values: list[list[float]] = []
    if sweep_store and isinstance(sweep_store, dict):
        metric_keys = list(sweep_store.get("metric_keys", []) or [])
        if "axes" in sweep_store and isinstance(sweep_store["axes"], list):
            axes_values = [list(a) for a in sweep_store["axes"]]
        else:
            for k in ("xs", "ys", "zs"):
                if k in sweep_store and sweep_store[k] is not None:
                    axes_values.append(list(sweep_store[k]))

    # Truncate axes_values to match metric_keys length (defensive).
    axes_values = axes_values[:len(metric_keys)]
    while len(axes_values) < len(metric_keys):
        axes_values.append([])

    options = _axis_dropdown_options(metric_keys)

    # Pick X and Y, preferring the user's last selection if still valid.
    x_value = prev_x if prev_x in metric_keys else (metric_keys[0] if metric_keys else None)
    y_value = prev_y if prev_y in metric_keys and prev_y != x_value else None
    if y_value is None:
        y_value = next((k for k in metric_keys if k != x_value), None)

    color_options = (
        [{"label": "FoM", "value": "fom"}, {"label": "None", "value": "none"}]
        + [{"label": _progress_label(k), "value": k} for k in metric_keys]
    )
    color_value = prev_color if prev_color in {"fom", "none", *metric_keys} else (
        metric_keys[0] if metric_keys else "fom"
    )

    # Per-slider config — index aligns with metric_keys[i] when active.
    mins = [0] * n_sliders
    maxs = [1] * n_sliders
    steps = [None] * n_sliders
    values = [0] * n_sliders
    marks_list: list[dict] = [{} for _ in range(n_sliders)]
    labels: list[str] = ["" for _ in range(n_sliders)]
    for i in range(min(n_sliders, len(metric_keys))):
        ax = sorted({float(v) for v in axes_values[i]
                     if v is not None and np.isfinite(float(v))}) if axes_values[i] else []
        if not ax:
            continue
        mins[i] = float(ax[0])
        maxs[i] = float(ax[-1]) if ax[-1] != ax[0] else float(ax[0]) + 1.0
        # ``step=None`` + sorted ``marks`` snaps the slider to the discrete
        # grid points — works correctly for integer axes (Cores) and unevenly
        # spaced log axes alike.
        steps[i] = None
        values[i] = float(ax[len(ax) // 2])
        marks_list[i] = _slider_marks(ax)
        labels[i] = _progress_label(metric_keys[i])

    return (
        options, x_value, options, y_value,
        color_options, color_value,
        mins, maxs, steps, values, marks_list, labels,
    )


@app.callback(
    Output({"type": "merit-frozen-slider-row", "index": ALL}, "style"),
    Input("merit-x-axis-dropdown", "value"),
    Input("merit-y-axis-dropdown", "value"),
    Input("sweep-result-store", "data"),
    State({"type": "merit-frozen-slider-row", "index": ALL}, "id"),
    prevent_initial_call=False,
)
def update_merit_frozen_row_visibility(x_axis, y_axis, sweep_store, row_ids):
    metric_keys = []
    if sweep_store and isinstance(sweep_store, dict):
        metric_keys = list(sweep_store.get("metric_keys", []) or [])
    visible = {"display": "block"}
    hidden = {"display": "none"}
    out = []
    for entry in row_ids:
        i = entry.get("index", -1)
        if i is None or i < 0 or i >= len(metric_keys):
            out.append(hidden)
            continue
        key = metric_keys[i]
        if key == x_axis or key == y_axis:
            out.append(hidden)
        else:
            out.append(visible)
    return out


@app.callback(
    Output({"type": "merit-frozen-slider-value", "index": MATCH}, "children"),
    Input({"type": "merit-frozen-slider", "index": MATCH}, "value"),
    prevent_initial_call=False,
)
def display_merit_frozen_slider_value(value):
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.4g}"


@app.callback(
    Output("main-plot", "figure", allow_duplicate=True),
    Input("merit-mode-store", "data"),
    Input("merit-x-axis-dropdown", "value"),
    Input("merit-y-axis-dropdown", "value"),
    Input({"type": "merit-frozen-slider", "index": ALL}, "value"),
    Input("merit-color-by-dropdown", "value"),
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("fom-config-store", "data"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    prevent_initial_call=True,
)
def replot_on_merit_view_change(
    mode, x_axis, y_axis, frozen_slider_values, color_by,
    sweep_store, view_type, fom_config, threshold_enable,
    t0, t1, t2, t3, t4, tc0, tc1, tc2, tc3, tc4,
):
    if view_type != "merit":
        return dash.no_update
    return _build_merit_figure(
        sweep_store, fom_config, threshold_enable,
        [t0, t1, t2, t3, t4], [tc0, tc1, tc2, tc3, tc4],
        mode, x_axis, y_axis, frozen_slider_values, color_by,
    )


# ---------------------------------------------------------------------------
# Callback: dropdown choice → update frozen-axis-store and rebuild figure
#           from the cached sweep result (no re-sweep).
# ---------------------------------------------------------------------------

@app.callback(
    Output("frozen-axis-store", "data"),
    Output("main-plot", "figure", allow_duplicate=True),
    Output("interp-grid-store", "data", allow_duplicate=True),
    Output("frozen-slider", "min", allow_duplicate=True),
    Output("frozen-slider", "max", allow_duplicate=True),
    Output("frozen-slider", "value", allow_duplicate=True),
    Input("frozen-axis-dropdown", "value"),
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("cfg-output-metric", "value"),
    State("cfg-view-mode", "value"),
    prevent_initial_call=True,
)
def on_frozen_axis_change(frozen_idx, sweep_store, view_type, output_key, view_mode):
    if frozen_idx is None or sweep_store is None:
        raise dash.exceptions.PreventUpdate
    full = _get_sweep(sweep_store)
    if full is None or len(full.get("metric_keys", [])) != 3:
        raise dash.exceptions.PreventUpdate
    f_idx = frozen_idx if frozen_idx in (0, 1, 2) else 2

    permuted = permute_sweep_for_frozen(full, f_idx)
    out_key = output_key or "overall_fidelity"
    interp_grid = sweep_to_interp_grid(permuted, out_key)
    fs_cfg = frozen_slider_config(permuted)
    fig = build_figure(
        3, permuted, out_key, view_type=view_type,
        view_mode=view_mode or "absolute",
    )

    new_min = fs_cfg["min"] if fs_cfg else dash.no_update
    new_max = fs_cfg["max"] if fs_cfg else dash.no_update
    new_val = fs_cfg["default"] if fs_cfg else dash.no_update
    return f_idx, fig, interp_grid, new_min, new_max, new_val


# ---------------------------------------------------------------------------
# Clientside: after session load, resync window._sweepDirty so the next
# user-driven input increment starts above the post-load high-water mark.
# ---------------------------------------------------------------------------

app.clientside_callback(
    """function(tick) {
        if (typeof tick === 'number' && tick > 0) {
            window._sweepDirty = tick;
            window._lastProcessed = tick;
            window._sweepPending = false;
            // Stamp the load timestamp so the sweep-trigger gate ignores
            // any cascading dirty bumps that arrive while downstream
            // callbacks (sliders, dropdowns, summary) settle around the
            // newly-loaded values.
            window._loadCompleteAt = Date.now();
        }
        return window.dash_clientside.no_update;
    }""",
    Output("sweep-trigger", "data", allow_duplicate=True),
    Input("session-loaded-tick", "data"),
    prevent_initial_call=True,
)


# Reset suppress-cascade=False once the load grace period expires, so
# subsequent user-driven dropdown changes use registry defaults again.
# Driven by the same sweep-check interval that owns the trigger gate.
app.clientside_callback(
    """function(n) {
        var loadAt = window._loadCompleteAt || 0;
        if (loadAt > 0 && (Date.now() - loadAt) > 1500) {
            window._loadCompleteAt = 0;
            return false;
        }
        return window.dash_clientside.no_update;
    }""",
    Output("suppress-cascade", "data", allow_duplicate=True),
    Input("sweep-check", "n_intervals"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Custom QASM uploaded → disable the num_logical_qubits sweep axis.
# ---------------------------------------------------------------------------

# (The existing _CUSTOM_QASM_DISABLED_KEYS handler in run_sweep already
# silently drops circuit-shape axes when QASM is loaded, so the heat-map
# never tries to sweep them. The sweep-axis dropdown also lives in the
# left sidebar; greying out the option there is a polish item handled by
# the dropdown's options callback below.)


# ---------------------------------------------------------------------------
# Callback: Topology view — toggle visibility against the main Plotly graph
# ---------------------------------------------------------------------------


_TOPOLOGY_HIDDEN_STYLE = {
    "display": "none",
    "position": "absolute",
    "top": "0", "left": "0", "right": "0", "bottom": "0",
    "background": COLORS["bg"],
    "zIndex": 5,
}
_TOPOLOGY_VISIBLE_STYLE = {**_TOPOLOGY_HIDDEN_STYLE, "display": "block"}


@app.callback(
    Output("topology-view-container", "style"),
    Output("main-plot", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def _toggle_topology_view(view_type):
    main_style_visible = {"flex": "1", "minHeight": "0", "height": "100%"}
    main_style_hidden = {**main_style_visible, "visibility": "hidden"}
    if view_type == "topology":
        return _TOPOLOGY_VISIBLE_STYLE, main_style_hidden
    return _TOPOLOGY_HIDDEN_STYLE, main_style_visible


# ---------------------------------------------------------------------------
# Callback: Topology view — initialise per-axis sliders from sweep result
# ---------------------------------------------------------------------------


_FACET_ROW_HIDDEN = {"display": "none"}
_FACET_ROW_VISIBLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "marginBottom": "10px",
}


@app.callback(
    Output("topology-sweep-controls", "style"),
    Output("topology-facet-row", "style"),
    Output("topology-facet-label", "children"),
    Output("topology-facet-selector", "options"),
    Output("topology-facet-selector", "value"),
    *[Output({"type": "topology-axis-row", "index": i}, "style") for i in range(MAX_METRICS)],
    *[Output({"type": "topology-axis-slider", "index": i}, "max") for i in range(MAX_METRICS)],
    *[Output({"type": "topology-axis-slider", "index": i}, "value") for i in range(MAX_METRICS)],
    *[Output({"type": "topology-axis-slider", "index": i}, "marks") for i in range(MAX_METRICS)],
    *[Output({"type": "topology-axis-label", "index": i}, "children") for i in range(MAX_METRICS)],
    Input("sweep-result-store", "data"),
    Input("num-metrics-store", "data"),
    prevent_initial_call=False,
)
def _init_topology_sliders(sweep_store, num_metrics):
    """Reveal one slider per active sweep axis after a sweep.

    Visible count is clamped to ``num_metrics`` (the live left-sidebar
    axis count) so axes the user has just removed disappear from the
    topology panel even before the next sweep runs.  Adding axes keeps
    the new row hidden until the next sweep populates data for it.
    """
    panel_hidden = {"display": "none"}
    panel_visible = {
        "display": "block",
        "position": "absolute", "top": "8px", "left": "8px", "zIndex": 10,
        "background": COLORS["surface"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "6px",
        "padding": "8px 10px",
        "minWidth": "320px", "maxWidth": "420px", "fontSize": "11px",
    }

    full = _get_sweep(sweep_store) if sweep_store else None
    facet_opts = _facet_options(full)
    facet_keys = (full.get("facet_keys") if isinstance(full, dict) else None) or []
    facet_label = ""
    if facet_keys:
        cat_def = CAT_METRIC_BY_KEY.get(facet_keys[0])
        facet_label = cat_def.label if cat_def else facet_keys[0]
    # Facets store per_qubit_data inside each entry; non-faceted sweeps keep
    # it at the top level.  Use facet 0 as the canonical source for axis
    # metadata since all facets share the same numeric sweep grid.
    pq = _facet_per_qubit_data(full, 0 if facet_opts else None)
    if not pq or not pq.get("axis_keys"):
        return (
            panel_hidden,
            _FACET_ROW_HIDDEN, "", [], None,
            *([{"display": "none"}] * MAX_METRICS),
            *([1] * MAX_METRICS),
            *([0] * MAX_METRICS),
            *([{}] * MAX_METRICS),
            *([""] * MAX_METRICS),
        )

    axis_keys = pq["axis_keys"]
    axis_values = pq.get("axis_values", [])
    shape = pq.get("shape", [len(v) for v in axis_values])
    mark_style = {"fontSize": "10px", "color": COLORS["text_muted"]}
    visible = min(len(axis_keys), int(num_metrics or len(axis_keys)))
    row_styles: list = []
    maxes: list = []
    values: list = []
    marks_out: list = []
    labels: list = []
    for d in range(MAX_METRICS):
        if d < visible:
            row_styles.append({})
            sz = max(1, int(shape[d]))
            maxes.append(max(0, sz - 1))
            # Default each slider to the LAST index — the richest end of
            # the swept range (largest cores, longest T1, etc.). This is
            # the most informative starting frame for the topology view;
            # the cold-config snapshot is typically the lowest value (1
            # core, 1 comm qubit, …) which renders an uninformative
            # single-node graph.  The comm-qubits clamp callback then
            # walks the comm slider back to its valid max for the
            # current cores selection.
            values.append(max(0, sz - 1))
            # End-marks formatted as actual axis magnitudes (e.g. "1 µs", "10 ms").
            if sz > 1 and d < len(axis_values) and len(axis_values[d]) >= 2:
                lo = _human_axis_value(axis_keys[d], axis_values[d][0])
                hi = _human_axis_value(axis_keys[d], axis_values[d][-1])
                marks_out.append({
                    0: {"label": lo, "style": mark_style},
                    sz - 1: {"label": hi, "style": mark_style},
                })
            elif sz == 1 and d < len(axis_values) and axis_values[d]:
                only = _human_axis_value(axis_keys[d], axis_values[d][0])
                marks_out.append({0: {"label": only, "style": mark_style}})
            else:
                marks_out.append({})
            metric = METRIC_BY_KEY.get(axis_keys[d])
            label = metric.label if metric else axis_keys[d]
            cold_tag = " (cold)" if metric and metric.is_cold_path else ""
            labels.append(f"{label}{cold_tag}")
        else:
            row_styles.append({"display": "none"})
            maxes.append(1)
            values.append(0)
            marks_out.append({})
            labels.append("")

    facet_row_style = _FACET_ROW_VISIBLE if facet_opts else _FACET_ROW_HIDDEN
    facet_default = facet_opts[0]["value"] if facet_opts else None
    return (
        panel_visible,
        facet_row_style,
        facet_label,
        facet_opts,
        facet_default,
        *row_styles,
        *maxes,
        *values,
        *marks_out,
        *labels,
    )


# ---------------------------------------------------------------------------
# Callback: keep the per-axis value read-outs in sync with the slider value
# ---------------------------------------------------------------------------


@app.callback(
    *[Output({"type": "topology-axis-value", "index": i}, "value") for i in range(MAX_METRICS)],
    *[Input({"type": "topology-axis-slider", "index": i}, "value") for i in range(MAX_METRICS)],
    Input("topology-facet-selector", "value"),
    State("sweep-result-store", "data"),
    prevent_initial_call=False,
)
def _topology_axis_value_labels(*args):
    slider_vals = list(args[:MAX_METRICS])
    facet_idx = args[MAX_METRICS]
    sweep_store = args[MAX_METRICS + 1]
    full = _get_sweep(sweep_store) if sweep_store else None
    pq = _facet_per_qubit_data(full, facet_idx)
    if not pq or not pq.get("axis_keys"):
        return [""] * MAX_METRICS
    axis_keys = pq["axis_keys"]
    axis_values = pq.get("axis_values", [])
    out: list[str] = []
    for d in range(MAX_METRICS):
        if d < len(axis_keys) and d < len(axis_values):
            i = int(slider_vals[d] or 0)
            i = max(0, min(i, len(axis_values[d]) - 1))
            raw = axis_values[d][i]
            out.append(_human_axis_value(axis_keys[d], raw))
        else:
            out.append("")
    return out


# ---------------------------------------------------------------------------
# Callback: typed value in a topology-axis chip → snap the slider to the
# closest axis cell.  Lets the user jump to a specific magnitude (e.g. type
# "8" to land on cores=8) instead of having to drag through every step.
# ---------------------------------------------------------------------------


@app.callback(
    Output({"type": "topology-axis-slider", "index": MATCH}, "value", allow_duplicate=True),
    Input({"type": "topology-axis-value", "index": MATCH}, "value"),
    State({"type": "topology-axis-slider", "index": MATCH}, "value"),
    State({"type": "topology-axis-value", "index": MATCH}, "id"),
    State("topology-facet-selector", "value"),
    State("sweep-result-store", "data"),
    prevent_initial_call=True,
)
def _topology_axis_value_to_slider(
    typed, current_slider_val, axis_id, facet_idx, sweep_store,
):
    """Map a user-typed magnitude back to the closest swept-cell index.

    Guards against the programmatic chip writes the sibling label callback
    issues on every slider move: when the typed string is exactly what the
    chip *should* read for the current cell, we leave the slider alone.
    Without this, formatted log-axis chip values like "100 µs" round-trip
    through ``float()`` as ``100.0`` and would teleport the slider to cell 0.
    """
    if typed is None or typed == "":
        return dash.no_update
    full = _get_sweep(sweep_store) if sweep_store else None
    pq = _facet_per_qubit_data(full, facet_idx)
    if not pq or not pq.get("axis_keys"):
        return dash.no_update
    d = int(axis_id["index"])
    axis_keys = pq["axis_keys"]
    axis_values = pq.get("axis_values", [])
    if d >= len(axis_values) or not axis_values[d]:
        return dash.no_update
    current_idx = int(current_slider_val or 0)
    current_idx = max(0, min(current_idx, len(axis_values[d]) - 1))
    if d < len(axis_keys):
        expected = _human_axis_value(axis_keys[d], axis_values[d][current_idx])
        if str(typed).strip() == str(expected).strip():
            return dash.no_update
    # Strip trailing unit (e.g. "10 ns", "1.5 µs", "200 MHz") — the value
    # chip formats with units, so the user might re-paste one back in.
    raw = str(typed).strip().split()[0] if str(typed).strip() else ""
    if not raw:
        return dash.no_update
    try:
        target = float(raw)
    except ValueError:
        return dash.no_update
    diffs = [abs(float(v) - target) for v in axis_values[d]]
    closest = min(range(len(diffs)), key=diffs.__getitem__)
    if closest == current_idx:
        return dash.no_update
    return closest


# ---------------------------------------------------------------------------
# Topology scrub clamp: the sweep grid encodes its own validity through
# NaN ``overall_fidelity`` cells (B>K, infeasible architecture, no nc
# satisfies the fixpoint, …). When the user drags a scrub slider into
# a NaN cell, the rebuild upstream raises PreventUpdate — leaving the
# slider visually pointing at an invalid cell while the cytoscape shows
# stale data. Walk the just-moved axis back to the nearest valid index
# instead.
#
# This subsumes the older B≤K-only clamp: B>K is just one of many ways a
# cell becomes NaN, and the grid's NaN signal is the single source of
# truth that doesn't need rule-by-rule coding.
# ---------------------------------------------------------------------------


def _cell_overall_fidelity(grid, idx_tuple):
    """Index ``grid`` (nested list / structured ndarray / flat list)
    by ``idx_tuple`` and return the cell's ``overall_fidelity``, or
    ``None`` if the lookup fails or the value is missing."""
    import numpy as _np
    try:
        v = grid
        for k in idx_tuple:
            v = v[k]
        if isinstance(v, dict):
            return v.get("overall_fidelity")
        if isinstance(v, _np.void):
            return float(v["overall_fidelity"])
        return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _is_finite(x) -> bool:
    import math
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


@app.callback(
    Output({"type": "topology-axis-slider", "index": ALL}, "value", allow_duplicate=True),
    Input({"type": "topology-axis-slider", "index": ALL}, "value"),
    State("sweep-result-store", "data"),
    State("topology-facet-selector", "value"),
    prevent_initial_call=True,
)
def _topology_clamp_to_valid_cells(slider_vals, sweep_store, facet_idx):
    """Snap the just-moved scrub slider back to the nearest non-NaN cell.

    The sweep grid's ``overall_fidelity`` is NaN exactly for the cells
    the engine refused to compute (B>K, infeasible deduction, etc.).
    On every slider change, build the resulting cell index tuple,
    check the grid, and walk the moved axis backward (then forward as
    a fallback) until a finite cell is found.
    """
    no = dash.no_update
    n = len(slider_vals)
    no_change = [no] * n

    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "topology-axis-slider":
        return no_change
    moved_d = triggered.get("index")
    if not isinstance(moved_d, int) or moved_d >= n:
        return no_change

    full = _get_sweep(sweep_store) if sweep_store else None
    if not isinstance(full, dict):
        return no_change
    pq = _facet_per_qubit_data(full, facet_idx)
    if not pq or not pq.get("axis_keys"):
        return no_change
    axis_keys = pq["axis_keys"]
    shape = pq.get("shape") or [
        len(v) for v in pq.get("axis_values", [])
    ]
    if moved_d >= len(shape):
        return no_change

    # Resolve the grid for this facet (top-level for non-faceted sweeps,
    # otherwise inside the chosen facet entry).
    facets = full.get("facets")
    if facets:
        i = int(facet_idx or 0)
        if i < 0 or i >= len(facets):
            i = 0
        grid = facets[i].get("grid")
    else:
        grid = full.get("grid")
    if grid is None:
        return no_change

    cell_idx = [
        max(0, min(int(slider_vals[d] or 0), max(0, int(shape[d]) - 1)))
        for d in range(len(axis_keys))
    ]
    fid = _cell_overall_fidelity(grid, tuple(cell_idx))
    if _is_finite(fid):
        return no_change

    # Walk the just-moved axis backward, then forward, until we land on
    # a finite cell. This keeps every other axis fixed so the user's
    # mental model (`I just moved K, only K should jump`) holds.
    original = cell_idx[moved_d]
    found = None
    for new_idx in range(original - 1, -1, -1):
        cell_idx[moved_d] = new_idx
        if _is_finite(_cell_overall_fidelity(grid, tuple(cell_idx))):
            found = new_idx
            break
    if found is None:
        for new_idx in range(original + 1, int(shape[moved_d])):
            cell_idx[moved_d] = new_idx
            if _is_finite(_cell_overall_fidelity(grid, tuple(cell_idx))):
                found = new_idx
                break
    if found is None or found == original:
        return no_change
    out = list(no_change)
    out[moved_d] = found
    return out


# ---------------------------------------------------------------------------
# Sweep-axis range sliders: same B≤K rule, applied to whichever metric-
# dropdown slot currently holds buffer_qubits. When comm's upper bound
# moves, buffer's upper bound follows so the user can't paint a buffer
# range whose high end is unreachable.
# ---------------------------------------------------------------------------


@app.callback(
    *[Output(f"metric-slider-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
    *[Input(f"metric-slider-{i}", "value") for i in range(MAX_METRICS)],
    *[State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    prevent_initial_call=True,
)
def _sweep_axis_clamp_buffer_to_comm(*args):
    """If both ``communication_qubits`` and ``buffer_qubits`` are on
    sweep axes, force buffer's upper bound ≤ comm's upper bound."""
    no = dash.no_update
    sliders = list(args[:MAX_METRICS])
    dropdowns = list(args[MAX_METRICS:2 * MAX_METRICS])
    no_change = [no] * MAX_METRICS

    try:
        d_K = dropdowns.index("communication_qubits")
        d_B = dropdowns.index("buffer_qubits")
    except ValueError:
        return no_change
    sv_K = sliders[d_K]
    sv_B = sliders[d_B]
    if not (isinstance(sv_K, (list, tuple)) and len(sv_K) == 2):
        return no_change
    if not (isinstance(sv_B, (list, tuple)) and len(sv_B) == 2):
        return no_change
    K_hi = float(sv_K[1])
    B_lo, B_hi = float(sv_B[0]), float(sv_B[1])
    if B_hi <= K_hi:
        return no_change
    # Clamp B's upper bound down to K's upper bound; if that would push
    # B_lo above the new B_hi, drop B_lo to match.
    new_B_hi = K_hi
    new_B_lo = min(B_lo, new_B_hi)
    out = list(no_change)
    out[d_B] = [new_B_lo, new_B_hi]
    return out


# ---------------------------------------------------------------------------
# Callback: Topology view — rebuild Cytoscape elements (structure + overlay)
# ---------------------------------------------------------------------------


@app.callback(
    Output("topology-cyto", "elements"),
    Input("cfg-qubits-per-core", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-communication-qubits", "value"),
    Input("cfg-buffer-qubits", "value"),
    Input("cfg-topology", "value"),
    Input("cfg-intracore-topology", "value"),
    Input("cfg-num-logical-qubits", "value"),
    Input("cfg-pin-axis", "data"),
    Input("topology-overlay-metric", "value"),
    Input("topology-facet-selector", "value"),
    Input({"type": "topology-axis-slider", "index": ALL}, "value"),
    Input("sweep-result-store", "data"),
    prevent_initial_call=False,
)
def _rebuild_topology_graph(
    qubits_per_core, num_cores, comm_qubits, buffer_qubits, topology,
    intracore_topology, num_logical_qubits, pin_axis,
    overlay_metric, facet_idx, axis_slider_vals, sweep_store,
):
    # Logical-first: derive the unpinned axis through the resolver so the
    # topology view matches the right-sidebar's derived-value badges
    # exactly. Without this, switching pin from cores to qpc leaves the
    # hidden cores slider at a stale value (e.g. 8) and the picture shows
    # 8 cores while the badge says "Cores: 1 (derived)".
    from quadris.dse.config import _resolve_architecture
    cfg = {
        "num_logical_qubits": int(num_logical_qubits or 16),
        "num_cores": int(num_cores or 1),
        "qubits_per_core": int(qubits_per_core or 16),
        "communication_qubits": int(comm_qubits or 1),
        "buffer_qubits": int(buffer_qubits or 1),
        "topology_type": topology or "ring",
        "pin_axis": pin_axis or "cores",
    }
    res = _resolve_architecture(cfg)
    if res["feasible"]:
        eff_cores = int(cfg["num_cores"])
        eff_phys = int(cfg["num_qubits"])
    else:
        eff_cores = max(1, int(num_cores or 1))
        eff_phys = eff_cores * max(1, int(qubits_per_core or 16))

    def _build_default_elements() -> list:
        return build_topology_elements(
            num_cores=eff_cores,
            num_qubits=eff_phys,
            communication_qubits=comm_qubits or 1,
            topology=topology or "ring",
            intracore_topology=intracore_topology or "all_to_all",
            buffer_qubits=buffer_qubits or 1,
        )

    def _clear_fidelity(els: list) -> list:
        # Cytoscape diff-merges node data on element updates, so a previous
        # overlay's ``fidelity`` field can persist on a node that we no
        # longer want coloured.  Setting the field to ``None`` makes the
        # ``node[fidelity]`` stylesheet selector miss, falling back to the
        # default ``.data`` grey.
        for el in els:
            d = el.get("data", {})
            if d.get("qtype") == "data":
                d["fidelity"] = None
                d["fid_algorithmic"] = None
                d["fid_routing"] = None
                d["fid_coherence"] = None
                d["fid_overall"] = None
                d["logical_qubit"] = None
        return els

    full = _get_sweep(sweep_store) if sweep_store else None
    pq = _facet_per_qubit_data(full, facet_idx)

    if not pq or not pq.get("axis_keys"):
        # No active sweep cell: graph follows the live right-sidebar config.
        return _clear_fidelity(_build_default_elements())

    cell_idx = tuple(int(v or 0) for v in axis_slider_vals[: len(pq["axis_keys"])])

    # Per-group rule check: if the swept axes include both K and B and
    # the current cell has B>K, the architecture is infeasible (white
    # cell in the heat-map, no per-qubit data). Preserve the previous
    # render rather than fall back to the right-panel cfg-derived
    # default — that default doesn't match the swept cell shape and
    # would flicker the device picture each time the slider transits
    # an infeasible row.
    axis_keys = pq["axis_keys"]
    axis_values = pq.get("axis_values", [])
    try:
        d_K = axis_keys.index("communication_qubits")
        d_B = axis_keys.index("buffer_qubits")
        if (d_K < len(cell_idx) and d_B < len(cell_idx)
                and d_K < len(axis_values) and d_B < len(axis_values)):
            K_val = float(axis_values[d_K][cell_idx[d_K]])
            B_val = float(axis_values[d_B][cell_idx[d_B]])
            if B_val > K_val:
                raise dash.exceptions.PreventUpdate
    except (ValueError, IndexError):
        pass

    # Memoize per (sweep token, facet, cell_idx). The sweep already paid the
    # compile cost for every cell during the run; keeping the per-qubit
    # output here means scrubbing through the same cells never re-enters
    # the engine. LRU eviction caps memory at _PER_CELL_CACHE_MAX entries.
    cache_key = _per_cell_cache_key(sweep_store, facet_idx, cell_idx)
    cell = _per_cell_cache_get(cache_key) if cache_key is not None else None
    if cell is None:
        try:
            cell = _compute_per_qubit_for_cell(full, cell_idx, facet_idx=facet_idx)
        except Exception:
            cell = None
        if cell is not None and cache_key is not None:
            _per_cell_cache_put(cache_key, cell)
    if cell is None or cell.get("infeasible"):
        # Same rationale as the B>K guard above.
        raise dash.exceptions.PreventUpdate

    # When a sweep cell is in scope, the architecture has to follow the
    # *cell's* cold config, not the right-sidebar — otherwise scrubbing a
    # cold-path axis (comm_qubits, num_cores, …) leaves the graph stuck on
    # the right-panel snapshot. The cell already carries the post-clamp
    # values that the engine actually used for compilation.
    elements = build_topology_elements(
        num_cores=int(cell.get("num_cores", eff_cores)),
        num_qubits=int(cell.get("num_physical", eff_phys)),
        communication_qubits=int(cell.get("communication_qubits", comm_qubits or 1)),
        topology=cell.get("topology_type", topology or "ring"),
        intracore_topology=cell.get(
            "intracore_topology", intracore_topology or "all_to_all"
        ),
        buffer_qubits=int(cell.get("buffer_qubits", buffer_qubits or 1)),
    )

    # Always clear first so a shrinking ``nlog`` (e.g. switching to a facet
    # whose num_logical_qubits is smaller) doesn't leave the tail
    # half-coloured.
    _clear_fidelity(elements)

    # Apply per-logical fidelity to the live structure's data-qubit slots
    # (in id order).  We tolerate a structural mismatch — colour as many
    # data slots as we have logical-qubit fidelities for, leave the rest
    # at the default.  All four metrics are stashed on each coloured node
    # so the hover panel shows the full breakdown.
    metric = overlay_metric or "overall_fidelity"
    fid_alg = _per_logical_fidelity(cell, "algorithmic_fidelity")
    fid_rt = _per_logical_fidelity(cell, "routing_fidelity")
    fid_coh = _per_logical_fidelity(cell, "coherence_fidelity")
    fid_overall = _per_logical_fidelity(cell, "overall_fidelity")
    metric_to_arr = {
        "algorithmic_fidelity": fid_alg,
        "routing_fidelity": fid_rt,
        "coherence_fidelity": fid_coh,
        "overall_fidelity": fid_overall,
    }
    selected = metric_to_arr.get(metric, fid_overall)
    nlog = min(int(cell.get("num_logical_qubits", 0)), len(selected))

    data_count = 0
    for el in elements:
        d = el.get("data", {})
        nid = d.get("id")
        if not nid or "source" in d:
            continue
        if d.get("qtype") != "data":
            continue
        if data_count < nlog:
            d["fidelity"] = float(selected[data_count])
            d["fid_algorithmic"] = float(fid_alg[data_count])
            d["fid_routing"] = float(fid_rt[data_count])
            d["fid_coherence"] = float(fid_coh[data_count])
            d["fid_overall"] = float(fid_overall[data_count])
            d["logical_qubit"] = data_count
        data_count += 1
    return elements


# ---------------------------------------------------------------------------
# Callback: Topology view — Re-layout button re-heats the cose simulation
# ---------------------------------------------------------------------------


@app.callback(
    Output("topology-cyto", "layout"),
    Input("topology-view-relayout", "n_clicks"),
    Input("view-type-store", "data"),
    Input("topology-cyto", "elements"),
    prevent_initial_call=False,
)
def _relayout_topology(n_clicks, view_type, _elements):
    # Re-apply preset layout (uses the explicit per-node positions) when:
    #   - the user clicks "Re-layout"
    #   - the user switches into the Topology view (a layout running on a
    #     0×0 hidden canvas leaves nodes bunched in the corner)
    #   - the graph elements changed (node count etc.) AND the view is active
    trig = ctx.triggered_id
    if trig is None:
        raise dash.exceptions.PreventUpdate
    if trig == "topology-cyto" and view_type != "topology":
        raise dash.exceptions.PreventUpdate
    if trig == "view-type-store" and view_type != "topology":
        raise dash.exceptions.PreventUpdate
    return {
        "name": "preset",
        "fit": True,
        "padding": 40,
    }


# ---------------------------------------------------------------------------
# Clientside: after switching into the Topology view, call cy.fit() once the
# cose layout has settled.  Without this, the layout runs while the canvas was
# hidden (0×0) and the resulting graph clusters in the corner at zoom=1.
# ---------------------------------------------------------------------------


app.clientside_callback(
    """function(view_type, _elements) {
        if (view_type !== 'topology') {
            return window.dash_clientside.no_update;
        }
        const fit = (attempt) => {
            const cyEl = document.getElementById('topology-cyto');
            const cy = cyEl && cyEl._cyreg ? cyEl._cyreg.cy : null;
            if (!cy && attempt < 30) {
                setTimeout(() => fit(attempt + 1), 100);
                return;
            }
            if (!cy) { return; }
            try {
                cy.resize();
                cy.fit(undefined, 30);
            } catch (e) { /* ignore */ }
        };
        // Fit once the layout pass completes (cose with numIter=1500 finishes
        // synchronously after a tick, but we re-fit a couple of times to
        // catch any post-layout reflow).
        setTimeout(() => fit(0), 50);
        setTimeout(() => fit(0), 600);
        setTimeout(() => fit(0), 1500);
        return window.dash_clientside.no_update;
    }""",
    Output("topology-cyto", "id"),  # dummy output (id never changes)
    Input("view-type-store", "data"),
    Input("topology-cyto", "elements"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Callback: Topology view — fidelity-overlay colour legend visibility/title
# ---------------------------------------------------------------------------


_OVERLAY_METRIC_LABELS = {
    "overall_fidelity": "Overall fidelity",
    "algorithmic_fidelity": "Algorithmic fidelity",
    "routing_fidelity": "Routing fidelity",
    "coherence_fidelity": "Coherence fidelity",
}


@app.callback(
    Output("topology-view-legend", "style"),
    Output("topology-legend-title", "children"),
    Input("topology-overlay-metric", "value"),
    State("topology-view-legend", "style"),
    prevent_initial_call=False,
)
def _topology_legend_title(overlay_metric, current_style):
    base = dict(current_style or {})
    base["display"] = "block"
    title = _OVERLAY_METRIC_LABELS.get(overlay_metric or "overall_fidelity", "Fidelity")
    return base, title


# ---------------------------------------------------------------------------
# Callback: Topology view — hover info readout
# ---------------------------------------------------------------------------


@app.callback(
    Output("topology-view-hover", "children"),
    Input("topology-cyto", "mouseoverNodeData"),
    prevent_initial_call=False,
)
def _topology_hover_label(node_data):
    if not node_data:
        return "Hover a node to inspect."
    core = node_data.get("core", "?")
    qtype = node_data.get("qtype", "?")
    label = node_data.get("label", "")
    descriptor = {"comm": "comm", "buffer": "buffer", "data": "data"}.get(qtype, "data")
    head = f"core c{core}  ·  {descriptor} qubit {label}"

    if qtype != "data" or "fid_overall" not in node_data:
        return head

    rows = [
        ("Overall",     node_data.get("fid_overall")),
        ("Algorithmic", node_data.get("fid_algorithmic")),
        ("Routing",     node_data.get("fid_routing")),
        ("Coherence",   node_data.get("fid_coherence")),
    ]
    metric_lines = [
        html.Div(
            style={"display": "flex", "gap": "8px"},
            children=[
                html.Span(name, style={
                    "width": "82px",
                    "color": COLORS["text_muted"],
                }),
                html.Span(_fmt_fid(val), style={
                    "color": COLORS["text"],
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                }),
            ],
        )
        for name, val in rows
    ]
    logical_q = node_data.get("logical_qubit")
    sub_head = (
        f"logical q{logical_q}" if logical_q is not None else "fidelity (cell)"
    )
    return [
        html.Div(head, style={"fontWeight": "600", "marginBottom": "2px"}),
        html.Div(sub_head, style={
            "color": COLORS["text_muted"],
            "marginBottom": "4px",
            "fontSize": "10px",
        }),
        *metric_lines,
    ]


# ---------------------------------------------------------------------------
# Register extracted callback modules (after the inline @app.callback chain
# above has wired up all primary outputs).
# ---------------------------------------------------------------------------

from gui.callbacks import register_all as _register_callbacks

_register_callbacks(app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console-script entry point (`quadris-dse`).

    Multi-user knobs:
      - debug=False: no hot-reload (which breaks under concurrent requests)
        and no traceback leaks to the browser.
      - host="127.0.0.1": only cloudflared (or local browser) can reach the
        port; nothing on the LAN can connect directly.
      - threaded=True: the Flask dev server serves each request on its own
        thread, so polling endpoints stay responsive while a sweep runs.
      - Override host via QUADRIS_HOST=0.0.0.0 if you need direct LAN access.
    """
    host = os.environ.get("QUADRIS_HOST", "127.0.0.1")
    port = int(os.environ.get("QUADRIS_PORT", "8050"))
    print(f"quadris DSE GUI starting at http://{host}:{port}")
    app.run(debug=False, host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
