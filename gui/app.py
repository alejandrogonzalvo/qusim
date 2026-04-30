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
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from gui.components import (
    COLORS,
    FEEDBACK_COLORS,
    _linear_marks,
    _log_marks,
    _tooltip_cfg,
    build_topology_elements,
    make_add_metric_button,
    make_custom_qasm_help_modal,
    make_fixed_config_panel,
    make_merit_controls,
    make_merit_view_controls,
    make_performance_panel,
    make_metric_selector,
    make_topology_view_panel,
    make_view_tab_bar,
)
from gui.constants import (
    ANALYSIS_TABS,
    CAT_METRIC_BY_KEY,
    CATEGORICAL_METRICS,
    MAX_SWEEP_AXES,
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

_engine = DSEEngine()
sweep_lock = threading.Lock()
MAX_METRICS = MAX_SWEEP_AXES  # Alias for the centralised cap

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
# Sized for ~3 concurrent users keeping their last few sweeps in memory.
_SWEEP_CACHE_MAX = 12
_sweep_token_counter = 0
_sweep_cache_lock = threading.Lock()

_session_load_counter = 0
_session_load_lock = threading.Lock()


def _next_session_hw(current_processed: int) -> int:
    """Return a value strictly greater than any plausible current sweep-dirty."""
    global _session_load_counter
    with _session_load_lock:
        _session_load_counter += 10000
        return (current_processed or 0) + _session_load_counter


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
    if "facet_keys" in data:
        slim["facet_keys"] = data["facet_keys"]
    # Light per-axis metadata for the topology-view sliders (axis keys + which
    # ones are cold).  The heavy ``cold_config``/``fixed_noise`` snapshots
    # stay server-side under the same token.
    pq = data.get("per_qubit_data")
    if isinstance(pq, dict):
        slim["per_qubit_meta"] = {
            "axis_keys": list(pq.get("axis_keys", [])),
            "axis_values": [list(v) for v in pq.get("axis_values", [])],
            "shape": list(pq.get("shape", [])),
        }
    return slim


def _facet_options(sweep_data: dict | None) -> list[dict]:
    """Build dropdown options for the topology facet selector.

    Returns one option per facet of a faceted sweep (label = the facet's
    human-readable categorical value, e.g. "HQA + Sabre"). Empty list for
    non-faceted sweeps so the caller can hide the row.
    """
    if not isinstance(sweep_data, dict):
        return []
    facets = sweep_data.get("facets")
    if not facets:
        return []
    facet_keys = sweep_data.get("facet_keys") or []
    opts: list[dict] = []
    for i, fc in enumerate(facets):
        label_dict = fc.get("label") or {}
        # Build "Routing: HQA + Sabre · Circuit: QFT" style label.
        parts: list[str] = []
        for k in facet_keys:
            v = label_dict.get(k)
            if v is None:
                continue
            cat_def = CAT_METRIC_BY_KEY.get(k)
            disp = v
            if cat_def:
                disp = next(
                    (o["label"] for o in cat_def.options if o["value"] == v), v
                )
            parts.append(str(disp))
        opts.append({"label": " · ".join(parts) or f"Facet {i}", "value": i})
    return opts


def _facet_per_qubit_data(
    sweep_data: dict | None, facet_idx: int | None,
) -> dict | None:
    """Return the per_qubit_data dict for the selected facet (or top-level).

    Faceted sweeps stash per_qubit_data inside each facet entry; non-faceted
    sweeps keep it at the top level.
    """
    if not isinstance(sweep_data, dict):
        return None
    facets = sweep_data.get("facets")
    if facets:
        i = int(facet_idx or 0)
        if i < 0 or i >= len(facets):
            i = 0
        return facets[i].get("per_qubit_data")
    return sweep_data.get("per_qubit_data")


_PER_CELL_CACHE_MAX = 256
_per_cell_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_per_cell_cache_lock = threading.Lock()


def _per_cell_cache_key(
    sweep_store: dict | None, facet_idx: int | None, cell_idx: tuple,
) -> tuple | None:
    """Build a cache key from the sweep token + facet + cell index.

    Returns None when there is no token (e.g. no sweep loaded yet) so the
    caller can fall back to direct computation without caching.
    """
    if not isinstance(sweep_store, dict):
        return None
    token = sweep_store.get("token")
    if not token:
        return None
    return (token, facet_idx, tuple(cell_idx))


def _per_cell_cache_get(key: tuple) -> dict | None:
    with _per_cell_cache_lock:
        hit = _per_cell_cache.get(key)
        if hit is not None:
            _per_cell_cache.move_to_end(key)
        return hit


def _per_cell_cache_put(key: tuple, value: dict) -> None:
    with _per_cell_cache_lock:
        _per_cell_cache[key] = value
        _per_cell_cache.move_to_end(key)
        while len(_per_cell_cache) > _PER_CELL_CACHE_MAX:
            _per_cell_cache.popitem(last=False)


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
    cfg["num_cores"] = min(int(cfg["num_cores"]), int(cfg["num_qubits"]))
    if "communication_qubits" in cfg:
        qpc = max(1, int(cfg["num_qubits"]) // max(1, int(cfg["num_cores"])))
        cfg["communication_qubits"] = max(
            1, min(int(cfg["communication_qubits"] or 1), math.isqrt(qpc))
        )
    if "num_logical_qubits" in cfg:
        cfg["num_logical_qubits"] = max(
            2, min(int(cfg["num_logical_qubits"]), int(cfg["num_qubits"]))
        )

    cached = _engine.run_cold(**cfg, noise=hot_noise)
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

# Per-user progress state — keyed by browser session id (sid).
# Sweeps are serialised by ``sweep_lock``, but multiple users may poll
# concurrently, so each user only sees the progress of the sweep they kicked
# off. The active sid for the current sweep is stored in a threadlocal that
# the sweep callback sets before running, so ``_update_progress`` (called
# deep inside the sweep machinery) can route writes to the right slot.
_sweep_progress_by_sid: dict[str, dict] = {}
_sweep_progress_lock = threading.Lock()
_sweep_progress_tls = threading.local()


def _progress_label(k: str) -> str:
    if k in METRIC_BY_KEY:
        return METRIC_BY_KEY[k].label
    if k in CAT_METRIC_BY_KEY:
        return CAT_METRIC_BY_KEY[k].label
    return k


def _set_progress(sid: str, payload: dict) -> None:
    with _sweep_progress_lock:
        if payload.get("running"):
            _sweep_progress_by_sid[sid] = payload
        else:
            _sweep_progress_by_sid.pop(sid, None)


def _get_progress(sid: str) -> dict:
    with _sweep_progress_lock:
        return _sweep_progress_by_sid.get(sid) or {"running": False}


def _update_progress(p: SweepProgress) -> None:
    """Progress callback passed into sweep methods."""
    sid = getattr(_sweep_progress_tls, "sid", None)
    if not sid:
        return
    _set_progress(sid, {
        "running": True,
        "completed": p.completed,
        "total": p.total,
        "percentage": p.percentage,
        "current_params": {
            _progress_label(k): v for k, v in p.current_params.items()
        },
        "cold_completed": p.cold_completed,
        "cold_total": p.cold_total,
        "phase": "sweeping",
    })


@server.route("/api/progress")
def _api_progress():
    import json
    from flask import request
    sid = request.args.get("sid", "")
    payload = _get_progress(sid) if sid else {"running": False}
    return json.dumps(payload), 200, {"Content-Type": "application/json"}


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
                className="quadris-lockup",
                children=[
                    html.Img(
                        src="/assets/quadris-mark.svg",
                        className="quadris-mark",
                        style={"width": "22px", "height": "22px"},
                        alt="Quadris",
                    ),
                    html.Span("quadris", className="quadris-wordmark"),
                    html.Span("DSE Explorer", className="quadris-submark"),
                ],
            ),
            html.Div(
                style={
                    "flex": "1",
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "baseline",
                    "justifyContent": "center",
                    "gap": "14px",
                    "minWidth": "0",
                },
                children=[
                    dcc.Input(
                        id="session-name",
                        type="text",
                        value="",
                        placeholder="Untitled session",
                        debounce=False,
                        spellCheck=False,
                        style={
                            "background": "transparent",
                            "border": f"1px dashed {COLORS['border']}",
                            "outline": "none",
                            "color": COLORS["text"],
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "textAlign": "center",
                            "padding": "3px 10px",
                            "borderRadius": "4px",
                            "width": "260px",
                            "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        id="status-bar",
                        children="Ready",
                        style={
                            "fontSize": "11px",
                            "color": COLORS["text_muted"],
                            "whiteSpace": "nowrap",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "flex": "0 1 auto",
                            "minWidth": "0",
                        },
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    html.Button(
                        "Save",
                        id="save-btn",
                        className="ghost-btn",
                        n_clicks=0,
                    ),
                    dcc.Upload(
                        id="session-upload",
                        children=html.Span(
                            "Load",
                            className="ghost-btn",
                            role="button",
                            tabIndex=0,
                            **{"aria-label": "Load session from file"},
                        ),
                        multiple=False,
                        accept=".gz,.json",
                        style_active={},
                    ),
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
                        className="run-btn",
                        n_clicks=0,
                        style={"display": "none"},
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
                        className="ghost-btn",
                        n_clicks=0,
                        style={"flex": "1", "padding": "6px"},
                    ),
                    html.Button(
                        "− Remove",
                        id="remove-metric-btn",
                        className="ghost-btn ghost-btn--dashed",
                        n_clicks=0,
                        style={"flex": "1", "padding": "6px"},
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
                        className="ghost-btn",
                        n_clicks=0,
                        style={
                            "borderRadius": "4px",
                            "padding": "4px 12px",
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
                    make_merit_view_controls(),
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
                    make_topology_view_panel(),
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
                            html.Div(
                                style={"width": "170px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="frozen-axis-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=[],
                                    value=2,
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
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
            html.Div(
                id="pareto-axis-container",
                style={"display": "none", "padding": "4px 16px 8px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "fontSize": "11px",
                            "color": COLORS["text_muted"],
                        },
                        children=[
                            html.Span("X axis", style={"flexShrink": "0"}),
                            html.Div(
                                style={"width": "180px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="pareto-x-axis-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=OUTPUT_METRICS,
                                    value="total_epr_pairs",
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                            html.Span(
                                "Y axis",
                                style={"flexShrink": "0", "marginLeft": "12px"},
                            ),
                            html.Div(
                                style={"width": "180px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="pareto-y-axis-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=OUTPUT_METRICS,
                                    value="overall_fidelity",
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            # Mode toggle for Parameter Importance: range-based (global
            # spread) vs gradient-based sensitivity (local rate of change
            # at the operating point). Visibility toggled by view_type.
            html.Div(
                id="importance-mode-container",
                style={"display": "none", "padding": "4px 16px 8px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "fontSize": "11px",
                            "color": COLORS["text_muted"],
                        },
                        children=[
                            html.Span("Importance metric", style={"flexShrink": "0"}),
                            html.Div(
                                style={"width": "320px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="importance-mode-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=[
                                        {"value": "range", "label": "Range  (max − min of mean projection)"},
                                        {"value": "sensitivity", "label": "Sensitivity  ⟨|∂F/∂x_i|⟩  (gradient)"},
                                    ],
                                    value="range",
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                            html.Span(
                                "Range = global structure;  Sensitivity = local rate of change.",
                                style={
                                    "flexShrink": "1",
                                    "marginLeft": "12px",
                                    "fontStyle": "italic",
                                    "color": COLORS["text_muted"],
                                },
                            ),
                        ],
                    ),
                ],
            ),
            # Mode toggle for the Correlation matrix: Spearman ρ
            # (input × output rank correlation, the historical view) vs
            # mean |∂²F/∂x_i ∂x_j| (interaction strength between axis pairs).
            html.Div(
                id="correlation-mode-container",
                style={"display": "none", "padding": "4px 16px 8px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "fontSize": "11px",
                            "color": COLORS["text_muted"],
                        },
                        children=[
                            html.Span("Matrix", style={"flexShrink": "0"}),
                            html.Div(
                                style={"width": "340px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="correlation-mode-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=[
                                        {"value": "spearman", "label": "Spearman ρ  (rank correlation, signed)"},
                                        {"value": "interaction", "label": "Interaction  ⟨|∂²F/∂x_i ∂x_j|⟩  (axis pairs)"},
                                    ],
                                    value="spearman",
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                            html.Span(
                                "Spearman compares input × output;  Interaction compares input × input.",
                                style={
                                    "flexShrink": "1",
                                    "marginLeft": "12px",
                                    "fontStyle": "italic",
                                    "color": COLORS["text_muted"],
                                },
                            ),
                        ],
                    ),
                ],
            ),
            # Trajectory selector for the Elasticity Comparison view — picks
            # which sweep axis lives on the X-axis (every other axis becomes
            # one curve). Options are populated from the active sweep's
            # metric_keys; visibility toggled by view_type.
            html.Div(
                id="elasticity-axis-container",
                style={"display": "none", "padding": "4px 16px 8px"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "fontSize": "11px",
                            "color": COLORS["text_muted"],
                        },
                        children=[
                            html.Span(
                                "Trajectory axis",
                                style={"flexShrink": "0"},
                            ),
                            html.Div(
                                style={"width": "220px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="elasticity-trajectory-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                            html.Span(
                                "Every other sweep axis becomes one curve.",
                                style={
                                    "flexShrink": "1",
                                    "marginLeft": "12px",
                                    "fontStyle": "italic",
                                    "color": COLORS["text_muted"],
                                },
                            ),
                        ],
                    ),
                ],
            ),
            make_merit_controls(),
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
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={"padding": "14px 12px 0"},
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
                ],
            ),
            html.Div(
                id="fixed-config-container",
                className="config-scroll",
                style={
                    "flex": "1 1 auto",
                    "overflow": "auto",
                    "minHeight": "80px",
                    "padding": "0 12px 12px",
                },
                children=[make_fixed_config_panel()],
            ),
            # Sweep Budget — panel-level footer, always visible across tabs.
            make_performance_panel(),
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
        # Per-tab session id, populated by a clientside callback on load.
        # Used to route sweep progress back to the user who started it.
        dcc.Store(id="user-sid", data=None, storage_type="session"),
        dcc.Interval(id="sid-init", interval=200, max_intervals=1),
        dcc.Store(id="num-metrics-store", data=3, storage_type="memory"),
        dcc.Store(id="sweep-result-store", data=None, storage_type="memory"),
        dcc.Store(id="view-type-store", data="isosurface", storage_type="memory"),
        dcc.Store(id="num-thresholds-store", data=3, storage_type="memory"),
        dcc.Store(id="sweep-dirty", data=1, storage_type="memory"),
        dcc.Store(id="sweep-processed", data=0, storage_type="memory"),
        dcc.Store(id="sweep-trigger", data=0, storage_type="memory"),
        dcc.Store(id="interp-grid-store", data=None, storage_type="memory"),
        dcc.Store(id="frozen-axis-store", data=2, storage_type="memory"),
        dcc.Store(id="session-loaded-tick", data=0, storage_type="memory"),
        dcc.Store(id="suppress-cascade", data=False, storage_type="memory"),
        dcc.Store(
            id="fom-config-store",
            data=DEFAULT_FOM.to_dict(),
            storage_type="memory",
        ),
        dcc.Store(id="merit-mode-store", data="heatmap", storage_type="memory"),
        dcc.Store(id="merit-frozen-values-store", data={}, storage_type="memory"),
        # Sink for the clientside callback that pushes (view-type, merit-mode)
        # into window.qusimUpdatePlotHelp — drives the modebar "?" popup text.
        dcc.Store(id="plot-help-sink", data=0, storage_type="memory"),
        dcc.Download(id="session-download"),
        dcc.Interval(id="sweep-check", interval=16, n_intervals=0),
        # Custom-circuit upload state.  When ``qasm`` is non-empty the sweep
        # callback bypasses ``circuit_type`` and feeds the QASM string into
        # the engine; the Circuit-tab dropdown / seed / logical-qubits rows
        # and the matching sweep-axis options are hidden.
        dcc.Store(
            id="custom-qasm-store",
            data={"qasm": None, "filename": None, "num_qubits": None, "error": None},
            storage_type="memory",
        ),
        make_custom_qasm_help_modal(),
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
# Callback: per-row "×" button → drop that specific sweep axis
# ---------------------------------------------------------------------------


@app.callback(
    *[Output(f"metric-row-wrap-{i}", "style", allow_duplicate=True) for i in range(MAX_METRICS)],
    *[Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
    *[Output(f"metric-slider-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
    *[Output(f"metric-checklist-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
    Output("add-metric-btn", "style", allow_duplicate=True),
    Output("remove-metric-btn", "style", allow_duplicate=True),
    Output("num-metrics-store", "data", allow_duplicate=True),
    Output("suppress-cascade", "data", allow_duplicate=True),
    Input({"type": "remove-metric-x", "index": ALL}, "n_clicks"),
    *[State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    *[State(f"metric-slider-{i}", "value") for i in range(MAX_METRICS)],
    *[State(f"metric-checklist-{i}", "value") for i in range(MAX_METRICS)],
    State("num-metrics-store", "data"),
    prevent_initial_call=True,
)
def on_remove_specific_metric(n_clicks_list, *all_states):
    triggered = ctx.triggered_id
    # Pattern-matched inputs can fire with n_clicks=None on component
    # (re)mount; only proceed when a real click registered for the
    # triggering row.
    if not isinstance(triggered, dict) or triggered.get("type") != "remove-metric-x":
        raise dash.exceptions.PreventUpdate
    clicked = triggered.get("index")
    if not isinstance(clicked, int):
        raise dash.exceptions.PreventUpdate
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate

    dropdown_vals = list(all_states[0:MAX_METRICS])
    slider_vals   = list(all_states[MAX_METRICS:2 * MAX_METRICS])
    checklist_vals = list(all_states[2 * MAX_METRICS:3 * MAX_METRICS])
    num_metrics = all_states[3 * MAX_METRICS] or 1

    # Never remove the last remaining axis, and never act on a hidden slot.
    if num_metrics <= 1 or clicked < 0 or clicked >= num_metrics:
        raise dash.exceptions.PreventUpdate

    # Shift every axis below the clicked row up by one slot.
    for k in range(clicked, num_metrics - 1):
        dropdown_vals[k] = dropdown_vals[k + 1]
        slider_vals[k]   = slider_vals[k + 1]
        checklist_vals[k] = checklist_vals[k + 1]

    new_num = num_metrics - 1

    row_styles = [
        {} if i < new_num else {"display": "none"}
        for i in range(MAX_METRICS)
    ]

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
        if new_num < MAX_METRICS
        else {"display": "none"}
    )
    remove_style = (
        {**_btn_base, "border": f"1px dashed {COLORS['border']}", "color": COLORS["text_muted"]}
        if new_num > 1
        else {"display": "none"}
    )

    # suppress-cascade=True tells the downstream dropdown listeners
    # (_reconfigure_slider, _toggle_slider_checklist) to preserve the slider
    # and checklist values we just shifted, instead of resetting them.
    return (
        *row_styles,
        *dropdown_vals,
        *slider_vals,
        *checklist_vals,
        add_style,
        remove_style,
        new_num,
        True,
    )


# ---------------------------------------------------------------------------
# Callback: per-axis "×" visibility — hide every cross when only one
# sweep axis is active (so the user can't remove the last one), show all
# of them otherwise. Runs on initial load via prevent_initial_call=False
# so the layout matches num-metrics-store after a session restore.
# ---------------------------------------------------------------------------


@app.callback(
    Output({"type": "remove-metric-x", "index": ALL}, "style"),
    Input("num-metrics-store", "data"),
    State({"type": "remove-metric-x", "index": ALL}, "id"),
    prevent_initial_call=False,
)
def toggle_axis_remove_visibility(num_metrics, ids):
    show = (num_metrics or 1) > 1
    style = {"display": "flex"} if show else {"display": "none"}
    return [style for _ in ids]


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
    + [
        Output("cfg-row-num-qubits", "style"),
        Output("cfg-row-num-cores", "style"),
        Output("cfg-row-communication-qubits", "style"),
        Output("cfg-row-num-logical-qubits", "style"),
    ]
    + [Output(f"cfg-row-cat-{cat.key}", "style") for cat in CATEGORICAL_METRICS],
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
    # The virtual ``qubits`` axis sweeps physical == logical, so it hides
    # both config rows for the duration of the sweep.
    qubits_alias_swept = "qubits" in swept
    qubits_style = (
        {"display": "none"}
        if ("num_qubits" in swept or qubits_alias_swept)
        else {}
    )
    cores_style = {"display": "none"} if "num_cores" in swept else {}
    comm_style = {"display": "none"} if "communication_qubits" in swept else {}
    logi_style = (
        {"display": "none"}
        if ("num_logical_qubits" in swept or qubits_alias_swept)
        else {}
    )
    cat_styles = [
        {"display": "none"} if cat.key in swept else {} for cat in CATEGORICAL_METRICS
    ]
    return noise_styles + [qubits_style, cores_style, comm_style, logi_style] + cat_styles


# ---------------------------------------------------------------------------
# Callback: live budget warning
#
# Re-runs `_compute_axis_counts` whenever the user changes budgets or axes
# so they see the budget error BEFORE clicking Run.
# ---------------------------------------------------------------------------

def _feedback_style(kind: str) -> dict:
    """Inline banner style driven by the shared FEEDBACK_COLORS palette."""
    palette = FEEDBACK_COLORS[kind]
    return {
        "background": palette["bg"],
        "border": f"1px solid {palette['border']}",
        "color": palette["text"],
        "borderRadius": "6px",
        "padding": "8px 10px",
        "marginTop": "-4px",
        "marginBottom": "10px",
        "fontSize": "12px",
        "lineHeight": "1.4",
    }


_BUDGET_WARNING_STYLE = _feedback_style("error")
# Softer amber style for an advisory (memory-cap) that doesn't block the run.
_BUDGET_INFO_STYLE = _feedback_style("warning")


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

    active = []
    seen: set = set()
    for i in range(n):
        k = dropdown_vals[i]
        if not k or k in seen:
            continue
        seen.add(k)
        if k in CAT_METRIC_BY_KEY:
            continue
        r = slider_vals[i]
        if r:
            active.append((k, float(r[0]), float(r[1])))

    if not active:
        return [], {"display": "none"}

    has_cold = any(ax[0] in _engine.COLD_PATH_KEYS for ax in active)
    max_hot_int = int(max_hot) if max_hot else None
    effective_max_hot, requested_max_hot = _engine._memory_capped_max_hot(max_hot_int)
    mem_capped = effective_max_hot < requested_max_hot

    try:
        _engine._compute_axis_counts(
            active, has_cold,
            max_cold=int(max_cold) if max_cold else None,
            max_hot=effective_max_hot,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if mem_capped:
            msg = (
                f"{msg} Memory cap: hot budget reduced from requested "
                f"{requested_max_hot:,} to {effective_max_hot:,} to fit in "
                f"available RAM."
            )
        return (
            [html.Span("⚠ ", style={"fontWeight": "700"}), msg],
            _BUDGET_WARNING_STYLE,
        )

    if mem_capped:
        return (
            [
                html.Span("ℹ ", style={"fontWeight": "700"}),
                f"Max hot evaluations capped to {effective_max_hot:,} "
                f"(from {requested_max_hot:,}) to fit in available RAM. "
                f"Close other apps or reduce the value to silence this.",
            ],
            _BUDGET_INFO_STYLE,
        )
    return [], {"display": "none"}


@app.callback(
    Output("sweep-workers-warning", "children"),
    Output("sweep-workers-warning", "style"),
    Input("cfg-max-workers", "value"),
    prevent_initial_call=False,
)
def update_workers_warning(max_workers):
    try:
        n = int(max_workers) if max_workers else 1
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        return [], {"display": "none"}
    return (
        [
            html.Span("⚠ ", style={"fontWeight": "700"}),
            f"{n} parallel workers will each hold an independent copy of "
            "the routed circuit in RAM. Dense logical circuits on "
            "constrained topologies (e.g. grid, ring) can allocate tens of "
            "GB per worker and exhaust system memory. Start at 1 and raise "
            "only when you know your circuit's memory footprint.",
        ],
        _BUDGET_INFO_STYLE,
    )


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
        "total_swaps": float(getattr(r, "total_swaps", 0)),
        "total_teleportations": float(getattr(r, "total_teleportations", 0)),
        "total_network_distance": float(getattr(r, "total_network_distance", 0)),
    }


def _slider_to_value(slider_pos: float, log_scale: bool) -> float:
    return 10.0**slider_pos if log_scale else slider_pos


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
    Input("cfg-num-qubits", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-communication-qubits", "value"),
    Input("cfg-num-logical-qubits", "value"),
    Input("cfg-seed", "value"),
    Input("cfg-dynamic-decoupling", "value"),
    Input("cfg-max-cold", "value"),
    Input("cfg-max-hot", "value"),
    Input("cfg-max-workers", "value"),
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


# Generate (or reuse) a per-tab session id so the server can route sweep
# progress back to the user who started it. Persists across reloads via
# sessionStorage (same tab) and is exposed on window so progress.js can
# read it for /api/progress polling.
app.clientside_callback(
    """function(_n, existing) {
        var sid = existing;
        if (!sid) {
            try { sid = sessionStorage.getItem('qusim_sid') || ''; } catch (e) { sid = ''; }
        }
        if (!sid) {
            if (window.crypto && window.crypto.randomUUID) {
                sid = window.crypto.randomUUID();
            } else {
                sid = 'sid-' + Math.random().toString(36).slice(2) + '-' + Date.now().toString(36);
            }
        }
        try { sessionStorage.setItem('qusim_sid', sid); } catch (e) {}
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
    State("cfg-num-qubits", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-communication-qubits", "value"),
    State("cfg-num-logical-qubits", "value"),
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
    num_qubits = all_args[idx]; idx += 1
    num_cores = all_args[idx]; idx += 1
    communication_qubits = all_args[idx]; idx += 1
    num_logical_qubits = all_args[idx]; idx += 1
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
        for i in range(int(num_metrics or 1)):
            k = dropdown_vals[i]
            if not k or k in seen:
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

        _phys = int(num_qubits or 16)
        _logi = int(num_logical_qubits) if num_logical_qubits else _phys
        cold_config = {
            "circuit_type": circuit_type or "qft",
            "num_qubits": _phys,
            "num_logical_qubits": max(2, min(_logi, _phys)),
            "num_cores": int(num_cores or 4),
            "communication_qubits": int(communication_qubits or 1),
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


def _count_points(sweep_data: dict) -> int:
    if "facets" in sweep_data:
        return sum(_count_points(f) for f in sweep_data["facets"])
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
# Callbacks: update range labels and reconfigure sliders when dropdown changes
# ---------------------------------------------------------------------------


# Inline-hint copy keyed by sweep-metric key.  Only metrics whose role is
# non-obvious from the label live here; everything else gets nothing under
# the dropdown.
_METRIC_INLINE_HINT = {
    "qubits": (
        "Sweeps physical qubits with logical = physical. "
        "Hides the Physical / Logical config rows while active."
    ),
}

_METRIC_HINT_VISIBLE_STYLE = {
    "fontSize": "11px",
    "color": COLORS["text_muted"],
    "background": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "4px",
    "padding": "6px 8px",
    "marginBottom": "10px",
    "lineHeight": "1.35",
    "fontStyle": "italic",
    "display": "block",
}


for _idx in range(MAX_METRICS):

    @app.callback(
        Output(f"metric-help-{_idx}", "children"),
        Output(f"metric-help-{_idx}", "style"),
        Input(f"metric-dropdown-{_idx}", "value"),
        prevent_initial_call=False,
    )
    def _update_metric_hint(metric_key, _i=_idx):
        hint = _METRIC_INLINE_HINT.get(metric_key)
        if not hint:
            return "", {"display": "none"}
        return hint, _METRIC_HINT_VISIBLE_STYLE


def _arch_clamped_max(
    metric_key: str | None,
    slider_min: float,
    slider_max: float,
    num_qubits: float | int | None,
    num_cores: float | int | None,
) -> float:
    """Cap a sweep slider's max to the architectural limit, if any.

    Currently only ``communication_qubits`` has an architectural cap:
    ``floor(sqrt(qubits_per_core))``.  The engine clamps anyway, but
    capping the *slider* prevents the user from spending sweep budget on
    cells that all collapse onto the clamp.
    """
    import math as _math
    if metric_key != "communication_qubits":
        return slider_max
    nq = int(num_qubits) if num_qubits else 16
    nc = max(1, int(num_cores) if num_cores else 1)
    qpc = max(1, nq // nc)
    cap = max(1, _math.isqrt(qpc))
    return float(min(slider_max, max(slider_min, float(cap))))


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
        State("suppress-cascade", "data"),
        State("cfg-num-qubits", "value"),
        State("cfg-num-cores", "value"),
        prevent_initial_call=True,
    )
    def _reconfigure_slider(metric_key, suppress, num_qubits, num_cores, _i=_idx):
        no = dash.no_update
        if not metric_key:
            return (no, no, no, no, no, no)
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return (no, no, no, no, no, no)
        # Clamp the registry max to the architectural limit when the metric
        # has one (comm_qubits is bounded by floor(sqrt(qubits/cores))).
        # Without this the user can sweep [1, 16] but values >= clamp all
        # collapse to the clamp inside the engine, wasting cells.
        smin, smax = float(m.slider_min), float(m.slider_max)
        smax = _arch_clamped_max(metric_key, smin, smax, num_qubits, num_cores)
        marks = (
            _log_marks(smin, smax, m.unit)
            if m.log_scale
            else _linear_marks(smin, smax, unit=m.unit)
        )
        if m.is_cold_path:
            step = 2 if m.key == "num_qubits" else 1
        else:
            step = (smax - smin) / 200
        # Preserve the slider value when the dropdown change came from a
        # session load (suppress=True); the load callback already wrote the
        # restored value and we'd otherwise clobber it with defaults.
        # ``_toggle_slider_checklist`` owns the suppress-cascade reset — one
        # writer per dropdown trigger is enough.
        if suppress:
            value = no
        else:
            lo = max(smin, min(float(m.slider_default_low), smax))
            hi = max(lo, min(float(m.slider_default_high), smax))
            value = [lo, hi]
        return (
            smin,
            smax,
            step,
            marks,
            value,
            _tooltip_cfg(m.log_scale, m.unit, always_visible=True),
        )

    # Re-clamp the slider's max when the architecture changes.  Only fires
    # for axes whose current metric has an architectural cap (currently:
    # comm_qubits, capped by floor(sqrt(qubits/cores))).  Preserves the
    # user's range — only clamps the value if it now exceeds the cap.
    @app.callback(
        Output(f"metric-slider-{_idx}", "max", allow_duplicate=True),
        Output(f"metric-slider-{_idx}", "marks", allow_duplicate=True),
        Output(f"metric-slider-{_idx}", "value", allow_duplicate=True),
        Input("cfg-num-qubits", "value"),
        Input("cfg-num-cores", "value"),
        State(f"metric-dropdown-{_idx}", "value"),
        State(f"metric-slider-{_idx}", "value"),
        State(f"metric-slider-{_idx}", "max"),
        prevent_initial_call=True,
    )
    def _reclamp_axis_to_arch(
        num_qubits, num_cores, metric_key, current_value, current_max,
        _i=_idx,
    ):
        no = dash.no_update
        if not metric_key:
            return (no, no, no)
        m = METRIC_BY_KEY.get(metric_key)
        if m is None:
            return (no, no, no)
        smin = float(m.slider_min)
        new_max = _arch_clamped_max(metric_key, smin, float(m.slider_max), num_qubits, num_cores)
        if current_max is not None and abs(float(current_max) - new_max) < 1e-9:
            # No-op: cap unchanged for the current architecture.
            return (no, no, no)
        marks = (
            _log_marks(smin, new_max, m.unit)
            if m.log_scale
            else _linear_marks(smin, new_max, unit=m.unit)
        )
        if isinstance(current_value, (list, tuple)) and len(current_value) == 2:
            lo = max(smin, min(float(current_value[0]), new_max))
            hi = max(lo, min(float(current_value[1]), new_max))
            new_value = [lo, hi]
        else:
            new_value = no
        return (new_max, marks, new_value)


# ---------------------------------------------------------------------------
# Callback: toggle slider / checklist when dropdown selects categorical
# ---------------------------------------------------------------------------

_RANGE_LABEL_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "fontSize": "11px",
    "color": COLORS["accent2"],
    "marginTop": "-20px",
}

for _idx in range(MAX_METRICS):

    @app.callback(
        Output(f"metric-slider-container-{_idx}", "style"),
        Output(f"metric-checklist-container-{_idx}", "style"),
        Output(f"metric-checklist-{_idx}", "options"),
        Output(f"metric-checklist-{_idx}", "value"),
        Output(f"metric-range-label-{_idx}", "style"),
        Output("suppress-cascade", "data", allow_duplicate=True),
        Input(f"metric-dropdown-{_idx}", "value"),
        State("suppress-cascade", "data"),
        prevent_initial_call=True,
    )
    def _toggle_slider_checklist(metric_key, suppress, _i=_idx):
        no = dash.no_update
        cat = CAT_METRIC_BY_KEY.get(metric_key)
        if cat:
            # See _reconfigure_slider: preserve the loaded checklist value
            # when the change came from a session load.
            value = no if suppress else [o["value"] for o in cat.options]
            return (
                {"display": "none"},
                {},
                cat.options,
                value,
                {"display": "none"},
                False,
            )
        return (
            {"paddingBottom": "22px"},
            {"display": "none"},
            [],
            no if suppress else [],
            _RANGE_LABEL_STYLE,
            False,
        )


# ---------------------------------------------------------------------------
# Callback: filter dropdown options so a metric can't be selected twice
# ---------------------------------------------------------------------------

_ALL_METRIC_OPTIONS = (
    [{"label": m.label, "value": m.key} for m in SWEEPABLE_METRICS]
    + [{"label": c.label, "value": c.key} for c in CATEGORICAL_METRICS]
)


# Sweep-axis keys that are meaningless once a custom QASM circuit is loaded:
# the algorithm size and gate sequence are fully determined by the uploaded
# file, so neither circuit type, logical qubit count, nor the (logical, physical)
# alias should be selectable on a sweep axis.
_CUSTOM_QASM_DISABLED_KEYS = {"circuit_type", "num_logical_qubits", "qubits"}


@app.callback(
    *[Output(f"metric-dropdown-{i}", "options") for i in range(MAX_METRICS)],
    *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
    Input("num-metrics-store", "data"),
    Input("custom-qasm-store", "data"),
    prevent_initial_call=True,
)
def _filter_dropdown_options(*args):
    values = args[:MAX_METRICS]
    num_metrics = args[MAX_METRICS] or 1
    custom_qasm = args[MAX_METRICS + 1] or {}
    custom_active = bool(custom_qasm.get("qasm"))
    # Only consider dropdowns from visible (active) rows as "taken"
    results = []
    for i in range(MAX_METRICS):
        taken = {
            values[j]
            for j in range(num_metrics)
            if j != i and values[j]
        }
        results.append([
            {
                **opt,
                "disabled": (
                    opt["value"] in taken
                    or (custom_active and opt["value"] in _CUSTOM_QASM_DISABLED_KEYS)
                ),
            }
            for opt in _ALL_METRIC_OPTIONS
        ])
    return results


# ---------------------------------------------------------------------------
# Callback: custom-QASM upload / clear → populate custom-qasm-store
# ---------------------------------------------------------------------------


@app.callback(
    Output("custom-qasm-store", "data"),
    Output("sweep-dirty", "data", allow_duplicate=True),
    Input("custom-qasm-upload", "contents"),
    Input({"type": "custom-qasm-clear", "index": ALL}, "n_clicks"),
    State("custom-qasm-upload", "filename"),
    State("sweep-dirty", "data"),
    prevent_initial_call=True,
)
def on_custom_qasm_change(contents, clear_clicks, filename, sweep_dirty):
    import base64

    triggered = ctx.triggered_id
    cleared = (
        isinstance(triggered, dict)
        and triggered.get("type") == "custom-qasm-clear"
        and any(clear_clicks or [])
    )
    if cleared or contents is None:
        return (
            {"qasm": None, "filename": None, "num_qubits": None, "error": None},
            (sweep_dirty or 0) + 1,
        )

    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        qasm_str = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        return (
            {"qasm": None, "filename": filename, "num_qubits": None,
             "error": f"Failed to decode upload: {exc}"},
            sweep_dirty or 0,
        )

    try:
        from qiskit import qasm2
        circ = qasm2.loads(qasm_str)
        num_qubits = int(circ.num_qubits)
    except Exception as exc:
        return (
            {"qasm": None, "filename": filename, "num_qubits": None,
             "error": f"Not a valid OpenQASM 2.0 circuit: {exc}"},
            sweep_dirty or 0,
        )

    return (
        {"qasm": qasm_str, "filename": filename, "num_qubits": num_qubits, "error": None},
        (sweep_dirty or 0) + 1,
    )


# ---------------------------------------------------------------------------
# Callback: render the custom-QASM status line + hide circuit-config rows
# ---------------------------------------------------------------------------


@app.callback(
    Output("custom-qasm-status", "children"),
    Output("custom-qasm-status", "style"),
    Output("custom-qasm-upload-label", "children"),
    Output("cfg-row-cat-circuit_type-wrap", "style"),
    Output("cfg-row-num-logical-qubits-wrap", "style"),
    Output("cfg-row-seed", "style"),
    Input("custom-qasm-store", "data"),
)
def render_custom_qasm_status(data):
    data = data or {}
    qasm = data.get("qasm")
    err = data.get("error")
    filename = data.get("filename") or "circuit.qasm"
    nq = data.get("num_qubits")

    hidden = {"display": "none"}
    visible = {}

    if err:
        status_children = [
            html.Div(
                err,
                style={
                    "fontSize": "11px",
                    "color": FEEDBACK_COLORS["error"]["text"],
                    "padding": "6px 8px",
                    "background": FEEDBACK_COLORS["error"]["bg"],
                    "border": f"1px solid {FEEDBACK_COLORS['error']['border']}",
                    "borderRadius": "6px",
                    "marginTop": "6px",
                },
            ),
        ]
        return (
            status_children,
            {"display": "block"},
            "Upload .qasm",
            visible, visible, visible,
        )

    if not qasm:
        return (
            [],
            {"display": "none"},
            "Upload .qasm",
            visible, visible, visible,
        )

    status_children = [
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "6px",
                "marginTop": "6px",
                "padding": "6px 8px",
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "6px",
                "background": COLORS["surface"],
            },
            children=[
                html.Div(
                    style={
                        "minWidth": "0",
                        "fontSize": "11px",
                        "color": COLORS["text"],
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap",
                    },
                    title=filename,
                    children=[
                        html.Span("✓ ", style={"color": COLORS["brand"]}),
                        html.Span(filename, style={"fontWeight": "600"}),
                        html.Span(
                            f"  ({nq} qubits)" if nq else "",
                            style={"color": COLORS["text_muted"]},
                        ),
                    ],
                ),
                html.Button(
                    "Clear",
                    id={"type": "custom-qasm-clear", "index": 0},
                    className="ghost-btn",
                    n_clicks=0,
                    style={"padding": "2px 8px", "fontSize": "11px"},
                ),
            ],
        ),
    ]
    return (
        status_children,
        {"display": "block"},
        f"Replace ({filename})",
        hidden, hidden, hidden,
    )


# ---------------------------------------------------------------------------
# Callback: open / close the custom-QASM help modal
# ---------------------------------------------------------------------------


@app.callback(
    Output("custom-qasm-help-modal", "is_open"),
    Input("custom-qasm-help-icon", "n_clicks"),
    Input("custom-qasm-help-close", "n_clicks"),
    State("custom-qasm-help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_custom_qasm_help_modal(open_clicks, close_clicks, is_open):
    triggered = ctx.triggered_id
    if triggered == "custom-qasm-help-icon":
        return True
    if triggered == "custom-qasm-help-close":
        return False
    return is_open


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
    State("cfg-num-qubits", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-communication-qubits", "value"),
    State("cfg-num-logical-qubits", "value"),
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
    cfg_num_qubits = all_args[idx]; idx += 1
    cfg_num_cores = all_args[idx]; idx += 1
    cfg_communication_qubits = all_args[idx]; idx += 1
    cfg_num_logical_qubits = all_args[idx]; idx += 1
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
        cfg_num_qubits=cfg_num_qubits,
        cfg_num_cores=cfg_num_cores,
        cfg_communication_qubits=cfg_communication_qubits,
        cfg_num_logical_qubits=cfg_num_logical_qubits,
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
    fname = _time.strftime(f"{stem}-%Y%m%d-%H%M%S.qusim.json.gz")
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
    Output("cfg-num-qubits", "value", allow_duplicate=True),
    Output("cfg-num-cores", "value", allow_duplicate=True),
    Output("cfg-communication-qubits", "value", allow_duplicate=True),
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
]

_THRESH_OUTPUTS = (
    [Output(f"cfg-threshold-{i}", "value", allow_duplicate=True) for i in range(5)]
    + [Output(f"cfg-threshold-color-{i}", "value", allow_duplicate=True) for i in range(5)]
)


def _value_to_slider(val: float, log_scale: bool) -> float:
    """Inverse of ``_slider_to_value``."""
    import math
    if log_scale and val is not None and val > 0:
        return math.log10(val)
    return val


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
    Input("session-upload", "contents"),
    State("session-upload", "filename"),
    prevent_initial_call=True,
)
def on_load_session(contents, filename):
    import base64
    from gui.session import load as session_load, apply_session, SessionError

    if contents is None:
        raise dash.exceptions.PreventUpdate

    # ``contents`` has shape 'data:<mime>;base64,<payload>'.
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
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
    cfg_out = [
        circuit["circuit_type"],
        circuit["num_qubits"],
        circuit["num_cores"],
        int(circuit.get("communication_qubits", 1) or 1),
        int(circuit.get("num_logical_qubits", circuit["num_qubits"]) or circuit["num_qubits"]),
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
    )


# Count of named scalar Outputs between thresholds and noise in the main
# decorator — see on_load_session: num-metrics, num-thresholds, hot-reload,
# view-type, frozen-axis.
_LOAD_SCALAR_OUTPUTS = 5
# Count of trailing Outputs: status-bar, banner.children, banner.style,
# fom-config-store, fom-name, fom-numerator, fom-denominator, fom-intermediates.
_LOAD_TRAILING_OUTPUTS = 8
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
    # Trailing order: status-bar, banner.children, banner.style, then 5 FoM outputs.
    stub[-8] = "Load failed"
    stub[-7] = banner_children
    stub[-6] = _error_banner_visible_style()
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

        // dcc.Graph wraps the plotly div in an outer container — only the
        // inner .js-plotly-plot element carries the live `.data` array.
        var outer = document.getElementById("main-plot");
        var plotDiv = outer ? outer.querySelector(".js-plotly-plot") : null;
        if (plotDiv && plotDiv.data && plotDiv.data.length > 0) {
            Plotly.restyle(plotDiv, {z: [slice2d]}, [0]);
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
# Callback: show/hide Pareto-axis dropdowns based on active view
# ---------------------------------------------------------------------------

@app.callback(
    Output("pareto-axis-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_pareto_axis_visibility(view_type):
    if view_type == "pareto":
        return {"padding": "4px 16px 8px"}
    return {"display": "none"}


@app.callback(
    Output("elasticity-axis-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_elasticity_axis_visibility(view_type):
    if view_type == "elasticity":
        return {"padding": "4px 16px 8px"}
    return {"display": "none"}


@app.callback(
    Output("importance-mode-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_importance_mode_visibility(view_type):
    if view_type == "importance":
        return {"padding": "4px 16px 8px"}
    return {"display": "none"}


@app.callback(
    Output("correlation-mode-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_correlation_mode_visibility(view_type):
    if view_type == "correlation":
        return {"padding": "4px 16px 8px"}
    return {"display": "none"}


# Populate the trajectory dropdown from the active sweep's numeric axes
# (categorical axes can't be elasticised — derivatives need an ordered
# coordinate). Default to the first axis when the prior selection is no
# longer valid for this sweep.
@app.callback(
    Output("elasticity-trajectory-dropdown", "options"),
    Output("elasticity-trajectory-dropdown", "value"),
    Input("sweep-result-store", "data"),
    State("elasticity-trajectory-dropdown", "value"),
    prevent_initial_call=False,
)
def update_elasticity_trajectory_options(sweep_store, current_value):
    full = _get_sweep(sweep_store)
    if not full:
        return [], None
    metric_keys = list(full.get("metric_keys", []))
    options = []
    for mk in metric_keys:
        m = METRIC_BY_KEY.get(mk)
        if m is None:
            continue  # categorical axes are excluded
        options.append({"label": m.label, "value": mk})
    if not options:
        return [], None
    valid_keys = {opt["value"] for opt in options}
    value = current_value if current_value in valid_keys else options[0]["value"]
    return options, value


# ---------------------------------------------------------------------------
# Callbacks: Figure of Merit (Merit view)
# ---------------------------------------------------------------------------


@app.callback(
    Output("merit-controls-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_merit_controls_visibility(view_type):
    if view_type == "merit":
        return {
            "display": "block",
            "padding": "6px 16px 10px",
            "borderTop": f"1px solid {COLORS['border']}",
        }
    return {"display": "none"}


def _parse_intermediates(text: str) -> list[tuple[str, str]]:
    """Parse the textarea format (``name = expr`` per line) into tuples."""
    out: list[tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, expr = line.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            continue
        out.append((name, expr))
    return out


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


def _resolve_thresholds(threshold_enable, all_t, all_c) -> tuple[list[float] | None, list[str] | None]:
    """Filter the right-sidebar threshold inputs into ``(values, colors)``.

    Returns ``(None, None)`` if the user disabled thresholds; otherwise both
    lists are aligned and contain only the populated entries.
    """
    if not (threshold_enable and "yes" in (threshold_enable or [])):
        return None, None
    vals: list[float] = []
    colors: list[str] = []
    for v, c in zip(all_t, all_c):
        if v is None:
            continue
        vals.append(float(v))
        colors.append(c or "")
    return (vals or None), (colors or None)


def _frozen_values_from_sliders(metric_keys, slider_values) -> dict[str, float]:
    """Map pattern-matched slider values back to ``{axis_key: value}``.

    Slider index ``i`` corresponds to ``metric_keys[i]`` while the axis is
    active; out-of-range or ``None`` values are dropped.
    """
    out: dict[str, float] = {}
    for i, val in enumerate(slider_values or []):
        if i >= len(metric_keys) or val is None:
            continue
        try:
            out[metric_keys[i]] = float(val)
        except (TypeError, ValueError):
            continue
    return out


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


# ---------------------------------------------------------------------------
# Merit view: two-mode UI (Heatmap + Pareto) with selectable X/Y axes,
# frozen sliders for the remaining axes, and a colour-by dropdown for the
# Pareto scatter.
# ---------------------------------------------------------------------------


_MERIT_VIEW_CONTAINER_STYLE_VISIBLE = {
    "display": "block",
    "padding": "6px 16px 6px",
    "borderBottom": f"1px solid {COLORS['border']}",
    "background": COLORS["bg"],
}
_MERIT_VIEW_CONTAINER_STYLE_HIDDEN = {"display": "none"}


@app.callback(
    Output("merit-view-controls-container", "style"),
    Input("view-type-store", "data"),
    prevent_initial_call=False,
)
def toggle_merit_view_controls_visibility(view_type):
    if view_type == "merit":
        return _MERIT_VIEW_CONTAINER_STYLE_VISIBLE
    return _MERIT_VIEW_CONTAINER_STYLE_HIDDEN


# Push the active (view-type, merit-mode) pair into the help-icon JS so the
# modebar "?" popup describes whatever the user is currently looking at.
app.clientside_callback(
    """function(viewType, meritMode) {
        if (window.qusimUpdatePlotHelp) {
            window.qusimUpdatePlotHelp(viewType, meritMode);
        }
        return window.dash_clientside.no_update;
    }""",
    Output("plot-help-sink", "data"),
    Input("view-type-store", "data"),
    Input("merit-mode-store", "data"),
    prevent_initial_call=False,
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


def _axis_dropdown_options(metric_keys: list[str]) -> list[dict]:
    return [{"label": _progress_label(k), "value": k} for k in metric_keys]


def _slider_marks(values: list[float]) -> dict:
    """Build slider tick marks for a discrete grid; label every value when ≤8,
    otherwise label first/last + a few interior ticks.
    """
    if not values:
        return {}
    if len(values) <= 8:
        return {float(v): {"label": f"{v:g}",
                            "style": {"fontSize": "9px", "color": COLORS["text_muted"]}}
                for v in values}
    # Sparse marks: first, last, and ~3 evenly-spaced interior points.
    indices = sorted({0, len(values) - 1,
                      len(values) // 4, len(values) // 2,
                      (3 * len(values)) // 4})
    return {float(values[i]): {"label": f"{values[i]:g}",
                                "style": {"fontSize": "9px",
                                          "color": COLORS["text_muted"]}}
            for i in indices}


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
        }
        return window.dash_clientside.no_update;
    }""",
    Output("sweep-trigger", "data", allow_duplicate=True),
    Input("session-loaded-tick", "data"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Callback: chevron toggle for OUTPUT / SWEEP BUDGET collapsible sections
# ---------------------------------------------------------------------------


def _make_section_toggle(section_id: str, default_open: bool = True) -> None:
    """Wire a header click → body display + chevron flip for one section."""
    body_id = f"{section_id}-body"
    chevron_id = f"{section_id}-chevron"
    header_id = f"{section_id}-header"

    @app.callback(
        Output(body_id, "style"),
        Output(chevron_id, "children"),
        Input(header_id, "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks: int | None):
        # n_clicks-based parity: even = initial state, odd = toggled.
        is_open = default_open if not n_clicks else (
            (n_clicks % 2 == 0) if default_open else (n_clicks % 2 == 1)
        )
        if is_open:
            return {"display": "block", "padding": "0 14px 12px"}, "▾"
        return {"display": "none", "padding": "0 14px 12px"}, "▸"


_make_section_toggle("sweep-budget-section", default_open=False)


# Sweep Budget summary strip — shown in the always-visible footer header.
# Format: "<cold> cold · <hot> hot · <workers>w" (e.g. "64 cold · 5,000 hot · 1w").
app.clientside_callback(
    """function(cold, hot, workers) {
        var fmt = function(n) {
            if (n === null || n === undefined || isNaN(n)) return "—";
            return Number(n).toLocaleString();
        };
        var c = (cold === null || cold === undefined) ? "—" : Number(cold);
        var h = (hot  === null || hot  === undefined) ? "—" : Number(hot);
        var w = (workers === null || workers === undefined) ? "—" : Number(workers);
        return fmt(c) + " cold · " + fmt(h) + " hot · " + w + "w";
    }""",
    Output("sweep-budget-summary", "children"),
    Input("cfg-max-cold", "value"),
    Input("cfg-max-hot", "value"),
    Input("cfg-max-workers", "value"),
    prevent_initial_call=False,
)


# ---------------------------------------------------------------------------
# Bidirectional sync: slider value <-> editable value-chip input
# ---------------------------------------------------------------------------
# Each `slider_row()` renders a slider with id `X` and an input with id
# `X-input`. The input shows the current value (formatted, parseable) and
# can be edited; the slider updates as the user drags. Clientside sync
# keeps both in step. Loops are avoided by writing only the non-triggered
# output and trusting Dash's value-equality dedup on the trigger side.

_SLIDER_INPUT_PAIRS: list[tuple[str, bool]] = []


def _bind_slider_input(slider_id: str, log_scale: bool = False) -> None:
    """Wire two-way sync between a slider and its `<id>-input` chip."""
    input_id = f"{slider_id}-input"
    if log_scale:
        # Slider position is the log10 exponent. Input shows the real value.
        slider_to_input = (
            "function(sv) { "
            "  if (sv === null || sv === undefined || isNaN(sv)) "
            "    return window.dash_clientside.no_update; "
            "  var v = Math.pow(10, sv); "
            "  if (v < 1e-3 || v >= 1e6) return v.toExponential(2); "
            "  if (v < 1) return v.toPrecision(3); "
            "  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v)); "
            "  return v.toPrecision(4); "
            "}"
        )
        input_to_slider = (
            "function(iv) { "
            "  if (iv === null || iv === undefined || iv === '') "
            "    return window.dash_clientside.no_update; "
            "  var n = parseFloat(iv); "
            "  if (!isFinite(n) || n <= 0) return window.dash_clientside.no_update; "
            "  return Math.log10(n); "
            "}"
        )
    else:
        slider_to_input = (
            "function(sv) { "
            "  if (sv === null || sv === undefined || isNaN(sv)) "
            "    return window.dash_clientside.no_update; "
            "  if (Math.abs(sv - Math.round(sv)) < 1e-9) return String(Math.round(sv)); "
            "  return String(Number(sv.toPrecision(6))); "
            "}"
        )
        input_to_slider = (
            "function(iv) { "
            "  if (iv === null || iv === undefined || iv === '') "
            "    return window.dash_clientside.no_update; "
            "  var n = parseFloat(iv); "
            "  if (!isFinite(n)) return window.dash_clientside.no_update; "
            "  return n; "
            "}"
        )

    app.clientside_callback(
        slider_to_input,
        Output(input_id, "value", allow_duplicate=True),
        Input(slider_id, "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        input_to_slider,
        Output(slider_id, "value", allow_duplicate=True),
        Input(input_id, "value"),
        prevent_initial_call=True,
    )
    _SLIDER_INPUT_PAIRS.append((slider_id, log_scale))


# Right-panel sliders that get a value-chip input.
from gui.constants import SWEEPABLE_METRICS as _SWEEPABLE_METRICS

_bind_slider_input("cfg-num-logical-qubits", log_scale=False)
_bind_slider_input("cfg-num-qubits", log_scale=False)
_bind_slider_input("cfg-num-cores", log_scale=False)
_bind_slider_input("cfg-communication-qubits", log_scale=False)
for _m in _SWEEPABLE_METRICS:
    _bind_slider_input(f"noise-{_m.key}", log_scale=_m.log_scale)


# ---------------------------------------------------------------------------
# Callback: dynamic max for cfg-num-logical-qubits = cfg-num-qubits (physical)
# ---------------------------------------------------------------------------


@app.callback(
    Output("cfg-num-logical-qubits", "max"),
    Output("cfg-num-logical-qubits", "marks"),
    Output("cfg-num-logical-qubits", "value", allow_duplicate=True),
    Input("cfg-num-qubits", "value"),
    State("cfg-num-logical-qubits", "value"),
    prevent_initial_call=True,
)
def _clamp_logical_to_physical(num_qubits, current_logical):
    from gui.components import _minmax_marks
    phys = int(num_qubits) if num_qubits else 16
    phys = max(2, phys)
    cur = int(current_logical) if current_logical else phys
    clamped = max(2, min(cur, phys))
    return phys, _minmax_marks(2, phys, log_scale=False), clamped


# ---------------------------------------------------------------------------
# Callback: dynamic max for cfg-communication-qubits = floor(sqrt(N/Cores))
# ---------------------------------------------------------------------------


@app.callback(
    Output("cfg-communication-qubits", "max"),
    Output("cfg-communication-qubits", "marks"),
    Output("cfg-communication-qubits", "value", allow_duplicate=True),
    Input("cfg-num-qubits", "value"),
    Input("cfg-num-cores", "value"),
    State("cfg-communication-qubits", "value"),
    prevent_initial_call=True,
)
def _update_comm_qubits_bound(num_qubits, num_cores, current):
    import math
    from gui.components import _minmax_marks
    nq = int(num_qubits) if num_qubits else 16
    nc = max(1, int(num_cores) if num_cores else 1)
    qpc = max(1, nq // nc)
    new_max = max(1, math.isqrt(qpc))
    cur = int(current) if current else 1
    clamped = max(1, min(cur, new_max))
    return new_max, _minmax_marks(1, new_max, log_scale=False), clamped


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


def _human_axis_value(axis_key: str, raw_val) -> str:
    """Format a swept axis value for the slider read-out."""
    metric = METRIC_BY_KEY.get(axis_key)
    if metric is None:
        return str(raw_val)
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        return str(raw_val)
    unit = metric.unit
    if metric.is_cold_path or not metric.log_scale:
        if v == int(v):
            return f"{int(v)}{(' ' + unit) if unit else ''}"
        return f"{v:.3f}{(' ' + unit) if unit else ''}"
    # Hot-path log axes have already been materialised by ``np.logspace``
    # — values are physical magnitudes, not exponents.
    if unit == "ns":
        if v >= 1e6:
            return f"{v/1e6:.2f} ms"
        if v >= 1e3:
            return f"{v/1e3:.2f} µs"
        return f"{v:.0f} ns"
    if unit == "Hz":
        if v >= 1e9:
            return f"{v/1e9:.2f} GHz"
        if v >= 1e6:
            return f"{v/1e6:.2f} MHz"
        return f"{v:.0f} Hz"
    return f"{v:.3g}"


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
            values.append(0)
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
    *[Output({"type": "topology-axis-value", "index": i}, "children") for i in range(MAX_METRICS)],
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
# Callback: Topology view — rebuild Cytoscape elements (structure + overlay)
# ---------------------------------------------------------------------------


@app.callback(
    Output("topology-cyto", "elements"),
    Input("cfg-num-qubits", "value"),
    Input("cfg-num-cores", "value"),
    Input("cfg-communication-qubits", "value"),
    Input("cfg-topology", "value"),
    Input("cfg-intracore-topology", "value"),
    Input("topology-overlay-metric", "value"),
    Input("topology-facet-selector", "value"),
    Input({"type": "topology-axis-slider", "index": ALL}, "value"),
    Input("sweep-result-store", "data"),
    prevent_initial_call=False,
)
def _rebuild_topology_graph(
    num_qubits, num_cores, comm_qubits, topology, intracore_topology,
    overlay_metric, facet_idx, axis_slider_vals, sweep_store,
):
    def _build_default_elements() -> list:
        return build_topology_elements(
            num_cores=num_cores or 1,
            num_qubits=num_qubits or 16,
            communication_qubits=comm_qubits or 1,
            topology=topology or "ring",
            intracore_topology=intracore_topology or "all_to_all",
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
    if cell is None:
        return _clear_fidelity(_build_default_elements())

    # When a sweep cell is in scope, the architecture has to follow the
    # *cell's* cold config, not the right-sidebar — otherwise scrubbing a
    # cold-path axis (comm_qubits, num_cores, …) leaves the graph stuck on
    # the right-panel snapshot. The cell already carries the post-clamp
    # values that the engine actually used for compilation.
    elements = build_topology_elements(
        num_cores=int(cell.get("num_cores", num_cores or 1)),
        num_qubits=int(cell.get("num_physical", num_qubits or 16)),
        communication_qubits=int(cell.get("communication_qubits", comm_qubits or 1)),
        topology=cell.get("topology_type", topology or "ring"),
        intracore_topology=cell.get(
            "intracore_topology", intracore_topology or "all_to_all"
        ),
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


def _fmt_fid(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 0.9999:
        return "1.000"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"


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
    descriptor = "comm" if qtype == "comm" else "data"
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Multi-user knobs:
    # - debug=False: no hot-reload (which breaks under concurrent requests)
    #   and no traceback leaks to the browser.
    # - host="127.0.0.1": only cloudflared (or local browser) can reach the
    #   port; nothing on the LAN can connect directly.
    # - threaded=True: the Flask dev server serves each request on its own
    #   thread, so polling endpoints stay responsive while a sweep runs.
    # Override host via QUSIM_HOST=0.0.0.0 if you need direct LAN access.
    host = os.environ.get("QUSIM_HOST", "127.0.0.1")
    port = int(os.environ.get("QUSIM_PORT", "8050"))
    print(f"qusim DSE GUI starting at http://{host}:{port}")
    app.run(debug=False, host=host, port=port, threaded=True)
