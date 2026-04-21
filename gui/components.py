"""
Reusable Dash UI component factories for the DSE GUI.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from .constants import (
    ANALYSIS_TABS,
    CIRCUIT_TYPES,
    DEFAULT_SWEEP_AXES,
    INTRACORE_TOPOLOGY_TYPES,
    MAX_COLD_COMPILATIONS,
    MAX_TOTAL_POINTS_HOT,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    OUTPUT_METRICS,
    PLACEMENT_OPTIONS,
    ROUTING_ALGORITHM_OPTIONS,
    SWEEPABLE_METRICS,
    TOPOLOGY_TYPES,
    VIEW_TAB_DEFAULTS,
    VIEW_TABS,
)

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F5F5F5",
    "surface2": "#EBEBEB",
    "border": "#D4D4D4",
    "accent": "#2B2B2B",
    "accent2": "#555555",
    "text": "#2B2B2B",
    "text_muted": "#888888",
}

# Banner / inline feedback palettes. Each entry carries its own bg / border /
# text triple so consumers don't have to recombine the pieces ad-hoc.
FEEDBACK_COLORS = {
    "error": {
        "bg": "#FDECEC",
        "border": "#E57373",
        "text": "#8B1A1A",
    },
    "warning": {
        "bg": "#FFF7E6",
        "border": "#F0B955",
        "text": "#8A5A00",
    },
}

CARD_STYLE = {
    "background": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "6px",
    "padding": "12px 10px",
    "marginBottom": "8px",
}


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _fmt_value(value: float, log_scale: bool) -> str:
    """Format a slider position as a human-readable value."""
    if log_scale:
        v = 10**value
        if v < 1e-3:
            return f"{v:.2e}"
        if v < 1:
            return f"{v:.4f}"
        return f"{v:.1f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def _fmt_log_mark(exp: int, unit: str) -> str:
    """Compact label for a log-scale mark at the given integer exponent."""
    v = 10.0**exp
    if unit == "ns":
        if exp < 3:
            return f"{v:g} ns"
        if exp < 6:
            return f"{v / 1e3:g} \u00b5s"
        if exp < 9:
            return f"{v / 1e6:g} ms"
        return f"{v / 1e9:g} s"
    if unit == "Hz":
        if exp < 3:
            return f"{v:g} Hz"
        if exp < 6:
            return f"{v / 1e3:g} kHz"
        if exp < 9:
            return f"{v / 1e6:g} MHz"
        return f"{v / 1e9:g} GHz"
    # Unitless (error rates etc.)
    if exp >= 0:
        return f"{v:g}"
    return f"1e{exp}"


_MARK_STYLE = {"color": COLORS["text_muted"], "fontSize": "9px"}


def _log_marks(slider_min: float, slider_max: float, unit: str = "") -> dict:
    """Generate integer-exponent marks showing actual values.

    Limits to at most 5 evenly-spaced marks to fit narrow panels.
    """
    lo, hi = int(slider_min), int(slider_max)
    all_exps = list(range(lo, hi + 1))
    # Thin out if more than 5 marks
    if len(all_exps) > 5:
        step = max(1, (hi - lo) // 4)
        exps = list(range(lo, hi, step)) + [hi]
    else:
        exps = all_exps
    marks: dict = {}
    for exp in exps:
        marks[str(exp)] = {"label": _fmt_log_mark(exp, unit), "style": _MARK_STYLE}
    return marks


def _linear_marks(slider_min: float, slider_max: float, n: int = 5, unit: str = "") -> dict:
    import numpy as np

    n = min(n, 5)
    vals = np.linspace(slider_min, slider_max, n)
    marks: dict = {}
    for v in vals:
        label = f"{v:.2f}" if v != int(v) else str(int(v))
        marks[str(round(v, 6))] = {"label": label, "style": _MARK_STYLE}
    return marks


# ---------------------------------------------------------------------------
# Tooltip helpers  (uses window.dccFunctions defined in assets/tooltip_transforms.js)
# ---------------------------------------------------------------------------

_TOOLTIP_TRANSFORM_MAP = {
    # (log_scale, unit) -> JS function name
    (True, "ns"): "logNs",
    (True, "Hz"): "logHz",
    (True, ""): "logRate",
    (False, "wires"): "linearWires",
    (False, "cycles"): "linearCycles",
    (False, ""): "linearFraction",
}


def _tooltip_transform_name(log_scale: bool, unit: str) -> str:
    return _TOOLTIP_TRANSFORM_MAP.get((log_scale, unit), "linearFraction")


def _tooltip_cfg(log_scale: bool, unit: str, always_visible: bool = False) -> dict:
    """Build the tooltip dict for a dcc.Slider / dcc.RangeSlider."""
    cfg: dict = {
        "placement": "bottom",
        "transform": _tooltip_transform_name(log_scale, unit),
        "style": {"whiteSpace": "nowrap"},
    }
    if always_visible:
        cfg["always_visible"] = True
    return cfg


# ---------------------------------------------------------------------------
# Metric selector (left sidebar row)
# ---------------------------------------------------------------------------


def make_metric_selector(index: int) -> html.Div:
    """
    One sweep metric row: dropdown (choose which param) + range slider.
    ``index`` is 0-based, supports up to len(SWEEPABLE_METRICS) rows.
    """
    if index < len(DEFAULT_SWEEP_AXES):
        default_key = DEFAULT_SWEEP_AXES[index]
    else:
        # Cycle through remaining metrics not in DEFAULT_SWEEP_AXES
        remaining = [
            m.key for m in SWEEPABLE_METRICS if m.key not in DEFAULT_SWEEP_AXES
        ]
        default_key = (
            remaining[(index - len(DEFAULT_SWEEP_AXES)) % len(remaining)]
            if remaining
            else SWEEPABLE_METRICS[0].key
        )
    m = METRIC_BY_KEY[default_key]

    marks = (
        _log_marks(m.slider_min, m.slider_max, m.unit)
        if m.log_scale
        else _linear_marks(m.slider_min, m.slider_max, unit=m.unit)
    )

    return html.Div(
        id=f"metric-row-{index}",
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "6px",
                },
                children=[
                    html.Span(
                        f"Metric {index + 1}",
                        style={
                            "color": COLORS["text_muted"],
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.05em",
                        },
                    ),
                    html.Button(
                        "×",
                        id=f"remove-metric-{index}",
                        n_clicks=0,
                        style={
                            "background": "transparent",
                            "border": f"1px solid {COLORS['border']}",
                            "color": COLORS["text_muted"],
                            "borderRadius": "4px",
                            "width": "22px",
                            "height": "22px",
                            "cursor": "pointer",
                            "fontSize": "14px",
                            "lineHeight": "1",
                            "padding": "0",
                            "display": "flex" if index > 0 else "none",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),
                ],
            ),
            dcc.Dropdown(
                id=f"metric-dropdown-{index}",
                options=[{"label": m.label, "value": m.key} for m in SWEEPABLE_METRICS],
                value=default_key,
                clearable=False,
                style={"marginBottom": "10px"},
                className="dse-dropdown",
            ),
            html.Div(
                id=f"metric-slider-container-{index}",
                children=[
                    dcc.RangeSlider(
                        id=f"metric-slider-{index}",
                        min=m.slider_min,
                        max=m.slider_max,
                        step=(m.slider_max - m.slider_min) / 200,
                        value=[m.slider_min, m.slider_max],
                        marks=marks,
                        tooltip=_tooltip_cfg(m.log_scale, m.unit, always_visible=True),
                        updatemode="drag",
                        className="dse-range-slider",
                    ),
                ],
                style={"paddingBottom": "22px"},
            ),
            html.Div(
                id=f"metric-range-label-{index}",
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontSize": "11px",
                    "color": COLORS["accent2"],
                    "marginTop": "-20px",
                },
            ),
        ],
        style=CARD_STYLE,
    )


# ---------------------------------------------------------------------------
# "Add metric" button
# ---------------------------------------------------------------------------


def make_add_metric_button() -> html.Div:
    return html.Div(
        html.Button(
            "+ Add Metric Axis",
            id="add-metric-btn",
            n_clicks=0,
            style={
                "width": "100%",
                "background": "transparent",
                "border": f"1px dashed {COLORS['border']}",
                "color": COLORS["text_muted"],
                "borderRadius": "8px",
                "padding": "8px",
                "cursor": "pointer",
                "fontSize": "12px",
                "transition": "border-color 0.2s, color 0.2s",
            },
        ),
        id="add-metric-btn-wrapper",
    )


# ---------------------------------------------------------------------------
# Right panel: fixed configuration
# ---------------------------------------------------------------------------

_tooltip_counter = 0


def _section_header(title: str, tooltip: str | None = None) -> html.Div:
    global _tooltip_counter
    header_style = {
        "fontSize": "10px",
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "0.08em",
        "color": COLORS["text_muted"],
        "marginBottom": "8px",
        "marginTop": "4px",
    }
    if tooltip is None:
        return html.Div(title, style=header_style)

    _tooltip_counter += 1
    target_id = f"help-icon-{_tooltip_counter}"

    return html.Div(
        style={**header_style, "display": "flex", "alignItems": "center", "gap": "6px"},
        children=[
            html.Span(title),
            html.Span(
                "?",
                id=target_id,
                className="help-icon",
            ),
            dbc.Tooltip(
                tooltip,
                target=target_id,
                placement="top",
                style={
                    "fontSize": "11px",
                    "maxWidth": "240px",
                    "textTransform": "none",
                    "letterSpacing": "normal",
                    "fontWeight": "400",
                },
            ),
        ],
    )


def _label(text: str, tooltip: str = "") -> html.Div:
    return html.Div(
        text,
        title=tooltip,
        style={
            "fontSize": "12px",
            "color": COLORS["text"],
            "marginBottom": "4px",
            "cursor": "help" if tooltip else "default",
        },
    )


_CONFIG_TAB_STYLE = {
    "padding": "6px 0",
    "fontSize": "11px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.05em",
    "borderBottom": "2px solid transparent",
    "color": COLORS["text_muted"],
    "background": "transparent",
    "border": "none",
    "cursor": "pointer",
}

_CONFIG_TAB_ACTIVE_STYLE = {
    **_CONFIG_TAB_STYLE,
    "color": COLORS["accent"],
    "borderBottom": f"2px solid {COLORS['accent']}",
}


def make_fixed_config_panel(swept_keys: set = None) -> html.Div:
    """
    Build the right-panel configuration controls as tabbed sections.

    All noise sliders are ALWAYS rendered so their IDs exist for Dash callbacks.
    Swept keys are visually hidden with ``display:none`` rather than removed.
    """
    import math

    swept_keys = swept_keys or set()

    # --- Circuit/Topology tab content ---
    circuit_content = html.Div(
        [
            _label("Circuit type"),
            dcc.Dropdown(
                id="cfg-circuit-type",
                options=CIRCUIT_TYPES,
                value="qft",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "10px"},
            ),
            html.Div(
                id="cfg-row-num-qubits",
                style={"display": "none"} if "num_qubits" in swept_keys else {},
                children=[
                    _label("Qubits"),
                    dcc.Slider(
                        id="cfg-num-qubits",
                        min=int(METRIC_BY_KEY["num_qubits"].slider_min),
                        max=int(METRIC_BY_KEY["num_qubits"].slider_max),
                        step=2,
                        value=16,
                        marks=_linear_marks(
                            METRIC_BY_KEY["num_qubits"].slider_min,
                            METRIC_BY_KEY["num_qubits"].slider_max,
                        ),
                        tooltip=_tooltip_cfg(False, "", always_visible=False),
                        updatemode="drag",
                        className="dse-slider",
                    ),
                    html.Div(style={"height": "10px"}),
                ],
            ),
            html.Div(
                id="cfg-row-num-cores",
                style={"display": "none"} if "num_cores" in swept_keys else {},
                children=[
                    _label("Cores"),
                    dcc.Slider(
                        id="cfg-num-cores",
                        min=int(METRIC_BY_KEY["num_cores"].slider_min),
                        max=int(METRIC_BY_KEY["num_cores"].slider_max),
                        step=1,
                        value=1,
                        marks={
                            str(i): str(i)
                            for i in range(
                                int(METRIC_BY_KEY["num_cores"].slider_min),
                                int(METRIC_BY_KEY["num_cores"].slider_max) + 1,
                            )
                        },
                        tooltip=_tooltip_cfg(False, "", always_visible=False),
                        updatemode="drag",
                        className="dse-slider",
                    ),
                    html.Div(style={"height": "10px"}),
                ],
            ),
            _label("Inter-core topology"),
            dcc.Dropdown(
                id="cfg-topology",
                options=TOPOLOGY_TYPES,
                value="ring",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "10px"},
            ),
            _label("Intra-core connectivity"),
            dcc.Dropdown(
                id="cfg-intracore-topology",
                options=INTRACORE_TOPOLOGY_TYPES,
                value="all_to_all",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "10px"},
            ),
            _label("Initial placement"),
            dcc.Dropdown(
                id="cfg-placement",
                options=PLACEMENT_OPTIONS,
                value="random",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "10px"},
            ),
            _label("Routing algorithm"),
            dcc.Dropdown(
                id="cfg-routing-algorithm",
                options=ROUTING_ALGORITHM_OPTIONS,
                value="hqa_sabre",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "10px"},
            ),
            _label("Seed"),
            dcc.Input(
                id="cfg-seed",
                type="number",
                value=42,
                min=0,
                debounce=True,
                className="dse-input",
                style={
                    "width": "100%",
                    "background": COLORS["surface2"],
                    "border": f"1px solid {COLORS['border']}",
                    "color": COLORS["text"],
                    "borderRadius": "6px",
                    "padding": "6px 10px",
                    "fontSize": "13px",
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                    "marginBottom": "10px",
                    "outline": "none",
                },
            ),
            _label("Dynamic decoupling"),
            dcc.Checklist(
                id="cfg-dynamic-decoupling",
                options=[{"label": " Enable", "value": "yes"}],
                value=[],
                style={
                    "color": COLORS["text"],
                    "fontSize": "13px",
                    "marginBottom": "12px",
                },
            ),
            _section_header(
                "Sweep Budget",
                tooltip=(
                    "Controls how many design points are evaluated. "
                    "Cold compilations are expensive and only "
                    "needed per unique structural (e.g. Qubits/Cores) combination. "
                    "Hot evaluations are near-free (batched in Rust) and cover "
                    "all noise parameter combinations. "
                ),
            ),
            _label(
                "Max cold compilations",
                "Caps unique (qubits, cores) combos. Each takes ~1-10s.",
            ),
            dcc.Input(
                id="cfg-max-cold",
                type="number",
                value=MAX_COLD_COMPILATIONS,
                min=1,
                max=1024,
                step=1,
                debounce=True,
                className="dse-input",
                style={
                    "width": "100%",
                    "background": COLORS["surface2"],
                    "border": f"1px solid {COLORS['border']}",
                    "color": COLORS["text"],
                    "borderRadius": "6px",
                    "padding": "6px 10px",
                    "fontSize": "13px",
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                    "marginBottom": "10px",
                    "outline": "none",
                },
            ),
            _label(
                "Max hot evaluations",
                "Total grid points. Hot path is batched in Rust (~free).",
            ),
            dcc.Input(
                id="cfg-max-hot",
                type="number",
                value=MAX_TOTAL_POINTS_HOT,
                min=100,
                max=100_000_000,
                step=100,
                debounce=True,
                className="dse-input",
                style={
                    "width": "100%",
                    "background": COLORS["surface2"],
                    "border": f"1px solid {COLORS['border']}",
                    "color": COLORS["text"],
                    "borderRadius": "6px",
                    "padding": "6px 10px",
                    "fontSize": "13px",
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                    "marginBottom": "10px",
                    "outline": "none",
                },
            ),
            html.Div(
                id="sweep-budget-warning",
                style={"display": "none"},
                children=[],
            ),
        ],
        style={"paddingTop": "8px"},
    )

    # --- Noise tab content ---
    noise_controls = []
    for m in SWEEPABLE_METRICS:
        marks = (
            _log_marks(m.slider_min, m.slider_max, m.unit)
            if m.log_scale
            else _linear_marks(m.slider_min, m.slider_max, unit=m.unit)
        )
        default_val = NOISE_DEFAULTS.get(m.key)
        if default_val is not None:
            default_slider = math.log10(default_val) if m.log_scale else default_val
        else:
            default_slider = m.slider_default_low
        hidden = m.is_cold_path or m.key in swept_keys
        noise_controls.append(
            html.Div(
                id=f"noise-row-{m.key}",
                style={"display": "none"} if hidden else {},
                children=[
                    _label(m.label, m.description),
                    dcc.Slider(
                        id=f"noise-{m.key}",
                        min=m.slider_min,
                        max=m.slider_max,
                        step=(m.slider_max - m.slider_min) / 200,
                        value=default_slider,
                        marks=marks,
                        tooltip=_tooltip_cfg(m.log_scale, m.unit),
                        updatemode="drag",
                        className="dse-slider",
                    ),
                    html.Div(style={"height": "12px"}),
                ],
            )
        )

    noise_content = html.Div(noise_controls, style={"paddingTop": "8px"})

    # --- Thresholds tab content ---
    threshold_content = html.Div(
        [
            _section_header("Output (Y-axis)"),
            dcc.Dropdown(
                id="cfg-output-metric",
                options=OUTPUT_METRICS,
                value="overall_fidelity",
                clearable=False,
                className="dse-dropdown",
                style={"marginBottom": "12px"},
            ),
            _section_header("Iso-levels"),
            dcc.Checklist(
                id="cfg-threshold-enable",
                options=[{"label": " Show on non-3D views", "value": "yes"}],
                value=[],
                style={
                    "color": COLORS["text"],
                    "fontSize": "12px",
                    "marginBottom": "8px",
                },
            ),
            *_make_threshold_inputs(),
        ],
        style={"paddingTop": "8px"},
    )

    return html.Div(
        id="fixed-config-panel",
        children=[
            dcc.Tabs(
                id="config-tabs",
                value="circuit",
                children=[
                    dcc.Tab(
                        label="Circuit",
                        value="circuit",
                        children=[circuit_content],
                        style=_CONFIG_TAB_STYLE,
                        selected_style=_CONFIG_TAB_ACTIVE_STYLE,
                    ),
                    dcc.Tab(
                        label="Noise",
                        value="noise",
                        children=[noise_content],
                        style=_CONFIG_TAB_STYLE,
                        selected_style=_CONFIG_TAB_ACTIVE_STYLE,
                    ),
                    dcc.Tab(
                        label="Thresholds",
                        value="thresholds",
                        children=[threshold_content],
                        style=_CONFIG_TAB_STYLE,
                        selected_style=_CONFIG_TAB_ACTIVE_STYLE,
                    ),
                ],
                style={"height": "auto"},
                content_style={"overflow": "auto"},
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Threshold inputs (up to 5 iso-levels, 3 pre-populated)
# ---------------------------------------------------------------------------

_THRESHOLD_DEFAULTS = [0.3, 0.6, 0.9]
_THRESHOLD_DEFAULT_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#91bfdb", "#4575b4"]
_MAX_THRESHOLDS = 5

_THRESHOLD_INPUT_STYLE = {
    "width": "60px",
    "background": COLORS["surface2"],
    "border": f"1px solid {COLORS['border']}",
    "color": COLORS["text"],
    "borderRadius": "4px",
    "padding": "4px 6px",
    "fontSize": "12px",
    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
    "outline": "none",
    "textAlign": "center",
}

_COLOR_SWATCH_STYLE = {
    "width": "24px",
    "height": "24px",
    "borderRadius": "4px",
    "border": f"1px solid {COLORS['border']}",
    "flexShrink": "0",
}

_COLOR_INPUT_STYLE = {
    "width": "70px",
    "background": COLORS["surface2"],
    "border": f"1px solid {COLORS['border']}",
    "color": COLORS["text"],
    "borderRadius": "4px",
    "padding": "3px 5px",
    "fontSize": "10px",
    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
    "outline": "none",
    "textAlign": "center",
}


def _make_threshold_inputs() -> list:
    children = []
    for i in range(_MAX_THRESHOLDS):
        default = _THRESHOLD_DEFAULTS[i] if i < len(_THRESHOLD_DEFAULTS) else None
        default_color = _THRESHOLD_DEFAULT_COLORS[i]
        visible = i < len(_THRESHOLD_DEFAULTS)
        children.append(
            html.Div(
                id=f"threshold-row-{i}",
                style={} if visible else {"display": "none"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "6px",
                            "marginBottom": "4px",
                        },
                        children=[
                            html.Div(
                                id=f"cfg-threshold-swatch-{i}",
                                style={
                                    **_COLOR_SWATCH_STYLE,
                                    "background": default_color,
                                },
                            ),
                            dcc.Input(
                                id=f"cfg-threshold-{i}",
                                type="number",
                                value=default,
                                min=0,
                                max=1,
                                step=0.01,
                                debounce=True,
                                style=_THRESHOLD_INPUT_STYLE,
                            ),
                            dcc.Input(
                                id=f"cfg-threshold-color-{i}",
                                type="text",
                                value=default_color,
                                debounce=True,
                                style=_COLOR_INPUT_STYLE,
                            ),
                        ],
                    ),
                ],
            )
        )
    children.append(
        html.Div(
            style={
                "display": "flex",
                "gap": "6px",
                "marginTop": "4px",
                "marginBottom": "12px",
            },
            children=[
                html.Button(
                    "+",
                    id="add-threshold-btn",
                    n_clicks=0,
                    style={
                        "background": "transparent",
                        "border": f"1px dashed {COLORS['border']}",
                        "color": COLORS["text_muted"],
                        "borderRadius": "4px",
                        "width": "28px",
                        "height": "28px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                    },
                ),
                html.Button(
                    "−",
                    id="remove-threshold-btn",
                    n_clicks=0,
                    style={
                        "background": "transparent",
                        "border": f"1px solid {COLORS['border']}",
                        "color": COLORS["text_muted"],
                        "borderRadius": "4px",
                        "width": "28px",
                        "height": "28px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "display": "none",
                    },
                ),
            ],
        )
    )
    return children


# ---------------------------------------------------------------------------
# View tab bar (above the plot area)
# ---------------------------------------------------------------------------


def _tab_button(tab: dict, is_active: bool) -> html.Button:
    return html.Button(
        tab["label"],
        id={"type": "view-tab-btn", "index": tab["value"]},
        n_clicks=0,
        style={
            "background": COLORS["accent"] if is_active else "transparent",
            "color": "#fff" if is_active else COLORS["text_muted"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "4px",
            "padding": "4px 14px",
            "fontSize": "12px",
            "fontWeight": "600" if is_active else "400",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
        },
    )


def make_view_tab_bar(num_metrics: int = 2, active: str | None = None) -> html.Div:
    sweep_tabs = VIEW_TABS.get(num_metrics, VIEW_TABS[1])
    if active is None:
        active = VIEW_TAB_DEFAULTS.get(num_metrics, sweep_tabs[0]["value"])

    children = [
        html.Span(
            "View",
            style={
                "fontSize": "10px",
                "fontWeight": "700",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
                "color": COLORS["text_muted"],
                "marginRight": "6px",
            },
        ),
    ]

    for tab in sweep_tabs:
        children.append(_tab_button(tab, tab["value"] == active))

    if ANALYSIS_TABS:
        children.append(
            html.Span(
                "|",
                style={
                    "color": COLORS["border"],
                    "margin": "0 4px",
                    "fontSize": "14px",
                },
            )
        )
        for tab in ANALYSIS_TABS:
            children.append(_tab_button(tab, tab["value"] == active))

    return html.Div(
        id="view-tab-bar",
        style={
            "display": "flex",
            "gap": "6px",
            "marginBottom": "8px",
            "alignItems": "center",
        },
        children=children,
    )
