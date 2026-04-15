"""
Reusable Dash UI component factories for the DSE GUI.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from .constants import (
    SWEEPABLE_METRICS,
    METRIC_BY_KEY,
    CIRCUIT_TYPES,
    TOPOLOGY_TYPES,
    PLACEMENT_OPTIONS,
    OUTPUT_METRICS,
    NOISE_DEFAULTS,
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
        v = 10 ** value
        if v < 1e-3:
            return f"{v:.2e}"
        if v < 1:
            return f"{v:.4f}"
        return f"{v:.1f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def _log_marks(slider_min: float, slider_max: float) -> dict:
    """Generate integer-exponent marks for a log-scale slider (string keys for orjson)."""
    marks = {}
    for exp in range(int(slider_min), int(slider_max) + 1):
        marks[str(exp)] = {"label": f"10^{exp}", "style": {"color": COLORS["text_muted"], "fontSize": "10px"}}
    return marks


def _linear_marks(slider_min: float, slider_max: float, n: int = 5) -> dict:
    import numpy as np
    vals = np.linspace(slider_min, slider_max, n)
    marks = {}
    for v in vals:
        label = f"{v:.2f}" if v != int(v) else str(int(v))
        marks[str(round(v, 6))] = {"label": label, "style": {"color": COLORS["text_muted"], "fontSize": "10px"}}
    return marks


# ---------------------------------------------------------------------------
# Metric selector (left sidebar row)
# ---------------------------------------------------------------------------

def make_metric_selector(index: int) -> html.Div:
    """
    One sweep metric row: dropdown (choose which param) + range slider.
    ``index`` is 0, 1, or 2.
    """
    default_metric = SWEEPABLE_METRICS[index % len(SWEEPABLE_METRICS)]
    m = default_metric

    marks = _log_marks(m.slider_min, m.slider_max) if m.log_scale else _linear_marks(m.slider_min, m.slider_max)

    return html.Div(
        id=f"metric-row-{index}",
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                children=[
                    html.Span(f"Metric {index + 1}", style={"color": COLORS["text_muted"], "fontSize": "11px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
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
                value=default_metric.key,
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
                        value=[m.slider_default_low, m.slider_default_high],
                        marks=marks,
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="dse-range-slider",
                    ),
                ],
                style={"paddingBottom": "22px"},
            ),
            html.Div(
                id=f"metric-range-label-{index}",
                style={"display": "flex", "justifyContent": "space-between", "fontSize": "11px", "color": COLORS["accent2"], "marginTop": "-20px"},
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

def _section_header(title: str) -> html.Div:
    return html.Div(
        title,
        style={
            "fontSize": "10px",
            "fontWeight": "700",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLORS["text_muted"],
            "marginBottom": "8px",
            "marginTop": "4px",
        },
    )


def _label(text: str, tooltip: str = "") -> html.Div:
    return html.Div(
        text,
        title=tooltip,
        style={"fontSize": "12px", "color": COLORS["text"], "marginBottom": "4px", "cursor": "help" if tooltip else "default"},
    )


def make_fixed_config_panel(swept_keys: set = None) -> html.Div:
    """
    Build the right-panel configuration controls.

    All noise sliders are ALWAYS rendered so their IDs exist for Dash callbacks.
    Swept keys are visually hidden with ``display:none`` rather than removed.
    """
    import math
    swept_keys = swept_keys or set()

    # --- Circuit/Topology section (always visible) ---
    circuit_section = html.Div([
        _section_header("Circuit"),

        _label("Type"),
        dcc.Dropdown(
            id="cfg-circuit-type",
            options=CIRCUIT_TYPES,
            value="qft",
            clearable=False,
            className="dse-dropdown",
            style={"marginBottom": "10px"},
        ),

        _label("Qubits"),
        dcc.Slider(
            id="cfg-num-qubits",
            min=4, max=80, step=2, value=16,
            marks={"4": "4", "20": "20", "40": "40", "60": "60", "80": "80"},
            tooltip={"placement": "bottom"},
            className="dse-slider",
        ),

        html.Div(style={"height": "14px"}),

        _section_header("Topology"),

        _label("Cores"),
        dcc.Slider(
            id="cfg-num-cores",
            min=1, max=16, step=1, value=4,
            marks={"1": "1", "4": "4", "8": "8", "12": "12", "16": "16"},
            tooltip={"placement": "bottom"},
            className="dse-slider",
        ),

        html.Div(style={"height": "18px"}),

        _label("Inter-core topology"),
        dcc.Dropdown(
            id="cfg-topology",
            options=TOPOLOGY_TYPES,
            value="ring",
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
            style={"color": COLORS["text"], "fontSize": "13px", "marginBottom": "12px"},
        ),
    ])

    # --- Noise parameters — always in the DOM, swept ones hidden ---
    noise_controls = []
    for m in SWEEPABLE_METRICS:
        marks = (
            _log_marks(m.slider_min, m.slider_max)
            if m.log_scale
            else _linear_marks(m.slider_min, m.slider_max)
        )
        default_slider = (
            math.log10(NOISE_DEFAULTS[m.key]) if m.log_scale else NOISE_DEFAULTS[m.key]
        )
        hidden = m.key in swept_keys
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
                        tooltip={"placement": "bottom"},
                        className="dse-slider",
                    ),
                    html.Div(style={"height": "12px"}),
                ],
            )
        )

    noise_section = html.Div([
        _section_header("Hardware noise"),
        *noise_controls,
    ])

    # --- Output metric selector ---
    output_section = html.Div([
        _section_header("Output (Y-axis)"),
        dcc.Dropdown(
            id="cfg-output-metric",
            options=OUTPUT_METRICS,
            value="overall_fidelity",
            clearable=False,
            className="dse-dropdown",
            style={"marginBottom": "10px"},
        ),
    ])

    return html.Div(
        id="fixed-config-panel",
        children=[circuit_section, noise_section, output_section],
    )
