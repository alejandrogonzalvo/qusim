"""
Reusable Dash UI component factories for the DSE GUI.
"""

import os

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import dcc, html

from .constants import (
    ANALYSIS_TABS,
    CAT_METRIC_BY_KEY,
    CATEGORICAL_METRICS,
    CIRCUIT_TYPES,
    DEFAULT_SWEEP_AXES,
    DEFAULT_VIEW_MODE,
    INTRACORE_TOPOLOGY_TYPES,
    MAX_COLD_COMPILATIONS,
    MAX_SWEEP_AXES,
    MAX_TOTAL_POINTS_HOT,
    MAX_WORKERS_DEFAULT,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    OUTPUT_METRICS,
    PLACEMENT_OPTIONS,
    ROUTING_ALGORITHM_OPTIONS,
    SWEEPABLE_METRICS,
    TOPOLOGY_TYPES,
    VIEW_MODES,
    VIEW_TAB_DEFAULTS,
    VIEW_TABS,
)

# Cap parallel cold-compile workers at half the CPU count so library thread
# pools (Rayon/OpenMP) — capped at 1 thread per worker elsewhere — still
# leave the main process and OS responsive even if a user cranks this up.
_WORKER_CAP = max(1, (os.cpu_count() or 2) // 2)

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
    # Brand accent — "Instrument Teal". Used sparingly: Run CTA,
    # active config-tab underline, axis chip, focus halo.
    "brand": "#0F5E6B",
    "brand_hover": "#0B4A54",
    "brand_press": "#083842",
    "brand_wash": "#E8F1F2",
    "brand_wash2": "#D3E4E6",
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


def _minmax_marks(slider_min: float, slider_max: float, log_scale: bool, unit: str = "") -> dict:
    """Return ONLY the two endpoint marks — drops intermediate noise.

    Used by all right-panel sliders so the user reads the exact value from
    the inline value-chip input, not from a forest of mid-axis tick labels.
    """
    if log_scale:
        return {
            str(int(slider_min)): {"label": _fmt_log_mark(int(slider_min), unit), "style": _MARK_STYLE},
            str(int(slider_max)): {"label": _fmt_log_mark(int(slider_max), unit), "style": _MARK_STYLE},
        }
    def _fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else f"{v:g}"
    return {
        str(round(float(slider_min), 6)): {"label": _fmt(slider_min), "style": _MARK_STYLE},
        str(round(float(slider_max), 6)): {"label": _fmt(slider_max), "style": _MARK_STYLE},
    }


def _format_value_for_input(value: float, log_scale: bool) -> str:
    """Display string for the value-chip input next to a slider.

    Mirrors ``_fmt_value`` but keeps the result parseable: scientific or
    plain decimal, no units, no separators.
    """
    if value is None:
        return ""
    if log_scale:
        return _fmt_value(value, True)
    if float(value) == int(value):
        return str(int(value))
    return f"{value:g}"


def slider_row(
    label: str,
    slider_id: str,
    *,
    min: float,
    max: float,
    value: float,
    step: float = 1,
    log_scale: bool = False,
    unit: str = "",
    tooltip: str = "",
    tooltip_visible: bool = False,
    row_id: str | None = None,
    row_style: dict | None = None,
) -> html.Div:
    """Build a slider with a label-row + editable value-chip + min/max marks.

    The input has id ``f"{slider_id}-input"``. A bidirectional sync
    callback (registered in ``app.py``) keeps slider and input in step.
    """
    input_id = f"{slider_id}-input"
    display = _format_value_for_input(value, log_scale)

    label_text = html.Span(
        label,
        style={"fontSize": "12px", "color": COLORS["text"]},
    )
    if tooltip:
        global _tooltip_counter
        _tooltip_counter += 1
        target_id = f"help-icon-{_tooltip_counter}"
        label_node = html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "6px",
                "minWidth": 0,
            },
            children=[
                label_text,
                html.Span("?", id=target_id, className="help-icon"),
                dbc.Tooltip(
                    tooltip,
                    target=target_id,
                    placement="top",
                    style={
                        "fontSize": "11px",
                        "maxWidth": "260px",
                        "textTransform": "none",
                        "letterSpacing": "normal",
                        "fontWeight": "400",
                    },
                ),
            ],
        )
    else:
        label_node = label_text

    header = html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "4px",
            "gap": "8px",
        },
        children=[
            label_node,
            dcc.Input(
                id=input_id,
                type="text",
                value=display,
                debounce=True,
                spellCheck=False,
                className="slider-value-chip",
            ),
        ],
    )
    slider = dcc.Slider(
        id=slider_id,
        min=min,
        max=max,
        step=step,
        value=value,
        marks=_minmax_marks(min, max, log_scale, unit),
        tooltip=_tooltip_cfg(log_scale, unit, always_visible=tooltip_visible),
        updatemode="drag",
        className="dse-slider",
    )
    children: list = [header, slider, html.Div(style={"height": "10px"})]
    div_kwargs: dict = {"children": children, "style": row_style or {}}
    if row_id is not None:
        div_kwargs["id"] = row_id
    return html.Div(**div_kwargs)


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
    One sweep metric row: dropdown (choose which param) + range slider (numeric)
    or checklist (categorical).  ``index`` is 0-based.
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

    all_options = (
        [{"label": nm.label, "value": nm.key} for nm in SWEEPABLE_METRICS]
        + [{"label": cat.label, "value": cat.key} for cat in CATEGORICAL_METRICS]
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
                        f"Axis {index + 1}",
                        className="axis-chip",
                    ),
                    html.Button(
                        "×",
                        id={"type": "remove-metric-x", "index": index},
                        n_clicks=0,
                        className="axis-remove-btn",
                        # `display` is the only style driven from Python — a
                        # callback in app.py keeps it in sync with
                        # num-metrics-store (hidden when only one axis remains).
                        style={"display": "flex"},
                    ),
                ],
            ),
            dcc.Dropdown(
                id=f"metric-dropdown-{index}",
                options=all_options,
                value=default_key,
                clearable=False,
                style={"marginBottom": "10px"},
                className="dse-dropdown",
            ),
            html.Div(
                id=f"metric-help-{index}",
                # Per-metric inline hint.  Hidden by default; populated by
                # the dropdown-change callback when the chosen metric has a
                # description worth surfacing inline (e.g. the ``qubits``
                # alias which folds Physical + Logical).
                style={"display": "none"},
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
                id=f"metric-checklist-container-{index}",
                children=[
                    dcc.Checklist(
                        id=f"metric-checklist-{index}",
                        options=[],
                        value=[],
                        style={"color": COLORS["text"], "fontSize": "12px"},
                        inputStyle={"marginRight": "4px"},
                    ),
                ],
                style={"display": "none"},
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
# Categorical section (left sidebar)
# ---------------------------------------------------------------------------


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


def _inline_toggle_row(label: str, checklist_id: str, tooltip: str = "") -> html.Div:
    """Single-line row: label (+ help icon) on the left, checkbox on the right.

    The checkbox is the existing ``dcc.Checklist`` with one option whose
    ``label`` is empty — just the box itself sits flush right.
    """
    global _tooltip_counter
    title_children: list = [
        html.Span(
            label,
            style={"fontSize": "13px", "color": COLORS["text"]},
        ),
    ]
    extras: list = []
    if tooltip:
        _tooltip_counter += 1
        target_id = f"help-icon-{_tooltip_counter}"
        title_children.append(
            html.Span(
                "?",
                id=target_id,
                className="help-icon",
                style={"marginLeft": "6px"},
            )
        )
        extras.append(
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
            )
        )
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "8px",
            "padding": "6px 0",
            "marginBottom": "6px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center"},
                children=title_children,
            ),
            dcc.Checklist(
                id=checklist_id,
                options=[{"label": "", "value": "yes"}],
                value=[],
                style={
                    "margin": "0",
                    "lineHeight": "1",
                },
                inputStyle={"margin": "0", "cursor": "pointer"},
            ),
            *extras,
        ],
    )


def _label(text: str, tooltip: str = "") -> html.Div:
    base_style = {
        "fontSize": "12px",
        "color": COLORS["text"],
        "marginBottom": "4px",
    }
    if not tooltip:
        return html.Div(text, style=base_style)

    global _tooltip_counter
    _tooltip_counter += 1
    target_id = f"help-icon-{_tooltip_counter}"
    return html.Div(
        style={**base_style, "display": "flex", "alignItems": "center", "gap": "6px"},
        children=[
            html.Span(text),
            html.Span("?", id=target_id, className="help-icon"),
            dbc.Tooltip(
                tooltip,
                target=target_id,
                placement="top",
                style={
                    "fontSize": "11px",
                    "maxWidth": "260px",
                    "textTransform": "none",
                    "letterSpacing": "normal",
                    "fontWeight": "400",
                },
            ),
        ],
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
    "color": COLORS["brand"],
    "borderBottom": f"2px solid {COLORS['brand']}",
}


_QISKIT_QASM_SNIPPET = """from qiskit import QuantumCircuit, qasm2

# Build (or load) your *logical* circuit — no transpiling, no
# coupling-map, no swap insertion. Quadris re-routes for you.
qc = QuantumCircuit(5)
qc.h(0)
for i in range(1, 5):
    qc.cx(0, i)

# OpenQASM 2.0 export — what the upload control expects.
qasm_str = qasm2.dumps(qc)

with open("my_circuit.qasm", "w") as f:
    f.write(qasm_str)
"""


def make_custom_qasm_help_modal() -> dbc.Modal:
    """Modal explaining how to export a logical OpenQASM file from Qiskit."""
    return dbc.Modal(
        id="custom-qasm-help-modal",
        is_open=False,
        size="lg",
        children=[
            dbc.ModalHeader(dbc.ModalTitle("Exporting a logical circuit from Qiskit")),
            dbc.ModalBody(
                children=[
                    html.P(
                        "Upload a logical OpenQASM 2.0 file — one that has not "
                        "been compiled for any specific topology or coupling "
                        "map. Quadris performs its own placement and routing, "
                        "so any pre-inserted SWAPs would be applied on top of "
                        "the routed program and skew the design-space results.",
                        style={"fontSize": "13px", "marginBottom": "10px"},
                    ),
                    html.Pre(
                        _QISKIT_QASM_SNIPPET,
                        style={
                            "background": COLORS["surface2"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "6px",
                            "padding": "10px 12px",
                            "fontSize": "12px",
                            "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                            "color": COLORS["text"],
                            "whiteSpace": "pre",
                            "overflowX": "auto",
                            "margin": "0",
                        },
                    ),
                ],
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Close",
                    id="custom-qasm-help-close",
                    className="ghost-btn",
                    n_clicks=0,
                ),
            ),
        ],
    )


def make_custom_qasm_row() -> html.Div:
    """
    "Custom circuit" row in the Circuit tab.  Combines a compact upload
    button, a help icon (opens the Qiskit-export modal), and a status
    line showing the currently uploaded file (with a Clear button).
    """
    return html.Div(
        id="cfg-row-custom-qasm",
        style={"marginBottom": "10px"},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginBottom": "4px",
                },
                children=[
                    html.Span(
                        "Custom circuit",
                        style={"fontSize": "12px", "color": COLORS["text"]},
                    ),
                    html.Span(
                        "?",
                        id="custom-qasm-help-icon",
                        className="help-icon",
                        n_clicks=0,
                        style={"cursor": "pointer"},
                    ),
                    dbc.Tooltip(
                        "How to export a logical OpenQASM 2.0 file from Qiskit",
                        target="custom-qasm-help-icon",
                        placement="top",
                        style={
                            "fontSize": "11px",
                            "maxWidth": "240px",
                            "fontWeight": "400",
                        },
                    ),
                ],
            ),
            dcc.Upload(
                id="custom-qasm-upload",
                multiple=False,
                accept=".qasm,.txt",
                children=html.Div(
                    id="custom-qasm-upload-label",
                    children="Upload .qasm",
                    style={
                        "border": f"1px dashed {COLORS['border']}",
                        "borderRadius": "6px",
                        "padding": "8px 10px",
                        "fontSize": "12px",
                        "color": COLORS["text_muted"],
                        "textAlign": "center",
                        "cursor": "pointer",
                        "background": COLORS["surface2"],
                    },
                ),
                style_active={},
            ),
            html.Div(
                id="custom-qasm-status",
                style={"display": "none"},
                children=[],
            ),
        ],
    )


def make_fixed_config_panel(swept_keys: set = None) -> html.Div:
    """
    Build the right-panel configuration controls as tabbed sections.

    All noise sliders are ALWAYS rendered so their IDs exist for Dash callbacks.
    Swept keys are visually hidden with ``display:none`` rather than removed.
    """
    import math

    swept_keys = swept_keys or set()

    # Architecture defaults — cores pinned (always feasible) at startup.
    _DEFAULT_NC = 1
    _DEFAULT_QPC = 16
    _DEFAULT_LOGICAL = 16

    # --- Circuit tab content ---
    circuit_content = html.Div(
        [
            make_custom_qasm_row(),
            html.Div(
                id="cfg-row-num-logical-qubits-wrap",
                children=slider_row(
                    label="Logical qubits",
                    slider_id="cfg-num-logical-qubits",
                    min=int(METRIC_BY_KEY["num_logical_qubits"].slider_min),
                    max=int(METRIC_BY_KEY["num_logical_qubits"].slider_max),
                    step=1,
                    value=_DEFAULT_LOGICAL,
                    log_scale=False,
                    tooltip=(
                        "Number of qubits used by the algorithm circuit. "
                        "Held constant during sweeps — the unpinned "
                        "architectural axis (cores or qubits/core) grows "
                        "to absorb comm/buffer overhead. Auto-set when a "
                        "custom QASM file is uploaded."
                    ),
                    row_id="cfg-row-num-logical-qubits",
                    row_style=({"display": "none"} if "num_logical_qubits" in swept_keys else {}),
                ),
            ),
            html.Div(
                id="cfg-row-seed",
                children=[
                    _label(
                        "Seed",
                        "Random seed for placement and routing. Same circuit "
                        "+ topology + seed always compile to the same schedule "
                        "— change it to explore alternative qubit mappings.",
                    ),
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
                ],
            ),
            html.Div(
                id="cfg-row-cat-circuit_type-wrap",
                children=html.Div(
                    id="cfg-row-cat-circuit_type",
                    children=[
                        _label(
                            "Circuit type",
                            "Algorithm to compile and simulate. QFT exercises "
                            "long-range entanglement; GHZ stresses comm qubits; "
                            "Random gives a synthetic high-depth benchmark. "
                            "Upload an OpenQASM file at the top of this tab to "
                            "run a custom circuit instead.",
                        ),
                        dcc.Dropdown(
                            id="cfg-circuit-type",
                            options=CIRCUIT_TYPES,
                            value="qft",
                            clearable=False,
                            className="dse-dropdown",
                            style={"marginBottom": "10px"},
                        ),
                    ],
                ),
            ),
            html.Div(
                id="cfg-row-cat-placement",
                children=[
                    _label(
                        "Placement",
                        "Initial mapping of logical qubits onto physical qubits, "
                        "before routing. Random shuffles uniformly; Spectral "
                        "Clustering uses a graph-Laplacian heuristic that keeps "
                        "frequently-interacting logical qubits on the same core.",
                    ),
                    dcc.Dropdown(
                        id="cfg-placement",
                        options=PLACEMENT_OPTIONS,
                        value="random",
                        clearable=False,
                        className="dse-dropdown",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            html.Div(
                id="cfg-row-cat-routing_algorithm",
                children=[
                    _label(
                        "Routing algorithm",
                        "Compiler pass that resolves coupling-map mismatches. "
                        "HQA + Sabre runs the Hierarchical Quantum Architecture "
                        "remapper followed by SABRE swap insertion; TeleSABRE "
                        "inserts inter-core teleportations when cheaper than "
                        "chains of SWAPs.",
                    ),
                    dcc.Dropdown(
                        id="cfg-routing-algorithm",
                        options=ROUTING_ALGORITHM_OPTIONS,
                        value="hqa_sabre",
                        clearable=False,
                        className="dse-dropdown",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            _inline_toggle_row(
                label="Dynamic decoupling",
                checklist_id="cfg-dynamic-decoupling",
                tooltip=(
                    "Insert idle-qubit refocusing pulses (CPMG / XY-4) "
                    "during routing waits to suppress dephasing. Trades a "
                    "small gate budget for higher coherence-limited fidelity "
                    "on long algorithms."
                ),
            ),
        ],
        style={"paddingTop": "8px"},
    )

    # --- Topology tab content ---
    # Pin toggle: exactly one of {Cores, Qubits per core} is the user-set
    # architectural input; the other is derived. A segmented control at
    # the top of the tab makes the choice unambiguous; only the active
    # slider renders, the inactive one shows a derived-value display.
    _pin_default = "cores"

    def _segmented_pin_toggle(pin_value: str) -> html.Div:
        """Two-button segmented control for the pin axis."""
        active_style = {
            "flex": "1",
            "padding": "8px 10px",
            "fontSize": "12px",
            "fontWeight": "600",
            "border": f"1px solid {COLORS['brand']}",
            "background": COLORS["brand"],
            "color": "white",
            "cursor": "pointer",
            "textAlign": "center",
            "userSelect": "none",
        }
        inactive_style = {
            "flex": "1",
            "padding": "8px 10px",
            "fontSize": "12px",
            "fontWeight": "500",
            "border": f"1px solid {COLORS['border']}",
            "background": COLORS["surface2"],
            "color": COLORS["text"],
            "cursor": "pointer",
            "textAlign": "center",
            "userSelect": "none",
        }
        cores_active = (pin_value == "cores")
        return html.Div(
            style={
                "display": "flex",
                "alignItems": "stretch",
                "gap": "0",
                "borderRadius": "6px",
                "overflow": "hidden",
                "marginBottom": "10px",
            },
            children=[
                html.Div(
                    "Cores",
                    id="cfg-pin-cores-btn",
                    n_clicks=0,
                    style=(active_style if cores_active else inactive_style),
                ),
                html.Div(
                    "Qubits per core",
                    id="cfg-pin-qpc-btn",
                    n_clicks=0,
                    style=(inactive_style if cores_active else active_style),
                ),
            ],
        )

    def _arch_row(
        *, label: str, slider_id: str,
        slider_min: int, slider_max: int, value: int,
        pin_value: str, axis_key: str, derived_id: str,
        row_id: str, swept_key: str, tooltip: str,
    ) -> html.Div:
        """One architectural axis: either the slider (when pinned) or a
        derived-value display (when the *other* axis is pinned). Hidden
        outright when this axis is being swept (handled upstream)."""
        is_active = (axis_key == pin_value)
        return html.Div(
            id=row_id,
            style=({"display": "none"} if (swept_key in swept_keys) else {}),
            children=[
                html.Div(
                    id=f"{row_id}-slider-row",
                    style=({} if is_active else {"display": "none"}),
                    children=[
                        slider_row(
                            label=label,
                            slider_id=slider_id,
                            min=slider_min,
                            max=slider_max,
                            step=1,
                            value=value,
                            log_scale=False,
                            tooltip=tooltip,
                            row_id=f"{row_id}-slider-inner",
                        ),
                    ],
                ),
                html.Div(
                    id=derived_id,
                    style=(
                        {"display": "none"} if is_active else {
                            "padding": "6px 8px",
                            "background": COLORS["surface2"],
                            "border": f"1px dashed {COLORS['border']}",
                            "borderRadius": "6px",
                            "fontSize": "12px",
                            "color": COLORS["text_muted"],
                            "marginBottom": "10px",
                        }
                    ),
                    children=f"{label}: (derived)",
                ),
            ],
        )

    topology_content = html.Div(
        [
            dcc.Store(id="cfg-pin-axis", data=_pin_default),
            _segmented_pin_toggle(_pin_default),
            _arch_row(
                label="Cores",
                slider_id="cfg-num-cores",
                slider_min=int(METRIC_BY_KEY["num_cores"].slider_min),
                slider_max=int(METRIC_BY_KEY["num_cores"].slider_max),
                value=_DEFAULT_NC,
                pin_value=_pin_default,
                axis_key="cores",
                derived_id="cfg-num-cores-derived",
                row_id="cfg-row-num-cores",
                swept_key="num_cores",
                tooltip=(
                    "Number of processor cores. Sweepable only when "
                    "*I set the cores* is selected."
                ),
            ),
            _arch_row(
                label="Qubits per core",
                slider_id="cfg-qubits-per-core",
                slider_min=int(METRIC_BY_KEY["qubits_per_core"].slider_min),
                slider_max=int(METRIC_BY_KEY["qubits_per_core"].slider_max),
                value=_DEFAULT_QPC,
                pin_value=_pin_default,
                axis_key="qubits_per_core",
                derived_id="cfg-qubits-per-core-derived",
                row_id="cfg-row-qubits-per-core",
                swept_key="qubits_per_core",
                tooltip=(
                    "Slots per core (uniform across the chip). Sweepable "
                    "only when *I set the qubits/core* is selected."
                ),
            ),
            html.Div(
                id="cfg-architecture-summary",
                style={
                    "fontSize": "12px",
                    "color": COLORS["text_muted"],
                    "padding": "6px 8px",
                    "background": COLORS["surface2"],
                    "borderRadius": "6px",
                    "margin": "4px 0 10px 0",
                },
                children="(architecture summary)",
            ),
            slider_row(
                label="Communication qubits",
                slider_id="cfg-communication-qubits",
                min=int(METRIC_BY_KEY["communication_qubits"].slider_min),
                max=int(METRIC_BY_KEY["communication_qubits"].slider_max),
                step=1,
                value=1,
                log_scale=False,
                tooltip=(
                    "Comm qubits per group (per inter-core link). Each of "
                    "a core's G_max(topology) inter-core neighbours each "
                    "reserves K + B slots (K comm + B buffer). Idle slots "
                    "at corner/edge cores in non-uniform topologies count "
                    "in the per-core reservation but carry no edges."
                ),
                row_id="cfg-row-communication-qubits",
                row_style=({"display": "none"} if "communication_qubits" in swept_keys else {}),
            ),
            slider_row(
                label="Buffer qubits",
                slider_id="cfg-buffer-qubits",
                min=int(METRIC_BY_KEY["buffer_qubits"].slider_min),
                max=int(METRIC_BY_KEY["buffer_qubits"].slider_max),
                step=1,
                value=1,
                log_scale=False,
                tooltip=(
                    "Buffer qubits per group. Sits adjacent to comm slots "
                    "as the local landing slot during teleportation. "
                    "Per-group rule: B ≤ K (clamped on this slider)."
                ),
                row_id="cfg-row-buffer-qubits",
                row_style=({"display": "none"} if "buffer_qubits" in swept_keys else {}),
            ),
            html.Div(
                id="cfg-row-cat-topology_type",
                children=[
                    _label(
                        "Inter-core topology",
                        "How cores are connected to each other. Ring connects "
                        "each core to its 2 neighbours; All-to-All connects "
                        "every pair (lots of EPR links); Linear Chain has "
                        "fixed endpoints. Affects teleportation hop count "
                        "between distant cores.",
                    ),
                    dcc.Dropdown(
                        id="cfg-topology",
                        options=TOPOLOGY_TYPES,
                        value="ring",
                        clearable=False,
                        className="dse-dropdown",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
            html.Div(
                id="cfg-row-cat-intracore_topology",
                children=[
                    _label(
                        "Intra-core topology",
                        "Coupling map within each core. All-to-All assumes any "
                        "pair of qubits can interact directly (1-cycle SWAPs); "
                        "Linear / Ring / Grid restrict interactions, forcing "
                        "more SWAPs but reflecting realistic hardware.",
                    ),
                    dcc.Dropdown(
                        id="cfg-intracore-topology",
                        options=INTRACORE_TOPOLOGY_TYPES,
                        value="all_to_all",
                        clearable=False,
                        className="dse-dropdown",
                        style={"marginBottom": "10px"},
                    ),
                ],
            ),
        ],
        style={"paddingTop": "8px"},
    )

    # --- Noise tab content ---
    noise_controls = []
    for m in SWEEPABLE_METRICS:
        default_val = NOISE_DEFAULTS.get(m.key)
        if default_val is not None:
            default_slider = math.log10(default_val) if m.log_scale else default_val
        else:
            default_slider = m.slider_default_low
        hidden = m.is_cold_path or m.key in swept_keys
        noise_controls.append(
            slider_row(
                label=m.label,
                slider_id=f"noise-{m.key}",
                min=m.slider_min,
                max=m.slider_max,
                step=(m.slider_max - m.slider_min) / 200,
                value=default_slider,
                log_scale=m.log_scale,
                unit=m.unit,
                tooltip=m.description,
                row_id=f"noise-row-{m.key}",
                row_style=({"display": "none"} if hidden else {}),
            )
        )

    noise_content = html.Div(noise_controls, style={"paddingTop": "8px"})

    output_content = html.Div(
        children=_output_tab_children(),
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
                        label="Topology",
                        value="topology",
                        children=[topology_content],
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
                        label="Output",
                        value="output",
                        children=[output_content],
                        style=_CONFIG_TAB_STYLE,
                        selected_style=_CONFIG_TAB_ACTIVE_STYLE,
                    ),
                ],
                style={"height": "auto"},
                content_style={"overflow": "auto"},
            ),
        ],
    )


def _output_tab_children() -> list:
    """Children for the Output config tab: Y-axis metric + iso-levels.

    Same component IDs as the legacy ``make_output_panel`` so existing
    threshold/iso-line callbacks keep working. The collapsible header is
    dropped — the tab itself is the affordance now.
    """
    return [
        _label(
            "Output (Y-axis)",
            "Metric to plot on the Y-axis (1D), the colour scale (2D / "
            "scatter), or the surface height (3D). Switching this re-renders "
            "the current view from the cached sweep — no re-run needed.",
        ),
        dcc.Dropdown(
            id="cfg-output-metric",
            options=OUTPUT_METRICS,
            value="overall_fidelity",
            clearable=False,
            className="dse-dropdown",
            style={"marginBottom": "10px"},
        ),
        _label(
            "View mode",
            "What every dimensional view actually plots. Absolute = the "
            "output value as is. |∇F| = gradient magnitude (single scalar "
            "per cell summarising how steep F is in any direction — low "
            "values mark robust regions, high values mark sensitive ones). "
            "Elasticity (1-D only) = (x/F)·dF/dx, the % change in F per % "
            "change in x. d²F/dx² (1-D only) = curvature, with the "
            "inflection point auto-marked as a vertical guide — that's the "
            "diminishing-returns sweet spot. ∂²F/∂x∂y (2-D only) = "
            "Savitzky-Golay-smoothed mixed partial; positive = the two "
            "axes synergise, negative = they substitute. For multi-axis "
            "elasticity comparison use the dedicated Elasticity tab.",
        ),
        dcc.Dropdown(
            id="cfg-view-mode",
            options=VIEW_MODES,
            value=DEFAULT_VIEW_MODE,
            clearable=False,
            className="dse-dropdown",
            style={"marginBottom": "10px"},
        ),
        _label(
            "Iso-levels",
            "Threshold values overlaid on the plot as contour lines (2D) or "
            "nested isosurfaces (3D). Each row is a (value, colour) pair; +/− "
            "below adds or removes rows. Use to mark target performance bands.",
        ),
        # Iso-levels are always shown on every applicable view. The hidden
        # Checklist preserves the ``cfg-threshold-enable`` callback wiring
        # (read by sweep / re-render / CSV / session callbacks) without
        # exposing a user-toggleable control.
        dcc.Checklist(
            id="cfg-threshold-enable",
            options=[{"label": "", "value": "yes"}],
            value=["yes"],
            style={"display": "none"},
        ),
        *_make_threshold_inputs(),
    ]


def _collapsible_header(title: str, section_id: str, tooltip: str | None = None) -> html.Div:
    """Header row with a chevron toggle for a collapsible bottom section."""
    label = [
        html.Span(
            id=f"{section_id}-chevron",
            children="▾",
            style={
                "fontSize": "10px",
                "color": COLORS["text_muted"],
                "marginRight": "6px",
                "display": "inline-block",
                "width": "10px",
            },
        ),
        html.Span(title),
    ]
    if tooltip:
        global _tooltip_counter
        _tooltip_counter += 1
        target_id = f"help-icon-{_tooltip_counter}"
        label.append(
            html.Span(
                "?",
                id=target_id,
                className="help-icon",
                style={"marginLeft": "6px"},
            )
        )
        return html.Div(
            id=f"{section_id}-header",
            n_clicks=0,
            style={
                "fontSize": "10px",
                "fontWeight": "700",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
                "color": COLORS["text_muted"],
                "display": "flex",
                "alignItems": "center",
                "padding": "6px 0 4px",
                "cursor": "pointer",
                "userSelect": "none",
                "borderTop": f"1px solid {COLORS['border']}",
            },
            children=[
                *label,
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
    return html.Div(
        id=f"{section_id}-header",
        n_clicks=0,
        style={
            "fontSize": "10px",
            "fontWeight": "700",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLORS["text_muted"],
            "display": "flex",
            "alignItems": "center",
            "padding": "6px 0 4px",
            "cursor": "pointer",
            "userSelect": "none",
            "borderTop": f"1px solid {COLORS['border']}",
        },
        children=label,
    )



def make_performance_panel() -> html.Div:
    """Sweep Budget — panel-level collapsible footer.

    Always visible at the bottom of the right sidebar across all config
    tabs. Default-collapsed: shows a one-line summary strip (``64 cold ·
    5,000 hot · 1w``); click expands the three inputs.
    """
    summary_default = (
        f"{MAX_COLD_COMPILATIONS} cold · "
        f"{MAX_TOTAL_POINTS_HOT:,} hot · {MAX_WORKERS_DEFAULT}w"
    )
    return html.Div(
        id="performance-panel",
        style={
            "borderTop": f"1px solid {COLORS['border']}",
            "background": COLORS["surface"],
            "flexShrink": "0",
        },
        children=[
            html.Div(
                id="sweep-budget-section-header",
                n_clicks=0,
                style={
                    "width": "100%",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "padding": "10px 14px",
                    "cursor": "pointer",
                    "userSelect": "none",
                },
                children=[
                    html.Span(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "6px",
                            "fontSize": "10px",
                            "fontWeight": "700",
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "color": COLORS["accent"],
                        },
                        children=[
                            html.Span(
                                id="sweep-budget-section-chevron",
                                children="▸",
                                style={
                                    "fontSize": "9px",
                                    "color": COLORS["text_muted"],
                                    "display": "inline-block",
                                    "width": "10px",
                                },
                            ),
                            html.Span("Sweep Budget"),
                        ],
                    ),
                    html.Span(
                        id="sweep-budget-summary",
                        children=summary_default,
                        style={
                            "fontSize": "10px",
                            "color": COLORS["text_muted"],
                            "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                        },
                    ),
                ],
            ),
            html.Div(
                id="sweep-budget-section-body",
                style={"display": "none", "padding": "0 14px 12px"},
                children=[
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
                    _label(
                        "Max workers",
                        "Parallel cold compilations. Each worker holds its own "
                        "copy of the routed circuit in RAM.",
                    ),
                    dcc.Input(
                        id="cfg-max-workers",
                        type="number",
                        value=MAX_WORKERS_DEFAULT,
                        min=1,
                        max=_WORKER_CAP,
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
                    html.Div(
                        id="sweep-workers-warning",
                        style={"display": "none"},
                        children=[],
                    ),
                    html.Div(
                        id="sweep-budget-warning",
                        style={"display": "none"},
                        children=[],
                    ),
                ],
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
            "background": COLORS["brand_wash"] if is_active else "transparent",
            "color": COLORS["brand"] if is_active else COLORS["text_muted"],
            "border": f"1px solid {COLORS['brand'] if is_active else COLORS['border']}",
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
            "flexWrap": "wrap",
            "gap": "6px",
            "rowGap": "4px",
            "marginBottom": "8px",
            "alignItems": "center",
            "minWidth": "0",
        },
        children=children,
    )


# ---------------------------------------------------------------------------
# Figure-of-Merit *view* controls — mode toggle + per-mode selectors that sit
# above the Plotly canvas in the Merit tab. The formula editor below the plot
# is rendered separately by ``make_merit_controls``.
# ---------------------------------------------------------------------------


def _merit_mode_button(label: str, mode: str, is_active: bool) -> html.Button:
    return html.Button(
        label,
        id={"type": "merit-mode-btn", "index": mode},
        n_clicks=0,
        style={
            "background": COLORS["accent"] if is_active else "transparent",
            "color": "#fff" if is_active else COLORS["text_muted"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "4px",
            "padding": "3px 12px",
            "fontSize": "11px",
            "fontWeight": "600" if is_active else "400",
            "cursor": "pointer",
            "transition": "all 0.15s ease",
        },
    )


def make_merit_view_controls() -> html.Div:
    """Mode toggle + per-mode selectors that render above the Plotly canvas
    in the Merit tab.

    All sub-rows are rendered eagerly so callbacks can flip ``display`` on
    the relevant containers without re-creating the DOM (avoids losing the
    Plotly graph wrapper between mode switches).
    """
    label_style = {
        "fontSize": "10px",
        "fontWeight": "700",
        "textTransform": "uppercase",
        "letterSpacing": "0.06em",
        "color": COLORS["text_muted"],
        "marginRight": "6px",
    }

    # One pre-rendered frozen-slider row per possible sweep axis. Each row is
    # toggled visible by the merit-frozen-slider-config callback when the
    # corresponding axis is active and not selected as X or Y.
    frozen_rows = []
    for i in range(MAX_SWEEP_AXES):
        frozen_rows.append(
            html.Div(
                id={"type": "merit-frozen-slider-row", "index": i},
                style={"display": "none"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "padding": "2px 0",
                        },
                        children=[
                            html.Span(
                                id={"type": "merit-frozen-slider-label", "index": i},
                                style={
                                    "width": "150px",
                                    "flexShrink": "0",
                                    "fontSize": "11px",
                                    "color": COLORS["text_muted"],
                                    "whiteSpace": "nowrap",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                },
                                children="",
                            ),
                            dcc.Slider(
                                id={"type": "merit-frozen-slider", "index": i},
                                min=0, max=1, step=None, value=0,
                                marks={},
                                tooltip={"placement": "bottom",
                                         "always_visible": False},
                                updatemode="drag",
                                included=False,
                            ),
                            html.Span(
                                id={"type": "merit-frozen-slider-value", "index": i},
                                style={
                                    "fontSize": "11px",
                                    "color": COLORS["accent"],
                                    "minWidth": "70px",
                                    "textAlign": "right",
                                },
                            ),
                        ],
                    ),
                ],
            )
        )

    return html.Div(
        id="merit-view-controls-container",
        style={
            "display": "none",
            "padding": "6px 16px 6px",
            "borderBottom": f"1px solid {COLORS['border']}",
            "background": COLORS["bg"],
        },
        children=[
            # Row 1: mode toggle (always visible when the container shows).
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "marginBottom": "6px",
                },
                children=[
                    html.Span("Mode", style=label_style),
                    _merit_mode_button("Heatmap", "heatmap", True),
                    _merit_mode_button("3D", "3d", False),
                    _merit_mode_button("Pareto", "pareto", False),
                ],
            ),
            # Row 2: heatmap controls — X/Y axis dropdowns.
            html.Div(
                id="merit-heatmap-controls",
                style={"display": "block"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "marginBottom": "4px",
                        },
                        children=[
                            html.Span("X", style=label_style),
                            html.Div(
                                style={"width": "180px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="merit-x-axis-dropdown",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                            html.Span("Y", style={**label_style, "marginLeft": "12px"}),
                            html.Div(
                                style={"width": "180px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="merit-y-axis-dropdown",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        id="merit-frozen-sliders-container",
                        style={"display": "block"},
                        children=frozen_rows,
                    ),
                ],
            ),
            # Row 3: pareto controls — color-by dropdown.
            html.Div(
                id="merit-pareto-controls",
                style={"display": "none"},
                children=[
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                        },
                        children=[
                            html.Span("Color by", style=label_style),
                            html.Div(
                                style={"width": "220px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="merit-color-by-dropdown",
                                    options=[
                                        {"label": "FoM", "value": "fom"},
                                        {"label": "None", "value": "none"},
                                    ],
                                    value="fom",
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Figure-of-Merit controls (shown above plot when the Merit view is active)
# ---------------------------------------------------------------------------


_FOM_FIELD_STYLE = {
    "width": "100%",
    "background": COLORS["surface2"],
    "border": f"1px solid {COLORS['border']}",
    "color": COLORS["text"],
    "borderRadius": "4px",
    "padding": "4px 8px",
    "fontSize": "12px",
    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
    "outline": "none",
}

_FOM_LABEL_STYLE = {
    "fontSize": "10px",
    "fontWeight": "700",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "color": COLORS["text_muted"],
    "marginBottom": "2px",
    "display": "block",
}


def make_merit_controls() -> html.Div:
    """Formula panel shown above the main plot when the Merit view is active.

    Importing ``PRESET_OPTIONS`` and ``DEFAULT_FOM`` locally keeps this module
    decoupled from ``gui.fom`` for consumers that don't render the panel.
    """
    from .fom import DEFAULT_FOM, PRESET_OPTIONS

    # Render intermediates as "name = expr" lines in a single textarea. That
    # avoids the complexity of pattern-matched dynamic rows and still round-
    # trips cleanly to/from the FoM store.
    intermediates_text = "\n".join(
        f"{name} = {expr}" for name, expr in DEFAULT_FOM.intermediates
    )

    row_style = {
        "display": "flex",
        "gap": "10px",
        "marginBottom": "6px",
    }

    return html.Div(
        id="merit-controls-container",
        style={
            "display": "none",
            "padding": "6px 16px 10px",
            "borderTop": f"1px solid {COLORS['border']}",
        },
        children=[
            html.Div(
                style=row_style,
                children=[
                    html.Div(
                        style={"flex": "0 0 220px"},
                        children=[
                            html.Label("Preset", style=_FOM_LABEL_STYLE),
                            dcc.Dropdown(
                                id="fom-preset",
                                className="dse-dropdown dse-dropdown-up",
                                options=PRESET_OPTIONS,
                                value="fidelity_per_epr",
                                clearable=False,
                                searchable=False,
                                style={"fontSize": "11px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 auto"},
                        children=[
                            html.Label("Name", style=_FOM_LABEL_STYLE),
                            dcc.Input(
                                id="fom-name",
                                type="text",
                                value=DEFAULT_FOM.name,
                                debounce=True,
                                style=_FOM_FIELD_STYLE,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style=row_style,
                children=[
                    html.Div(
                        style={"flex": "1 1 0"},
                        children=[
                            html.Label("Numerator", style=_FOM_LABEL_STYLE),
                            dcc.Input(
                                id="fom-numerator",
                                type="text",
                                value=DEFAULT_FOM.numerator,
                                debounce=True,
                                style=_FOM_FIELD_STYLE,
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 0"},
                        children=[
                            html.Label("Denominator", style=_FOM_LABEL_STYLE),
                            dcc.Input(
                                id="fom-denominator",
                                type="text",
                                value=DEFAULT_FOM.denominator,
                                debounce=True,
                                style=_FOM_FIELD_STYLE,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"marginBottom": "4px"},
                children=[
                    html.Label(
                        "Intermediates  (one per line, e.g. cost = total_epr_pairs + 1e-9 * total_circuit_time_ns)",
                        style=_FOM_LABEL_STYLE,
                    ),
                    dcc.Textarea(
                        id="fom-intermediates",
                        value=intermediates_text,
                        style={
                            **_FOM_FIELD_STYLE,
                            "height": "54px",
                            "resize": "vertical",
                        },
                    ),
                ],
            ),
            html.Div(
                id="fom-status",
                style={
                    "fontSize": "11px",
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                    "color": COLORS["text_muted"],
                    "marginTop": "4px",
                    "whiteSpace": "nowrap",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                children="",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Topology view (Cytoscape force-directed graph)
# ---------------------------------------------------------------------------


_NODE_SPACING = 55  # px between adjacent qubits inside a core
_CORE_GAP = 2.0     # extra core-spacing factor (multiplier on core bounding box)


def _local_qubit_positions(size: int, intracore: str) -> list[tuple[float, float]]:
    """Per-core (x, y) layout for ``size`` qubits, centred at the origin.

    Position pattern matches the intra-core topology so that drawing the
    intra-core edges over these coordinates minimises crossings.
    """
    import math
    if size <= 0:
        return []
    if size == 1:
        return [(0.0, 0.0)]
    intracore = (intracore or "all_to_all").lower()
    if intracore == "linear":
        return [((q - (size - 1) / 2) * _NODE_SPACING, 0.0) for q in range(size)]
    if intracore == "ring":
        r = _NODE_SPACING * size / (2 * math.pi)
        # Start from -π/2 so the first node sits at the top of the ring.
        return [
            (r * math.cos(-math.pi / 2 + 2 * math.pi * q / size),
             r * math.sin(-math.pi / 2 + 2 * math.pi * q / size))
            for q in range(size)
        ]
    # grid (and the all_to_all fallback): fill row-by-row.
    side = math.isqrt(size)
    if side * side < size:
        side += 1
    pos = []
    for q in range(size):
        row, col = divmod(q, side)
        x = (col - (side - 1) / 2) * _NODE_SPACING
        y = (row - (side - 1) / 2) * _NODE_SPACING
        pos.append((x, y))
    return pos


def _core_centres(num_cores: int, inter: str, core_box: float) -> list[tuple[float, float]]:
    """(x, y) centre for each core, ``core_box`` apart."""
    import math
    if num_cores <= 1:
        return [(0.0, 0.0)]
    inter = (inter or "ring").lower()
    if inter == "linear":
        return [((c - (num_cores - 1) / 2) * core_box, 0.0) for c in range(num_cores)]
    if inter == "grid":
        side = math.isqrt(num_cores)
        if side * side < num_cores:
            side += 1
        return [
            (((c % side) - (side - 1) / 2) * core_box,
             ((c // side) - (side - 1) / 2) * core_box)
            for c in range(num_cores)
        ]
    # ring / all_to_all
    r = max(core_box, num_cores * core_box / (2 * math.pi))
    return [
        (r * math.cos(-math.pi / 2 + 2 * math.pi * c / num_cores),
         r * math.sin(-math.pi / 2 + 2 * math.pi * c / num_cores))
        for c in range(num_cores)
    ]


def build_topology_elements(
    num_cores: int,
    num_qubits: int,
    communication_qubits: int,
    topology: str,
    intracore_topology: str = "all_to_all",
    buffer_qubits: int = 1,
) -> list:
    """Build the Cytoscape ``elements`` array for the multi-core topology view.

    Each core lays out its qubits as
    ``[D data, group_0, group_1, …, group_{G-1}]`` where ``G`` is the
    number of inter-core neighbours and each group contributes ``K`` comm
    slots followed by ``1`` buffer slot.  ``communication_qubits`` is the
    *per-group* count ``K`` (matching the dse_pau convention), so a ring
    core (G = 2) hosts ``2K`` comm qubits and ``2`` buffer qubits.

    Visual placement: every qubit (data, comm, buffer) sits in the
    per-core local grid that ``intracore_topology`` defines — the core's
    shape is a fixed square (or whatever the topology dictates).  For a
    grid intra-core topology, comm qubits cluster on the **left** and
    **right** edges of the local grid (one per group; further groups
    spill onto the top/bottom edges) and the buffer of each group sits
    on an interior cell **adjacent** to one of its comm qubits.

    Intra-core edges follow ``intracore_topology`` over the full slot
    list (matching the engine's coupling map exactly), so what you see
    is what the simulator routes through.

    Inter-core edges pair comm qubit ``i`` of one core's
    group-toward-partner with comm qubit ``i`` of the partner's
    group-toward-this — so each comm qubit hosts exactly one inter-core
    link.
    """
    import math
    from gui.dse_engine import (
        inter_core_neighbors as _nbrs_for_view,
        inter_core_edges,
        assign_core_slots,
    )

    num_cores = max(1, int(num_cores or 1))
    num_qubits = max(num_cores, int(num_qubits or num_cores))

    # Per-core sizes mirror the engine's distribution (uneven by remainder).
    base = num_qubits // num_cores
    remainder = num_qubits % num_cores
    core_sizes = [base + (1 if c < remainder else 0) for c in range(num_cores)]

    nbrs = list(_nbrs_for_view(num_cores, topology))
    groups_per_core = [len(n) for n in nbrs]

    # Logical-first model: K and B are taken as-is. The resolver has
    # already guaranteed the architecture is feasible upstream, so this
    # view just draws what it's told. ``num_cores < 2`` collapses to a
    # single-core device with no comm slots.
    K = int(communication_qubits or 0) if num_cores >= 2 else 0
    B = int(buffer_qubits or 0) if K >= 1 else 0

    intracore_topology = (intracore_topology or "all_to_all").lower()

    # Resolve per-core role assignments (data / comm / buffer slots).
    per_core_layout = [
        assign_core_slots(core_sizes[c], intracore_topology,
                          groups_per_core[c] if K >= 1 else 0,
                          K if K >= 1 else 0,
                          b_per_group=B if K >= 1 else 0)
        for c in range(num_cores)
    ]

    # Local positions for the *full* per-core grid (so the shape is fixed).
    local_pos_per_core = [
        _local_qubit_positions(core_sizes[c], intracore_topology)
        for c in range(num_cores)
    ]
    if local_pos_per_core and local_pos_per_core[0]:
        max_extent = max(
            max(abs(x), abs(y))
            for poss in local_pos_per_core for x, y in poss
        )
    else:
        max_extent = _NODE_SPACING
    core_box = max(_NODE_SPACING * 3, max_extent * 2 + _NODE_SPACING * _CORE_GAP)
    cores_xy = _core_centres(num_cores, topology, core_box)

    # Per-core rotation: align "right edge" (group 1's home for G=2) with
    # the direction toward partner 1 — this orients the comm rows of each
    # core toward their partners across the chip, so inter-core lines run
    # cleanly along chip-radial directions instead of zig-zagging.  Always
    # snap to a multiple of 90° so the per-core grid stays *square-aligned*
    # (a non-cardinal angle rotates a square grid into a diamond, which
    # makes "comm on left/right edges" look like "comm clustered in
    # corners"; the user-visible result is hard to read).
    snap_to_cardinal = True

    def _rotation_for_core(c: int) -> float:
        if num_cores < 2 or groups_per_core[c] == 0 or K < 1:
            return 0.0
        cx, cy = cores_xy[c]
        # We want group 0 to face partner 0 and group 1 to face partner 1.
        # In the unrotated grid, group 0 lives on the LEFT edge (-x) and
        # group 1 on the RIGHT edge (+x), so the natural orientation is
        # angle 0 = +x toward partner 1.
        partner_1 = nbrs[c][0] if groups_per_core[c] == 1 else nbrs[c][1]
        pcx, pcy = cores_xy[partner_1]
        dx, dy = pcx - cx, pcy - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        target = math.atan2(dy, dx)
        if snap_to_cardinal:
            quarter = math.pi / 2
            target = round(target / quarter) * quarter
        return target

    elements: list = []

    # ---- Nodes: data + comm + buffer, all at their local grid positions ----
    for c in range(num_cores):
        cx, cy = cores_xy[c]
        layout = per_core_layout[c]
        local_pos = local_pos_per_core[c]
        rotation = _rotation_for_core(c)
        cr, sr = math.cos(rotation), math.sin(rotation)

        def _rot(p):
            x, y = p
            return (cr * x - sr * y, sr * x + cr * y)

        # Resolve role for each slot.
        slot_role: list[tuple[str, int | None, int | None]] = [
            ("data", None, None) for _ in range(core_sizes[c])
        ]
        for q in layout["data"]:
            slot_role[q] = ("data", None, None)
        for g, grp in enumerate(layout["groups"]):
            for k_idx, q in enumerate(grp["comm"]):
                slot_role[q] = ("comm", g, k_idx)
            for b_idx, buf_q in enumerate(grp["buffer"]):
                if 0 <= buf_q < core_sizes[c]:
                    slot_role[buf_q] = ("buffer", g, b_idx)

        for q in range(core_sizes[c]):
            role, g, k_idx = slot_role[q]
            lx, ly = _rot(local_pos[q])
            if role == "data":
                node_id = f"c{c}_d{q}"
                label = f"c{c} d{q}"
            elif role == "comm":
                node_id = f"c{c}_g{g}_k{k_idx}"
                label = f"c{c} k{k_idx}"
            else:  # buffer
                node_id = f"c{c}_g{g}_b{k_idx}"
                label = f"c{c} buf{g}.{k_idx}" if k_idx > 0 else f"c{c} buf{g}"
            data_dict = {
                "id": node_id,
                "label": label,
                "core": c,
                "qtype": role,
                "slot": q,
            }
            if g is not None:
                data_dict["group"] = g
            elements.append({
                "data": data_dict,
                "position": {"x": cx + lx, "y": cy + ly},
                "classes": role,
            })

    # ---- Intra-core edges: full ``intracore_topology`` over slot order -----
    # This matches the engine's _add_intracore_edges so the visualization
    # mirrors the actual coupling map.  We need a slot-index → cytoscape
    # node-id map per core.
    def _slot_to_id(c: int, slot: int) -> str:
        role, g, k_idx = (
            ("data", None, None) if slot in per_core_layout[c]["data"]
            else _resolve_role(c, slot)
        )
        if role == "data":
            return f"c{c}_d{slot}"
        if role == "comm":
            return f"c{c}_g{g}_k{k_idx}"
        return f"c{c}_g{g}_b{k_idx}"

    def _resolve_role(c: int, slot: int):
        for g, grp in enumerate(per_core_layout[c]["groups"]):
            for k_idx, q in enumerate(grp["comm"]):
                if q == slot:
                    return ("comm", g, k_idx)
            for b_idx, q in enumerate(grp["buffer"]):
                if q == slot:
                    return ("buffer", g, b_idx)
        return ("data", None, None)

    for c in range(num_cores):
        size = core_sizes[c]
        if size < 2:
            continue
        ids = [_slot_to_id(c, q) for q in range(size)]
        if intracore_topology == "all_to_all":
            for i in range(size):
                for j in range(i + 1, size):
                    elements.append({
                        "data": {"source": ids[i], "target": ids[j]},
                        "classes": "intra",
                    })
        elif intracore_topology == "linear":
            for i in range(size - 1):
                elements.append({
                    "data": {"source": ids[i], "target": ids[i + 1]},
                    "classes": "intra",
                })
        elif intracore_topology == "ring":
            for i in range(size):
                elements.append({
                    "data": {"source": ids[i], "target": ids[(i + 1) % size]},
                    "classes": "intra",
                })
        elif intracore_topology == "grid":
            from gui.dse_engine import _grid_side
            side = _grid_side(size)
            for q in range(size):
                row, col = divmod(q, side)
                if col + 1 < side and q + 1 < size:
                    elements.append({
                        "data": {"source": ids[q], "target": ids[q + 1]},
                        "classes": "intra",
                    })
                if q + side < size:
                    elements.append({
                        "data": {"source": ids[q], "target": ids[q + side]},
                        "classes": "intra",
                    })
        else:
            for i in range(size):
                for j in range(i + 1, size):
                    elements.append({
                        "data": {"source": ids[i], "target": ids[j]},
                        "classes": "intra",
                    })

    # ---- Inter-core edges: K parallel one-to-one links per neighbour pair --
    if num_cores < 2 or K < 1:
        return elements
    for (a_core, a_g, a_k), (b_core, b_g, b_k) in inter_core_edges(
        num_cores, K, topology,
    ):
        elements.append({
            "data": {
                "source": f"c{a_core}_g{a_g}_k{a_k}",
                "target": f"c{b_core}_g{b_g}_k{b_k}",
            },
            "classes": "inter",
        })

    return elements


_TOPOLOGY_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "font-size": "8px",
            "color": COLORS["text_muted"],
            "text-valign": "center",
            "text-halign": "center",
            "text-margin-y": -10,
            "text-opacity": 0,
        },
    },
    {
        "selector": ".data",
        "style": {
            "background-color": "#9ca3af",
            "width": 12,
            "height": 12,
            "border-width": 1,
            "border-color": "#6b7280",
        },
    },
    {
        "selector": ".comm",
        "style": {
            "background-color": "#3b82f6",
            "width": 18,
            "height": 18,
            "border-width": 1.5,
            "border-color": "#1d4ed8",
        },
    },
    # Buffer qubits: reserved per-group teleportation buffer, drawn in a
    # darker grey + square shape so they're clearly distinct from the
    # light-grey data qubit circles.
    {
        "selector": ".buffer",
        "style": {
            "background-color": "#6b7280",
            "shape": "round-rectangle",
            "width": 14,
            "height": 14,
            "border-width": 1.5,
            "border-color": "#374151",
        },
    },
    # When the fidelity overlay is active each node carries a ``fidelity``
    # data field in [0, 1].  Two-stop selectors give a red→amber→green
    # gradient that matches the on-screen legend; cytoscape ``mapData`` is
    # linear so we split the range at 0.5 to get a true 3-colour ramp.
    {
        "selector": "node[fidelity][fidelity < 0.5]",
        "style": {
            "background-color": "mapData(fidelity, 0, 0.5, #b91c1c, #f59e0b)",
            "border-color": "mapData(fidelity, 0, 0.5, #7f1d1d, #b45309)",
        },
    },
    {
        "selector": "node[fidelity][fidelity >= 0.5]",
        "style": {
            "background-color": "mapData(fidelity, 0.5, 1, #f59e0b, #15803d)",
            "border-color": "mapData(fidelity, 0.5, 1, #b45309, #14532d)",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "straight",
        },
    },
    {
        "selector": ".intra",
        "style": {
            "line-color": "#d1d5db",
            "width": 0.6,
            "opacity": 0.4,
        },
    },
    {
        "selector": ".inter",
        "style": {
            "line-color": "#3b82f6",
            "width": 1.6,
            "opacity": 0.85,
        },
    },
    {
        "selector": "node:selected, .highlighted",
        "style": {
            "border-color": "#f59e0b",
            "border-width": 3,
            "text-opacity": 1,
        },
    },
    {
        "selector": ".dimmed",
        "style": {
            "opacity": 0.15,
        },
    },
]


def make_topology_view_panel() -> html.Div:
    """Topology view container (Cytoscape).  Hidden unless the Topology view
    tab is active.

    Reads ``num_cores`` / ``num_qubits`` / ``communication_qubits`` /
    ``topology`` from the right sidebar and renders an interactive
    force-directed graph of the multi-core architecture.
    """
    return html.Div(
        id="topology-view-container",
        style={
            "display": "none",
            "position": "absolute",
            "top": "0",
            "left": "0",
            "right": "0",
            "bottom": "0",
            "background": COLORS["bg"],
            "zIndex": 5,
        },
        children=[
            html.Div(
                style={
                    "position": "absolute",
                    "top": "8px",
                    "right": "8px",
                    "zIndex": 10,
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                },
                children=[
                    # Help "?" — bound by help_icon_modebar.js to show the
                    # same popup as the Plotly modebar help icon.
                    html.Span(
                        "?",
                        id="topology-help-icon",
                        className="help-icon",
                        title="What is this view?",
                    ),
                    html.Button(
                        "Re-layout",
                        id="topology-view-relayout",
                        n_clicks=0,
                        style={
                            "background": COLORS["surface"],
                            "border": f"1px solid {COLORS['border']}",
                            "color": COLORS["text"],
                            "borderRadius": "4px",
                            "padding": "4px 12px",
                            "fontSize": "11px",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
            # Sweep navigation + fidelity-overlay controls.  The whole panel
            # is hidden until a sweep with per_qubit metadata is loaded.
            html.Div(
                id="topology-sweep-controls",
                style={
                    "display": "none",
                    "position": "absolute",
                    "top": "8px",
                    "left": "8px",
                    "zIndex": 10,
                    "background": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "6px",
                    "padding": "14px 22px 18px",
                    "width": "440px",
                    "fontSize": "11px",
                },
                children=[
                    # Facet selector — visible only when the sweep is faceted
                    # (e.g. across routing algorithm).  Picks which slice of
                    # the categorical product the topology view renders.
                    html.Div(
                        id="topology-facet-row",
                        style={
                            "display": "none",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "10px",
                        },
                        children=[
                            html.Span(
                                id="topology-facet-label",
                                children="Facet",
                                style={
                                    "fontSize": "10px",
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                    "color": COLORS["text_muted"],
                                },
                            ),
                            dcc.Dropdown(
                                id="topology-facet-selector",
                                options=[],
                                value=None,
                                clearable=False,
                                searchable=False,
                                className="dse-dropdown",
                                style={"width": "220px", "fontSize": "11px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "10px",
                            "marginBottom": "10px",
                        },
                        children=[
                            html.Span(
                                "Fidelity",
                                style={
                                    "fontSize": "10px",
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                    "color": COLORS["text_muted"],
                                },
                            ),
                            dcc.Dropdown(
                                id="topology-overlay-metric",
                                options=[
                                    {"label": "Overall", "value": "overall_fidelity"},
                                    {"label": "Algorithmic", "value": "algorithmic_fidelity"},
                                    {"label": "Routing", "value": "routing_fidelity"},
                                    {"label": "Coherence", "value": "coherence_fidelity"},
                                ],
                                value="overall_fidelity",
                                clearable=False,
                                searchable=False,
                                className="dse-dropdown",
                                style={"width": "180px", "fontSize": "11px"},
                            ),
                        ],
                    ),
                    html.Div(
                        id="topology-axis-sliders",
                        children=[
                            html.Div(
                                id={"type": "topology-axis-row", "index": i},
                                style={"display": "none"},
                                children=[
                                    html.Div(
                                        style={
                                            "padding": "12px 14px 26px",
                                        },
                                        children=[
                                            html.Span(
                                                id={"type": "topology-axis-label", "index": i},
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": "600",
                                                    "color": COLORS["text_muted"],
                                                    "textTransform": "uppercase",
                                                    "letterSpacing": "0.05em",
                                                    "display": "block",
                                                    "marginBottom": "10px",
                                                },
                                                children="",
                                            ),
                                            html.Div(
                                                style={
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "gap": "10px",
                                                },
                                                children=[
                                                    html.Div(
                                                        dcc.Slider(
                                                            id={"type": "topology-axis-slider", "index": i},
                                                            min=0, max=1, step=1, value=0,
                                                            marks={},
                                                            # The slider's value
                                                            # is the 0-based cell
                                                            # index, not a user-
                                                            # readable magnitude,
                                                            # so we hide Dash's
                                                            # default input chip
                                                            # and the rc-slider
                                                            # tooltip — the
                                                            # axis-magnitude chip
                                                            # to the right is
                                                            # the canonical
                                                            # readout.
                                                            updatemode="drag",
                                                            included=False,
                                                            allow_direct_input=False,
                                                            className="dse-slider",
                                                        ),
                                                        style={"flex": "1", "minWidth": 0},
                                                    ),
                                                    dcc.Input(
                                                        id={"type": "topology-axis-value", "index": i},
                                                        type="text",
                                                        debounce=True,
                                                        spellCheck=False,
                                                        className="slider-value-chip",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                            for i in range(MAX_SWEEP_AXES)
                        ],
                    ),
                ],
            ),
            cyto.Cytoscape(
                id="topology-cyto",
                layout={
                    "name": "preset",
                    "fit": True,
                    "padding": 40,
                },
                stylesheet=_TOPOLOGY_STYLESHEET,
                elements=[],
                style={
                    "width": "100%",
                    "height": "100%",
                },
                minZoom=0.2,
                maxZoom=3.0,
                wheelSensitivity=0.2,
            ),
            html.Div(
                id="topology-view-hover",
                style={
                    "position": "absolute",
                    "bottom": "32px",
                    "left": "12px",
                    "fontSize": "11px",
                    "color": COLORS["text"],
                    "background": COLORS["surface"],
                    "padding": "8px 12px",
                    "borderRadius": "6px",
                    "border": f"1px solid {COLORS['border']}",
                    "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                    "pointerEvents": "none",
                    "minWidth": "240px",
                    "lineHeight": "1.5",
                },
                children="Hover a node to inspect.",
            ),
            # Colour legend for the fidelity overlay — gradient bar with
            # tick labels.  Hidden until the overlay is enabled.
            html.Div(
                id="topology-view-legend",
                style={
                    "display": "none",
                    "position": "absolute",
                    "bottom": "32px",
                    "right": "12px",
                    "background": COLORS["surface"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "6px",
                    "padding": "8px 12px",
                    "fontSize": "11px",
                    "color": COLORS["text"],
                    "pointerEvents": "none",
                    "minWidth": "240px",
                },
                children=[
                    html.Div(
                        id="topology-legend-title",
                        children="Overall fidelity",
                        style={
                            "fontSize": "10px",
                            "fontWeight": "700",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.06em",
                            "color": COLORS["text_muted"],
                            "marginBottom": "4px",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(
                        style={
                            "height": "12px",
                            "borderRadius": "3px",
                            "background": "linear-gradient(to right, #b91c1c, #f59e0b, #15803d)",
                            "border": f"1px solid {COLORS['border']}",
                        },
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "marginTop": "3px",
                            "fontSize": "10px",
                            "fontFamily": "'JetBrains Mono', 'SF Mono', monospace",
                            "color": COLORS["text_muted"],
                        },
                        children=[
                            html.Span("0.0"),
                            html.Span("0.25"),
                            html.Span("0.5"),
                            html.Span("0.75"),
                            html.Span("1.0"),
                        ],
                    ),
                ],
            ),
            html.Div(
                # Future hook: user-uploaded coupling map will replace the
                # synthetic elements built from the right-sidebar config.
                style={
                    "position": "absolute",
                    "bottom": "8px",
                    "left": "12px",
                    "fontSize": "10px",
                    "color": COLORS["text_muted"],
                    "fontStyle": "italic",
                },
                children=(
                    "Visualizing the current architecture. "
                    "Custom coupling map upload coming soon."
                ),
            ),
        ],
    )
