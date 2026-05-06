"""Dash layout for the qusim DSE GUI.

Pure UI: builds the topbar, sidebars, centre panel, and the root
``html.Div`` that becomes ``app.layout``. Holds no callbacks and no
runtime state; ``app.py`` calls ``build_layout()`` once during startup.
"""

from __future__ import annotations

from dash import dcc, html

from gui.components import (
    COLORS,
    make_custom_qasm_help_modal,
    make_fixed_config_panel,
    make_merit_controls,
    make_merit_view_controls,
    make_metric_selector,
    make_performance_panel,
    make_topology_view_panel,
    make_view_tab_bar,
)
from gui.constants import MAX_SWEEP_AXES, OUTPUT_METRICS
from gui.examples import example_options as _example_options
from gui.fom import DEFAULT_FOM
from gui.plotting import plot_empty


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
                    html.Div(
                        style={"width": "240px"},
                        children=dcc.Dropdown(
                            id="examples-dropdown",
                            className="dse-dropdown",
                            options=_example_options(),
                            value=None,
                            placeholder="Load example…",
                            clearable=True,
                            searchable=False,
                            style={"fontSize": "12px"},
                        ),
                    ),
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
            # Always render all MAX_SWEEP_AXES rows; show/hide via 'display'
            *[
                html.Div(
                    id=f"metric-row-wrap-{i}",
                    children=[make_metric_selector(i)],
                    style={} if i < 3 else {"display": "none"},
                )
                for i in range(MAX_SWEEP_AXES)
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


def build_layout() -> html.Div:
    """Assemble the root layout (panels + state stores).

    Called once during ``app.py`` startup.
    """
    return html.Div(
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
