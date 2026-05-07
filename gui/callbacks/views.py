"""View-mode and view-tab visibility callbacks.

These callbacks manage which set of controls is visible for each view tab
(pareto / elasticity / importance / correlation / frozen-slider / merit),
populate the view-mode dropdown options for the active sweep
dimensionality, and push the (view, merit-mode) pair into the modebar
help popup.

The big main-plot.figure rewriters (replot_on_output_change,
on_view_tab_click, replot_frozen_slider_in_derivative_mode, the FoM/merit
replot chain) still live in app.py for now — they ride alongside the
inline run_sweep primary so callback registration order is naturally
preserved.
"""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State

from gui.components import COLORS
from gui.constants import METRIC_BY_KEY, view_modes_for_dim
from gui.interpolation import is_frozen_view
from gui.server_state import _get_sweep


_MERIT_VIEW_CONTAINER_STYLE_VISIBLE = {
    "display": "block",
    "padding": "6px 16px 6px",
    "borderBottom": f"1px solid {COLORS['border']}",
    "background": COLORS["bg"],
}
_MERIT_VIEW_CONTAINER_STYLE_HIDDEN = {"display": "none"}

_PADDED_VISIBLE = {"padding": "4px 16px 8px"}
_HIDDEN = {"display": "none"}


def _view_eq_style(target_view: str):
    """Build a style closure that shows the panel when view-type matches."""
    def _toggle(view_type):
        return _PADDED_VISIBLE if view_type == target_view else _HIDDEN
    return _toggle


def register(app: Any) -> None:

    @app.callback(
        Output("frozen-slider-container", "style", allow_duplicate=True),
        Input("view-type-store", "data"),
        State("sweep-result-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_frozen_slider_visibility(view_type, sweep_data):
        if sweep_data is None:
            return _HIDDEN
        num_metrics = len(sweep_data.get("metric_keys", []))
        if num_metrics == 3 and is_frozen_view(view_type):
            return _PADDED_VISIBLE
        return _HIDDEN

    # Per-view side-panel visibility toggles. Each one shows its panel
    # only when the active view-type matches.
    for output_id, target_view in [
        ("pareto-axis-container", "pareto"),
        ("elasticity-axis-container", "elasticity"),
        ("importance-mode-container", "importance"),
        ("correlation-mode-container", "correlation"),
    ]:
        app.callback(
            Output(output_id, "style"),
            Input("view-type-store", "data"),
            prevent_initial_call=False,
        )(_view_eq_style(target_view))

    # View-mode dropdown filtering — hide derivative modes that don't make
    # sense for the current sweep dimensionality (e.g. d²F/dx² is 1-D
    # only, ∂²F/∂x∂y is 2-D only). Falls back to "absolute" if the
    # previously picked mode is no longer valid (e.g. user picked
    # Elasticity in a 1-D sweep then added a second axis).
    @app.callback(
        Output("cfg-view-mode", "options"),
        Output("cfg-view-mode", "value"),
        Input("num-metrics-store", "data"),
        State("cfg-view-mode", "value"),
        prevent_initial_call=False,
    )
    def update_view_mode_options(num_metrics, current_value):
        n = int(num_metrics or 1)
        options = view_modes_for_dim(n)
        valid = {opt["value"] for opt in options}
        value = current_value if current_value in valid else "absolute"
        return options, value

    # Populate the trajectory dropdown from the active sweep's numeric
    # axes (categorical axes can't be elasticised — derivatives need an
    # ordered coordinate). Default to the first axis when the prior
    # selection is no longer valid for this sweep.
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
        options = []
        for mk in full.get("metric_keys", []):
            m = METRIC_BY_KEY.get(mk)
            if m is None:
                continue  # categorical axes are excluded
            options.append({"label": m.label, "value": mk})
        if not options:
            return [], None
        valid_keys = {opt["value"] for opt in options}
        value = current_value if current_value in valid_keys else options[0]["value"]
        return options, value

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
        return _HIDDEN

    @app.callback(
        Output("merit-view-controls-container", "style"),
        Input("view-type-store", "data"),
        prevent_initial_call=False,
    )
    def toggle_merit_view_controls_visibility(view_type):
        if view_type == "merit":
            return _MERIT_VIEW_CONTAINER_STYLE_VISIBLE
        return _MERIT_VIEW_CONTAINER_STYLE_HIDDEN

    # Push the active (view-type, merit-mode) pair into the help-icon JS
    # so the modebar "?" popup describes whatever the user is currently
    # looking at.
    app.clientside_callback(
        """function(viewType, meritMode) {
            if (window.quadrisUpdatePlotHelp) {
                window.quadrisUpdatePlotHelp(viewType, meritMode);
            }
            return window.dash_clientside.no_update;
        }""",
        Output("plot-help-sink", "data"),
        Input("view-type-store", "data"),
        Input("merit-mode-store", "data"),
        prevent_initial_call=False,
    )
