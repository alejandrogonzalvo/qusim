"""Left-sidebar callbacks: metric / threshold / noise rows + per-axis slider config.

Stores are the single source of truth: ``num-metrics-store`` drives row
visibility for sweep axes; ``num-thresholds-store`` does the same for
threshold rows.  Mutators (the +/− buttons and the per-row × button)
write only those stores so manual clicks and session loads produce the
same on-screen state.
"""

from __future__ import annotations

from typing import Any

import dash
from dash import ALL, Input, Output, State, ctx, html

from gui.components import (
    COLORS,
    _linear_marks,
    _log_marks,
    _tooltip_cfg,
)
from gui.constants import (
    CAT_METRIC_BY_KEY,
    CATEGORICAL_METRICS,
    MAX_SWEEP_AXES,
    METRIC_BY_KEY,
    SWEEPABLE_METRICS,
)
from gui.utils import _slider_to_value


MAX_METRICS = MAX_SWEEP_AXES


_AXIS_BTN_BASE = {
    "flex": "1",
    "background": "transparent",
    "borderRadius": "6px",
    "padding": "6px",
    "cursor": "pointer",
    "fontSize": "12px",
}

_THRESHOLD_REMOVE_BTN_STYLE = {
    "background": "transparent",
    "border": f"1px solid {COLORS['border']}",
    "color": COLORS["text_muted"],
    "borderRadius": "4px",
    "width": "28px",
    "height": "28px",
    "cursor": "pointer",
    "fontSize": "14px",
}

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

_RANGE_LABEL_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "fontSize": "11px",
    "color": COLORS["accent2"],
    "marginTop": "-20px",
}

_ALL_METRIC_OPTIONS = (
    [{"label": m.label, "value": m.key} for m in SWEEPABLE_METRICS]
    + [{"label": c.label, "value": c.key} for c in CATEGORICAL_METRICS]
)

# Sweep-axis keys that are meaningless once a custom QASM circuit is loaded:
# the algorithm size and gate sequence are fully determined by the uploaded
# file, so neither circuit type, logical qubit count, nor the (logical, physical)
# alias should be selectable on a sweep axis.
_CUSTOM_QASM_DISABLED_KEYS = {"circuit_type", "num_logical_qubits"}


# ---------------------------------------------------------------------------
# Axis-availability registry
# ---------------------------------------------------------------------------
#
# A *rule* maps a snapshot of the GUI state (an ``AvailabilityContext``) to
# a ``{metric_key: human_reason}`` dict listing every metric the rule wants
# disabled.  Three callbacks consume the union of all rule outputs:
#
#   * ``_filter_dropdown_options`` greys the disabled options out (preserving
#     the user's current selection so a pin-flip doesn't silently change it).
#   * ``toggle_metric_rows`` skips disabled keys when auto-assigning a
#     newly-revealed sweep-axis row, so the +Add button never lands on an
#     option that's immediately greyed.
#   * ``run_sweep`` (in ``app.py``) refuses to start a sweep that includes a
#     disabled axis, surfacing the rule's reason in an error banner.
#
# Adding a new rule is *one* dataclass entry plus, if needed, a new
# ``AvailabilityContext`` field. Adding a new consumer is one extra import
# of :func:`axis_availability`.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AvailabilityContext:
    """All GUI inputs any rule might depend on.  Add a field here only
    when introducing a rule that consumes new state — the wiring stays
    contained because every consumer constructs the context the same
    way."""
    pin_axis: str = "cores"          # ``cfg-pin-axis.data``
    custom_qasm_active: bool = False  # ``custom-qasm-store.data["qasm"]`` truthy


def _rule_pin_axis(ctx: AvailabilityContext) -> dict[str, str]:
    """Pin axis owns Cores/Qpc; the unpinned one is *derived* per cell
    by the resolver and would be silently overwritten if swept."""
    if ctx.pin_axis == "cores":
        return {"qubits_per_core":
                "Pin axis is Cores; qubits-per-core is derived per cell"}
    return {"num_cores":
            "Pin axis is Qubits/core; cores is derived per cell"}


def _rule_custom_qasm(ctx: AvailabilityContext) -> dict[str, str]:
    """A custom QASM upload pins the circuit shape, so sweeping it
    produces identical cells."""
    if ctx.custom_qasm_active:
        return {k: "Custom QASM uploaded; circuit shape is fixed by the file"
                for k in _CUSTOM_QASM_DISABLED_KEYS}
    return {}


_AVAILABILITY_RULES: list[Callable[[AvailabilityContext], dict[str, str]]] = [
    _rule_pin_axis,
    _rule_custom_qasm,
]


def axis_availability(ctx: AvailabilityContext) -> dict[str, str]:
    """Union of every rule's disabled-key dict.  First reason wins on
    collision."""
    out: dict[str, str] = {}
    for rule in _AVAILABILITY_RULES:
        for k, reason in rule(ctx).items():
            out.setdefault(k, reason)
    return out


def register(app: Any) -> None:
    _register_metric_rows(app)
    _register_threshold_rows(app)
    _register_noise_rows(app)
    _register_per_axis_loops(app)
    _register_dropdown_filter(app)


# ---------------------------------------------------------------------------
# Metric (sweep-axis) row visibility, +/-, per-row × button
# ---------------------------------------------------------------------------

def _register_metric_rows(app: Any) -> None:

    @app.callback(
        *[Output(f"metric-row-wrap-{i}", "style", allow_duplicate=True) for i in range(MAX_METRICS)],
        Output("add-metric-btn", "style", allow_duplicate=True),
        Output("remove-metric-btn", "style", allow_duplicate=True),
        Input("num-metrics-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render_metric_rows(num_metrics):
        """Single source of truth for sweep-axis row visibility + +/− button styles."""
        n = max(1, int(num_metrics or 1))
        row_styles = [{} if i < n else {"display": "none"} for i in range(MAX_METRICS)]
        add_style = (
            {**_AXIS_BTN_BASE,
             "border": f"1px solid {COLORS['border']}",
             "color": COLORS["text_muted"]}
            if n < MAX_METRICS
            else {"display": "none"}
        )
        remove_style = (
            {**_AXIS_BTN_BASE,
             "border": f"1px dashed {COLORS['border']}",
             "color": COLORS["text_muted"]}
            if n > 1
            else {"display": "none"}
        )
        return *row_styles, add_style, remove_style

    @app.callback(
        *[Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
        Output("num-metrics-store", "data"),
        Input("add-metric-btn", "n_clicks"),
        Input("remove-metric-btn", "n_clicks"),
        State("num-metrics-store", "data"),
        *[State(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
        State("cfg-pin-axis", "data"),
        State("custom-qasm-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_metric_rows(add_clicks, remove_clicks, num_metrics, *all_states):
        dropdown_vals = all_states[:MAX_METRICS]
        pin_axis = all_states[MAX_METRICS] or "cores"
        custom_qasm = all_states[MAX_METRICS + 1] or {}
        triggered = ctx.triggered_id
        old_num = num_metrics
        if triggered == "add-metric-btn":
            num_metrics = min(MAX_METRICS, num_metrics + 1)
        elif triggered == "remove-metric-btn":
            num_metrics = max(1, num_metrics - 1)

        taken = {dropdown_vals[i] for i in range(old_num) if dropdown_vals[i]}
        # Ask the registry which keys are globally unavailable so a freshly-
        # revealed row never auto-picks something that's immediately greyed.
        disabled_globally = set(axis_availability(AvailabilityContext(
            pin_axis=pin_axis,
            custom_qasm_active=bool(custom_qasm.get("qasm")),
        )).keys())
        all_keys = [m.key for m in SWEEPABLE_METRICS]

        new_values = list(dropdown_vals)
        for i in range(old_num, num_metrics):
            current = new_values[i]
            if current in taken or current in disabled_globally:
                available = [
                    k for k in all_keys
                    if k not in taken and k not in disabled_globally
                ]
                new_values[i] = available[0] if available else current
            taken.add(new_values[i])

        no = dash.no_update
        dropdown_outs = [
            new_values[i] if i < num_metrics else no
            for i in range(MAX_METRICS)
        ]
        return *dropdown_outs, num_metrics

    @app.callback(
        *[Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
        *[Output(f"metric-slider-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
        *[Output(f"metric-checklist-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)],
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
        if not isinstance(triggered, dict) or triggered.get("type") != "remove-metric-x":
            raise dash.exceptions.PreventUpdate
        clicked = triggered.get("index")
        if not isinstance(clicked, int):
            raise dash.exceptions.PreventUpdate
        if not any(n_clicks_list):
            raise dash.exceptions.PreventUpdate

        dropdown_vals = list(all_states[0:MAX_METRICS])
        slider_vals = list(all_states[MAX_METRICS:2 * MAX_METRICS])
        checklist_vals = list(all_states[2 * MAX_METRICS:3 * MAX_METRICS])
        num_metrics = all_states[3 * MAX_METRICS] or 1

        if num_metrics <= 1 or clicked < 0 or clicked >= num_metrics:
            raise dash.exceptions.PreventUpdate

        for k in range(clicked, num_metrics - 1):
            dropdown_vals[k] = dropdown_vals[k + 1]
            slider_vals[k] = slider_vals[k + 1]
            checklist_vals[k] = checklist_vals[k + 1]

        # suppress-cascade=True tells the per-axis dropdown listeners
        # (_reconfigure_slider, _toggle_slider_checklist) to preserve the
        # slider and checklist values we just shifted, instead of resetting
        # them.
        return (
            *dropdown_vals,
            *slider_vals,
            *checklist_vals,
            num_metrics - 1,
            True,
        )

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
# Threshold-row visibility + +/-
# ---------------------------------------------------------------------------

def _register_threshold_rows(app: Any) -> None:

    @app.callback(
        *[Output(f"threshold-row-{i}", "style", allow_duplicate=True) for i in range(5)],
        Output("remove-threshold-btn", "style", allow_duplicate=True),
        Input("num-thresholds-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _render_threshold_rows(num_thresholds):
        n = max(1, int(num_thresholds or 1))
        row_styles = [{} if i < n else {"display": "none"} for i in range(5)]
        remove_style = _THRESHOLD_REMOVE_BTN_STYLE if n > 1 else {"display": "none"}
        return *row_styles, remove_style

    @app.callback(
        Output("num-thresholds-store", "data"),
        Input("add-threshold-btn", "n_clicks"),
        Input("remove-threshold-btn", "n_clicks"),
        State("num-thresholds-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_threshold_rows(add_clicks, remove_clicks, num_thresholds):
        triggered = ctx.triggered_id
        if triggered == "add-threshold-btn":
            return min(5, num_thresholds + 1)
        if triggered == "remove-threshold-btn":
            return max(1, num_thresholds - 1)
        return num_thresholds


# ---------------------------------------------------------------------------
# Noise-row + categorical-row visibility (driven by which axes are swept)
# ---------------------------------------------------------------------------

def _register_noise_rows(app: Any) -> None:

    @app.callback(
        [Output(f"noise-row-{m.key}", "style") for m in SWEEPABLE_METRICS]
        + [
            Output("cfg-row-num-cores", "style"),
            Output("cfg-row-qubits-per-core", "style"),
            Output("cfg-row-communication-qubits", "style"),
            Output("cfg-row-buffer-qubits", "style"),
            Output("cfg-row-num-logical-qubits", "style"),
        ]
        + [Output(f"cfg-row-cat-{cat.key}", "style") for cat in CATEGORICAL_METRICS],
        *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
        Input("num-metrics-store", "data"),
        Input("cfg-num-cores", "value"),
        prevent_initial_call=False,
    )
    def toggle_noise_rows(*args):
        dropdown_vals = args[:MAX_METRICS]
        num_metrics = args[MAX_METRICS] or 1
        cfg_num_cores = args[MAX_METRICS + 1]
        swept = {val for i, val in enumerate(dropdown_vals) if i < num_metrics and val}

        # Comm / buffer qubits are inter-core constructs — meaningless at
        # cores=1.  Hide both rows when the active cores value is 1 (unless
        # the cores axis is itself being swept, in which case the user is
        # varying cores and we keep the rows visible to drive the cells with
        # cores>1).
        cores_is_one = (
            cfg_num_cores is not None
            and int(cfg_num_cores) == 1
            and "num_cores" not in swept
        )

        hide = {"display": "none"}
        show: dict = {}

        noise_styles = [
            hide if m.is_cold_path or m.key in swept else show
            for m in SWEEPABLE_METRICS
        ]
        cores_style = hide if "num_cores" in swept else show
        qpc_style = hide if "qubits_per_core" in swept else show
        comm_style = hide if ("communication_qubits" in swept or cores_is_one) else show
        buffer_style = hide if ("buffer_qubits" in swept or cores_is_one) else show
        logi_style = hide if "num_logical_qubits" in swept else show
        cat_styles = [hide if cat.key in swept else show for cat in CATEGORICAL_METRICS]
        return (
            noise_styles
            + [cores_style, qpc_style, comm_style, buffer_style, logi_style]
            + cat_styles
        )


# ---------------------------------------------------------------------------
# Per-axis loops: hint, range label, slider reconfig, checklist toggle
# ---------------------------------------------------------------------------

def _register_per_axis_loops(app: Any) -> None:

    for _idx in range(MAX_METRICS):

        @app.callback(
            Output(f"metric-help-{_idx}", "children"),
            Output(f"metric-help-{_idx}", "style"),
            *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
            prevent_initial_call=False,
        )
        def _update_metric_hint(*dropdown_vals, _i=_idx):
            metric_key = dropdown_vals[_i] if _i < len(dropdown_vals) else None
            static_hint = _METRIC_INLINE_HINT.get(metric_key)
            if static_hint:
                return static_hint, _METRIC_HINT_VISIBLE_STYLE
            return "", {"display": "none"}

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

        # Logical-first model: sweep-axis sliders carry no architectural cap
        # — infeasible cells render as white in the heat-map, and the slider's
        # range is whatever the user (or load) set. The previous re-clamp
        # callback fired on every cfg-* change and rewrote slider values
        # whenever the cap moved, which (a) clobbered loaded sweep ranges
        # during session load and (b) bumped sweep-dirty, triggering a
        # phantom recompile. So this only consumes the dropdown + suppress
        # flag; the cfg-* states are deliberately not Inputs.
        @app.callback(
            Output(f"metric-slider-{_idx}", "min"),
            Output(f"metric-slider-{_idx}", "max"),
            Output(f"metric-slider-{_idx}", "step"),
            Output(f"metric-slider-{_idx}", "marks"),
            Output(f"metric-slider-{_idx}", "value"),
            Output(f"metric-slider-{_idx}", "tooltip"),
            Input(f"metric-dropdown-{_idx}", "value"),
            State("suppress-cascade", "data"),
            prevent_initial_call=True,
        )
        def _reconfigure_slider(metric_key, suppress, _i=_idx):
            no = dash.no_update
            if not metric_key:
                return (no, no, no, no, no, no)
            m = METRIC_BY_KEY.get(metric_key)
            if m is None:
                return (no, no, no, no, no, no)
            smin, smax = float(m.slider_min), float(m.slider_max)
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
            # session load (suppress=True); the load callback already wrote
            # the restored value and we'd otherwise clobber it with defaults.
            # ``_toggle_slider_checklist`` owns the suppress-cascade reset —
            # one writer per dropdown trigger is enough.
            if suppress:
                value = no
            else:
                lo = max(smin, min(float(m.slider_default_low), smax))
                hi = max(lo, min(float(m.slider_default_high), smax))
                value = [lo, hi]
            return (
                smin, smax, step, marks, value,
                _tooltip_cfg(m.log_scale, m.unit, always_visible=True),
            )

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
            # ``suppress`` output is left untouched (no_update) so that a
            # session load's suppress=True flag persists for *every* axis's
            # ``_reconfigure_slider`` call.  Otherwise the per-axis race
            # between this callback and the slider one resets suppress for
            # axis 0 before axis 1 reads it, clobbering the loaded axis-1
            # slider value to defaults.  The grace-period gate elsewhere
            # resets suppress=False once the load cascade settles.
            no = dash.no_update
            cat = CAT_METRIC_BY_KEY.get(metric_key)
            if cat:
                # See _reconfigure_slider: preserve the loaded checklist
                # value when the change came from a session load.
                value = no if suppress else [o["value"] for o in cat.options]
                return (
                    {"display": "none"}, {}, cat.options, value,
                    {"display": "none"}, no,
                )
            return (
                {"paddingBottom": "22px"}, {"display": "none"}, [],
                no if suppress else [], _RANGE_LABEL_STYLE, no,
            )


# ---------------------------------------------------------------------------
# Dropdown filter: prevent picking the same metric twice across axes
# ---------------------------------------------------------------------------

def _register_dropdown_filter(app: Any) -> None:

    @app.callback(
        *[Output(f"metric-dropdown-{i}", "options") for i in range(MAX_METRICS)],
        *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
        Input("num-metrics-store", "data"),
        Input("custom-qasm-store", "data"),
        Input("cfg-pin-axis", "data"),
        prevent_initial_call=True,
    )
    def _filter_dropdown_options(*args):
        values = args[:MAX_METRICS]
        num_metrics = args[MAX_METRICS] or 1
        custom_qasm = args[MAX_METRICS + 1] or {}
        pin_axis = args[MAX_METRICS + 2] or "cores"
        ctx = AvailabilityContext(
            pin_axis=pin_axis,
            custom_qasm_active=bool(custom_qasm.get("qasm")),
        )
        # ``disabled_globally`` flags incompatibility regardless of which
        # axis the user is looking at; ``taken`` is the per-axis "another
        # row already picked this metric" rule.
        disabled_globally = axis_availability(ctx)
        results = []
        for i in range(MAX_METRICS):
            taken = {values[j] for j in range(num_metrics) if j != i and values[j]}
            own = values[i]
            results.append([
                {
                    **opt,
                    # The user's current selection on this axis is never
                    # greyed: a pin flip should make the now-incompatible
                    # selection visible, not erase it.  Run-time
                    # validation refuses the sweep instead.
                    "disabled": (
                        opt["value"] != own
                        and (opt["value"] in taken
                             or opt["value"] in disabled_globally)
                    ),
                    # Tooltip-on-option isn't supported by all dropdown
                    # renderers, but we set ``title`` anyway — it shows
                    # in dev tools / future dropdown widgets.
                    "title": disabled_globally.get(opt["value"]),
                }
                for opt in _ALL_METRIC_OPTIONS
            ])
        return results
