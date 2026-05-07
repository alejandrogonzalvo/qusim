"""Right-panel callbacks: budget warnings, pin axis, architecture summary,
collapsible sections, slider/input two-way sync, and the B≤K buffer clamp.
"""

from __future__ import annotations

from typing import Any

from dash import Input, Output, State, ctx, html

from gui.components import (
    COLORS,
    FEEDBACK_COLORS,
    _minmax_marks,
)
from gui.constants import CAT_METRIC_BY_KEY, MAX_SWEEP_AXES, SWEEPABLE_METRICS
from gui.server_state import _engine

MAX_METRICS = MAX_SWEEP_AXES


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


_PIN_TOGGLE_BTN_ACTIVE = {
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
_PIN_TOGGLE_BTN_INACTIVE = {
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
# Same as INACTIVE but greyed out: shown when an architectural axis is on a
# sweep, so flipping the pin would un-sweep it. The button is still clickable
# (Dash has no built-in disabled state for ``html.Div`` toggles); the callback
# blocks the flip and surfaces a toast.
_PIN_TOGGLE_BTN_LOCKED_INACTIVE = {
    **_PIN_TOGGLE_BTN_INACTIVE,
    "color": COLORS["text_muted"],
    "cursor": "not-allowed",
    "opacity": "0.55",
}
_PIN_TOAST_VISIBLE_STYLE = {
    **_feedback_style("error"),
    "marginTop": "8px",
    "marginBottom": "0",
    "transition": "opacity 0.4s ease-out",
    "opacity": "1",
}
_DERIVED_BADGE_STYLE = {
    "padding": "6px 8px",
    "background": COLORS["surface2"],
    "border": f"1px dashed {COLORS['border']}",
    "borderRadius": "6px",
    "fontSize": "12px",
    "color": COLORS["text_muted"],
    "marginBottom": "10px",
}


# Slider <-> input chip JS templates (factored out of the bind helper so the
# bodies are read once instead of rebuilt per slider).
_SLIDER_TO_INPUT_LOG = (
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
_INPUT_TO_SLIDER_LOG = (
    "function(iv) { "
    "  if (iv === null || iv === undefined || iv === '') "
    "    return window.dash_clientside.no_update; "
    "  var n = parseFloat(iv); "
    "  if (!isFinite(n) || n <= 0) return window.dash_clientside.no_update; "
    "  return Math.log10(n); "
    "}"
)
_SLIDER_TO_INPUT_LIN = (
    "function(sv) { "
    "  if (sv === null || sv === undefined || isNaN(sv)) "
    "    return window.dash_clientside.no_update; "
    "  if (Math.abs(sv - Math.round(sv)) < 1e-9) return String(Math.round(sv)); "
    "  return String(Number(sv.toPrecision(6))); "
    "}"
)
_INPUT_TO_SLIDER_LIN = (
    "function(iv) { "
    "  if (iv === null || iv === undefined || iv === '') "
    "    return window.dash_clientside.no_update; "
    "  var n = parseFloat(iv); "
    "  if (!isFinite(n)) return window.dash_clientside.no_update; "
    "  return n; "
    "}"
)

_SLIDER_INPUT_PAIRS: list[tuple[str, bool]] = []


def register(app: Any) -> None:
    _register_budget_warnings(app)
    _register_section_toggles(app)
    _register_slider_input_pairs(app)
    _register_buffer_qubits_clamp(app)
    _register_pin_axis(app)
    _register_architecture_summary(app)


# ---------------------------------------------------------------------------
# Budget warnings (live preview before clicking Run)
# ---------------------------------------------------------------------------

def _register_budget_warnings(app: Any) -> None:

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
# Collapsible section chevron toggles + budget summary strip
# ---------------------------------------------------------------------------

def _register_section_toggles(app: Any) -> None:

    def _make_section_toggle(section_id: str, default_open: bool = True) -> None:
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
            is_open = default_open if not n_clicks else (
                (n_clicks % 2 == 0) if default_open else (n_clicks % 2 == 1)
            )
            if is_open:
                return {"display": "block", "padding": "0 14px 12px"}, "▾"
            return {"display": "none", "padding": "0 14px 12px"}, "▸"

    _make_section_toggle("sweep-budget-section", default_open=False)

    # Sweep Budget summary strip — shown in the always-visible footer header.
    # Format: "<cold> cold · <hot> hot · <workers>w".
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
# can be edited; the slider updates as the user drags. Clientside sync keeps
# both in step. Loops are avoided by writing only the non-triggered output
# and trusting Dash's value-equality dedup on the trigger side.

def _register_slider_input_pairs(app: Any) -> None:

    def _bind(slider_id: str, log_scale: bool = False) -> None:
        input_id = f"{slider_id}-input"
        if log_scale:
            slider_to_input, input_to_slider = _SLIDER_TO_INPUT_LOG, _INPUT_TO_SLIDER_LOG
        else:
            slider_to_input, input_to_slider = _SLIDER_TO_INPUT_LIN, _INPUT_TO_SLIDER_LIN

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

    _bind("cfg-num-logical-qubits", log_scale=False)
    _bind("cfg-qubits-per-core", log_scale=False)
    _bind("cfg-num-cores", log_scale=False)
    _bind("cfg-communication-qubits", log_scale=False)
    _bind("cfg-buffer-qubits", log_scale=False)
    for m in SWEEPABLE_METRICS:
        _bind(f"noise-{m.key}", log_scale=m.log_scale)


# ---------------------------------------------------------------------------
# Buffer-qubit clamp: B ≤ K
# ---------------------------------------------------------------------------
# Logical-first model: the only runtime slider clamp left is B ≤ K
# (per-group rule). Comm qubits and logical qubits range freely; the
# resolver grows the unpinned architecture axis to fit, and any
# truly-infeasible cells render as NaN/white in the heat-map.

def _register_buffer_qubits_clamp(app: Any) -> None:

    @app.callback(
        Output("cfg-buffer-qubits", "max"),
        Output("cfg-buffer-qubits", "marks"),
        Output("cfg-buffer-qubits", "value", allow_duplicate=True),
        Input("cfg-communication-qubits", "value"),
        State("cfg-buffer-qubits", "value"),
        prevent_initial_call=True,
    )
    def _update_buffer_qubits_bound(comm_qubits, current):
        K = max(1, int(comm_qubits or 1))
        cur = int(current) if current else 1
        return K, _minmax_marks(1, K, log_scale=False), max(1, min(cur, K))


# ---------------------------------------------------------------------------
# Pin-toggle: clicking the lock next to Cores or Qubits-per-core flips
# which axis is the user-set input and which is the derived output.
# ---------------------------------------------------------------------------

def _register_pin_axis(app: Any) -> None:

    @app.callback(
        Output("cfg-pin-axis", "data"),
        Output("cfg-pin-cores-btn", "style"),
        Output("cfg-pin-qpc-btn", "style"),
        Output("cfg-row-num-cores-slider-row", "style"),
        Output("cfg-row-qubits-per-core-slider-row", "style"),
        Output("cfg-num-cores-derived", "style"),
        Output("cfg-qubits-per-core-derived", "style"),
        Output("cfg-pin-toast", "children"),
        Output("cfg-pin-toast", "style"),
        Input("cfg-pin-cores-btn", "n_clicks"),
        Input("cfg-pin-qpc-btn", "n_clicks"),
        Input("cfg-pin-axis", "data"),
        Input("num-metrics-store", "data"),
        *[Input(f"metric-dropdown-{i}", "value") for i in range(MAX_METRICS)],
        prevent_initial_call=True,
    )
    def _toggle_pin_axis(_n_cores, _n_qpc, current, num_metrics, *dropdown_vals):
        """Flip the pin axis when either toggle button is clicked, or
        refresh visuals when ``cfg-pin-axis`` is set programmatically (e.g.
        by the load-session callback).

        When an architectural axis (``num_cores`` or ``qubits_per_core``)
        is active on a sweep, flipping the pin to the *other* axis would
        un-sweep it (the unpinned axis becomes derived per cell). To
        prevent that we lock the pin to whichever axis is being swept,
        grey out the other button, and surface a brief toast if the user
        clicks the locked side anyway.
        """
        triggered = ctx.triggered_id

        n = int(num_metrics or 1)
        swept_keys: set[str] = set()
        for i in range(min(n, MAX_METRICS)):
            k = dropdown_vals[i] if i < len(dropdown_vals) else None
            if k:
                swept_keys.add(k)
        if "num_cores" in swept_keys:
            pin_lock: str | None = "cores"
        elif "qubits_per_core" in swept_keys:
            pin_lock = "qubits_per_core"
        else:
            pin_lock = None

        current = current or "cores"
        if triggered == "cfg-pin-cores-btn":
            target = "cores"
        elif triggered == "cfg-pin-qpc-btn":
            target = "qubits_per_core"
        else:
            target = current  # programmatic / dropdown / num-metrics trigger

        toast_text = ""
        toast_style: dict = {"display": "none"}
        is_button_click = triggered in ("cfg-pin-cores-btn", "cfg-pin-qpc-btn")
        if pin_lock is not None and is_button_click and target != pin_lock:
            new_pin = current
            swept_label = "num_cores" if pin_lock == "cores" else "qubits_per_core"
            target_label = "Cores" if target == "cores" else "Qubits/core"
            toast_text = (
                f"Can't switch to {target_label}: '{swept_label}' is on a "
                f"sweep axis. Remove that axis first."
            )
            toast_style = _PIN_TOAST_VISIBLE_STYLE
        else:
            new_pin = target

        cores_active = new_pin == "cores"
        if pin_lock is not None:
            cores_btn = (
                _PIN_TOGGLE_BTN_ACTIVE if cores_active
                else _PIN_TOGGLE_BTN_LOCKED_INACTIVE
            )
            qpc_btn = (
                _PIN_TOGGLE_BTN_LOCKED_INACTIVE if cores_active
                else _PIN_TOGGLE_BTN_ACTIVE
            )
        else:
            cores_btn = (
                _PIN_TOGGLE_BTN_ACTIVE if cores_active
                else _PIN_TOGGLE_BTN_INACTIVE
            )
            qpc_btn = (
                _PIN_TOGGLE_BTN_INACTIVE if cores_active
                else _PIN_TOGGLE_BTN_ACTIVE
            )
        cores_slider_style = {} if cores_active else {"display": "none"}
        qpc_slider_style = {"display": "none"} if cores_active else {}
        cores_derived_style = {"display": "none"} if cores_active else _DERIVED_BADGE_STYLE
        qpc_derived_style = _DERIVED_BADGE_STYLE if cores_active else {"display": "none"}
        return (
            new_pin, cores_btn, qpc_btn,
            cores_slider_style, qpc_slider_style,
            cores_derived_style, qpc_derived_style,
            toast_text, toast_style,
        )

    # Clientside fade-out for the pin-axis toast: when its text is set,
    # fade opacity to 0 after ~2.5 s. Mirrors the topology toast pattern.
    app.clientside_callback(
        """function(text) {
            if (!text) return window.dash_clientside.no_update;
            if (window._pinToastTimer) clearTimeout(window._pinToastTimer);
            window._pinToastTimer = setTimeout(function () {
                var el = document.getElementById("cfg-pin-toast");
                if (el) el.style.opacity = "0";
            }, 2500);
            return window.dash_clientside.no_update;
        }""",
        Output("cfg-pin-toast", "children", allow_duplicate=True),
        Input("cfg-pin-toast", "children"),
        prevent_initial_call=True,
    )


# ---------------------------------------------------------------------------
# Architecture summary: live-updates the derived-metrics panel and the
# derived-value badges as the user moves logical / cores / qpc / K / B.
# ---------------------------------------------------------------------------

def _register_architecture_summary(app: Any) -> None:

    @app.callback(
        Output("cfg-architecture-summary", "children"),
        Output("cfg-num-cores-derived", "children"),
        Output("cfg-qubits-per-core-derived", "children"),
        Input("cfg-num-logical-qubits", "value"),
        Input("cfg-num-cores", "value"),
        Input("cfg-qubits-per-core", "value"),
        Input("cfg-communication-qubits", "value"),
        Input("cfg-buffer-qubits", "value"),
        Input("cfg-topology", "value"),
        Input("cfg-pin-axis", "data"),
        prevent_initial_call=False,
    )
    def _update_architecture_summary(logical, cores, qpc, K, B, topo, pin_axis):
        """Live architecture-summary line + derived-value badges."""
        from quadris.dse.config import _resolve_architecture
        cfg = {
            "num_logical_qubits": int(logical or 16),
            "num_cores": int(cores or 1),
            "qubits_per_core": int(qpc or 16),
            "communication_qubits": int(K or 1),
            "buffer_qubits": int(B or 1),
            "topology_type": topo or "ring",
            "pin_axis": pin_axis or "cores",
        }
        res = _resolve_architecture(cfg)
        if not res["feasible"]:
            return (
                f"⚠ This configuration is not buildable: {res['reason']}",
                "Cores → ?",
                "Qubits/core → ?",
            )
        nq = int(cfg["num_qubits"])
        nc = int(cfg["num_cores"])
        qpc_v = int(cfg["qubits_per_core"])
        wasted = int(cfg["idle_reserved_qubits"])
        summary = (
            f"→ {nq} physical qubit{'s' if nq != 1 else ''} "
            f"({nc} core{'s' if nc != 1 else ''} × {qpc_v} per core)"
        )
        if wasted > 0:
            summary += (
                f" · {wasted} unused comm slots at edge cores "
                f"(non-uniform inter-core topology — corner cores carry "
                f"fewer real comm links than the chip-uniform reservation)"
            )
        return summary, f"Cores → {nc}", f"Qubits/core → {qpc_v}"
