"""Custom QASM upload callbacks: parse upload, render status, modal toggle."""

from __future__ import annotations

import base64
from typing import Any

from dash import ALL, Input, Output, State, ctx, html

from gui.components import COLORS, FEEDBACK_COLORS

_EMPTY_QASM_STORE = {"qasm": None, "filename": None, "num_qubits": None, "error": None}


def _qasm_error_payload(filename, error):
    return {"qasm": None, "filename": filename, "num_qubits": None, "error": error}


def _decode_upload(contents: str) -> str:
    _, b64 = contents.split(",", 1)
    return base64.b64decode(b64).decode("utf-8")


def _parse_qasm(qasm_str: str) -> int:
    from qiskit import qasm2

    circ = qasm2.loads(qasm_str)
    return int(circ.num_qubits)


def _render_status_loaded(filename: str, nq: int | None) -> list:
    return [
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


def _render_status_error(err: str) -> list:
    return [
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


def register(app: Any) -> None:

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
        triggered = ctx.triggered_id
        cleared = (
            isinstance(triggered, dict)
            and triggered.get("type") == "custom-qasm-clear"
            and any(clear_clicks or [])
        )
        if cleared or contents is None:
            return _EMPTY_QASM_STORE, (sweep_dirty or 0) + 1

        try:
            qasm_str = _decode_upload(contents)
        except (ValueError, UnicodeDecodeError) as exc:
            return (
                _qasm_error_payload(filename, f"Failed to decode upload: {exc}"),
                sweep_dirty or 0,
            )

        try:
            num_qubits = _parse_qasm(qasm_str)
        except Exception as exc:
            return (
                _qasm_error_payload(filename, f"Not a valid OpenQASM 2.0 circuit: {exc}"),
                sweep_dirty or 0,
            )

        return (
            {"qasm": qasm_str, "filename": filename, "num_qubits": num_qubits, "error": None},
            (sweep_dirty or 0) + 1,
        )

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

        visible = {}
        hidden = {"display": "none"}

        if err:
            return (
                _render_status_error(err),
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

        return (
            _render_status_loaded(filename, nq),
            {"display": "block"},
            f"Replace ({filename})",
            hidden, hidden, hidden,
        )

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
