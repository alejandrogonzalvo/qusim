"""CSV export callback."""

from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State

from gui.plotting import sweep_to_csv
from gui.server_state import _get_sweep


def register(app: Any) -> None:
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
