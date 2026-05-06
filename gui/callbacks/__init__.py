"""Dash callback registration entry points for the qusim GUI.

Each submodule exposes a ``register(app)`` function that attaches its
callbacks to the given Dash app. ``register_all`` calls them in the
order required by Dash's shared-output rules (the primary writer of
each ``allow_duplicate`` chain must register before its duplicates).
"""

from __future__ import annotations

from typing import Any


def register_all(app: Any) -> None:
    # Import inside the function so importing the package doesn't
    # trigger any of the submodule side-effects until app.py is ready.
    from . import config_panel, custom_qasm, export, sidebar, views

    sidebar.register(app)
    config_panel.register(app)
    views.register(app)
    custom_qasm.register(app)
    export.register(app)
