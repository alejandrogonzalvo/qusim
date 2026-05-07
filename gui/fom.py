"""
Backwards-compatibility shim — FoM now lives in :mod:`quadris.analysis`.

Existing imports continue to work::

    from gui.fom import FomConfig, compute_for_sweep

…but new code should prefer the library path::

    from quadris.analysis import FomConfig, compute_for_sweep
"""

from quadris.analysis.fom import *  # noqa: F401, F403
from quadris.analysis.fom import (  # noqa: F401  — re-export helpers used by gui/plotting
    _compile_expr,
    _eval_compiled,
    _referenced_names,
    _trim_label,
    _validate,
)
