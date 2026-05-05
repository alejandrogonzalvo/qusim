"""
Analysis primitives over sweep results: Figure-of-Merit and Pareto frontiers.

These are the algorithmic helpers behind the GUI's Merit / Pareto views, but
work on any :class:`qusim.dse.SweepResult` (or its ``.as_dict()`` form), so
custom Python scripts get the same machinery as the Dash app.
"""

from .fom import (
    DEFAULT_FOM,
    PRESETS,
    PRESET_OPTIONS,
    FomBreakdown,
    FomConfig,
    FomError,
    FomResult,
    SAFE_FUNCS,
    compute_breakdown,
    compute_for_sweep,
    evaluate,
    primitives_for_sweep,
)
from .pareto import pareto_front, pareto_front_mask

__all__ = [
    # FoM
    "FomConfig",
    "FomResult",
    "FomBreakdown",
    "FomError",
    "PRESETS",
    "PRESET_OPTIONS",
    "DEFAULT_FOM",
    "SAFE_FUNCS",
    "evaluate",
    "compute_for_sweep",
    "compute_breakdown",
    "primitives_for_sweep",
    # Pareto
    "pareto_front_mask",
    "pareto_front",
]
