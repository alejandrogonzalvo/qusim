"""
Pareto-front utilities for sweep results.

Pure-numpy: no Plotly / Dash dependency. Used by both the GUI's Merit view
and any user script wanting to filter sweep points by Pareto-optimality.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qusim.dse.axes import PARETO_METRIC_ORIENTATION
from qusim.dse.flatten import flatten_sweep_to_table


def pareto_front_mask(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return a boolean mask selecting Pareto-optimal points under
    *maximize* ``num`` and *minimize* ``den``.

    O(N²) — fine for the ≤4096-point sweeps the engine produces. For
    ten-thousand-plus sweeps swap in a dimension-sweep algorithm.

    A point ``i`` is dominated iff there exists ``j`` with
    ``num[j] >= num[i]`` and ``den[j] <= den[i]`` and at least one
    inequality is strict.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    n = num.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    num_arr = num.reshape(-1, 1)
    den_arr = den.reshape(-1, 1)
    ge_num = num_arr.T >= num_arr
    le_den = den_arr.T <= den_arr
    strict = (num_arr.T > num_arr) | (den_arr.T < den_arr)
    dominated = (ge_num & le_den & strict).any(axis=1)
    return ~dominated


def pareto_front(
    sweep: Any,
    objective_x: str,
    objective_y: str,
) -> dict[str, np.ndarray]:
    """Compute the Pareto front of a sweep over two output metrics.

    Parameters
    ----------
    sweep
        A :class:`qusim.dse.SweepResult` (or its ``.as_dict()``) — anything
        :func:`flatten_sweep_to_table` accepts.
    objective_x, objective_y
        Output metric keys (e.g. ``"total_epr_pairs"``,
        ``"overall_fidelity"``). Each axis is min/max-imised according to
        :data:`qusim.dse.axes.PARETO_METRIC_ORIENTATION`.

    Returns
    -------
    dict
        ``{"x": ..., "y": ..., "mask": ..., "axes": {<axis_key>: ...}}``
        where ``mask`` is the Pareto-optimal selector over the flattened
        sweep, and ``axes`` carries the swept-axis values for each point
        (handy for hover labels in custom plots).
    """
    if hasattr(sweep, "as_dict"):
        sweep_dict = sweep.as_dict()
    else:
        sweep_dict = sweep

    metric_keys, outputs, rows = flatten_sweep_to_table(sweep_dict)
    if rows.size == 0:
        return {"x": np.empty(0), "y": np.empty(0), "mask": np.empty(0, dtype=bool), "axes": {}}

    if objective_x not in outputs or objective_y not in outputs:
        raise ValueError(
            f"output metric not in sweep — got {objective_x!r}/{objective_y!r}, "
            f"sweep has {outputs}"
        )

    x = rows[:, len(metric_keys) + outputs.index(objective_x)]
    y = rows[:, len(metric_keys) + outputs.index(objective_y)]

    # Convert to (max num, min den) form expected by pareto_front_mask.
    x_dir = PARETO_METRIC_ORIENTATION.get(objective_x, "max")
    y_dir = PARETO_METRIC_ORIENTATION.get(objective_y, "max")
    num = y if y_dir == "max" else -y
    den = x if x_dir == "min" else -x

    mask = pareto_front_mask(num, den)

    axes_cols: dict[str, np.ndarray] = {
        k: rows[:, j] for j, k in enumerate(metric_keys)
    }

    return {"x": x, "y": y, "mask": mask, "axes": axes_cols}
