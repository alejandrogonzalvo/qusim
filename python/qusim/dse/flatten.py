"""
Flatten an N-D :class:`SweepResult` (or sweep dict) into a column-wise table.

Used by FoM evaluation, Pareto-front analysis, CSV export, and the GUI's
plotting layer. Pure-numpy: no Plotly / Dash imports.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


# Scalar output keys carried in every sweep cell. Mirrors
# ``qusim.dse.engine._RESULT_SCALAR_KEYS``; duplicated here so this module
# stays a leaf with no dependency on the engine.
_OUTPUT_KEYS: tuple[str, ...] = (
    "overall_fidelity",
    "algorithmic_fidelity",
    "routing_fidelity",
    "coherence_fidelity",
    "readout_fidelity",
    "total_circuit_time_ns",
    "total_epr_pairs",
    "total_swaps",
    "total_teleportations",
    "total_network_distance",
)


def flatten_sweep_to_table(sweep_data: dict) -> tuple[list[str], list[str], np.ndarray]:
    """Flatten a sweep's results into a ``(total_points, ndim + n_outputs)`` matrix.

    Parameters
    ----------
    sweep_data
        A sweep dict — typically ``SweepResult.as_dict()`` or the result of
        ``DSEEngine.sweep_nd(...).as_dict()``. Must contain ``metric_keys``,
        ``grid``, plus axis values (``xs/ys/zs`` or ``axes``).

    Returns
    -------
    (metric_keys, available_outputs, data)
        ``data`` is a ``float64`` ndarray; do not re-wrap it.
    """
    # Fast path: pre-built table from _flatten_facets_for_analysis.
    if "_prebuilt_table" in sweep_data:
        return sweep_data["_prebuilt_table"]

    metric_keys = sweep_data["metric_keys"]
    grid = sweep_data["grid"]
    ndim = len(metric_keys)

    sample = _find_sample(grid, ndim)
    available_outputs = [k for k in _OUTPUT_KEYS if k in sample] if sample else []

    axes = _resolve_axes(sweep_data, ndim)

    if isinstance(grid, np.ndarray) and grid.dtype.names:
        data = _flatten_structured(grid, axes, ndim, available_outputs)
        return metric_keys, available_outputs, data

    rows: list[list[float]] = []
    if ndim <= 3:
        _flatten_nested(grid, axes, ndim, available_outputs, rows)
    else:
        shape = tuple(sweep_data.get("shape", [len(ax) for ax in axes]))
        _flatten_nd(grid, axes, shape, ndim, available_outputs, rows)

    if len(rows) == 0:
        return metric_keys, available_outputs, np.empty(
            (0, ndim + len(available_outputs)), dtype=np.float64,
        )
    return metric_keys, available_outputs, np.asarray(rows, dtype=np.float64)


def _flatten_structured(
    grid: np.ndarray,
    axes: list,
    ndim: int,
    outputs: list[str],
) -> np.ndarray:
    shape = grid.shape
    total = int(np.prod(shape))
    n_out = len(outputs)
    data = np.empty((total, ndim + n_out), dtype=np.float64)

    for d in range(ndim):
        ax = np.asarray(axes[d], dtype=np.float64)
        reshape = [1] * ndim
        reshape[d] = shape[d]
        data[:, d] = np.broadcast_to(ax.reshape(reshape), shape).ravel()

    for i, k in enumerate(outputs):
        data[:, ndim + i] = grid[k].ravel().astype(np.float64, copy=False)

    return data


def _find_sample(grid: Any, ndim: int) -> dict | None:
    if isinstance(grid, np.ndarray) and grid.dtype.names:
        return {name: 0.0 for name in grid.dtype.names}
    if not grid:
        return None
    item = grid[0]
    if isinstance(item, dict):
        return item
    nested = item
    while isinstance(nested, list) and nested:
        nested = nested[0]
    return nested if isinstance(nested, dict) else None


def _resolve_axes(sweep_data: dict, ndim: int) -> list[list]:
    if "axes" in sweep_data and len(sweep_data["axes"]) == ndim:
        return sweep_data["axes"]
    axes = [sweep_data["xs"]]
    if ndim >= 2:
        axes.append(sweep_data["ys"])
    if ndim >= 3:
        axes.append(sweep_data["zs"])
    return axes


def _flatten_nested(
    grid: Any, axes: list, ndim: int, outputs: list[str], rows: list,
) -> None:
    axis_values = [list(enumerate(ax)) for ax in axes[:ndim]]
    for combo in itertools.product(*axis_values):
        indices = [c[0] for c in combo]
        values = [float(c[1]) for c in combo]
        r = grid
        for idx in indices:
            r = r[idx]
        row = values[:]
        for k in outputs:
            row.append(float(r.get(k, 0.0) if isinstance(r, dict) else getattr(r, k, 0.0)))
        rows.append(row)


def _flatten_nd(
    grid: list, axes: list, shape: tuple, ndim: int,
    outputs: list[str], rows: list,
) -> None:
    ranges = [range(s) for s in shape]
    flat_idx = 0
    for combo in itertools.product(*ranges):
        values = [float(axes[d][combo[d]]) for d in range(ndim)]
        r = grid[flat_idx]
        row = values[:]
        for k in outputs:
            row.append(float(r.get(k, 0.0) if isinstance(r, dict) else getattr(r, k, 0.0)))
        rows.append(row)
        flat_idx += 1
