"""
Client-side interpolation utilities for the frozen-slider hot-reload pattern.

Python implementations used for:
  1. Server-side validation of interpolated values
  2. Reference implementations mirrored by the JS clientside callbacks

Grid convention
---------------
  1D: values[ix]
  2D: values[iy][ix]          (row = y, col = x)
  3D: values[iz][iy][ix]
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bisect helper — find the left index for interpolation
# ---------------------------------------------------------------------------

def _bisect(xs: np.ndarray, x: float) -> int:
    """Return index i such that xs[i] <= x < xs[i+1], clamped to valid range."""
    n = len(xs)
    if n < 2:
        return 0
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return n - 2
    idx = int(np.searchsorted(xs, x, side="right")) - 1
    return min(idx, n - 2)


# ---------------------------------------------------------------------------
# 1-D linear interpolation
# ---------------------------------------------------------------------------

def lerp(xs: np.ndarray, vs: np.ndarray, x: float) -> float:
    """Linearly interpolate vs at position x given axis values xs.

    Clamps to the boundary values when x is outside [xs[0], xs[-1]].
    """
    i = _bisect(xs, x)
    if xs[i + 1] == xs[i]:
        return float(vs[i])
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    t = max(0.0, min(1.0, t))
    return float(vs[i] * (1 - t) + vs[i + 1] * t)


# ---------------------------------------------------------------------------
# 2-D bilinear interpolation
# ---------------------------------------------------------------------------

def bilerp(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Bilinear interpolation on a 2D grid.

    grid shape: (ny, nx).  Clamps to boundaries.
    """
    i = _bisect(xs, x)
    j = _bisect(ys, y)

    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[j], ys[j + 1]

    fx = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    fy = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))

    v00 = grid[j, i]
    v10 = grid[j, i + 1]
    v01 = grid[j + 1, i]
    v11 = grid[j + 1, i + 1]

    return float(
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )


def bilerp_mesh(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
) -> np.ndarray:
    """Evaluate bilinear interpolation over a mesh of query points.

    Returns a (len(qy), len(qx)) array.
    """
    result = np.empty((len(qy), len(qx)))
    for jj, y in enumerate(qy):
        for ii, x in enumerate(qx):
            result[jj, ii] = bilerp(grid, xs, ys, x, y)
    return result


# ---------------------------------------------------------------------------
# 3-D trilinear interpolation
# ---------------------------------------------------------------------------

def trilerp(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    """Trilinear interpolation on a 3D grid.

    grid shape: (nz, ny, nx).  Clamps to boundaries.
    """
    i = _bisect(xs, x)
    j = _bisect(ys, y)
    k = _bisect(zs, z)

    fx = (x - xs[i]) / (xs[i + 1] - xs[i]) if xs[i + 1] != xs[i] else 0.0
    fy = (y - ys[j]) / (ys[j + 1] - ys[j]) if ys[j + 1] != ys[j] else 0.0
    fz = (z - zs[k]) / (zs[k + 1] - zs[k]) if zs[k + 1] != zs[k] else 0.0
    fx = max(0.0, min(1.0, fx))
    fy = max(0.0, min(1.0, fy))
    fz = max(0.0, min(1.0, fz))

    c = np.zeros(8)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                c[dz * 4 + dy * 2 + dx] = grid[k + dz, j + dy, i + dx]

    return float(
        c[0] * (1 - fx) * (1 - fy) * (1 - fz)
        + c[1] * fx * (1 - fy) * (1 - fz)
        + c[2] * (1 - fx) * fy * (1 - fz)
        + c[3] * fx * fy * (1 - fz)
        + c[4] * (1 - fx) * (1 - fy) * fz
        + c[5] * fx * (1 - fy) * fz
        + c[6] * (1 - fx) * fy * fz
        + c[7] * fx * fy * fz
    )


# ---------------------------------------------------------------------------
# Frozen slice: extract 2D slice from 3D grid at a given z value
# ---------------------------------------------------------------------------

def frozen_slice(
    grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    z_value: float,
) -> np.ndarray:
    """Extract a 2D (ny, nx) slice from a 3D grid by interpolating along z.

    This is the core of the frozen-slider pattern: the user drags a slider
    for the z axis, and we produce a fresh 2D heatmap without a server call.
    """
    k = _bisect(zs, z_value)
    fz = (z_value - zs[k]) / (zs[k + 1] - zs[k]) if zs[k + 1] != zs[k] else 0.0
    fz = max(0.0, min(1.0, fz))

    plane0 = grid[k]
    plane1 = grid[k + 1]
    return plane0 * (1 - fz) + plane1 * fz


# ---------------------------------------------------------------------------
# Grid format conversion: sweep_data → compact interp-ready dict
# ---------------------------------------------------------------------------

def sweep_to_interp_grid(sweep_data: dict, output_key: str) -> dict:
    """Convert a sweep_data store dict into a compact interpolation grid.

    The returned dict is JSON-serialisable and suitable for storage in a
    dcc.Store for use by clientside_callbacks.
    """
    metric_keys = sweep_data["metric_keys"]
    ndim = len(metric_keys)
    xs = sweep_data["xs"]

    if ndim == 1:
        raw = sweep_data["grid"]
        values = [float(r.get(output_key, 0.0)) for r in raw]
        return {"ndim": 1, "metric_keys": metric_keys, "xs": xs, "values": values}

    ys = sweep_data["ys"]
    if ndim == 2:
        raw = sweep_data["grid"]
        nx = len(xs)
        ny = len(ys)
        values = [[0.0] * nx for _ in range(ny)]
        for i in range(nx):
            for j in range(ny):
                values[j][i] = float(raw[i][j].get(output_key, 0.0))
        return {
            "ndim": 2,
            "metric_keys": metric_keys,
            "xs": xs,
            "ys": ys,
            "values": values,
        }

    zs = sweep_data["zs"]
    nx, ny, nz = len(xs), len(ys), len(zs)
    raw = sweep_data["grid"]
    values = [[[0.0] * nx for _ in range(ny)] for _ in range(nz)]
    for i in range(nx):
        for j in range(ny):
            for k_idx in range(nz):
                values[k_idx][j][i] = float(
                    raw[i][j][k_idx].get(output_key, 0.0)
                )
    return {
        "ndim": 3,
        "metric_keys": metric_keys,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Frozen axis selection
# ---------------------------------------------------------------------------

_2D_VIEWS = {"heatmap", "contour"}
_FROZEN_VIEWS = {"frozen_heatmap", "frozen_contour"}
_FROZEN_TO_BASE = {"frozen_heatmap": "heatmap", "frozen_contour": "contour"}


def pick_frozen_axis(num_axes: int, view_type: str) -> int | None:
    """Decide which axis to freeze when displaying a 2D view of 3D data.

    Returns the axis index (0-based) to freeze, or None if no freezing needed.
    """
    if num_axes == 3 and view_type in _2D_VIEWS:
        return 2
    return None


def is_frozen_view(view_type: str) -> bool:
    """Return True if the view type is a frozen-slider view."""
    return view_type in _FROZEN_VIEWS


def frozen_view_base(view_type: str) -> str:
    """Map a frozen view type to its underlying 2D view type."""
    return _FROZEN_TO_BASE.get(view_type, view_type)


def frozen_slider_config(sweep_data: dict) -> dict | None:
    """Return frozen slider configuration for a 3D sweep, or None if not 3D.

    Returns dict with keys: min, max, default, metric_key, step.
    """
    metric_keys = sweep_data.get("metric_keys", [])
    if len(metric_keys) != 3:
        return None
    zs = sweep_data.get("zs", [])
    if len(zs) < 2:
        return None
    z_min = float(zs[0])
    z_max = float(zs[-1])
    return {
        "min": z_min,
        "max": z_max,
        "default": (z_min + z_max) / 2,
        "metric_key": metric_keys[2],
        "step": (z_max - z_min) / max(1, len(zs) - 1) / 4,
    }


def permute_sweep_for_frozen(sweep_data: dict, frozen_idx: int) -> dict:
    """Return a sweep_data dict rearranged so that ``frozen_idx`` ends at axis 2.

    Downstream consumers (build_figure, sweep_to_interp_grid, frozen_slice) all
    assume the frozen axis is the third one (``zs``). For 3D sweeps this helper
    permutes ``xs``, ``ys``, ``zs``, ``grid`` and ``metric_keys`` so that the
    user's chosen ``frozen_idx`` becomes axis 2 while the other two axes keep
    their original relative order at positions 0 and 1.

    Returns the input unchanged for non-3D sweeps or when ``frozen_idx == 2``.
    """
    metric_keys = sweep_data.get("metric_keys", [])
    if len(metric_keys) != 3 or frozen_idx == 2:
        return sweep_data

    if frozen_idx not in (0, 1):
        return sweep_data

    axes_data = [sweep_data["xs"], sweep_data["ys"], sweep_data["zs"]]
    free = [i for i in range(3) if i != frozen_idx]
    new_order = free + [frozen_idx]  # e.g. frozen=0 → [1, 2, 0]
    new_xs = list(axes_data[new_order[0]])
    new_ys = list(axes_data[new_order[1]])
    new_zs = list(axes_data[new_order[2]])
    new_keys = [metric_keys[i] for i in new_order]

    # Original grid is indexed as grid[i_x][i_y][i_z].
    # We want new_grid[i_x'][i_y'][i_z'] where the indices correspond to new_order.
    nx, ny, nz = len(sweep_data["xs"]), len(sweep_data["ys"]), len(sweep_data["zs"])
    sizes = [nx, ny, nz]
    new_sizes = [sizes[new_order[0]], sizes[new_order[1]], sizes[new_order[2]]]
    old_grid = sweep_data["grid"]

    new_grid = [
        [
            [None for _ in range(new_sizes[2])]
            for _ in range(new_sizes[1])
        ]
        for _ in range(new_sizes[0])
    ]
    # idx[d] = old-axis-d position, computed from new indices.
    for a in range(new_sizes[0]):
        for b in range(new_sizes[1]):
            for c in range(new_sizes[2]):
                old_idx = [0, 0, 0]
                old_idx[new_order[0]] = a
                old_idx[new_order[1]] = b
                old_idx[new_order[2]] = c
                new_grid[a][b][c] = old_grid[old_idx[0]][old_idx[1]][old_idx[2]]

    out = dict(sweep_data)
    out["xs"] = new_xs
    out["ys"] = new_ys
    out["zs"] = new_zs
    out["metric_keys"] = new_keys
    out["grid"] = new_grid
    return out


def frozen_slider_config_nd(
    sweep_data: dict,
    free_axes: list[int] | None = None,
) -> list[dict]:
    """Return frozen slider configs for N-D sweeps.

    Given *free_axes* (the axes to keep free for plotting), returns one
    slider config per frozen axis.

    Parameters
    ----------
    sweep_data : dict with ``metric_keys`` and ``axes`` (or zs for 3D).
    free_axes : indices of axes to keep free (default: [0, 1]).

    Returns
    -------
    List of dicts, each with keys: min, max, default, metric_key, step.
    """
    metric_keys = sweep_data.get("metric_keys", [])
    ndim = len(metric_keys)
    if ndim < 3:
        return []

    if free_axes is None:
        free_axes = [0, 1]

    # Get axis values
    axes = sweep_data.get("axes", [])
    if not axes:
        # Fall back to legacy format
        axes = [sweep_data.get("xs", [])]
        if ndim >= 2:
            axes.append(sweep_data.get("ys", []))
        if ndim >= 3:
            axes.append(sweep_data.get("zs", []))

    configs = []
    for d in range(ndim):
        if d in free_axes:
            continue
        ax = axes[d] if d < len(axes) else []
        if len(ax) < 2:
            continue
        ax_min = float(ax[0])
        ax_max = float(ax[-1])
        configs.append({
            "min": ax_min,
            "max": ax_max,
            "default": (ax_min + ax_max) / 2,
            "metric_key": metric_keys[d],
            "step": (ax_max - ax_min) / max(1, len(ax) - 1) / 4,
        })

    return configs
