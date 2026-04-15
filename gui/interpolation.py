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

def pick_frozen_axis(num_axes: int, view_type: str) -> int | None:
    """Decide which axis to freeze when displaying a 2D view of 3D data.

    Returns the axis index (0-based) to freeze, or None if no freezing needed.
    """
    if num_axes == 3 and view_type in _2D_VIEWS:
        return 2
    return None
