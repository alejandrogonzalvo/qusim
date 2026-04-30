"""Finite-difference derivative views over sweep data.

Researchers iterating on hardware co-design need to know not just *what* the
fidelity (or FoM) is at each design point but *how sensitive* it is to a
change in each parameter — where the leverage lives. This module computes
two derivative views from an existing sweep grid (no re-running the engine):

* **Elasticity** ``(x / F) · dF/dx`` — dimensionless, "% change in F per %
  change in x". Lets you compare across parameters with wildly different
  units (T1 in seconds, gate_error fraction, qubit count). Used by the 1-D
  Line view and the new Elasticity Comparison analysis tab.
* **Gradient magnitude** ``|∇F| = sqrt(Σ (∂F/∂x_i)²)`` — a single scalar
  per cell summarising how steep the response is, regardless of direction.
  Used as a 2-D / 3-D heatmap or isosurface mode for "robustness" analysis.

Both views are computed as numpy finite differences (``np.gradient``) over
the existing grid; the engine is never re-invoked.

Log-axis correction
-------------------
Several sweep axes are sampled on a log10 grid (T1, gate_error, …). For
those axes we compute derivatives in log-coordinate space, i.e. on
``np.log10(x)`` rather than ``x``. The elasticity formula then becomes
``(x/F) · dF/dx = (1/(F · ln 10)) · dF/d(log10 x)`` — handled internally
so callers don't have to know whether an axis is log-scaled.

Numerical caveats
-----------------
Finite differences amplify noise. For small grids (< 10 points per axis),
treat second-order quantities (mixed partials, Hessians) as qualitative.
We don't apply any smoothing here; first-order quantities are robust
enough on the 10–80-point sweeps the GUI typically runs.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .constants import METRIC_BY_KEY


_EPS = 1e-12  # avoid divide-by-zero in elasticity when F crosses zero


def _is_log_axis(axis_key: str) -> bool:
    """True when this axis is sampled on a log10 grid (per METRIC_BY_KEY)."""
    m = METRIC_BY_KEY.get(axis_key)
    return bool(m and getattr(m, "log_scale", False))


def _coord_array(values: Iterable[float], log_scale: bool) -> np.ndarray:
    """Return the coordinate array used for finite differences.

    For linear axes this is just the raw values. For log axes we use the
    log10 of the values so that ``np.gradient(F, coord)`` directly produces
    ``dF/d(log10 x)`` — the natural "per decade" derivative on a log-spaced
    grid.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(arr > 0, arr, np.nan))
    return arr


# ---------------------------------------------------------------------------
# 1-D elasticity along a single sweep axis
# ---------------------------------------------------------------------------


def elasticity_1d(
    F: np.ndarray, x_values: np.ndarray, axis_key: str,
) -> np.ndarray:
    """Elasticity ``(x/F) · dF/dx`` evaluated at every grid point.

    For log-scaled axes this is equivalent to ``(1 / (F · ln 10)) ·
    dF/d(log10 x)``; both forms produce the same dimensionless number — a
    1 % increase in *x* yields ``elasticity %`` change in *F*.

    Returns an array the same shape as ``F``. Cells where ``F`` is too
    close to zero return NaN — Plotly renders those as gaps.
    """
    F = np.asarray(F, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if F.shape != x.shape:
        raise ValueError(
            f"elasticity_1d: F shape {F.shape} != x shape {x.shape}",
        )
    log_axis = _is_log_axis(axis_key)
    coord = _coord_array(x, log_axis)
    dF_dcoord = np.gradient(F, coord)
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_axis:
            # dF/d(log10 x) → divide by F·ln(10) to get elasticity
            elast = dF_dcoord / (F * math.log(10.0))
        else:
            # raw dF/dx → multiply by x/F
            elast = (x / F) * dF_dcoord
    elast = np.where(np.abs(F) > _EPS, elast, np.nan)
    return elast


# ---------------------------------------------------------------------------
# N-D gradient magnitude
# ---------------------------------------------------------------------------


def gradient_magnitude(
    F: np.ndarray, axes: list[np.ndarray], axis_keys: list[str],
) -> np.ndarray:
    """Compute ``|∇F| = sqrt(Σ (∂F/∂x_i)²)`` on the sweep grid.

    Each axis's partial derivative is taken in its natural coordinate
    (log10 for log axes, raw values otherwise) so the magnitude is
    dimensionless-per-decade rather than dominated by axes whose units
    happen to be small. This makes the heatmap visually consistent with
    the absolute-value view (which uses the same log-decade rendering on
    its log axes).

    ``axes`` and ``axis_keys`` must match the dimensionality of ``F``.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != len(axes) or F.ndim != len(axis_keys):
        raise ValueError(
            "gradient_magnitude: F.ndim must match len(axes) and len(axis_keys); "
            f"got F.ndim={F.ndim}, axes={len(axes)}, keys={len(axis_keys)}",
        )
    sq_sum = np.zeros_like(F)
    for d, (ax_vals, ax_key) in enumerate(zip(axes, axis_keys)):
        coord = _coord_array(ax_vals, _is_log_axis(ax_key))
        # np.gradient with a coordinate array handles non-uniform spacing.
        partial = np.gradient(F, coord, axis=d)
        sq_sum = sq_sum + partial * partial
    return np.sqrt(sq_sum)


# ---------------------------------------------------------------------------
# Elasticity comparison — one curve per non-trajectory parameter
# ---------------------------------------------------------------------------


def elasticity_comparison(
    F_grid: np.ndarray,
    axes: list[np.ndarray],
    axis_keys: list[str],
    trajectory_axis: int,
) -> dict:
    """For each non-trajectory axis, compute its mean-over-other-axes
    elasticity *as a function of the trajectory axis*.

    The trajectory axis is plotted on the X-axis of the resulting view;
    every other axis becomes one curve overlaid on that X-axis, showing
    how that parameter's leverage on F evolves as you sweep the trajectory.
    Crossovers on this plot mark physical regime transitions
    (e.g. T1-limited → gate-limited).

    Returns
    -------
    dict with keys:
        ``trajectory_axis``  — int (echoed back)
        ``trajectory_key``   — axis_keys[trajectory_axis]
        ``trajectory_values``— 1-D ndarray, the X-axis points
        ``curves``           — dict ``{axis_key: 1-D ndarray}`` of elasticity
                              per non-trajectory axis, evaluated by averaging
                              over all other (non-trajectory, non-self) axes
                              at every trajectory grid point. For a 2-D
                              sweep the average is trivial (no other axes).
    """
    F = np.asarray(F_grid, dtype=np.float64)
    ndim = F.ndim
    if not 0 <= trajectory_axis < ndim:
        raise ValueError(
            f"trajectory_axis {trajectory_axis} out of range for {ndim}-D grid",
        )
    if len(axes) != ndim or len(axis_keys) != ndim:
        raise ValueError("axes / axis_keys length must match F.ndim")

    traj_vals = np.asarray(axes[trajectory_axis], dtype=np.float64)

    curves: dict[str, np.ndarray] = {}
    for d in range(ndim):
        if d == trajectory_axis:
            continue
        ax_vals = np.asarray(axes[d], dtype=np.float64)
        ax_key = axis_keys[d]
        log_axis = _is_log_axis(ax_key)
        coord = _coord_array(ax_vals, log_axis)
        # Partial derivative along this axis at every grid point.
        partial = np.gradient(F, coord, axis=d)
        # Broadcast x_d across F to compute elasticity per cell.
        x_shape = [1] * ndim
        x_shape[d] = ax_vals.shape[0]
        x_b = ax_vals.reshape(x_shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            if log_axis:
                cell_elast = partial / (F * math.log(10.0))
            else:
                cell_elast = (x_b / F) * partial
        cell_elast = np.where(np.abs(F) > _EPS, cell_elast, np.nan)
        # Average over every axis except the trajectory axis to collapse
        # to a 1-D curve indexed by the trajectory.
        other_axes = tuple(i for i in range(ndim) if i != trajectory_axis)
        with np.errstate(invalid="ignore"):
            curve = np.nanmean(cell_elast, axis=other_axes) if other_axes else cell_elast
        curves[ax_key] = curve.astype(np.float64)

    return {
        "trajectory_axis": trajectory_axis,
        "trajectory_key": axis_keys[trajectory_axis],
        "trajectory_values": traj_vals,
        "curves": curves,
    }


# ---------------------------------------------------------------------------
# Second derivative + inflection point (1-D Line view)
# ---------------------------------------------------------------------------


def second_derivative_1d(
    F: np.ndarray, x_values: np.ndarray, axis_key: str,
) -> np.ndarray:
    """Second derivative ``d²F/dx²`` evaluated at every grid point.

    For log-scaled axes this returns ``d²F/d(log10 x)²``, which is what the
    Line view actually visualises (its X-axis is log-rendered, so the
    coordinate-space curvature is what's meaningful — a positive value
    means the curve is concave-up *as drawn*, not as drawn against raw x).
    """
    F = np.asarray(F, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if F.shape != x.shape:
        raise ValueError(
            f"second_derivative_1d: F shape {F.shape} != x shape {x.shape}",
        )
    coord = _coord_array(x, _is_log_axis(axis_key))
    dF = np.gradient(F, coord)
    d2F = np.gradient(dF, coord)
    return d2F


def find_inflection_x(
    d2F: np.ndarray, x_values: np.ndarray,
) -> float | None:
    """Return the X-value of the *first* sign change in ``d²F/dx²``.

    Inflection = curvature flip = where the response transitions from
    accelerating to decelerating (or vice versa). For monotonic
    diminishing-returns curves there's typically exactly one inflection
    and it's the natural "sweet spot" beyond which extra investment in
    the swept parameter buys progressively less.

    Returns ``None`` when no sign change is detectable (curve is purely
    convex or concave, or all-NaN).
    """
    d2 = np.asarray(d2F, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if d2.size < 3:
        return None
    finite = np.isfinite(d2)
    if not finite.all():
        d2 = np.where(finite, d2, 0.0)
    # Sign change between consecutive samples — linear interpolate the
    # zero crossing between (x[i], d2[i]) and (x[i+1], d2[i+1]).
    for i in range(d2.size - 1):
        a, b = d2[i], d2[i + 1]
        if a == 0.0:
            return float(x[i])
        if a * b < 0.0:
            t = a / (a - b)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


# ---------------------------------------------------------------------------
# Mixed partial derivative (2-D Heatmap interaction map)
# ---------------------------------------------------------------------------


def _savgol_2d(
    grid: np.ndarray, window: int = 5, order: int = 2,
) -> np.ndarray:
    """Apply a separable Savitzky-Golay smoothing pass along both axes.

    Mixed partials amplify finite-difference noise — without smoothing
    the heatmap looks like static. We use the smallest credible window
    (length 5, order 2) so genuine interaction structure isn't smeared
    away. Falls back to the raw grid when an axis is too short for the
    chosen window.
    """
    from scipy.signal import savgol_filter

    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 2:
        return g
    w = min(window, g.shape[0] | 1, g.shape[1] | 1)  # force odd, fit shape
    if w < 3 or w <= order:
        return g
    smoothed = savgol_filter(g, window_length=w, polyorder=order, axis=0, mode="nearest")
    smoothed = savgol_filter(smoothed, window_length=w, polyorder=order, axis=1, mode="nearest")
    return smoothed


def mixed_partial_2d(
    F: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_key: str,
    y_key: str,
    smooth: bool = True,
) -> np.ndarray:
    """Mixed partial ``∂²F/∂x∂y`` over a 2-D sweep grid.

    Sign convention mirrors the elasticity-ish per-axis derivative: each
    partial is taken in its natural coordinate (log10 for log axes), so
    a positive entry means "increasing x makes y *more* effective at
    raising F" — synergy. Negative = parameters substitute for each
    other. Near-zero = independent effects.

    Optional Savitzky-Golay smoothing keeps the result readable on the
    coarse grids the GUI typically runs (≤ 50 points per axis).
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError(f"mixed_partial_2d: expected 2-D F, got shape {F.shape}")
    x_coord = _coord_array(x_values, _is_log_axis(x_key))
    y_coord = _coord_array(y_values, _is_log_axis(y_key))
    # ∂F/∂x first, then ∂(∂F/∂x)/∂y. F is shaped (Nx, Ny) per the rest of
    # the codebase (sweep_data["grid"][i][j] indexes axis-1 first).
    dF_dx = np.gradient(F, x_coord, axis=0)
    d2F = np.gradient(dF_dx, y_coord, axis=1)
    if smooth:
        d2F = _savgol_2d(d2F)
    return d2F


# ---------------------------------------------------------------------------
# Sensitivity ranking (gradient-based Parameter Importance)
# ---------------------------------------------------------------------------


def sensitivity_ranking(
    F_grid: np.ndarray, axes: list[np.ndarray], axis_keys: list[str],
) -> np.ndarray:
    """Per-axis ``mean(|∂F/∂x_i|)`` averaged over every other axis.

    Returns an array indexed in the order of ``axis_keys`` — caller
    typically zips this with the metric labels to render a horizontal
    bar chart, sorted by magnitude. Complements the existing
    variance-based Importance view: variance answers "how much does
    varying x change F overall", sensitivity answers "what's the local
    rate of change at this operating point" — both useful, this is the
    local one.

    Per-axis log-coordinate correction is applied (so log-scaled axes'
    sensitivities are per-decade, comparable across multi-scale param
    combinations).
    """
    F = np.asarray(F_grid, dtype=np.float64)
    if F.ndim != len(axes) or F.ndim != len(axis_keys):
        raise ValueError("sensitivity_ranking: ndim mismatch")
    out = np.zeros(F.ndim, dtype=np.float64)
    for d, (ax_vals, ax_key) in enumerate(zip(axes, axis_keys)):
        coord = _coord_array(ax_vals, _is_log_axis(ax_key))
        partial = np.gradient(F, coord, axis=d)
        with np.errstate(invalid="ignore"):
            out[d] = float(np.nanmean(np.abs(partial)))
    return out


# ---------------------------------------------------------------------------
# Interaction matrix (mean |mixed partial| per axis pair)
# ---------------------------------------------------------------------------


def interaction_matrix(
    F_grid: np.ndarray, axes: list[np.ndarray], axis_keys: list[str],
    smooth: bool = True,
) -> np.ndarray:
    """N×N matrix of ``mean(|∂²F/∂x_i∂x_j|)`` over the grid.

    Diagonal entries hold the mean of ``|∂²F/∂x_i²|`` (curvature). Off-
    diagonals hold mixed-partial magnitudes — entry ``(i, j)`` summarises
    "how strongly do axes i and j interact across the sweep". Symmetric
    by construction (Schwarz's theorem on the smoothed field).

    Same partial-coordinate convention as :func:`mixed_partial_2d`:
    every partial is taken in log10-space for log axes, raw otherwise,
    so cross-axis comparisons are meaningful even when units differ.
    """
    F = np.asarray(F_grid, dtype=np.float64)
    n = F.ndim
    if n != len(axes) or n != len(axis_keys):
        raise ValueError("interaction_matrix: ndim mismatch")

    coords = [
        _coord_array(ax_vals, _is_log_axis(ax_key))
        for ax_vals, ax_key in zip(axes, axis_keys)
    ]
    partials = [np.gradient(F, c, axis=d) for d, c in enumerate(coords)]

    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d2 = np.gradient(partials[i], coords[j], axis=j)
            if smooth and n == 2:
                d2 = _savgol_2d(d2)
            with np.errstate(invalid="ignore"):
                v = float(np.nanmean(np.abs(d2)))
            M[i, j] = v
            M[j, i] = v
    return M


# ---------------------------------------------------------------------------
# Grid extraction helper — pulls F out of the nested-list sweep grid into a
# numpy ndarray with the natural shape (matches the order of metric_keys).
# ---------------------------------------------------------------------------


def extract_grid_values(
    grid, ndim: int, output_key: str,
) -> np.ndarray:
    """Walk a nested-list sweep grid into an ndarray of the given output key.

    Returns shape ``(len(axis_0), len(axis_1), …, len(axis_{ndim-1}))`` with
    NaN substituted for missing cells.
    """
    def _walk(node, depth):
        if depth == ndim:
            if isinstance(node, dict):
                v = node.get(output_key)
            else:
                v = getattr(node, output_key, None)
            return float(v) if v is not None else float("nan")
        return [_walk(child, depth + 1) for child in node]

    nested = _walk(grid, 0)
    return np.asarray(nested, dtype=np.float64)
