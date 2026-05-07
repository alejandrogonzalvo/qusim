"""
Plotly figure builders for 1-D, 2-D, and 3-D fidelity sweep results.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from .constants import (
    CAT_METRIC_BY_KEY,
    FIDELITY_METRICS,
    METRIC_BY_KEY,
    OUTPUT_METRIC_LABEL,
    OUTPUT_METRICS,
    PARETO_METRIC_ORIENTATION,
)
from quadris.dse.flatten import (
    flatten_sweep_to_table as _flatten_sweep_to_table,
    _OUTPUT_KEYS,
    _find_sample,
    _flatten_nd,
    _flatten_nested,
    _flatten_structured,
    _resolve_axes,
)
from quadris.analysis.pareto import pareto_front_mask as _pareto_front_mask

# Map output metric key → display label
_OUTPUT_LABELS = {m["value"]: m["label"] for m in OUTPUT_METRICS}

_BG = "#FFFFFF"
_PLOT_BG = "#FAFAFA"
_GRID_COLOR = "#E8E8E8"
_TEXT_COLOR = "#2B2B2B"
_TEXT_MUTED = "#888888"
_ACCENT = "#2B2B2B"
_ACCENT2 = "#555555"
_LINE_COLOR = "#2B2B2B"

_THRESHOLD_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#91bfdb", "#4575b4"]
_DEFAULT_THRESHOLDS = [0.3, 0.6, 0.9]

# ---------------------------------------------------------------------------
# Unified colorscales — every continuous-value plot (heatmap, scatter3d,
# parallel coords, merit, isosurface fallback) draws from the same
# red→amber→blue ramp. Discrete iso-level markers (``_THRESHOLD_COLORS``)
# share the same five stops so contours land on visually consistent hues.
# ---------------------------------------------------------------------------

# Primary continuous scale: red (low) → amber (mid) → blue (high). Matches
# ``_THRESHOLD_COLORS`` and reads "more is better" for fidelity / FoM.
_COLORSCALE = [
    [0.00, "#d73027"],
    [0.25, "#fc8d59"],
    [0.50, "#fee08b"],
    [0.75, "#91bfdb"],
    [1.00, "#4575b4"],
]

# Diverging variant for signed metrics (correlation, deltas) — same end
# colours as ``_COLORSCALE`` but white at the midpoint so 0 reads neutral.
_COLORSCALE_DIVERGING = [
    [0.00, "#d73027"],
    [0.25, "#fc8d59"],
    [0.50, "#FFFFFF"],
    [0.75, "#91bfdb"],
    [1.00, "#4575b4"],
]

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(color=_TEXT_COLOR, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=55, r=20, t=50, b=45),
)


def _extract(results: list, output_key: str) -> np.ndarray:
    """Extract a scalar output value from a list of fidelity result dicts."""
    values = []
    for r in results:
        if isinstance(r, dict):
            values.append(float(r.get(output_key, 0.0)))
        else:
            values.append(float(getattr(r, output_key, 0.0)))
    return np.array(values)


def _axis_label(metric_key: str) -> str:
    m = METRIC_BY_KEY.get(metric_key) or CAT_METRIC_BY_KEY.get(metric_key)
    if m is None:
        return metric_key
    unit = f" ({m.unit})" if hasattr(m, "unit") and m.unit else ""
    return f"{m.label}{unit}"


# ---------------------------------------------------------------------------
# 1-D line plot
# ---------------------------------------------------------------------------

def plot_1d(
    x_values: np.ndarray,
    results: list,
    metric_key: str,
    output_key: str,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    inflection_x: float | None = None,
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    y = _extract(results, output_key)
    m = METRIC_BY_KEY.get(metric_key)
    x_log = m.log_scale if m else False

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values, y=y,
            mode="lines",
            line=dict(color=_LINE_COLOR, width=2, shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(43, 43, 43, 0.05)",
            name=_OUTPUT_LABELS.get(output_key, output_key),
            hovertemplate="%{x:.3e}<br><b>%{y:.4f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values, y=y,
            mode="markers",
            marker=dict(color=_LINE_COLOR, size=3.5, opacity=0.4, line=dict(width=0)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    xaxis_cfg = dict(
        title=dict(text=_axis_label(metric_key), font=dict(size=12, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
        tickfont=dict(size=10, color=_TEXT_MUTED),
    )
    if x_log:
        xaxis_cfg["type"] = "log"

    yaxis_cfg = dict(
        title=dict(text=_OUTPUT_LABELS.get(output_key, output_key), font=dict(size=12, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
        range=[0, 1] if "fidelity" in output_key else None,
        tickfont=dict(size=10, color=_TEXT_MUTED),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=xaxis_cfg, yaxis=yaxis_cfg,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
        showlegend=False,
    )

    if thresholds:
        for i, t in enumerate(thresholds):
            fig.add_shape(
                type="line", xref="paper", x0=0, x1=1, y0=t, y1=t,
                line=dict(color=_colors[i % len(_colors)], width=1.5, dash="dash"),
            )
        lowest = min(thresholds)
        fig.add_shape(
            type="rect", xref="paper", x0=0, x1=1, y0=0, y1=lowest,
            fillcolor="rgba(215, 48, 39, 0.08)", line=dict(width=0),
        )

    # Diminishing-returns marker: when the second-derivative view computed
    # an inflection x, draw a vertical guide + label so the saturation
    # point is impossible to miss.
    if inflection_x is not None and np.isfinite(inflection_x):
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0=inflection_x, x1=inflection_x, y0=0, y1=1,
            line=dict(color=_THRESHOLD_COLORS[0], width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=inflection_x, y=1.0, xref="x", yref="paper",
            text=f"inflection x ≈ {inflection_x:.3g}",
            showarrow=False,
            xanchor="left", yanchor="top",
            xshift=4, yshift=-4,
            font=dict(size=10, color=_THRESHOLD_COLORS[0]),
        )

    return fig


# ---------------------------------------------------------------------------
# 2-D heatmap (legacy no-iso-line variant, kept as a thin wrapper that adds
# the iso-line overlay so any external caller migrating from this signature
# gets the unified Heatmap rendering).
# ---------------------------------------------------------------------------

def plot_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: list,          # list[list[dict]]  shape: [Nx][Ny]
    metric_key1: str,
    metric_key2: str,
    output_key: str,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    return plot_2d_contour(
        x_values=x_values, y_values=y_values, grid=grid,
        metric_key1=metric_key1, metric_key2=metric_key2,
        output_key=output_key,
        thresholds=thresholds, threshold_colors=threshold_colors,
    )



# ---------------------------------------------------------------------------
# 2-D contour heatmap (iso-lines overlay)
# ---------------------------------------------------------------------------

def plot_2d_contour(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: list,
    metric_key1: str,
    metric_key2: str,
    output_key: str,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    z = np.zeros((len(y_values), len(x_values)))
    for i, row in enumerate(grid):
        for j, r in enumerate(row):
            val = r.get(output_key, 0.0) if isinstance(r, dict) else getattr(r, output_key, 0.0)
            z[j, i] = float(val)

    m1 = METRIC_BY_KEY.get(metric_key1)
    m2 = METRIC_BY_KEY.get(metric_key2)
    x_log = m1.log_scale if m1 else False
    y_log = m2.log_scale if m2 else False

    x_plot = np.log10(x_values) if x_log else x_values
    y_plot = np.log10(y_values) if y_log else y_values

    is_fidelity = "fidelity" in output_key
    zmin, zmax = (0.0, 1.0) if is_fidelity else (float(z.min()), float(z.max()))

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_plot, y=y_plot, z=z,
            zmin=zmin, zmax=zmax,
            colorscale=_COLORSCALE,
            hovertemplate=(
                _axis_label(metric_key1) + ": %{x:.3g}<br>"
                + _axis_label(metric_key2) + ": %{y:.3g}<br>"
                + "<b>%{z:.4f}</b><extra></extra>"
            ),
            colorbar=dict(
                title=dict(text=_OUTPUT_LABELS.get(output_key, output_key), side="right",
                           font=dict(size=11, color=_TEXT_MUTED)),
                tickfont=dict(color=_TEXT_MUTED, size=10),
                bgcolor="rgba(0,0,0,0)",
                outlinewidth=0,
                thickness=14,
                len=0.85,
            ),
        )
    )

    fig.add_trace(
        go.Contour(
            x=x_plot, y=y_plot, z=z,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color=_TEXT_COLOR),
                coloring="none",
            ),
            line=dict(color="rgba(43,43,43,0.6)", width=1.5),
            showscale=False,
            hoverinfo="skip",
        )
    )

    if thresholds:
        for i, t in enumerate(thresholds):
            c = _colors[i % len(_colors)]
            fig.add_trace(
                go.Contour(
                    x=x_plot, y=y_plot, z=z,
                    contours=dict(
                        start=t, end=t, size=0,
                        showlabels=True,
                        labelfont=dict(size=11, color=c),
                        coloring="none",
                    ),
                    line=dict(color=c, width=2.5),
                    showscale=False,
                    hoverinfo="skip",
                    name=f"threshold {t}",
                )
            )

    x_title = _axis_label(metric_key1) + (" (log\u2081\u2080)" if x_log else "")
    y_title = _axis_label(metric_key2) + (" (log\u2081\u2080)" if y_log else "")

    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(title=dict(text=x_title, font=dict(size=12, color=_TEXT_MUTED)),
                   gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED)),
        yaxis=dict(title=dict(text=y_title, font=dict(size=12, color=_TEXT_MUTED)),
                   gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D surface
# ---------------------------------------------------------------------------

def plot_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    grid: list,          # list[list[list[dict]]]  shape: [Nx][Ny][Nz]
    metric_key1: str,
    metric_key2: str,
    metric_key3: str,
    output_key: str,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    point_cap: int | None = None,
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    m1 = METRIC_BY_KEY.get(metric_key1)
    m2 = METRIC_BY_KEY.get(metric_key2)
    m3 = METRIC_BY_KEY.get(metric_key3)

    # Downsample heavy grids so the browser doesn't stall rendering 50k+ markers.
    cap = point_cap if point_cap is not None else _MAX_BROWSER_3D_POINTS
    x_values, y_values, z_values, grid, _strides = _downsample_grid_3d(
        np.asarray(x_values), np.asarray(y_values), np.asarray(z_values), grid,
        cap=cap,
    )

    xs_all, ys_all, zs_all, fs_all = [], [], [], []

    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            for k, z_val in enumerate(z_values):
                r = grid[i][j][k]
                f = r.get(output_key, 0.0) if isinstance(r, dict) else getattr(r, output_key, 0.0)

                x_plot = float(np.log10(x_val) if (m1 and m1.log_scale) else x_val)
                y_plot = float(np.log10(y_val) if (m2 and m2.log_scale) else y_val)
                z_plot = float(np.log10(z_val) if (m3 and m3.log_scale) else z_val)

                xs_all.append(x_plot)
                ys_all.append(y_plot)
                zs_all.append(z_plot)
                fs_all.append(float(f))

    xs_all = np.array(xs_all)
    ys_all = np.array(ys_all)
    zs_all = np.array(zs_all)
    fs_all = np.array(fs_all)

    fmin = 0.0 if "fidelity" in output_key else float(fs_all.min())
    fmax = 1.0 if "fidelity" in output_key else float(fs_all.max())

    x_title = _axis_label(metric_key1) + (" (log₁₀)" if (m1 and m1.log_scale) else "")
    y_title = _axis_label(metric_key2) + (" (log₁₀)" if (m2 and m2.log_scale) else "")
    z_title = _axis_label(metric_key3) + (" (log₁₀)" if (m3 and m3.log_scale) else "")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs_all.tolist(), y=ys_all.tolist(), z=zs_all.tolist(),
        mode="markers",
        marker=dict(
            size=3.5, color=fs_all.tolist(),
            cmin=fmin, cmax=fmax, colorscale=_COLORSCALE,
            colorbar=dict(
                title=dict(text=_OUTPUT_LABELS.get(output_key, output_key),
                           font=dict(size=11, color=_TEXT_MUTED)),
                tickfont=dict(color=_TEXT_MUTED, size=10),
                outlinewidth=0, thickness=14, len=0.75,
            ),
            opacity=0.85, line=dict(width=0),
        ),
        hovertemplate=(
            x_title + ": %{x:.3g}<br>"
            + y_title + ": %{y:.3g}<br>"
            + z_title + ": %{z:.3g}<br>"
            + "<b>fidelity: %{marker.color:.4f}</b><extra></extra>"
        ),
    ))

    if thresholds:
        fs_grid = fs_all.reshape(len(x_values), len(y_values), len(z_values))
        for i, t in enumerate(thresholds):
            above = fs_grid >= t
            crosses = np.zeros_like(above, dtype=bool)
            if above.shape[0] > 1:
                diff = above[1:, :, :] ^ above[:-1, :, :]
                crosses[1:, :, :] |= diff
                crosses[:-1, :, :] |= diff
            if above.shape[1] > 1:
                diff = above[:, 1:, :] ^ above[:, :-1, :]
                crosses[:, 1:, :] |= diff
                crosses[:, :-1, :] |= diff
            if above.shape[2] > 1:
                diff = above[:, :, 1:] ^ above[:, :, :-1]
                crosses[:, :, 1:] |= diff
                crosses[:, :, :-1] |= diff
            near = crosses.flatten()
            if not near.any():
                continue
            color = _colors[i % len(_colors)]
            fig.add_trace(go.Scatter3d(
                x=xs_all[near].tolist(), y=ys_all[near].tolist(), z=zs_all[near].tolist(),
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.9,
                            line=dict(width=1, color="#FFFFFF")),
                name=f"≈{t}",
                hovertemplate=(
                    f"threshold {t}<br>"
                    + x_title + ": %{x:.3g}<br>"
                    + y_title + ": %{y:.3g}<br>"
                    + z_title + ": %{z:.3g}<extra></extra>"
                ),
            ))

    _SCENE_AXIS = lambda title: dict(
        title=dict(text=title, font=dict(size=11, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR,
        backgroundcolor=_PLOT_BG,
        color=_TEXT_MUTED,
        tickfont=dict(size=9, color=_TEXT_MUTED),
        showspikes=False,
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(color=_TEXT_COLOR, family="Inter, system-ui, sans-serif", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            bgcolor=_PLOT_BG,
            xaxis=_SCENE_AXIS(x_title),
            yaxis=_SCENE_AXIS(y_title),
            zaxis=_SCENE_AXIS(z_title),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D isosurface (volumetric iso-level rendering)
# ---------------------------------------------------------------------------

_MIN_GRID_FOR_ISOSURFACE = 3 * 3 * 3

# Maximum 3D grid points handed to Plotly. Beyond this the browser stalls
# on JSON parse + WebGL mesh build. The downsampler strides the longest
# axis first so short axes keep full resolution.
_MAX_BROWSER_3D_POINTS = 20_000


def _downsample_strides_3d(nx: int, ny: int, nz: int, cap: int) -> tuple[int, int, int]:
    """Pick per-axis strides so ceil(nx/sx)*ceil(ny/sy)*ceil(nz/sz) <= cap.

    Strides the current longest (after-stride) axis until under budget, so
    small axes keep full resolution.
    """
    s = [1, 1, 1]
    lens = [nx, ny, nz]

    def remaining(i: int) -> int:
        return (lens[i] + s[i] - 1) // s[i]

    def total() -> int:
        return remaining(0) * remaining(1) * remaining(2)

    while total() > cap:
        longest = max(range(3), key=remaining)
        if remaining(longest) <= 1:
            break
        s[longest] += 1
    return s[0], s[1], s[2]


def _downsample_grid_3d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    grid: list,
    cap: int = _MAX_BROWSER_3D_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, tuple[int, int, int]]:
    """Return strided axes + grid when total points exceed ``cap``.

    The returned strides are (1, 1, 1) when no downsampling was needed.
    """
    nx, ny, nz = len(x_values), len(y_values), len(z_values)
    if nx * ny * nz <= cap:
        return x_values, y_values, z_values, grid, (1, 1, 1)
    sx, sy, sz = _downsample_strides_3d(nx, ny, nz, cap)
    xs_d = x_values[::sx]
    ys_d = y_values[::sy]
    zs_d = z_values[::sz]
    grid_d = [
        [
            [grid[i][j][k] for k in range(0, nz, sz)]
            for j in range(0, ny, sy)
        ]
        for i in range(0, nx, sx)
    ]
    return xs_d, ys_d, zs_d, grid_d, (sx, sy, sz)


def _flatten_3d_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    grid: list,
    metric_key1: str,
    metric_key2: str,
    metric_key3: str,
    output_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    m1 = METRIC_BY_KEY.get(metric_key1)
    m2 = METRIC_BY_KEY.get(metric_key2)
    m3 = METRIC_BY_KEY.get(metric_key3)

    xs, ys, zs, fs = [], [], [], []
    for i, x_val in enumerate(x_values):
        for j, y_val in enumerate(y_values):
            for k, z_val in enumerate(z_values):
                r = grid[i][j][k]
                f = r.get(output_key, 0.0) if isinstance(r, dict) else getattr(r, output_key, 0.0)
                xs.append(float(np.log10(x_val) if (m1 and m1.log_scale) else x_val))
                ys.append(float(np.log10(y_val) if (m2 and m2.log_scale) else y_val))
                zs.append(float(np.log10(z_val) if (m3 and m3.log_scale) else z_val))
                fs.append(float(f))

    x_title = _axis_label(metric_key1) + (" (log\u2081\u2080)" if (m1 and m1.log_scale) else "")
    y_title = _axis_label(metric_key2) + (" (log\u2081\u2080)" if (m2 and m2.log_scale) else "")
    z_title = _axis_label(metric_key3) + (" (log\u2081\u2080)" if (m3 and m3.log_scale) else "")

    return np.array(xs), np.array(ys), np.array(zs), np.array(fs), x_title, y_title, z_title


def _scene_axis(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=11, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR,
        backgroundcolor=_PLOT_BG,
        color=_TEXT_MUTED,
        tickfont=dict(size=9, color=_TEXT_MUTED),
        showspikes=False,
    )


def plot_3d_isosurface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    grid: list,
    metric_key1: str,
    metric_key2: str,
    metric_key3: str,
    output_key: str,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    point_cap: int | None = None,
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    total_points = len(x_values) * len(y_values) * len(z_values)

    if total_points < _MIN_GRID_FOR_ISOSURFACE:
        return plot_3d(x_values, y_values, z_values, grid,
                       metric_key1, metric_key2, metric_key3, output_key,
                       thresholds=thresholds, threshold_colors=threshold_colors)

    # Downsample when total > cap so Plotly's marching-cubes + WebGL stays
    # on the main thread for a reasonable time.
    cap = point_cap if point_cap is not None else _MAX_BROWSER_3D_POINTS
    x_values, y_values, z_values, grid, _strides = _downsample_grid_3d(
        np.asarray(x_values), np.asarray(y_values), np.asarray(z_values), grid,
        cap=cap,
    )

    xs, ys, zs, fs, x_title, y_title, z_title = _flatten_3d_grid(
        x_values, y_values, z_values, grid,
        metric_key1, metric_key2, metric_key3, output_key,
    )

    is_fidelity = "fidelity" in output_key
    fmin = 0.0 if is_fidelity else float(fs.min())
    fmax = 1.0 if is_fidelity else float(fs.max())

    levels = thresholds if thresholds else _DEFAULT_THRESHOLDS

    _hover = (
        x_title + ": %{x:.3g}<br>"
        + y_title + ": %{y:.3g}<br>"
        + z_title + ": %{z:.3g}<br>"
        + "<b>%{value:.4f}</b><extra></extra>"
    )

    fig = go.Figure()

    for i, level in enumerate(sorted(levels)):
        color = _colors[i % len(_colors)]
        opacity = 0.3 + 0.15 * (i / max(len(levels) - 1, 1))
        is_last = (i == len(levels) - 1)
        fig.add_trace(
            go.Isosurface(
                x=xs, y=ys, z=zs,
                value=fs,
                isomin=level, isomax=level,
                surface_count=1,
                opacity=opacity,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorscale=[[0.0, color], [1.0, color]],
                cmin=fmin, cmax=fmax,
                showscale=is_last,
                colorbar=dict(
                    title=dict(text=_OUTPUT_LABELS.get(output_key, output_key),
                               font=dict(size=11, color=_TEXT_MUTED)),
                    tickfont=dict(color=_TEXT_MUTED, size=10),
                    outlinewidth=0, thickness=14, len=0.75,
                ) if is_last else None,
                hovertemplate=_hover,
                name=f"≈{level}",
            )
        )

    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(color=_TEXT_COLOR, family="Inter, system-ui, sans-serif", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            bgcolor=_PLOT_BG,
            xaxis=_scene_axis(x_title),
            yaxis=_scene_axis(y_title),
            zaxis=_scene_axis(z_title),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
    )
    return fig


# ---------------------------------------------------------------------------
# Parallel coordinates (analysis view — works on any sweep dimensionality)
# ---------------------------------------------------------------------------

# Maximum rows passed to per-point plot traces (parallel coords, Pareto
# dominated cloud, slice scatter). Beyond this, we random-sample — both
# because Plotly/WebGL struggle past a few hundred k markers and because
# `col.tolist()` on a 100 M-point column builds a ~3 GB Python list.
_MAX_PLOT_POINTS = 200_000


def _downsample_rows(data: np.ndarray, cap: int = _MAX_PLOT_POINTS) -> np.ndarray:
    """Return ``data`` unchanged if it fits, otherwise a random-sampled view."""
    if data.shape[0] <= cap:
        return data
    rng = np.random.default_rng(0)
    idx = rng.choice(data.shape[0], size=cap, replace=False)
    idx.sort()  # preserves axis ordering where possible
    return data[idx]


def _add_sample_annotation(fig: go.Figure, shown: int, total: int) -> None:
    """Overlay a 'showing X of Y' note on the figure when downsampling was applied."""
    if shown >= total:
        return
    fig.add_annotation(
        text=f"Showing {shown:,} of {total:,} points (random sample)",
        xref="paper", yref="paper",
        x=1.0, y=1.02,
        xanchor="right", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color=_TEXT_MUTED, family="Inter, system-ui, sans-serif"),
    )


def plot_parallel_coordinates(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for parallel coordinates")

    full_rows = rows.shape[0]
    data = _downsample_rows(rows)  # Parcoords + col.tolist() OOMs past ~1M
    num_param_cols = len(metric_keys)

    # Only show swept parameters + the selected output metric (not all outputs).
    # This keeps the plot readable even with many swept axes.
    visible_output_keys = [output_key] if output_key in available_outputs else available_outputs[:1]
    visible_col_indices = list(range(num_param_cols))
    for ok in visible_output_keys:
        if ok in available_outputs:
            visible_col_indices.append(num_param_cols + available_outputs.index(ok))

    col_names = metric_keys + available_outputs
    color_col_idx = None
    if output_key in available_outputs:
        color_col_idx = num_param_cols + available_outputs.index(output_key)
    elif available_outputs:
        color_col_idx = num_param_cols

    cat_ticks = sweep_data.get("_cat_tick_labels", {})

    dimensions = []
    for i in visible_col_indices:
        name = col_names[i]
        m = METRIC_BY_KEY.get(name) or CAT_METRIC_BY_KEY.get(name)
        label = m.label if m else _OUTPUT_LABELS.get(name, name)
        col = data[:, i]
        col_range = [float(col.min()), float(col.max())]
        # Avoid degenerate range
        if col_range[0] == col_range[1]:
            col_range[1] = col_range[0] + 1.0
        dim = dict(
            label=label,
            values=col.tolist(),
            range=col_range,
        )
        if name in cat_ticks:
            labels = cat_ticks[name]
            dim["tickvals"] = list(range(len(labels)))
            dim["ticktext"] = labels
        dimensions.append(dim)

    color_vals = data[:, color_col_idx].tolist() if color_col_idx is not None else None

    fig = go.Figure(
        go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=color_vals,
                colorscale=_COLORSCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(text=_OUTPUT_LABELS.get(output_key, output_key),
                               font=dict(size=11, color=_TEXT_MUTED)),
                    tickfont=dict(color=_TEXT_MUTED, size=10),
                    outlinewidth=0, thickness=14, len=0.75,
                ),
            ),
            unselected=dict(line=dict(color="#F0F0F0", opacity=0.02)),
            labelfont=dict(size=12, color=_TEXT_COLOR, family="Inter, system-ui, sans-serif"),
            tickfont=dict(size=10, color=_TEXT_MUTED, family="Inter, system-ui, sans-serif"),
            rangefont=dict(size=10, color=_ACCENT, family="Inter, system-ui, sans-serif"),
        )
    )

    # Scale left/right margins with axis count to keep labels from clipping
    n_axes = len(dimensions)
    lr_margin = max(80, min(140, n_axes * 12))
    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=lr_margin, r=lr_margin, t=50, b=40)},
    )
    _add_sample_annotation(fig, data.shape[0], full_rows)
    return fig


# ---------------------------------------------------------------------------
# Slice plot (marginal effects — one subplot per swept parameter)
# ---------------------------------------------------------------------------

def plot_slice(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for slice plot")

    full_rows = rows.shape[0]
    data = _downsample_rows(rows)  # slice filters + scatter per axis
    num_params = len(metric_keys)
    out_col = num_params + available_outputs.index(output_key) if output_key in available_outputs else num_params

    from plotly.subplots import make_subplots
    cols = min(num_params, 3)
    subplot_rows = (num_params + cols - 1) // cols

    fig = make_subplots(
        rows=subplot_rows, cols=cols,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
        subplot_titles=[getattr(METRIC_BY_KEY.get(k) or CAT_METRIC_BY_KEY.get(k), "label", k) for k in metric_keys],
    )

    for idx, param_key in enumerate(metric_keys):
        r = idx // cols + 1
        c = idx % cols + 1

        center_indices = []
        for p in range(num_params):
            if p == idx:
                center_indices.append(None)
            else:
                unique_vals = sorted(set(data[:, p]))
                center_indices.append(unique_vals[len(unique_vals) // 2])

        mask = np.ones(len(data), dtype=bool)
        for p in range(num_params):
            if center_indices[p] is not None:
                mask &= data[:, p] == center_indices[p]

        subset = data[mask]
        if len(subset) == 0:
            continue

        order = np.argsort(subset[:, idx])
        x_vals = subset[order, idx]
        y_vals = subset[order, out_col]

        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                line=dict(color=_LINE_COLOR, width=1.5),
                marker=dict(size=4, color=_LINE_COLOR),
                showlegend=False,
                hovertemplate="%{x:.3g}<br><b>%{y:.4f}</b><extra></extra>",
            ),
            row=r, col=c,
        )

    is_fidelity = "fidelity" in output_key
    y_range = [0, 1] if is_fidelity else None

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=55, r=20, t=60, b=45)},
    )
    fig.update_yaxes(range=y_range, gridcolor=_GRID_COLOR, tickfont=dict(size=9, color=_TEXT_MUTED))
    fig.update_xaxes(gridcolor=_GRID_COLOR, tickfont=dict(size=9, color=_TEXT_MUTED))

    # Set categorical tick labels on x-axes.
    cat_ticks = sweep_data.get("_cat_tick_labels", {})
    for idx, param_key in enumerate(metric_keys):
        if param_key in cat_ticks:
            labels = cat_ticks[param_key]
            r = idx // cols + 1
            c = idx % cols + 1
            fig.update_xaxes(
                tickvals=list(range(len(labels))),
                ticktext=labels,
                row=r, col=c,
            )
    if is_fidelity:
        fig.update_yaxes(title_text=_OUTPUT_LABELS.get(output_key, output_key), row=1, col=1)

    _add_sample_annotation(fig, data.shape[0], full_rows)
    return fig


# ---------------------------------------------------------------------------
# Parameter importance (range-based sensitivity — horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_importance(
    sweep_data: dict, output_key: str, mode: str = "range",
) -> go.Figure:
    """Horizontal bar chart ranking parameters by their effect on F.

    Two ranking modes:

    * ``"range"`` (default) — global structure: for each parameter, take
      the spread (``max - min``) of F's mean projected onto that axis.
      Answers "how much does varying this parameter change F overall?".
    * ``"sensitivity"`` — local structure: ``mean(|∂F/∂x_i|)`` averaged
      over the grid, with log-axis correction. Answers "what's the local
      rate of change at this operating point?". Complements the range
      mode, especially when the response saturates or has a sharp
      transition that the range averages over.
    """
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for importance plot")

    if mode == "sensitivity":
        try:
            from gui.derivatives import (
                extract_grid_values, sensitivity_ranking,
            )

            mks = list(sweep_data.get("metric_keys", []))
            ndim = len(mks)
            if ndim == 0 or "grid" not in sweep_data:
                raise ValueError("Sensitivity mode requires a structured N-D grid.")
            axes = _resolve_axes(sweep_data, ndim)
            F = extract_grid_values(sweep_data["grid"], ndim, output_key)
            sens = sensitivity_ranking(F, [np.asarray(a) for a in axes], mks)
            importances = []
            for i, param_key in enumerate(mks):
                m = METRIC_BY_KEY.get(param_key) or CAT_METRIC_BY_KEY.get(param_key)
                label = m.label if m else param_key
                importances.append((label, float(sens[i])))
            importances.sort(key=lambda t: t[1])
            x_title = (
                f"⟨|∂{_OUTPUT_LABELS.get(output_key, output_key)} / ∂x|⟩"
                "  (mean magnitude over grid)"
            )
        except Exception as exc:
            return plot_empty(f"Sensitivity mode error: {exc}")
    else:
        data = rows  # already an ndarray from _flatten_sweep_to_table
        num_params = len(metric_keys)
        out_col = (
            num_params + available_outputs.index(output_key)
            if output_key in available_outputs else num_params
        )

        importances = []
        for idx, param_key in enumerate(metric_keys):
            unique_vals = sorted(set(data[:, idx]))
            means = []
            for v in unique_vals:
                mask = data[:, idx] == v
                means.append(data[mask, out_col].mean())
            importance = max(means) - min(means) if means else 0.0
            m = METRIC_BY_KEY.get(param_key) or CAT_METRIC_BY_KEY.get(param_key)
            label = m.label if m else param_key
            importances.append((label, importance))
        importances.sort(key=lambda x: x[1])
        x_title = f"Range of {_OUTPUT_LABELS.get(output_key, output_key)}"

    labels = [x[0] for x in importances]
    values = [x[1] for x in importances]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=_LINE_COLOR),
            hovertemplate="%{y}: <b>%{x:.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=120, r=20, t=50, b=45)},
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=_TEXT_COLOR),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Pareto front (user-chosen X/Y axes — dominated points dimmed)
# ---------------------------------------------------------------------------


def _hover_format(key: str) -> str:
    # Fidelities (0..1) want 4 decimals; integer-valued cost metrics
    # (pair counts, swaps) want no decimals; times fall between.
    if key in FIDELITY_METRICS:
        return ":.4f"
    if key == "total_circuit_time_ns":
        return ":.1f"
    return ":.0f"


def plot_pareto(
    sweep_data: dict,
    x_key: str = "total_epr_pairs",
    y_key: str = "overall_fidelity",
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for Pareto plot")

    x_label = OUTPUT_METRIC_LABEL.get(x_key, x_key)
    y_label = OUTPUT_METRIC_LABEL.get(y_key, y_key)

    if x_key == y_key:
        return plot_empty("Pick two different metrics for the Pareto axes")

    data = rows  # already an ndarray from _flatten_sweep_to_table
    num_params = len(metric_keys)

    if x_key not in available_outputs or y_key not in available_outputs:
        missing = [k for k in (x_key, y_key) if k not in available_outputs]
        return plot_empty(f"Missing metric(s) in sweep data: {', '.join(missing)}")

    x_col = num_params + available_outputs.index(x_key)
    y_col = num_params + available_outputs.index(y_key)

    x_vals = data[:, x_col]
    y_vals = data[:, y_col]

    x_orient = PARETO_METRIC_ORIENTATION.get(x_key, "min")
    y_orient = PARETO_METRIC_ORIENTATION.get(y_key, "max")

    # Normalise to a "lower is better" frame so a single algorithm handles
    # every axis-orientation combo. Sign-flipping preserves ordering and
    # tie structure while letting the existing O(N log N) scan work.
    x_norm = x_vals if x_orient == "min" else -x_vals
    y_norm = y_vals if y_orient == "min" else -y_vals

    # Vectorised Pareto front: sort by x_norm asc, then for each x-tie group
    # a point is dominated iff
    #   y_norm >= min_y_from_strictly_lower_x   (strict-x dominator) OR
    #   y_norm >  min_y_within_same_x_group     (same-x strictly-lower-y dominator)
    # O(N log N) numpy; drops multi-second 50k-point runs to ~ms.
    is_pareto = np.ones(len(data), dtype=bool)
    if len(data) > 0:
        order = np.argsort(x_norm, kind="stable")
        s_x = x_norm[order]
        s_y = y_norm[order]

        change = np.concatenate(([True], s_x[1:] != s_x[:-1]))
        group_id = np.cumsum(change) - 1
        n_groups = int(group_id[-1]) + 1

        group_min = np.full(n_groups, np.inf)
        np.minimum.at(group_min, group_id, s_y)

        if n_groups >= 2:
            prev_min_strict = np.concatenate(
                ([np.inf], np.minimum.accumulate(group_min[:-1]))
            )
        else:
            prev_min_strict = np.array([np.inf])

        s_prev = prev_min_strict[group_id]
        s_group_min = group_min[group_id]
        dominated_sorted = (s_y >= s_prev) | (s_y > s_group_min)
        is_pareto[order[dominated_sorted]] = False

    fig = go.Figure()

    dominated_mask = ~is_pareto
    # Scattergl for the (typically very large) dominated cloud — WebGL
    # scales to ~100 k markers; plain Scatter (SVG) freezes the browser
    # past a few thousand. The Pareto front itself is computed on the full
    # grid above; we only sample the cloud display.
    dom_x = x_vals[dominated_mask]
    dom_y = y_vals[dominated_mask]
    dom_full = dom_x.size
    if dom_x.size > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(0)
        sample = rng.choice(dom_x.size, size=_MAX_PLOT_POINTS, replace=False)
        dom_x = dom_x[sample]
        dom_y = dom_y[sample]

    x_fmt = _hover_format(x_key)
    y_fmt = _hover_format(y_key)
    hover = (
        f"{x_label}: %{{x{x_fmt}}}<br>"
        f"{y_label}: <b>%{{y{y_fmt}}}</b><extra></extra>"
    )

    fig.add_trace(
        go.Scattergl(
            x=dom_x, y=dom_y,
            mode="markers",
            marker=dict(size=5, color="#CCCCCC", opacity=0.5),
            name="Dominated",
            hovertemplate=hover,
        )
    )

    pareto_idx = np.where(is_pareto)[0]
    # Sort the front by the normalised x so the connecting line is
    # monotonic regardless of axis orientation.
    pareto_order = np.argsort(x_norm[pareto_idx])
    pareto_sorted = pareto_idx[pareto_order]

    fig.add_trace(
        go.Scatter(
            x=x_vals[pareto_sorted], y=y_vals[pareto_sorted],
            mode="lines+markers",
            line=dict(color="#4575b4", width=2),
            marker=dict(size=7, color="#4575b4", line=dict(width=1, color="#FFFFFF")),
            name="Pareto front",
            hovertemplate=hover,
        )
    )

    y_axis = dict(
        title=dict(text=y_label, font=dict(size=12, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR,
        tickfont=dict(size=10, color=_TEXT_MUTED),
    )
    if y_key in FIDELITY_METRICS:
        y_axis["range"] = [0, 1]

    x_axis = dict(
        title=dict(text=x_label, font=dict(size=12, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR,
        tickfont=dict(size=10, color=_TEXT_MUTED),
    )
    if x_key in FIDELITY_METRICS:
        x_axis["range"] = [0, 1]

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=55, r=20, t=50, b=50)},
        xaxis=x_axis,
        yaxis=y_axis,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
    )

    # Threshold lines live in fidelity-space, so only draw them when a
    # fidelity metric is actually on the Y axis.
    if thresholds and y_key in FIDELITY_METRICS:
        _colors = threshold_colors or _THRESHOLD_COLORS
        for i, t in enumerate(thresholds):
            fig.add_shape(
                type="line", xref="paper", x0=0, x1=1, y0=t, y1=t,
                line=dict(color=_colors[i % len(_colors)], width=1.5, dash="dash"),
            )

    # Front is exact; only the dominated cloud is sampled for display.
    _add_sample_annotation(fig, dom_x.size, dom_full)
    return fig


# ---------------------------------------------------------------------------
# Correlation matrix (Spearman rank correlations — annotated heatmap)
# ---------------------------------------------------------------------------

def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def _rankdata(a: np.ndarray) -> np.ndarray:
        unique_vals, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        if len(unique_vals) == len(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(a) + 1, dtype=float)
            return ranks
            
        cum_counts = np.cumsum(counts)
        prev_counts = np.insert(cum_counts[:-1], 0, 0)
        avg_ranks = (prev_counts + 1 + cum_counts) / 2.0
        return avg_ranks[inverse]

    if len(x) < 3:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    if den == 0:
        return 0.0
    return float(num / den)


def plot_correlation(
    sweep_data: dict, output_key: str, mode: str = "spearman",
) -> go.Figure:
    """Two-mode parameter relationship matrix.

    * ``"spearman"`` (default) — rank correlation between each input axis
      and each output metric (the historical Corr. view).
    * ``"interaction"`` — symmetric N×N matrix of mean ``|∂²F/∂x_i∂x_j|``
      magnitudes for the active output metric. Diagonal entries are pure
      curvature; hot off-diagonals mark axis pairs that can't be tuned
      independently.
    """
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for correlation matrix")

    if mode == "interaction":
        try:
            from gui.derivatives import extract_grid_values, interaction_matrix

            mks = list(sweep_data.get("metric_keys", []))
            n = len(mks)
            if n < 2 or "grid" not in sweep_data:
                return plot_empty(
                    "Interaction matrix needs ≥ 2 sweep axes and a structured grid."
                )
            axes = _resolve_axes(sweep_data, n)
            F = extract_grid_values(sweep_data["grid"], n, output_key)
            M = interaction_matrix(F, [np.asarray(a) for a in axes], mks)
        except Exception as exc:
            return plot_empty(f"Interaction matrix error: {exc}")

        labels = [
            (METRIC_BY_KEY.get(k).label if METRIC_BY_KEY.get(k) else k) for k in mks
        ]
        m_max = float(np.nanmax(np.abs(M))) if M.size else 1.0
        if not np.isfinite(m_max) or m_max <= 0:
            m_max = 1.0

        annotations = []
        for i in range(n):
            for j in range(n):
                annotations.append(dict(
                    x=j, y=i,
                    text=f"{M[i, j]:.2f}",
                    showarrow=False,
                    font=dict(
                        size=9,
                        color=_TEXT_COLOR if abs(M[i, j]) < 0.7 * m_max else "#FFFFFF",
                    ),
                ))

        out_label = _OUTPUT_LABELS.get(output_key, output_key)
        fig = go.Figure(
            go.Heatmap(
                z=M,
                x=labels, y=labels,
                zmin=0.0, zmax=m_max,
                colorscale=_COLORSCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"⟨|∂²{out_label}/∂x_i∂x_j|⟩",
                        font=dict(size=11, color=_TEXT_MUTED),
                    ),
                    tickfont=dict(color=_TEXT_MUTED, size=10),
                    outlinewidth=0, thickness=14, len=0.75,
                ),
                hovertemplate="%{x} × %{y}: <b>%{z:.3f}</b><extra></extra>",
            )
        )
        fig.update_layout(
            **{**_LAYOUT_BASE, "margin": dict(l=100, r=20, t=50, b=100)},
            xaxis=dict(tickfont=dict(size=9, color=_TEXT_MUTED), tickangle=45),
            yaxis=dict(tickfont=dict(size=9, color=_TEXT_MUTED), autorange="reversed"),
            annotations=annotations,
        )
        return fig

    data = rows  # already an ndarray from _flatten_sweep_to_table
    n_inputs = len(metric_keys)
    n_outputs = len(available_outputs)

    x_labels = []
    for name in metric_keys:
        m = METRIC_BY_KEY.get(name) or CAT_METRIC_BY_KEY.get(name)
        x_labels.append(m.label if m else _OUTPUT_LABELS.get(name, name))

    y_labels = []
    for name in available_outputs:
        m = METRIC_BY_KEY.get(name) or CAT_METRIC_BY_KEY.get(name)
        y_labels.append(m.label if m else _OUTPUT_LABELS.get(name, name))

    corr = np.zeros((n_outputs, n_inputs))
    for i in range(n_outputs):
        out_idx = n_inputs + i
        for j in range(n_inputs):
            in_idx = j
            c = _spearman_corr(data[:, out_idx], data[:, in_idx])
            corr[i, j] = c

    annotations = []
    for i in range(n_outputs):
        for j in range(n_inputs):
            annotations.append(dict(
                x=j, y=i,
                text=f"{corr[i, j]:.2f}",
                showarrow=False,
                font=dict(size=9, color=_TEXT_COLOR if abs(corr[i, j]) < 0.7 else "#FFFFFF"),
            ))

    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=x_labels, y=y_labels,
            zmin=-1.0, zmax=1.0,
            colorscale=_COLORSCALE_DIVERGING,
            showscale=True,
            colorbar=dict(
                title=dict(text="Spearman ρ", font=dict(size=11, color=_TEXT_MUTED)),
                tickfont=dict(color=_TEXT_MUTED, size=10),
                outlinewidth=0, thickness=14, len=0.75,
            ),
            hovertemplate="%{x} vs %{y}: <b>%{z:.3f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=100, r=20, t=50, b=100)},
        xaxis=dict(tickfont=dict(size=9, color=_TEXT_MUTED), tickangle=45),
        yaxis=dict(tickfont=dict(size=9, color=_TEXT_MUTED), autorange="reversed"),
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Elasticity Comparison view (analysis tab) — one curve per parameter
# overlaid on the chosen "trajectory" axis. Crossovers reveal regime
# transitions (e.g. T1-limited vs gate-limited). Most useful for hardware
# co-design papers because it answers "which parameter has the highest
# leverage at each operating point?" in a single dimensionless plot.
# ---------------------------------------------------------------------------

# Up to 8 distinguishable hues for the per-parameter curves — sampled from
# the unified red→blue ramp at evenly spaced stops so they match the rest
# of the GUI palette while staying visually distinct.
_ELASTICITY_LINE_COLORS = [
    "#d73027", "#f46d43", "#fdae61", "#fee08b",
    "#a8d4a0", "#91bfdb", "#4575b4", "#313695",
]


def plot_elasticity_comparison(
    sweep_data: dict,
    output_key: str,
    trajectory_key: str | None = None,
) -> go.Figure:
    """Plot one elasticity curve per parameter against a chosen trajectory axis.

    Each curve is the dimensionless local elasticity of *output_key* with
    respect to that parameter, averaged over every sweep axis except the
    trajectory. A horizontal "0" reference line is drawn so positive
    (parameter helps) and negative (parameter hurts) regimes are obvious.
    """
    from gui.derivatives import elasticity_comparison, extract_grid_values

    metric_keys = list(sweep_data.get("metric_keys", []))
    if len(metric_keys) < 2:
        return plot_empty(
            "Elasticity needs ≥ 2 sweep axes — one trajectory plus at least "
            "one parameter to compare against it."
        )

    # Pick the trajectory: caller-supplied if valid, otherwise the first axis.
    if trajectory_key not in metric_keys:
        trajectory_key = metric_keys[0]
    trajectory_idx = metric_keys.index(trajectory_key)

    ndim = len(metric_keys)
    axes = _resolve_axes(sweep_data, ndim)

    try:
        F = extract_grid_values(sweep_data["grid"], ndim, output_key)
        result = elasticity_comparison(
            F, [np.asarray(a) for a in axes], metric_keys, trajectory_idx,
        )
    except Exception as exc:
        return plot_empty(f"Elasticity error: {exc}")

    # Degenerate case: F is NaN / saturated to zero across the entire grid
    # (e.g. a deep circuit at low T1 — the engine reports F = 0 / NaN
    # everywhere, so every elasticity cell is NaN and every curve is empty).
    # Show the diagnostic explicitly instead of a silent empty plot.
    finite_F = int(np.sum(np.isfinite(F) & (np.abs(F) > 1e-12)))
    any_finite_curve = any(
        np.isfinite(c).any() for c in result["curves"].values()
    )
    if not any_finite_curve:
        if finite_F == 0:
            return plot_empty(
                "F is zero or NaN across the entire sweep grid — elasticity "
                "is undefined here. Widen the sweep ranges or relax the noise "
                "budget so at least some cells produce a non-trivial fidelity, "
                "then this view will show one curve per parameter."
            )
        return plot_empty(
            "All elasticity curves are NaN — the partial derivatives are "
            "indeterminate at this operating point. Try a different "
            "trajectory axis, or widen the sweep ranges."
        )

    traj_vals = result["trajectory_values"]
    traj_metric = METRIC_BY_KEY.get(trajectory_key)
    traj_log = bool(traj_metric and traj_metric.log_scale)

    fig = go.Figure()

    for idx, (param_key, curve) in enumerate(result["curves"].items()):
        param_metric = METRIC_BY_KEY.get(param_key)
        label = param_metric.label if param_metric else param_key
        color = _ELASTICITY_LINE_COLORS[idx % len(_ELASTICITY_LINE_COLORS)]
        fig.add_trace(go.Scatter(
            x=traj_vals, y=curve,
            mode="lines+markers",
            line=dict(color=color, width=2, shape="spline", smoothing=0.4),
            marker=dict(size=4, color=color, line=dict(width=0)),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                + _axis_label(trajectory_key) + ": %{x:.3g}<br>"
                + "elasticity: <b>%{y:.3f}</b><extra></extra>"
            ),
        ))

    # Zero reference line — separates "parameter helps" (above) from
    # "parameter hurts" (below). Uses xref="paper" so it spans the full
    # axis range no matter what the data covers.
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, y0=0, y1=0,
        line=dict(color=_GRID_COLOR, width=1.5, dash="dot"),
    )

    output_label = _OUTPUT_LABELS.get(output_key, output_key)
    x_title = _axis_label(trajectory_key) + (" (log₁₀)" if traj_log else "")
    y_title = (
        f"Elasticity of {output_label}  ·  Δlog F / Δlog x"
    )

    xaxis_cfg = dict(
        title=dict(text=x_title, font=dict(size=12, color=_TEXT_MUTED)),
        gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
        tickfont=dict(size=10, color=_TEXT_MUTED),
    )
    if traj_log:
        xaxis_cfg["type"] = "log"

    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=xaxis_cfg,
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR, zerolinecolor=_TEXT_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
        legend=dict(
            font=dict(size=10, color=_TEXT_COLOR),
            bgcolor="rgba(255,255,255,0.8)", bordercolor=_GRID_COLOR, borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# View-mode transforms (Absolute / |∇F| / Elasticity)
# ---------------------------------------------------------------------------

# Synthetic output keys used by the derivative view modes — chosen to match
# the ``__fom__`` convention so the existing plot dispatch (which switches on
# ``output_key``) needs no knowledge of derivative semantics. Labels are set
# per-figure so the colorbar / Y-axis carry the right title.
_DERIV_OUTPUT_KEY = "__deriv__"


def _rebuild_grid_with_field(
    sweep_data: dict, values: np.ndarray, key: str = _DERIV_OUTPUT_KEY,
) -> dict:
    """Synthesize a new sweep_data whose grid stores ``values`` under ``key``.

    Mirrors :func:`_rebuild_grid_with_fom` but for derivative outputs.
    """
    ndim = len(sweep_data["metric_keys"])
    axes = _resolve_axes(sweep_data, ndim)
    shape = tuple(len(ax) for ax in axes)
    if values.shape != shape:
        raise ValueError(
            f"derivative shape {values.shape} != sweep grid shape {shape}",
        )

    def _build(idx: tuple[int, ...], d: int):
        if d == ndim:
            return {key: float(values[idx])}
        return [_build(idx + (i,), d + 1) for i in range(shape[d])]

    new_sweep = dict(sweep_data)
    new_sweep["grid"] = _build((), 0)
    new_sweep.pop("facets", None)
    new_sweep.pop("facet_keys", None)
    return new_sweep


def _apply_view_mode(
    sweep_data: dict, output_key: str, view_mode: str, num_metrics: int,
) -> tuple[dict, str]:
    """Apply the derivative view mode to ``sweep_data``.

    Returns ``(new_sweep_data, new_output_key)``. When ``view_mode`` is
    ``"absolute"`` (or invalid for the active dimensionality) the inputs are
    returned unchanged.
    """
    from gui.derivatives import (
        elasticity_1d, extract_grid_values, gradient_magnitude,
        mixed_partial_2d, second_derivative_1d,
    )

    if view_mode == "absolute" or not view_mode:
        return sweep_data, output_key

    # Only N-D dimensional views (Line / Heatmap / Isosurface / Frozen) are
    # transformed; analysis tabs (parallel / slices / pareto / corr) read the
    # sweep table directly and are unaffected.
    metric_keys = sweep_data.get("metric_keys", [])
    ndim = min(num_metrics, len(metric_keys))
    if ndim < 1:
        return sweep_data, output_key

    axes = _resolve_axes(sweep_data, ndim)
    F = extract_grid_values(sweep_data["grid"], ndim, output_key)

    if view_mode == "gradient_magnitude":
        if ndim < 1:
            return sweep_data, output_key
        magnitudes = gradient_magnitude(F, [np.asarray(a) for a in axes], metric_keys[:ndim])
        new_data = _rebuild_grid_with_field(sweep_data, magnitudes)
        _OUTPUT_LABELS[_DERIV_OUTPUT_KEY] = (
            f"|∇{_OUTPUT_LABELS.get(output_key, output_key)}|"
        )
        return new_data, _DERIV_OUTPUT_KEY

    if view_mode == "elasticity":
        # Only 1-D Line views support elasticity-as-a-mode; for higher
        # dimensions the user picks a trajectory axis via the Elasticity tab.
        if ndim != 1:
            return sweep_data, output_key
        elast = elasticity_1d(F, np.asarray(axes[0]), metric_keys[0])
        new_data = _rebuild_grid_with_field(sweep_data, elast)
        _OUTPUT_LABELS[_DERIV_OUTPUT_KEY] = (
            f"Elasticity of {_OUTPUT_LABELS.get(output_key, output_key)}"
        )
        return new_data, _DERIV_OUTPUT_KEY

    if view_mode == "second_derivative":
        # Curvature of the 1-D response — sign change marks the inflection
        # point (the diminishing-returns sweet spot). The Line plotter
        # auto-annotates that x value as a vertical guide.
        if ndim != 1:
            return sweep_data, output_key
        from gui.derivatives import find_inflection_x

        d2 = second_derivative_1d(F, np.asarray(axes[0]), metric_keys[0])
        new_data = _rebuild_grid_with_field(sweep_data, d2)
        new_data["_inflection_x"] = find_inflection_x(d2, np.asarray(axes[0]))
        _OUTPUT_LABELS[_DERIV_OUTPUT_KEY] = (
            f"d²{_OUTPUT_LABELS.get(output_key, output_key)} / dx²"
        )
        return new_data, _DERIV_OUTPUT_KEY

    if view_mode == "mixed_partial":
        # ∂²F/∂x∂y interaction heatmap — positive = synergy, negative =
        # substitution, zero = independent. Smoothed with Savitzky-Golay
        # so coarse-grid finite-difference noise stays readable.
        if ndim != 2:
            return sweep_data, output_key
        mp = mixed_partial_2d(
            F,
            np.asarray(axes[0]), np.asarray(axes[1]),
            metric_keys[0], metric_keys[1],
            smooth=True,
        )
        new_data = _rebuild_grid_with_field(sweep_data, mp)
        _OUTPUT_LABELS[_DERIV_OUTPUT_KEY] = (
            f"∂²{_OUTPUT_LABELS.get(output_key, output_key)} / ∂x∂y"
        )
        return new_data, _DERIV_OUTPUT_KEY

    return sweep_data, output_key


# ---------------------------------------------------------------------------
# Figure of Merit view
# ---------------------------------------------------------------------------

_FOM_OUTPUT_KEY = "__fom__"


def _rebuild_grid_with_fom(sweep_data: dict, fom_values: np.ndarray) -> dict:
    """Return a shallow-copied sweep_data whose grid is a nested list of
    ``{"__fom__": value}`` dicts — ready to feed into the existing 1/2/3-D
    plotters by passing ``output_key="__fom__"``.

    ``fom_values`` is the flat C-order array produced by
    :func:`gui.fom.compute_for_sweep`; it must match the grid shape implied
    by the sweep's axes.
    """
    ndim = len(sweep_data["metric_keys"])
    axes = _resolve_axes(sweep_data, ndim)
    shape = tuple(len(ax) for ax in axes)
    total = int(np.prod(shape)) if shape else 0
    if total != fom_values.size:
        raise ValueError(
            f"FoM value count ({fom_values.size}) does not match grid shape {shape}"
        )
    arr = fom_values.reshape(shape)

    def build(idx: tuple[int, ...], d: int):
        if d == ndim:
            v = float(arr[idx])
            return {_FOM_OUTPUT_KEY: v}
        return [build(idx + (i,), d + 1) for i in range(shape[d])]

    new_sweep = dict(sweep_data)
    new_sweep["grid"] = build((), 0)
    new_sweep.pop("facets", None)
    new_sweep.pop("facet_keys", None)
    return new_sweep


# ---------------------------------------------------------------------------
# Merit view: shared helpers used by both Heatmap and Pareto modes.
# ---------------------------------------------------------------------------


# Neutral grey shown in the heatmap where the FoM evaluated to NaN/Inf.
_MERIT_NODATA_COLOR = "rgba(180,180,180,0.55)"


def _build_merit_breakdown(sweep_data: dict, fom_config):
    """Resolve sweep + FoM into a ``FomBreakdown`` or an empty-figure on error.

    Returns ``(breakdown, fom_config_normalized, sweep_data_resolved,
    error_figure_or_None)``. When the last element is non-None, callers
    should return it directly.
    """
    from gui.fom import FomConfig, compute_breakdown

    if not isinstance(fom_config, FomConfig):
        fom_config = FomConfig.from_dict(fom_config or {})

    if sweep_data is None:
        return None, fom_config, None, plot_empty("No sweep loaded — run a sweep first")

    if "facets" in sweep_data and sweep_data["facets"]:
        sweep_data = _flatten_facets_for_analysis(sweep_data)

    bd = compute_breakdown(sweep_data, fom_config)
    if bd.error is not None:
        return None, fom_config, sweep_data, plot_empty(f"FoM error: {bd.error}")
    return bd, fom_config, sweep_data, None


def _snap_to_grid(value: float, grid_values: np.ndarray) -> float:
    """Snap ``value`` to the nearest discrete point in ``grid_values``."""
    if grid_values.size == 0:
        return float(value)
    idx = int(np.argmin(np.abs(grid_values - value)))
    return float(grid_values[idx])


def _frozen_mask(
    primitives: dict[str, np.ndarray],
    frozen_values: dict[str, float] | None,
    n_rows: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return a boolean row-mask selecting points whose frozen-axis values
    match (after snapping) the requested ``frozen_values``. Also returns the
    snapped values actually used, so the UI can echo them back.
    """
    mask = np.ones(n_rows, dtype=bool)
    snapped: dict[str, float] = {}
    for axis_key, value in (frozen_values or {}).items():
        col = primitives.get(axis_key)
        if col is None or value is None:
            continue
        unique_vals = np.unique(col[np.isfinite(col)])
        if unique_vals.size == 0:
            continue
        snap_v = _snap_to_grid(float(value), unique_vals)
        snapped[axis_key] = snap_v
        # rtol/atol tuned for typical sweep axis spacings — tight enough to
        # avoid neighbouring grid points, loose enough to absorb float noise.
        mask &= np.isclose(col, snap_v, rtol=1e-9, atol=1e-12)
    return mask, snapped


def _pick_axis_scale(values: np.ndarray) -> str:
    """Return ``'log'`` or ``'linear'`` based on the value distribution."""
    if values.size == 0:
        return "linear"
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not (finite > 0).all():
        return "linear"
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin > 0 and vmax / vmin > 100:
        return "log"
    return "linear"


# ---------------------------------------------------------------------------
# Merit view: shared 2-D grid construction (used by Heatmap and 3D modes)
# ---------------------------------------------------------------------------


def _build_merit_2d_grid(
    bd, sweep_axes, fom_arr, num_arr, den_arr,
    x_axis, y_axis, frozen_values,
):
    """Slice the FoM down to a regular ``(len(y_unique), len(x_unique))`` 2-D
    grid by pinning the non-XY axes to the user-chosen frozen values.

    Returns the tuple ``(x_axis, y_axis, x_unique, y_unique, z, num_grid,
    den_grid, snapped, frozen_keys)`` on success or a ``go.Figure`` carrying
    an empty-state message on failure (caller forwards it to the user).
    """
    n_rows = fom_arr.shape[0]

    if x_axis not in sweep_axes:
        x_axis = sweep_axes[0]
    if y_axis not in sweep_axes or y_axis == x_axis:
        y_axis = next((k for k in sweep_axes if k != x_axis), x_axis)

    other_axes = [k for k in sweep_axes if k not in (x_axis, y_axis)]
    frozen = dict(frozen_values or {})
    # Default any missing frozen value to the median grid point so the slice
    # is reproducible before the user touches a slider.
    for k in other_axes:
        if k not in frozen:
            uvals = np.unique(bd.primitives[k][np.isfinite(bd.primitives[k])])
            if uvals.size:
                frozen[k] = float(uvals[uvals.size // 2])

    mask, snapped = _frozen_mask(
        bd.primitives,
        {k: frozen[k] for k in other_axes if k in frozen},
        n_rows,
    )
    if not mask.any():
        return plot_empty("No sweep points match the selected frozen-axis values.")

    x_col = bd.primitives[x_axis][mask]
    y_col = bd.primitives[y_axis][mask]
    fom_col = fom_arr[mask]
    num_col = num_arr[mask]
    den_col = den_arr[mask]

    x_unique = np.unique(x_col[np.isfinite(x_col)])
    y_unique = np.unique(y_col[np.isfinite(y_col)])
    if x_unique.size == 0 or y_unique.size == 0:
        return plot_empty("FoM landscape is empty for this slice.")

    z = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
    num_grid = np.full_like(z, np.nan)
    den_grid = np.full_like(z, np.nan)
    x_idx_map = {float(v): i for i, v in enumerate(x_unique)}
    y_idx_map = {float(v): i for i, v in enumerate(y_unique)}
    for xi, yi, fi, ni, di in zip(x_col, y_col, fom_col, num_col, den_col):
        ix = x_idx_map.get(float(xi))
        iy = y_idx_map.get(float(yi))
        if ix is None or iy is None:
            continue
        z[iy, ix] = fi
        num_grid[iy, ix] = ni
        den_grid[iy, ix] = di

    if not np.isfinite(z).any():
        return plot_empty(
            "FoM produced no finite values — check for divide-by-zero or constant expressions."
        )

    return (x_axis, y_axis, x_unique, y_unique, z,
            num_grid, den_grid, snapped, list(other_axes))


def _merit_grid_customdata(
    n_y: int, n_x: int, num_grid: np.ndarray, den_grid: np.ndarray,
    frozen_keys: list[str], snapped: dict[str, float],
) -> np.ndarray:
    """Stack per-cell ``(num, den, *frozen_values)`` into a Plotly customdata
    cube of shape ``(n_y, n_x, 2 + len(frozen_keys))``.
    """
    customdata = np.empty((n_y, n_x, 2 + len(frozen_keys)), dtype=float)
    customdata[:, :, 0] = num_grid
    customdata[:, :, 1] = den_grid
    for i, fk in enumerate(frozen_keys):
        customdata[:, :, 2 + i] = snapped.get(fk, float("nan"))
    return customdata


def _merit_grid_hovertemplate(
    x_axis: str, y_axis: str, fom_name: str,
    frozen_keys: list[str], z_var: str = "z",
) -> str:
    """Hover template shared by Heatmap and Surface traces. ``z_var`` is the
    Plotly variable holding the FoM value (``z`` for heatmap, surface).
    """
    lines = [
        f"{_axis_label(x_axis)}: %{{x:.4g}}",
        f"{_axis_label(y_axis)}: %{{y:.4g}}",
        f"<b>{fom_name}: %{{{z_var}:.4g}}</b>",
        "  numerator = %{customdata[0]:.4g}",
        "  denominator = %{customdata[1]:.4g}",
    ]
    if frozen_keys:
        lines.append("<i>frozen</i>")
        for i, fk in enumerate(frozen_keys):
            lines.append(f"  {_axis_label(fk)} = %{{customdata[{2 + i}]:.4g}}")
    return "<br>".join(lines) + "<extra></extra>"


# ---------------------------------------------------------------------------
# Merit view: Heatmap mode
# ---------------------------------------------------------------------------


def plot_merit_heatmap(
    sweep_data: dict,
    fom_config,
    x_axis: str | None = None,
    y_axis: str | None = None,
    frozen_values: dict[str, float] | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    """Render the FoM as a 2-D heatmap over two selected sweep axes, with
    the remaining sweep axes pinned to specific values via ``frozen_values``.

    When only one sweep axis is active the figure degrades to a 1-D line
    plot; with zero sweep axes a clear empty-state message is shown.
    """
    # Guard before evaluation: a sweep with zero axes can't plot anything,
    # and the FoM evaluator would raise "sweep is empty" with a less useful
    # message. Catching it here keeps the empty-state UX consistent.
    if (sweep_data is not None
            and not (sweep_data.get("metric_keys") or [])):
        return plot_empty(
            "Add at least one sweep axis to view the FoM heatmap."
        )

    bd, fom_config, _resolved, err = _build_merit_breakdown(sweep_data, fom_config)
    if err is not None:
        return err

    sweep_axes = list(bd.sweep_axes)
    if not sweep_axes:
        return plot_empty(
            "Add at least one sweep axis to view the FoM heatmap."
        )

    fom_arr = np.asarray(bd.fom, dtype=float)
    num_arr = np.asarray(bd.numerator, dtype=float)
    den_arr = np.asarray(bd.denominator, dtype=float)
    n_rows = fom_arr.shape[0]

    fom_name = fom_config.name or "Figure of Merit"

    # --- 1-D fallback ------------------------------------------------------
    if len(sweep_axes) == 1:
        x_key = sweep_axes[0]
        x_col = bd.primitives[x_key]
        finite_mask = np.isfinite(fom_arr)
        order = np.argsort(x_col[finite_mask])
        x_sorted = x_col[finite_mask][order]
        fom_sorted = fom_arr[finite_mask][order]
        num_sorted = num_arr[finite_mask][order]
        den_sorted = den_arr[finite_mask][order]

        custom = np.column_stack([num_sorted, den_sorted])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_sorted, y=fom_sorted,
            mode="lines+markers",
            line=dict(color=_LINE_COLOR, width=2),
            marker=dict(size=5, color=_LINE_COLOR, opacity=0.8),
            customdata=custom,
            hovertemplate=(
                f"{_axis_label(x_key)}: %{{x:.4g}}<br>"
                f"<b>{fom_name}: %{{y:.4g}}</b><br>"
                "  numerator = %{customdata[0]:.4g}<br>"
                "  denominator = %{customdata[1]:.4g}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.update_layout(
            **_LAYOUT_BASE,
            xaxis=dict(
                title=dict(text=_axis_label(x_key), font=dict(size=12, color=_TEXT_MUTED)),
                gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED),
            ),
            yaxis=dict(
                title=dict(text=fom_name, font=dict(size=12, color=_TEXT_MUTED)),
                gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED),
            ),
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR,
                            font_color=_TEXT_COLOR),
        )
        return fig

    # --- ≥2-D heatmap ------------------------------------------------------
    grid = _build_merit_2d_grid(
        bd, sweep_axes, fom_arr, num_arr, den_arr,
        x_axis, y_axis, frozen_values,
    )
    if isinstance(grid, go.Figure):
        return grid
    x_axis, y_axis, x_unique, y_unique, z, num_grid, den_grid, snapped, frozen_keys = grid
    finite_z = z[np.isfinite(z)]

    customdata = _merit_grid_customdata(
        y_unique.size, x_unique.size, num_grid, den_grid, frozen_keys, snapped,
    )
    hovertemplate = _merit_grid_hovertemplate(
        x_axis, y_axis, fom_name, frozen_keys, z_var="z",
    )

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x_unique, y=y_unique, z=z,
        colorscale=_COLORSCALE,
        zmin=float(finite_z.min()), zmax=float(finite_z.max()),
        colorbar=dict(
            title=dict(text=fom_name, side="right",
                       font=dict(size=11, color=_TEXT_MUTED)),
            tickfont=dict(color=_TEXT_MUTED, size=10),
            outlinewidth=0, thickness=14, len=0.85,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate,
        # Plotly renders NaN cells transparent; the plot bg shows through as
        # the "no data" colour.
        hoverongaps=False,
        zsmooth=False,
    ))

    # Iso-level contour overlay using the user-picked threshold colours.
    if thresholds:
        colors = threshold_colors or _THRESHOLD_COLORS
        zmin = float(finite_z.min())
        zmax = float(finite_z.max())
        for i, t in enumerate(thresholds):
            if t is None:
                continue
            t_f = float(t)
            if t_f < zmin or t_f > zmax:
                # Outside the data range — Plotly would silently draw nothing.
                continue
            color = colors[i % len(colors)]
            fig.add_trace(go.Contour(
                x=x_unique, y=y_unique, z=z,
                contours=dict(
                    start=t_f, end=t_f, size=0,
                    showlines=True, coloring="lines",
                    showlabels=True,
                    labelfont=dict(size=11, color=color),
                ),
                line=dict(color=color, width=2),
                showscale=False,
                hoverinfo="skip",
                name=f"FoM = {t_f:g}",
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(
            title=dict(text=_axis_label(x_axis), font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            title=dict(text=_axis_label(y_axis), font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR, tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR,
                        font_color=_TEXT_COLOR, align="left"),
    )
    return fig


# ---------------------------------------------------------------------------
# Merit view: 3D surface mode — same XY/frozen-slider mechanics as Heatmap
# but renders the FoM as both height and colour, with optional iso-FoM
# threshold lines projected onto the surface.
# ---------------------------------------------------------------------------


def plot_merit_surface(
    sweep_data: dict,
    fom_config,
    x_axis: str | None = None,
    y_axis: str | None = None,
    frozen_values: dict[str, float] | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    """Render the FoM landscape as a Plotly Surface where Z = colour = FoM.

    Shares the XY-axis selection and frozen-slider mechanics with the
    Heatmap mode; only the trace type differs.
    """
    if (sweep_data is not None
            and not (sweep_data.get("metric_keys") or [])):
        return plot_empty(
            "Add at least one sweep axis to view the FoM surface."
        )

    bd, fom_config, _resolved, err = _build_merit_breakdown(sweep_data, fom_config)
    if err is not None:
        return err

    sweep_axes = list(bd.sweep_axes)
    if not sweep_axes:
        return plot_empty("Add at least one sweep axis to view the FoM surface.")

    fom_arr = np.asarray(bd.fom, dtype=float)
    num_arr = np.asarray(bd.numerator, dtype=float)
    den_arr = np.asarray(bd.denominator, dtype=float)
    fom_name = fom_config.name or "Figure of Merit"

    # 1-D fallback: a true surface needs two axes — degrade to the same
    # line-plot the Heatmap mode shows so the view stays usable.
    if len(sweep_axes) == 1:
        return plot_merit_heatmap(
            sweep_data, fom_config,
            x_axis=x_axis, y_axis=y_axis,
            frozen_values=frozen_values,
            thresholds=thresholds, threshold_colors=threshold_colors,
        )

    grid = _build_merit_2d_grid(
        bd, sweep_axes, fom_arr, num_arr, den_arr,
        x_axis, y_axis, frozen_values,
    )
    if isinstance(grid, go.Figure):
        return grid
    x_axis, y_axis, x_unique, y_unique, z, num_grid, den_grid, snapped, frozen_keys = grid
    finite_z = z[np.isfinite(z)]
    zmin = float(finite_z.min())
    zmax = float(finite_z.max())

    customdata = _merit_grid_customdata(
        y_unique.size, x_unique.size, num_grid, den_grid, frozen_keys, snapped,
    )
    hovertemplate = _merit_grid_hovertemplate(
        x_axis, y_axis, fom_name, frozen_keys, z_var="z",
    )

    # Surface contours: project FoM iso-lines onto the surface itself so the
    # height-colour duality stays readable. ``usecolormap=True`` keeps the
    # projected lines aligned with the shared ``_COLORSCALE`` ramp.
    contour_z_cfg = dict(
        show=True,
        usecolormap=True,
        highlightcolor="#404040",
        project=dict(z=True),
    )

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_unique, y=y_unique, z=z,
        surfacecolor=z,
        cmin=zmin, cmax=zmax,
        colorscale=_COLORSCALE,
        colorbar=dict(
            title=dict(text=fom_name, side="right",
                       font=dict(size=11, color=_TEXT_MUTED)),
            tickfont=dict(color=_TEXT_MUTED, size=10),
            outlinewidth=0, thickness=14, len=0.85,
        ),
        contours=dict(z=contour_z_cfg),
        customdata=customdata,
        hovertemplate=hovertemplate,
        showscale=True,
        opacity=0.95,
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1, roughness=0.6),
    ))

    # Threshold iso-FoM rings: drawn as thin rims at z = threshold using
    # Scatter3d traces so each gets its user-picked colour. Cells outside
    # the data range are silently skipped.
    if thresholds:
        colors = threshold_colors or _THRESHOLD_COLORS
        for i, t in enumerate(thresholds):
            if t is None:
                continue
            t_f = float(t)
            if t_f < zmin or t_f > zmax:
                continue
            color = colors[i % len(colors)]
            rings = _surface_iso_segments(x_unique, y_unique, z, t_f)
            if rings.size == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=rings[:, 0], y=rings[:, 1], z=rings[:, 2],
                mode="lines",
                line=dict(color=color, width=4),
                hoverinfo="skip",
                showlegend=True,
                name=f"FoM = {t_f:g}",
            ))

    fig.update_layout(
        **_LAYOUT_BASE,
        scene=dict(
            xaxis=dict(
                title=dict(text=_axis_label(x_axis), font=dict(size=11, color=_TEXT_MUTED)),
                gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
                tickfont=dict(size=10, color=_TEXT_MUTED),
            ),
            yaxis=dict(
                title=dict(text=_axis_label(y_axis), font=dict(size=11, color=_TEXT_MUTED)),
                gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
                tickfont=dict(size=10, color=_TEXT_MUTED),
            ),
            zaxis=dict(
                title=dict(text=fom_name, font=dict(size=11, color=_TEXT_MUTED)),
                gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
                tickfont=dict(size=10, color=_TEXT_MUTED),
            ),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR,
                        font_color=_TEXT_COLOR, align="left"),
    )
    return fig


def _surface_iso_segments(
    x_unique: np.ndarray, y_unique: np.ndarray, z: np.ndarray, level: float,
) -> np.ndarray:
    """Marching-squares-style segment extractor for one iso-level.

    Returns an ``(N, 3)`` array of ``(x, y, level)`` triples ordered as
    line-segment endpoints (Plotly's Scatter3d ``mode='lines'`` joins
    consecutive points; we insert NaN rows between disjoint segments to
    break the polyline).
    """
    pts: list[tuple[float, float, float]] = []
    nan_break = (float("nan"), float("nan"), float("nan"))
    ny, nx = z.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners = [
                (x_unique[i],     y_unique[j],     z[j,     i]),
                (x_unique[i + 1], y_unique[j],     z[j,     i + 1]),
                (x_unique[i + 1], y_unique[j + 1], z[j + 1, i + 1]),
                (x_unique[i],     y_unique[j + 1], z[j + 1, i]),
            ]
            # Skip cells with any NaN — can't infer a crossing.
            if any(not np.isfinite(c[2]) for c in corners):
                continue
            crossings: list[tuple[float, float]] = []
            for k in range(4):
                a = corners[k]
                b = corners[(k + 1) % 4]
                za = a[2] - level
                zb = b[2] - level
                if za == 0 and zb == 0:
                    continue
                if (za > 0) != (zb > 0):
                    # Linear interpolation along the cell edge.
                    t = za / (za - zb)
                    cx = a[0] + t * (b[0] - a[0])
                    cy = a[1] + t * (b[1] - a[1])
                    crossings.append((cx, cy))
            # Either 2 (single segment) or 4 (saddle — emit both pairs).
            if len(crossings) >= 2:
                for k in range(0, len(crossings) - 1, 2):
                    pts.append((crossings[k][0], crossings[k][1], level))
                    pts.append((crossings[k + 1][0], crossings[k + 1][1], level))
                    pts.append(nan_break)
    if not pts:
        return np.empty((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)


# ---------------------------------------------------------------------------
# Merit view: Pareto mode (numerator vs denominator scatter, with iso-FoM
# guide lines, optional colour-by-input, and Pareto-front highlighting).
# ---------------------------------------------------------------------------


def plot_merit_pareto(
    sweep_data: dict,
    fom_config,
    color_by: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    """Numerator-vs-denominator scatter with iso-FoM guide lines, optional
    colour-by-input, and Pareto-front highlighting (max numerator, min
    denominator).
    """
    bd, fom_config, _resolved, err = _build_merit_breakdown(sweep_data, fom_config)
    if err is not None:
        return err

    num = np.asarray(bd.numerator, dtype=float)
    den = np.asarray(bd.denominator, dtype=float)
    fom = np.asarray(bd.fom, dtype=float)

    finite_mask = np.isfinite(num) & np.isfinite(den) & np.isfinite(fom)
    if not finite_mask.any():
        return plot_empty(
            "FoM produced no finite values — check for divide-by-zero or constant expressions."
        )

    num_f = num[finite_mask]
    den_f = den[finite_mask]
    fom_f = fom[finite_mask]

    # --- Build customdata + hover template ---------------------------------
    # Columns, in order: num, den, fom, <sweep axis values>, <output primitives
    # referenced in the formulas>, <intermediate values>.
    referenced = (
        _fom_referenced_names(fom_config)
        | set(bd.intermediates.keys())
    )
    axis_cols: list[tuple[str, np.ndarray]] = [
        (name, bd.primitives[name][finite_mask])
        for name in bd.sweep_axes
        if name in bd.primitives
    ]
    output_cols: list[tuple[str, np.ndarray]] = [
        (name, bd.primitives[name][finite_mask])
        for name in bd.output_keys
        if name in bd.primitives and name in referenced
    ]
    inter_cols: list[tuple[str, np.ndarray]] = [
        (name, vals[finite_mask]) for name, vals in bd.intermediates.items()
    ]

    columns = (
        [("__num__", num_f), ("__den__", den_f), ("__fom__", fom_f)]
        + axis_cols + output_cols + inter_cols
    )
    customdata = np.column_stack([c[1] for c in columns])

    fom_name = fom_config.name or "Figure of Merit"
    num_expr = (fom_config.numerator or "").strip() or "1"
    den_expr = (fom_config.denominator or "").strip() or "1"

    _OUTPUT_LABELS[_FOM_OUTPUT_KEY] = fom_name

    hover_lines: list[str] = [
        f"<b>{fom_name}: %{{customdata[2]:.4g}}</b>",
        f"  numerator ({_truncate_expr(num_expr)}) = %{{customdata[0]:.4g}}",
        f"  denominator ({_truncate_expr(den_expr)}) = %{{customdata[1]:.4g}}",
    ]
    col_idx = 3
    if axis_cols:
        hover_lines.append("<br><i>sweep point</i>")
        for name, _ in axis_cols:
            hover_lines.append(
                f"  {_axis_label(name)} = %{{customdata[{col_idx}]:.4g}}"
            )
            col_idx += 1
    if output_cols:
        hover_lines.append("<br><i>outputs</i>")
        for name, _ in output_cols:
            label = _OUTPUT_LABELS.get(name, name)
            hover_lines.append(f"  {label} = %{{customdata[{col_idx}]:.4g}}")
            col_idx += 1
    if inter_cols:
        hover_lines.append("<br><i>intermediates</i>")
        for name, _ in inter_cols:
            hover_lines.append(f"  {name} = %{{customdata[{col_idx}]:.4g}}")
            col_idx += 1
    hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

    x_type = _pick_axis_scale(den_f)
    y_type = _pick_axis_scale(num_f)

    # --- Pareto front: maximise numerator, minimise denominator. -----------
    front_mask = _pareto_front_mask(num_f, den_f)

    # --- Resolve colour-by source. -----------------------------------------
    color_by = (color_by or "").strip().lower()
    color_mode = "fom"
    color_values: np.ndarray = fom_f
    color_label = fom_name
    if color_by == "none":
        color_mode = "none"
    elif color_by and color_by != "fom":
        # Try to match against an active sweep axis (case-insensitive).
        axis_lookup = {a.lower(): a for a in bd.sweep_axes}
        axis_key = axis_lookup.get(color_by)
        if axis_key and axis_key in bd.primitives:
            color_mode = "axis"
            color_values = bd.primitives[axis_key][finite_mask]
            color_label = _axis_label(axis_key)
    # else fall through to FoM colouring

    fig = go.Figure()

    # --- Iso-FoM guide lines from user-picked thresholds. ------------------
    # On a (denominator, numerator) plot, FoM = num/den = c is the line
    # num = c·den — straight on log-log, straight on linear.
    iso_levels: list[tuple[float, str]] = []
    if thresholds:
        colors = threshold_colors or _THRESHOLD_COLORS
        for i, t in enumerate(thresholds):
            if t is None:
                continue
            iso_levels.append((float(t), colors[i % len(colors)]))
    if not iso_levels and fom_f.size >= 4:
        # Fallback: faint percentile guide lines so the plot stays readable
        # even when the user hasn't enabled thresholds.
        pct_levels = sorted({
            round(float(np.percentile(fom_f, q)), 12)
            for q in (25, 50, 75)
        })
        iso_levels = [(lvl, "rgba(43,43,43,0.18)") for lvl in pct_levels]

    if x_type == "log":
        x_line = np.array([float(den_f.min()), float(den_f.max())])
    else:
        x_line = np.array([0.0, float(den_f.max())])
    for lvl, color in iso_levels:
        if lvl <= 0 and (x_type == "log" or y_type == "log"):
            continue
        y_line = lvl * x_line
        is_threshold = color != "rgba(43,43,43,0.18)"
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines+text" if is_threshold else "lines",
            text=["", f"FoM={lvl:g}"] if is_threshold else None,
            textposition="top left" if is_threshold else None,
            textfont=dict(color=color, size=10) if is_threshold else None,
            line=dict(
                color=color,
                width=1.6 if is_threshold else 1,
                dash="dash" if is_threshold else "dot",
            ),
            hoverinfo="skip",
            showlegend=False,
            name=f"FoM={lvl:.3g}",
        ))

    # --- Markers: dominated points (faint, small) + Pareto-front (large). --
    dom_mask = ~front_mask
    if dom_mask.any():
        marker_kwargs = dict(
            size=5,
            line=dict(width=0),
            opacity=0.35,
        )
        if color_mode == "none":
            marker_kwargs["color"] = _TEXT_MUTED
        else:
            marker_kwargs.update(dict(
                color=color_values[dom_mask],
                colorscale=_COLORSCALE,
                cmin=float(np.nanmin(color_values)),
                cmax=float(np.nanmax(color_values)),
                showscale=False,
            ))
        fig.add_trace(go.Scatter(
            x=den_f[dom_mask], y=num_f[dom_mask],
            mode="markers",
            marker=marker_kwargs,
            customdata=customdata[dom_mask],
            hovertemplate=hovertemplate,
            showlegend=False,
            name="dominated",
        ))

    # Pareto-front line connecting front points sorted by denominator.
    front_idx = np.where(front_mask)[0]
    if front_idx.size:
        order = np.argsort(den_f[front_idx])
        fx = den_f[front_idx][order]
        fy = num_f[front_idx][order]
        fig.add_trace(go.Scatter(
            x=fx, y=fy,
            mode="lines",
            line=dict(color=_ACCENT, width=1.2, dash="solid"),
            hoverinfo="skip",
            showlegend=False,
            opacity=0.6,
            name="Pareto front",
        ))

        front_marker = dict(
            size=10,
            line=dict(width=1.5, color=_ACCENT),
            opacity=1.0,
        )
        if color_mode == "none":
            front_marker["color"] = "#FFFFFF"
        else:
            front_marker.update(dict(
                color=color_values[front_mask],
                colorscale=_COLORSCALE,
                cmin=float(np.nanmin(color_values)),
                cmax=float(np.nanmax(color_values)),
                colorbar=dict(
                    title=dict(text=color_label, side="right",
                               font=dict(size=11, color=_TEXT_MUTED)),
                    tickfont=dict(color=_TEXT_MUTED, size=10),
                    outlinewidth=0, thickness=14, len=0.85,
                ),
                showscale=True,
            ))
        fig.add_trace(go.Scatter(
            x=den_f[front_mask], y=num_f[front_mask],
            mode="markers",
            marker=front_marker,
            customdata=customdata[front_mask],
            hovertemplate=hovertemplate,
            showlegend=False,
            name="Pareto-optimal",
        ))

    x_title = f"Denominator  ·  {_truncate_expr(den_expr, 80)}"
    y_title = f"Numerator  ·  {_truncate_expr(num_expr, 80)}"
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=_TEXT_MUTED)),
            type=x_type,
            gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=_TEXT_MUTED)),
            type=y_type,
            gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=_GRID_COLOR,
            font=dict(
                color=_TEXT_COLOR,
                family="'JetBrains Mono', 'SF Mono', monospace",
                size=11,
            ),
            align="left",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Merit view: top-level dispatcher (Heatmap by default, Pareto on request).
# ---------------------------------------------------------------------------


def plot_merit(
    sweep_data: dict,
    fom_config,
    view_hint: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    point_cap: int | None = None,
    mode: str = "heatmap",
    x_axis: str | None = None,
    y_axis: str | None = None,
    frozen_values: dict[str, float] | None = None,
    color_by: str | None = None,
) -> go.Figure:
    """Dispatch to either the heatmap or the (numerator-vs-denominator)
    Pareto-style scatter view of the FoM, based on ``mode``.
    """
    m = (mode or "heatmap").lower()
    if m == "pareto":
        return plot_merit_pareto(
            sweep_data, fom_config,
            color_by=color_by,
            thresholds=thresholds,
            threshold_colors=threshold_colors,
        )
    if m in ("3d", "surface", "3d_surface"):
        return plot_merit_surface(
            sweep_data, fom_config,
            x_axis=x_axis, y_axis=y_axis,
            frozen_values=frozen_values,
            thresholds=thresholds,
            threshold_colors=threshold_colors,
        )
    return plot_merit_heatmap(
        sweep_data, fom_config,
        x_axis=x_axis, y_axis=y_axis,
        frozen_values=frozen_values,
        thresholds=thresholds,
        threshold_colors=threshold_colors,
    )


def _truncate_expr(expr: str, limit: int = 60) -> str:
    """Shorten an expression string for display in axis/hover labels."""
    s = (expr or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _fom_referenced_names(fom_config) -> set[str]:
    """Free variable names referenced anywhere in the FoM formula.

    Used to pick which primitive columns to include in hover text.
    """
    from gui.fom import _referenced_names
    refs: set[str] = set()
    for _, expr in fom_config.intermediates:
        refs |= _referenced_names(expr)
    refs |= _referenced_names(fom_config.numerator or "")
    refs |= _referenced_names(fom_config.denominator or "")
    return refs


# ---------------------------------------------------------------------------
# Empty / error placeholder
# ---------------------------------------------------------------------------

def plot_empty(message: str = "Select sweep axes and click Run") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=_TEXT_MUTED, family="Inter, system-ui, sans-serif"),
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def sweep_to_csv(sweep_data: dict) -> str:
    if "facets" in sweep_data and sweep_data["facets"]:
        return _faceted_sweep_to_csv(sweep_data)

    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    col_names = []
    for k in metric_keys:
        m = METRIC_BY_KEY.get(k)
        col_names.append(m.label if m else k)
    for k in available_outputs:
        col_names.append(_OUTPUT_LABELS.get(k, k))

    lines = [",".join(col_names)]
    for row in rows:
        lines.append(",".join(f"{v}" for v in row))
    return "\n".join(lines)


def _faceted_sweep_to_csv(sweep_data: dict) -> str:
    """Export faceted sweep data with extra columns per facet key."""
    facet_keys = sweep_data.get("facet_keys", [])
    facets = sweep_data["facets"]

    all_lines: list[str] = []
    header_written = False

    for facet in facets:
        label_dict = facet.get("label", {})
        metric_keys, available_outputs, rows = _flatten_sweep_to_table(facet)

        if not header_written:
            col_names = list(facet_keys)
            for k in metric_keys:
                m = METRIC_BY_KEY.get(k)
                col_names.append(m.label if m else k)
            for k in available_outputs:
                col_names.append(_OUTPUT_LABELS.get(k, k))
            all_lines.append(",".join(col_names))
            header_written = True

        facet_prefix = [str(label_dict.get(fk, "")) for fk in facet_keys]
        for row in rows:
            all_lines.append(",".join(facet_prefix + [f"{v}" for v in row]))

    return "\n".join(all_lines)


def _facet_grid_dims(facet_keys: list[str], n_facets: int, sweep_data: dict) -> tuple[int, int, list[str]]:
    """Compute (rows, cols, subplot_titles) for the facet grid.

    * 1 facet key  → 1 row × N cols
    * 2 facet keys → first key → rows, second → cols
    * 3+ keys      → rows = product of all-but-last, cols = last
    """
    facets = sweep_data["facets"]
    if len(facet_keys) == 1:
        rows, cols = 1, n_facets
    elif len(facet_keys) == 2:
        from gui.constants import CAT_METRIC_BY_KEY as _cm
        cols = len(_cm[facet_keys[1]].options)
        rows = (n_facets + cols - 1) // cols
    else:
        from gui.constants import CAT_METRIC_BY_KEY as _cm
        cols = len(_cm[facet_keys[-1]].options)
        rows = (n_facets + cols - 1) // cols
    titles = [" / ".join(str(v) for v in f["label"].values()) for f in facets]
    return rows, cols, titles


def _build_faceted_figure(
    num_metrics: int,
    sweep_data: dict,
    output_key: str,
    view_type: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    view_mode: str = "absolute",
) -> go.Figure:
    """Build a subplot grid, one panel per facet."""
    from plotly.subplots import make_subplots

    facets = sweep_data["facets"]
    facet_keys = sweep_data.get("facet_keys", [])
    n_facets = len(facets)
    rows, cols, titles = _facet_grid_dims(facet_keys, n_facets, sweep_data)

    is_3d = num_metrics == 3 and view_type in ("scatter3d", "isosurface", None)
    specs = None
    if is_3d:
        specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]

    h_spacing = max(0.02, 0.12 / cols) if not is_3d else max(0.03, 0.20 / cols)
    v_spacing = max(0.04, 0.12 / rows) if not is_3d else max(0.06, 0.20 / rows)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles + [""] * (rows * cols - n_facets),
        specs=specs,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
    )

    # Apply the derivative view mode per-facet *before* the per-panel
    # recursion (and before the global zmin/zmax pass) so the recursive
    # build_figure call below sees an already-transformed sweep_data with
    # ``output_key="__deriv__"`` and short-circuits its own view_mode
    # branch. Drops thresholds in derivative mode for the same reason as
    # the non-faceted path — they're absolute-scale values.
    transformed_facets = []
    transformed_output_key = output_key
    transformed_thresholds = thresholds
    apply_mode = (
        view_mode and view_mode != "absolute"
        and num_metrics in (1, 2, 3)
        and view_type not in {
            "parallel", "slices", "importance", "pareto", "correlation",
            "elasticity", "merit", "topology",
        }
    )
    if apply_mode:
        try:
            for facet in facets:
                tf, tk = _apply_view_mode(facet, output_key, view_mode, num_metrics)
                transformed_facets.append(tf)
                transformed_output_key = tk
        except Exception:
            transformed_facets = list(facets)
            transformed_output_key = output_key
    else:
        transformed_facets = list(facets)

    # Compute global zmin/zmax for shared colorscale (2D views) using the
    # *transformed* output key so derivative-mode heatmaps share a coherent
    # range across panels.
    global_zmin = float("inf")
    global_zmax = float("-inf")
    if num_metrics == 2:
        for facet in transformed_facets:
            grid = facet.get("grid", [])
            for row_data in grid:
                for cell in row_data:
                    v = (
                        float(cell.get(transformed_output_key, 0.0))
                        if isinstance(cell, dict) else 0.0
                    )
                    if v < global_zmin:
                        global_zmin = v
                    if v > global_zmax:
                        global_zmax = v

    # For 3D faceted views, divide the point budget across facets so the
    # combined figure stays within the browser's WebGL/JSON limits.
    facet_3d_cap = _MAX_BROWSER_3D_POINTS // max(n_facets, 1) if is_3d else None

    for idx, facet in enumerate(transformed_facets):
        r = idx // cols + 1
        c = idx % cols + 1

        # Build a single-facet figure, then transplant traces into the grid.
        # ``view_mode="absolute"`` because we already transformed the facet's
        # sweep_data above — the recursive call must not re-apply.
        panel_fig = build_figure(
            num_metrics, facet, transformed_output_key,
            view_type=view_type,
            thresholds=transformed_thresholds,
            threshold_colors=threshold_colors,
            _3d_point_cap=facet_3d_cap,
            view_mode="absolute",
        )
        for trace in panel_fig.data:
            if is_3d:
                scene_name = f"scene{idx + 1}" if idx > 0 else "scene"
                trace.scene = scene_name
            if hasattr(trace, "showlegend"):
                trace.showlegend = False
            fig.add_trace(trace, row=r, col=c)

    # Synchronize colorscale range across heatmaps/contours.
    if num_metrics == 2 and global_zmin < global_zmax:
        for trace in fig.data:
            if hasattr(trace, "zmin"):
                trace.zmin = global_zmin
                trace.zmax = global_zmax

    row_height = 500 if is_3d else 350
    fig.update_layout(
        **_LAYOUT_BASE,
        height=max(500, row_height * rows),
        showlegend=False,
    )
    return fig


def _flatten_facets_for_analysis(sweep_data: dict) -> dict:
    """Merge all facets into a single flat sweep_data for analysis views.

    Each facet's grid is flattened and tagged with numeric-encoded categorical
    values so that analysis views (parallel coordinates, slices, importance,
    etc.) can treat categoricals as extra dimensions alongside the numeric axes.

    Returns a sweep_data dict with a ``_prebuilt_table`` key containing the
    already-flattened ``(metric_keys, available_outputs, data)`` tuple so
    that ``_flatten_sweep_to_table`` can skip re-flattening.
    """
    facets = sweep_data["facets"]
    facet_keys = sweep_data.get("facet_keys", [])
    base_metric_keys = facets[0].get("metric_keys", [])
    ndim = len(base_metric_keys)

    # Build value→index maps for each categorical key so they become numeric.
    cat_value_maps: list[dict[str, int]] = []
    for fk in facet_keys:
        seen_vals: dict[str, int] = {}
        for f in facets:
            v = f.get("label", {}).get(fk, "")
            if v not in seen_vals:
                seen_vals[v] = len(seen_vals)
        cat_value_maps.append(seen_vals)

    # Resolve axes from the first facet (all facets share the same numeric axes).
    first = facets[0]
    if ndim == 1:
        axes = [first.get("xs", [])]
    elif ndim == 2:
        axes = [first.get("xs", []), first.get("ys", [])]
    elif ndim == 3:
        axes = [first.get("xs", []), first.get("ys", []), first.get("zs", [])]
    else:
        axes = first.get("axes", [])

    # Determine available output keys from a sample cell.
    sample = _find_sample(first.get("grid", []), ndim)
    available_outputs = [k for k in _OUTPUT_KEYS if k in sample] if sample else []

    # Flatten every facet's grid into a combined row list.
    combined_rows: list[list[float]] = []
    for f in facets:
        label = f.get("label", {})
        cat_vals = [float(cat_value_maps[ci].get(label.get(fk, ""), 0))
                    for ci, fk in enumerate(facet_keys)]

        grid = f.get("grid", [])

        # Structured numpy grid (N >= 4 production path) → vectorised flatten.
        if isinstance(grid, np.ndarray) and grid.dtype.names:
            flat_data = _flatten_structured(grid, axes, ndim, available_outputs)
            for ri in range(flat_data.shape[0]):
                row = flat_data[ri].tolist()
                row[ndim:ndim] = cat_vals
                combined_rows.append(row)
        else:
            facet_rows: list[list[float]] = []
            if ndim <= 3:
                _flatten_nested(grid, axes, ndim, available_outputs, facet_rows)
            else:
                shape = tuple(f.get("shape", [len(ax) for ax in axes]))
                _flatten_nd(grid, axes, shape, ndim, available_outputs, facet_rows)
            for row in facet_rows:
                row[ndim:ndim] = cat_vals
                combined_rows.append(row)

    combined_metric_keys = list(base_metric_keys) + list(facet_keys)

    if not combined_rows:
        data = np.empty((0, len(combined_metric_keys) + len(available_outputs)),
                        dtype=np.float64)
    else:
        data = np.asarray(combined_rows, dtype=np.float64)

    # Build index→label maps so views can show option names instead of numbers.
    cat_tick_labels: dict[str, list[str]] = {}
    for ci, fk in enumerate(facet_keys):
        inv = {idx: name for name, idx in cat_value_maps[ci].items()}
        cat_tick_labels[fk] = [inv[j] for j in range(len(inv))]

    return {
        "metric_keys": combined_metric_keys,
        "xs": [],
        "grid": [],
        "_prebuilt_table": (combined_metric_keys, available_outputs, data),
        "_cat_tick_labels": cat_tick_labels,
    }


def build_figure(
    num_metrics: int,
    sweep_data: dict,
    output_key: str,
    view_type: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    frozen_z: float | None = None,
    _3d_point_cap: int | None = None,
    pareto_x: str | None = None,
    pareto_y: str | None = None,
    fom_config: dict | None = None,
    merit_mode: str = "heatmap",
    merit_x_axis: str | None = None,
    merit_y_axis: str | None = None,
    merit_frozen_values: dict | None = None,
    merit_color_by: str | None = None,
    view_mode: str = "absolute",
    elasticity_trajectory: str | None = None,
    importance_mode: str = "range",
    correlation_mode: str = "spearman",
) -> go.Figure:
    if sweep_data is None:
        return plot_empty()

    # New "Elasticity Comparison" analysis view — handled before any other
    # dispatch since it builds its own multi-line figure from the raw grid.
    if view_type == "elasticity":
        return plot_elasticity_comparison(
            sweep_data, output_key, trajectory_key=elasticity_trajectory,
        )

    # Merit view: custom FoM laid out over the sweep axes — handled before
    # facet dispatch because it builds its own flattened sweep internally.
    if view_type == "merit":
        return plot_merit(
            sweep_data,
            fom_config or {},
            thresholds=thresholds,
            threshold_colors=threshold_colors,
            point_cap=_3d_point_cap,
            mode=merit_mode,
            x_axis=merit_x_axis,
            y_axis=merit_y_axis,
            frozen_values=merit_frozen_values,
            color_by=merit_color_by,
        )

    # Faceted data → delegate to subplot grid builder for spatial views,
    # or flatten into a single dataset for analysis views.
    if "facets" in sweep_data and sweep_data["facets"]:
        _analysis_views = {"parallel", "slices", "importance", "pareto", "correlation", "elasticity"}
        if view_type in _analysis_views:
            sweep_data = _flatten_facets_for_analysis(sweep_data)
        else:
            return _build_faceted_figure(
                num_metrics, sweep_data, output_key,
                view_type=view_type,
                thresholds=thresholds,
                threshold_colors=threshold_colors,
                view_mode=view_mode or "absolute",
            )

    _tc = threshold_colors

    # Derivative view mode: replace F in the grid with |∇F| / elasticity /
    # second derivative / mixed partial before dispatching, so every
    # dimensional plotter renders the derivative without knowing it.
    # Threshold iso-levels are passed through unchanged — they trace level
    # sets of the *displayed* scalar field, so in derivative mode the same
    # threshold values (e.g. 0.3) draw contours where |∇F| = 0.3 instead
    # of where F = 0.3. Same legend toggle UX in both modes; the user
    # reinterprets the threshold numbers against the colorbar that now
    # says "|∇Overall Fidelity|" (or similar).
    _analysis_views = {
        "parallel", "slices", "importance", "pareto", "correlation",
        "elasticity", "merit", "topology",
    }
    _is_dimensional_view = view_type is None or view_type not in _analysis_views
    if (
        view_mode and view_mode != "absolute"
        and _is_dimensional_view
        and num_metrics in (1, 2, 3)
    ):
        try:
            sweep_data, output_key = _apply_view_mode(
                sweep_data, output_key, view_mode, num_metrics,
            )
        except Exception as exc:
            return plot_empty(f"Derivative error: {exc}")

    fig: go.Figure

    if view_type == "parallel":
        fig = plot_parallel_coordinates(sweep_data, output_key)
    elif view_type == "slices":
        fig = plot_slice(sweep_data, output_key)
    elif view_type == "importance":
        fig = plot_importance(sweep_data, output_key, mode=importance_mode or "range")
    elif view_type == "pareto":
        fig = plot_pareto(
            sweep_data,
            x_key=pareto_x or "total_epr_pairs",
            y_key=pareto_y or "overall_fidelity",
            thresholds=thresholds,
            threshold_colors=_tc,
        )
    elif view_type == "correlation":
        fig = plot_correlation(sweep_data, output_key, mode=correlation_mode or "spearman")
    else:
        try:
            if num_metrics == 1:
                fig = plot_1d(
                    x_values=np.array(sweep_data["xs"]),
                    results=sweep_data["grid"],
                    metric_key=sweep_data["metric_keys"][0],
                    output_key=output_key,
                    thresholds=thresholds,
                    threshold_colors=_tc,
                    inflection_x=sweep_data.get("_inflection_x"),
                )
            elif num_metrics == 2:
                # Heatmap with iso-line overlay (the previous "contour" view —
                # the no-iso-line variant was removed). Old saved sessions with
                # ``view_type="contour"`` land here too.
                fig = plot_2d_contour(
                    x_values=np.array(sweep_data["xs"]),
                    y_values=np.array(sweep_data["ys"]),
                    grid=sweep_data["grid"],
                    metric_key1=sweep_data["metric_keys"][0],
                    metric_key2=sweep_data["metric_keys"][1],
                    output_key=output_key,
                    thresholds=thresholds,
                    threshold_colors=_tc,
                )
            elif num_metrics == 3 and view_type in ("frozen_heatmap", "frozen_contour"):
                from gui.interpolation import (
                    frozen_slice,
                    sweep_to_interp_grid,
                )
                igrid = sweep_to_interp_grid(sweep_data, output_key)
                z_val = frozen_z if frozen_z is not None else (
                    (sweep_data["zs"][0] + sweep_data["zs"][-1]) / 2
                )
                vals_3d = np.array(igrid["values"])
                slice_2d = frozen_slice(
                    vals_3d,
                    np.array(igrid["xs"]),
                    np.array(igrid["ys"]),
                    np.array(igrid["zs"]),
                    z_val,
                )
                slice_grid = [[{output_key: float(slice_2d[j, i])}
                               for j in range(slice_2d.shape[0])]
                              for i in range(slice_2d.shape[1])]
                fig = plot_2d_contour(
                    x_values=np.array(sweep_data["xs"]),
                    y_values=np.array(sweep_data["ys"]),
                    grid=slice_grid,
                    metric_key1=sweep_data["metric_keys"][0],
                    metric_key2=sweep_data["metric_keys"][1],
                    output_key=output_key,
                    thresholds=thresholds,
                    threshold_colors=_tc,
                )
            elif num_metrics == 3:
                _3d_args = dict(
                    x_values=np.array(sweep_data["xs"]),
                    y_values=np.array(sweep_data["ys"]),
                    z_values=np.array(sweep_data["zs"]),
                    grid=sweep_data["grid"],
                    metric_key1=sweep_data["metric_keys"][0],
                    metric_key2=sweep_data["metric_keys"][1],
                    metric_key3=sweep_data["metric_keys"][2],
                    output_key=output_key,
                    thresholds=thresholds,
                    threshold_colors=_tc,
                )
                if view_type == "isosurface":
                    fig = plot_3d_isosurface(**_3d_args, point_cap=_3d_point_cap)
                else:
                    fig = plot_3d(**_3d_args, point_cap=_3d_point_cap)
            elif num_metrics >= 4:
                # For N >= 4, default to parallel coordinates
                fig = plot_parallel_coordinates(sweep_data, output_key)
            else:
                return plot_empty("Add at least one metric axis on the left to start a sweep")
        except Exception as exc:
            return plot_empty(f"Plot error: {exc}")

    fig.update_layout(uirevision="keep")
    return fig
