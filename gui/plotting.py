"""
Plotly figure builders for 1-D, 2-D, and 3-D fidelity sweep results.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from .constants import CAT_METRIC_BY_KEY, METRIC_BY_KEY, OUTPUT_METRICS

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

    return fig


# ---------------------------------------------------------------------------
# 2-D heatmap
# ---------------------------------------------------------------------------

def plot_2d(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: list,          # list[list[dict]]  shape: [Nx][Ny]
    metric_key1: str,
    metric_key2: str,
    output_key: str,
) -> go.Figure:
    # Build Z matrix: shape (Ny, Nx) for heatmap (y-axis → rows)
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

    zmin, zmax = (0.0, 1.0) if "fidelity" in output_key else (float(z.min()), float(z.max()))

    _COLORSCALE = [
        [0.0,  "#2B2B2B"],
        [0.2,  "#5a5a5a"],
        [0.4,  "#888888"],
        [0.6,  "#B3B3B3"],
        [0.8,  "#D4D4D4"],
        [1.0,  "#F0F0F0"],
    ]

    fig = go.Figure(
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

    _COLORSCALE = [
        [0.0, "#d73027"],
        [0.25, "#fc8d59"],
        [0.5, "#fee08b"],
        [0.75, "#91bfdb"],
        [1.0, "#4575b4"],
    ]

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

    _COLORSCALE = [
        [0.0,  "#2B2B2B"],
        [0.2,  "#5a5a5a"],
        [0.4,  "#888888"],
        [0.6,  "#B3B3B3"],
        [0.8,  "#D4D4D4"],
        [1.0,  "#F0F0F0"],
    ]

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

_OUTPUT_KEYS = ["overall_fidelity", "algorithmic_fidelity", "routing_fidelity",
                "coherence_fidelity", "total_circuit_time_ns", "total_epr_pairs"]

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


def _flatten_sweep_to_table(sweep_data: dict) -> tuple[list[str], list[str], np.ndarray]:
    """Flatten a sweep's results into a ``(total_points, ndim + n_outputs)`` matrix.

    Returns ``(metric_keys, available_outputs, data)``. ``data`` is always a
    numpy float64 array — callers should treat it as such and avoid
    re-wrapping with ``np.array(...)`` (which would copy).
    """
    # Fast path: pre-built table from _flatten_facets_for_analysis.
    if "_prebuilt_table" in sweep_data:
        return sweep_data["_prebuilt_table"]

    metric_keys = sweep_data["metric_keys"]
    grid = sweep_data["grid"]
    ndim = len(metric_keys)

    # Find a sample result to determine available outputs
    sample = _find_sample(grid, ndim)
    available_outputs = [k for k in _OUTPUT_KEYS if k in sample] if sample else []

    # Resolve axis values for all dimensions
    axes = _resolve_axes(sweep_data, ndim)

    # Structured numpy grid (N >= 4 production path) → vectorised flatten.
    # Builds the output matrix directly without allocating per-cell dicts.
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
    """Build the flattened (total, ndim + n_outputs) matrix from a structured grid.

    Vectorised over numpy — no Python loop, no intermediate dict objects.
    Peak extra memory is the output matrix itself (~N × 16 × 8 B) plus one
    transient param column at a time (~N × 8 B).
    """
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


def _find_sample(grid, ndim: int) -> dict | None:
    """Extract one sample result dict from the grid.

    Supports both the legacy list-of-dicts form and the structured numpy
    array form; callers only use the returned mapping to discover which
    output keys are available.
    """
    if isinstance(grid, np.ndarray) and grid.dtype.names:
        return {name: 0.0 for name in grid.dtype.names}
    if not grid:
        return None
    item = grid[0]
    if isinstance(item, dict):
        return item
    # Nested list: drill down
    nested = item
    while isinstance(nested, list) and nested:
        nested = nested[0]
    return nested if isinstance(nested, dict) else None


def _resolve_axes(sweep_data: dict, ndim: int) -> list[list]:
    """Get axis values for each dimension from sweep_data."""
    if "axes" in sweep_data and len(sweep_data["axes"]) == ndim:
        return sweep_data["axes"]
    axes = [sweep_data["xs"]]
    if ndim >= 2:
        axes.append(sweep_data["ys"])
    if ndim >= 3:
        axes.append(sweep_data["zs"])
    return axes


def _flatten_nested(
    grid, axes: list, ndim: int, outputs: list[str], rows: list,
) -> None:
    """Flatten 1-3D nested-list grids into rows."""
    import itertools
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
    """Flatten N-D (N >= 4) flat grid list into rows."""
    import itertools
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

    _PARCOORDS_COLORSCALE = [
        [0.0, "#d73027"],
        [0.25, "#fc8d59"],
        [0.5, "#fee08b"],
        [0.75, "#91bfdb"],
        [1.0, "#4575b4"],
    ]

    fig = go.Figure(
        go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=color_vals,
                colorscale=_PARCOORDS_COLORSCALE,
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

def plot_importance(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for importance plot")

    data = rows  # already an ndarray from _flatten_sweep_to_table
    num_params = len(metric_keys)
    out_col = num_params + available_outputs.index(output_key) if output_key in available_outputs else num_params

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
            title=dict(text=f"Range of {_OUTPUT_LABELS.get(output_key, output_key)}",
                       font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=_TEXT_COLOR),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Pareto front (fidelity vs EPR pairs — dominated points dimmed)
# ---------------------------------------------------------------------------

def plot_pareto(sweep_data: dict, output_key: str, thresholds: list[float] | None = None,
                threshold_colors: list[str] | None = None) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for Pareto plot")

    data = rows  # already an ndarray from _flatten_sweep_to_table
    num_params = len(metric_keys)

    fid_col = num_params + available_outputs.index("overall_fidelity") if "overall_fidelity" in available_outputs else None
    epr_col = num_params + available_outputs.index("total_epr_pairs") if "total_epr_pairs" in available_outputs else None

    if fid_col is None or epr_col is None:
        return plot_empty("Need both fidelity and EPR metrics for Pareto")

    fidelity = data[:, fid_col]
    epr = data[:, epr_col]

    # Vectorised Pareto front: sort by EPR asc, then for each epr-tie group
    # mark a point as dominated iff
    #   fid <= max_fid_from_strictly_lower_epr   (strict-epr dominator) OR
    #   fid <  max_fid_within_same_epr_group     (same-epr strictly-higher dominator)
    # O(N log N) numpy, drops multi-second 50k-point runs to ~ms.
    is_pareto = np.ones(len(data), dtype=bool)
    if len(data) > 0:
        order = np.argsort(epr, kind="stable")
        s_epr = epr[order]
        s_fid = fidelity[order]

        change = np.concatenate(([True], s_epr[1:] != s_epr[:-1]))
        group_id = np.cumsum(change) - 1
        n_groups = int(group_id[-1]) + 1

        group_max = np.full(n_groups, -np.inf)
        np.maximum.at(group_max, group_id, s_fid)

        if n_groups >= 2:
            prev_max_strict = np.concatenate(
                ([-np.inf], np.maximum.accumulate(group_max[:-1]))
            )
        else:
            prev_max_strict = np.array([-np.inf])

        s_prev = prev_max_strict[group_id]
        s_group_max = group_max[group_id]
        dominated_sorted = (s_fid <= s_prev) | (s_fid < s_group_max)
        is_pareto[order[dominated_sorted]] = False

    fig = go.Figure()

    dominated_mask = ~is_pareto
    # Scattergl for the (typically very large) dominated cloud — WebGL
    # scales to ~100 k markers; plain Scatter (SVG) freezes the browser
    # past a few thousand. The Pareto front itself is computed on the full
    # grid above; we only sample the cloud display.
    dom_x = epr[dominated_mask]
    dom_y = fidelity[dominated_mask]
    dom_full = dom_x.size
    if dom_x.size > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(0)
        sample = rng.choice(dom_x.size, size=_MAX_PLOT_POINTS, replace=False)
        dom_x = dom_x[sample]
        dom_y = dom_y[sample]
    fig.add_trace(
        go.Scattergl(
            x=dom_x, y=dom_y,
            mode="markers",
            marker=dict(size=5, color="#CCCCCC", opacity=0.5),
            name="Dominated",
            hovertemplate="EPR: %{x:.0f}<br>Fidelity: <b>%{y:.4f}</b><extra></extra>",
        )
    )

    pareto_idx = np.where(is_pareto)[0]
    pareto_order = np.argsort(epr[pareto_idx])
    pareto_sorted = pareto_idx[pareto_order]

    fig.add_trace(
        go.Scatter(
            x=epr[pareto_sorted], y=fidelity[pareto_sorted],
            mode="lines+markers",
            line=dict(color="#4575b4", width=2),
            marker=dict(size=7, color="#4575b4", line=dict(width=1, color="#FFFFFF")),
            name="Pareto front",
            hovertemplate="EPR: %{x:.0f}<br>Fidelity: <b>%{y:.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=55, r=20, t=50, b=50)},
        xaxis=dict(
            title=dict(text="Total EPR Pairs", font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
        ),
        yaxis=dict(
            title=dict(text="Overall Fidelity", font=dict(size=12, color=_TEXT_MUTED)),
            gridcolor=_GRID_COLOR,
            tickfont=dict(size=10, color=_TEXT_MUTED),
            range=[0, 1],
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=_GRID_COLOR, font_color=_TEXT_COLOR),
    )

    if thresholds:
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
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        return ranks

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


def plot_correlation(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if len(rows) == 0:
        return plot_empty("No data for correlation matrix")

    data = rows  # already an ndarray from _flatten_sweep_to_table
    col_names = metric_keys + available_outputs
    labels = []
    for name in col_names:
        m = METRIC_BY_KEY.get(name) or CAT_METRIC_BY_KEY.get(name)
        labels.append(m.label if m else _OUTPUT_LABELS.get(name, name))

    n = len(col_names)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _spearman_corr(data[:, i], data[:, j])
            corr[i, j] = c
            corr[j, i] = c

    _DIVERGING = [
        [0.0, "#d73027"],
        [0.25, "#fc8d59"],
        [0.5, "#FFFFFF"],
        [0.75, "#91bfdb"],
        [1.0, "#4575b4"],
    ]

    annotations = []
    for i in range(n):
        for j in range(n):
            annotations.append(dict(
                x=j, y=i,
                text=f"{corr[i, j]:.2f}",
                showarrow=False,
                font=dict(size=9, color=_TEXT_COLOR if abs(corr[i, j]) < 0.7 else "#FFFFFF"),
            ))

    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=labels, y=labels,
            zmin=-1.0, zmax=1.0,
            colorscale=_DIVERGING,
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

    # Compute global zmin/zmax for shared colorscale (2D views).
    global_zmin = float("inf")
    global_zmax = float("-inf")
    if num_metrics == 2:
        for facet in facets:
            grid = facet.get("grid", [])
            for row_data in grid:
                for cell in row_data:
                    v = float(cell.get(output_key, 0.0)) if isinstance(cell, dict) else 0.0
                    if v < global_zmin:
                        global_zmin = v
                    if v > global_zmax:
                        global_zmax = v

    # For 3D faceted views, divide the point budget across facets so the
    # combined figure stays within the browser's WebGL/JSON limits.
    facet_3d_cap = _MAX_BROWSER_3D_POINTS // max(n_facets, 1) if is_3d else None

    for idx, facet in enumerate(facets):
        r = idx // cols + 1
        c = idx % cols + 1

        # Build a single-facet figure, then transplant traces into the grid.
        panel_fig = build_figure(
            num_metrics, facet, output_key,
            view_type=view_type,
            thresholds=thresholds,
            threshold_colors=threshold_colors,
            _3d_point_cap=facet_3d_cap,
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
) -> go.Figure:
    if sweep_data is None:
        return plot_empty()

    # Faceted data → delegate to subplot grid builder for spatial views,
    # or flatten into a single dataset for analysis views.
    if "facets" in sweep_data and sweep_data["facets"]:
        _analysis_views = {"parallel", "slices", "importance", "pareto", "correlation"}
        if view_type in _analysis_views:
            sweep_data = _flatten_facets_for_analysis(sweep_data)
        else:
            return _build_faceted_figure(
                num_metrics, sweep_data, output_key,
                view_type=view_type,
                thresholds=thresholds,
                threshold_colors=threshold_colors,
            )

    _tc = threshold_colors

    fig: go.Figure

    if view_type == "parallel":
        fig = plot_parallel_coordinates(sweep_data, output_key)
    elif view_type == "slices":
        fig = plot_slice(sweep_data, output_key)
    elif view_type == "importance":
        fig = plot_importance(sweep_data, output_key)
    elif view_type == "pareto":
        fig = plot_pareto(sweep_data, output_key, thresholds=thresholds, threshold_colors=_tc)
    elif view_type == "correlation":
        fig = plot_correlation(sweep_data, output_key)
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
                )
            elif num_metrics == 2:
                if view_type == "contour":
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
                else:
                    fig = plot_2d(
                        x_values=np.array(sweep_data["xs"]),
                        y_values=np.array(sweep_data["ys"]),
                        grid=sweep_data["grid"],
                        metric_key1=sweep_data["metric_keys"][0],
                        metric_key2=sweep_data["metric_keys"][1],
                        output_key=output_key,
                    )
            elif num_metrics == 3 and view_type in ("frozen_heatmap", "frozen_contour"):
                from gui.interpolation import (
                    frozen_slice,
                    frozen_view_base,
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
                base_view = frozen_view_base(view_type)
                slice_grid = [[{output_key: float(slice_2d[j, i])}
                               for j in range(slice_2d.shape[0])]
                              for i in range(slice_2d.shape[1])]
                if base_view == "contour":
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
                else:
                    fig = plot_2d(
                        x_values=np.array(sweep_data["xs"]),
                        y_values=np.array(sweep_data["ys"]),
                        grid=slice_grid,
                        metric_key1=sweep_data["metric_keys"][0],
                        metric_key2=sweep_data["metric_keys"][1],
                        output_key=output_key,
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
