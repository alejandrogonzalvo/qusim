"""
Plotly figure builders for 1-D, 2-D, and 3-D fidelity sweep results.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from .constants import METRIC_BY_KEY, OUTPUT_METRICS

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
    m = METRIC_BY_KEY.get(metric_key)
    if m is None:
        return metric_key
    unit = f" ({m.unit})" if m.unit else ""
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
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    m1 = METRIC_BY_KEY.get(metric_key1)
    m2 = METRIC_BY_KEY.get(metric_key2)
    m3 = METRIC_BY_KEY.get(metric_key3)

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
        frange = fmax - fmin
        band = max(frange * 0.05, 0.01)
        for i, t in enumerate(thresholds):
            near = np.abs(fs_all - t) <= band
            if not near.any():
                near = np.zeros(len(fs_all), dtype=bool)
                near[np.argmin(np.abs(fs_all - t))] = True
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
) -> go.Figure:
    _colors = threshold_colors or _THRESHOLD_COLORS
    total_points = len(x_values) * len(y_values) * len(z_values)

    if total_points < _MIN_GRID_FOR_ISOSURFACE:
        return plot_3d(x_values, y_values, z_values, grid,
                       metric_key1, metric_key2, metric_key3, output_key,
                       thresholds=thresholds, threshold_colors=threshold_colors)

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


def _flatten_sweep_to_table(sweep_data: dict) -> tuple[list[str], list[str], list[list[float]]]:
    metric_keys = sweep_data["metric_keys"]
    grid = sweep_data["grid"]
    xs = sweep_data["xs"]

    available_outputs = []
    sample = grid[0] if isinstance(grid[0], dict) else None
    if sample is None:
        nested = grid[0]
        while isinstance(nested, list):
            nested = nested[0]
        sample = nested
    for k in _OUTPUT_KEYS:
        if k in sample:
            available_outputs.append(k)

    rows: list[list[float]] = []
    ndim = len(metric_keys)

    if ndim == 1:
        for i, x in enumerate(xs):
            r = grid[i]
            row = [float(x)]
            for k in available_outputs:
                row.append(float(r.get(k, 0.0) if isinstance(r, dict) else getattr(r, k, 0.0)))
            rows.append(row)
    elif ndim == 2:
        ys = sweep_data["ys"]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                r = grid[i][j]
                row = [float(x), float(y)]
                for k in available_outputs:
                    row.append(float(r.get(k, 0.0) if isinstance(r, dict) else getattr(r, k, 0.0)))
                rows.append(row)
    elif ndim == 3:
        ys = sweep_data["ys"]
        zs = sweep_data["zs"]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for ki, z in enumerate(zs):
                    r = grid[i][j][ki]
                    row = [float(x), float(y), float(z)]
                    for k in available_outputs:
                        row.append(float(r.get(k, 0.0) if isinstance(r, dict) else getattr(r, k, 0.0)))
                    rows.append(row)

    return metric_keys, available_outputs, rows


def plot_parallel_coordinates(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if not rows:
        return plot_empty("No data for parallel coordinates")

    data = np.array(rows)
    col_names = metric_keys + available_outputs
    num_param_cols = len(metric_keys)

    color_col_idx = None
    if output_key in available_outputs:
        color_col_idx = num_param_cols + available_outputs.index(output_key)
    elif available_outputs:
        color_col_idx = num_param_cols

    dimensions = []
    for i, name in enumerate(col_names):
        m = METRIC_BY_KEY.get(name)
        label = m.label if m else _OUTPUT_LABELS.get(name, name)
        col = data[:, i]
        dimensions.append(dict(
            label=label,
            values=col.tolist(),
            range=[float(col.min()), float(col.max())],
        ))

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
            labelfont=dict(size=11, color=_TEXT_COLOR),
            tickfont=dict(size=9, color=_TEXT_MUTED),
            rangefont=dict(size=9, color=_TEXT_MUTED),
        )
    )

    fig.update_layout(
        **{**_LAYOUT_BASE, "margin": dict(l=60, r=30, t=50, b=30)},
    )
    return fig


# ---------------------------------------------------------------------------
# Slice plot (marginal effects — one subplot per swept parameter)
# ---------------------------------------------------------------------------

def plot_slice(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if not rows:
        return plot_empty("No data for slice plot")

    data = np.array(rows)
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
        subplot_titles=[METRIC_BY_KEY[k].label if k in METRIC_BY_KEY else k for k in metric_keys],
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
    if is_fidelity:
        fig.update_yaxes(title_text=_OUTPUT_LABELS.get(output_key, output_key), row=1, col=1)

    return fig


# ---------------------------------------------------------------------------
# Parameter importance (range-based sensitivity — horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_importance(sweep_data: dict, output_key: str) -> go.Figure:
    metric_keys, available_outputs, rows = _flatten_sweep_to_table(sweep_data)

    if not rows:
        return plot_empty("No data for importance plot")

    data = np.array(rows)
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
        m = METRIC_BY_KEY.get(param_key)
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

    if not rows:
        return plot_empty("No data for Pareto plot")

    data = np.array(rows)
    num_params = len(metric_keys)

    fid_col = num_params + available_outputs.index("overall_fidelity") if "overall_fidelity" in available_outputs else None
    epr_col = num_params + available_outputs.index("total_epr_pairs") if "total_epr_pairs" in available_outputs else None

    if fid_col is None or epr_col is None:
        return plot_empty("Need both fidelity and EPR metrics for Pareto")

    fidelity = data[:, fid_col]
    epr = data[:, epr_col]

    is_pareto = np.ones(len(data), dtype=bool)
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            if fidelity[j] >= fidelity[i] and epr[j] <= epr[i] and (fidelity[j] > fidelity[i] or epr[j] < epr[i]):
                is_pareto[i] = False
                break

    fig = go.Figure()

    dominated_mask = ~is_pareto
    fig.add_trace(
        go.Scatter(
            x=epr[dominated_mask], y=fidelity[dominated_mask],
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

    if not rows:
        return plot_empty("No data for correlation matrix")

    data = np.array(rows)
    col_names = metric_keys + available_outputs
    labels = []
    for name in col_names:
        m = METRIC_BY_KEY.get(name)
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


def build_figure(
    num_metrics: int,
    sweep_data: dict,
    output_key: str,
    view_type: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
) -> go.Figure:
    if sweep_data is None:
        return plot_empty()

    _tc = threshold_colors

    if view_type == "parallel":
        return plot_parallel_coordinates(sweep_data, output_key)
    if view_type == "slices":
        return plot_slice(sweep_data, output_key)
    if view_type == "importance":
        return plot_importance(sweep_data, output_key)
    if view_type == "pareto":
        return plot_pareto(sweep_data, output_key, thresholds=thresholds, threshold_colors=_tc)
    if view_type == "correlation":
        return plot_correlation(sweep_data, output_key)

    try:
        if num_metrics == 1:
            return plot_1d(
                x_values=np.array(sweep_data["xs"]),
                results=sweep_data["grid"],
                metric_key=sweep_data["metric_keys"][0],
                output_key=output_key,
                thresholds=thresholds,
                threshold_colors=_tc,
            )
        elif num_metrics == 2:
            if view_type == "contour":
                return plot_2d_contour(
                    x_values=np.array(sweep_data["xs"]),
                    y_values=np.array(sweep_data["ys"]),
                    grid=sweep_data["grid"],
                    metric_key1=sweep_data["metric_keys"][0],
                    metric_key2=sweep_data["metric_keys"][1],
                    output_key=output_key,
                    thresholds=thresholds,
                    threshold_colors=_tc,
                )
            return plot_2d(
                x_values=np.array(sweep_data["xs"]),
                y_values=np.array(sweep_data["ys"]),
                grid=sweep_data["grid"],
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
                return plot_3d_isosurface(**_3d_args)
            return plot_3d(**_3d_args)
        else:
            return plot_empty("Add 1\u20133 metric axes on the left to start a sweep")
    except Exception as exc:
        return plot_empty(f"Plot error: {exc}")
