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
) -> go.Figure:
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
) -> go.Figure:
    m1 = METRIC_BY_KEY.get(metric_key1)
    m2 = METRIC_BY_KEY.get(metric_key2)
    m3 = METRIC_BY_KEY.get(metric_key3)

    # For 3D, fix the middle value of the 3rd axis and show a surface,
    # then provide a slider note. Full volumetric 3D iso-surface is complex;
    # instead we render a scatter3d for all points coloured by fidelity.
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

    fmin = 0.0 if "fidelity" in output_key else min(fs_all)
    fmax = 1.0 if "fidelity" in output_key else max(fs_all)

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

    fig = go.Figure(
        go.Scatter3d(
            x=xs_all, y=ys_all, z=zs_all,
            mode="markers",
            marker=dict(
                size=3.5,
                color=fs_all,
                cmin=fmin, cmax=fmax,
                colorscale=_COLORSCALE,
                colorbar=dict(
                    title=dict(text=_OUTPUT_LABELS.get(output_key, output_key),
                               font=dict(size=11, color=_TEXT_MUTED)),
                    tickfont=dict(color=_TEXT_MUTED, size=10),
                    outlinewidth=0, thickness=14, len=0.75,
                ),
                opacity=0.85,
                line=dict(width=0),
            ),
            hovertemplate=(
                x_title + ": %{x:.3g}<br>"
                + y_title + ": %{y:.3g}<br>"
                + z_title + ": %{z:.3g}<br>"
                + "<b>fidelity: %{marker.color:.4f}</b><extra></extra>"
            ),
        )
    )

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

def build_figure(
    num_metrics: int,
    sweep_data: dict,
    output_key: str,
) -> go.Figure:
    """
    Route to the correct plot type based on the number of active sweep metrics.

    ``sweep_data`` is the dict stored in the dcc.Store after a sweep completes.
    Keys: xs, ys, zs, grid, metric_keys  (only the ones used for this sweep).
    """
    if sweep_data is None:
        return plot_empty()

    try:
        if num_metrics == 1:
            return plot_1d(
                x_values=np.array(sweep_data["xs"]),
                results=sweep_data["grid"],
                metric_key=sweep_data["metric_keys"][0],
                output_key=output_key,
            )
        elif num_metrics == 2:
            return plot_2d(
                x_values=np.array(sweep_data["xs"]),
                y_values=np.array(sweep_data["ys"]),
                grid=sweep_data["grid"],
                metric_key1=sweep_data["metric_keys"][0],
                metric_key2=sweep_data["metric_keys"][1],
                output_key=output_key,
            )
        elif num_metrics == 3:
            return plot_3d(
                x_values=np.array(sweep_data["xs"]),
                y_values=np.array(sweep_data["ys"]),
                z_values=np.array(sweep_data["zs"]),
                grid=sweep_data["grid"],
                metric_key1=sweep_data["metric_keys"][0],
                metric_key2=sweep_data["metric_keys"][1],
                metric_key3=sweep_data["metric_keys"][2],
                output_key=output_key,
            )
        else:
            return plot_empty("Add 1–3 metric axes on the left to start a sweep")
    except Exception as exc:
        return plot_empty(f"Plot error: {exc}")
