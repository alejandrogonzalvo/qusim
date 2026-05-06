"""Pure formatting / conversion helpers used across callback modules.

Everything here is referentially transparent (or close to it): no Dash,
no global state, no side effects beyond reading the constants tables.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from gui.components import COLORS
from gui.constants import CAT_METRIC_BY_KEY, METRIC_BY_KEY
from gui.server_state import _progress_label


# ---------------------------------------------------------------------------
# Slider <-> physical value
# ---------------------------------------------------------------------------

def _slider_to_value(slider_pos: float, log_scale: bool) -> float:
    return 10.0**slider_pos if log_scale else slider_pos


def _value_to_slider(val: float, log_scale: bool) -> float:
    """Inverse of ``_slider_to_value``."""
    if log_scale and val is not None and val > 0:
        return math.log10(val)
    return val


# ---------------------------------------------------------------------------
# Sweep result helpers
# ---------------------------------------------------------------------------

def _result_to_dict(r: Any) -> dict:
    if isinstance(r, dict):
        return {
            k: float(v)
            for k, v in r.items()
            if isinstance(v, (int, float, np.floating))
        }
    return {
        "overall_fidelity": float(r.overall_fidelity),
        "algorithmic_fidelity": float(r.algorithmic_fidelity),
        "routing_fidelity": float(r.routing_fidelity),
        "coherence_fidelity": float(r.coherence_fidelity),
        "total_circuit_time_ns": float(r.total_circuit_time_ns),
        "total_epr_pairs": float(getattr(r, "total_epr_pairs", 0)),
        "total_swaps": float(getattr(r, "total_swaps", 0)),
        "total_teleportations": float(getattr(r, "total_teleportations", 0)),
        "total_network_distance": float(getattr(r, "total_network_distance", 0)),
    }


def _count_points(sweep_data: dict) -> int:
    if "facets" in sweep_data:
        return sum(_count_points(f) for f in sweep_data["facets"])
    if "shape" in sweep_data:
        result = 1
        for s in sweep_data["shape"]:
            result *= s
        return result
    if "axes" in sweep_data:
        result = 1
        for ax in sweep_data["axes"]:
            result *= len(ax)
        return result
    xs = sweep_data.get("xs", [])
    ys = sweep_data.get("ys", [xs])
    zs = sweep_data.get("zs", [xs])
    return len(xs) * len(ys) * len(zs)


# ---------------------------------------------------------------------------
# Facet helpers
# ---------------------------------------------------------------------------

def _facet_options(sweep_data: dict | None) -> list[dict]:
    """Build dropdown options for the topology facet selector.

    Returns one option per facet of a faceted sweep (label = the facet's
    human-readable categorical value, e.g. "HQA + Sabre"). Empty list for
    non-faceted sweeps so the caller can hide the row.
    """
    if not isinstance(sweep_data, dict):
        return []
    facets = sweep_data.get("facets")
    if not facets:
        return []
    facet_keys = sweep_data.get("facet_keys") or []
    opts: list[dict] = []
    for i, fc in enumerate(facets):
        label_dict = fc.get("label") or {}
        # Build "Routing: HQA + Sabre · Circuit: QFT" style label.
        parts: list[str] = []
        for k in facet_keys:
            v = label_dict.get(k)
            if v is None:
                continue
            cat_def = CAT_METRIC_BY_KEY.get(k)
            disp = v
            if cat_def:
                disp = next(
                    (o["label"] for o in cat_def.options if o["value"] == v), v
                )
            parts.append(str(disp))
        opts.append({"label": " · ".join(parts) or f"Facet {i}", "value": i})
    return opts


def _facet_per_qubit_data(
    sweep_data: dict | None, facet_idx: int | None,
) -> dict | None:
    """Return the per_qubit_data dict for the selected facet (or top-level).

    Faceted sweeps stash per_qubit_data inside each facet entry; non-faceted
    sweeps keep it at the top level.
    """
    if not isinstance(sweep_data, dict):
        return None
    facets = sweep_data.get("facets")
    if facets:
        i = int(facet_idx or 0)
        if i < 0 or i >= len(facets):
            i = 0
        return facets[i].get("per_qubit_data")
    return sweep_data.get("per_qubit_data")


# ---------------------------------------------------------------------------
# FoM / threshold / merit helpers
# ---------------------------------------------------------------------------

def _parse_intermediates(text: str) -> list[tuple[str, str]]:
    """Parse the textarea format (``name = expr`` per line) into tuples."""
    out: list[tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, expr = line.split("=", 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            continue
        out.append((name, expr))
    return out


def _resolve_thresholds(
    threshold_enable, all_t, all_c,
) -> tuple[list[float] | None, list[str] | None]:
    """Filter the right-sidebar threshold inputs into ``(values, colors)``.

    Returns ``(None, None)`` if the user disabled thresholds; otherwise both
    lists are aligned and contain only the populated entries.
    """
    if not (threshold_enable and "yes" in (threshold_enable or [])):
        return None, None
    vals: list[float] = []
    colors: list[str] = []
    for v, c in zip(all_t, all_c):
        if v is None:
            continue
        vals.append(float(v))
        colors.append(c or "")
    return (vals or None), (colors or None)


def _frozen_values_from_sliders(metric_keys, slider_values) -> dict[str, float]:
    """Map pattern-matched slider values back to ``{axis_key: value}``.

    Slider index ``i`` corresponds to ``metric_keys[i]`` while the axis is
    active; out-of-range or ``None`` values are dropped.
    """
    out: dict[str, float] = {}
    for i, val in enumerate(slider_values or []):
        if i >= len(metric_keys) or val is None:
            continue
        try:
            out[metric_keys[i]] = float(val)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Dropdown / slider mark builders
# ---------------------------------------------------------------------------

def _axis_dropdown_options(metric_keys: list[str]) -> list[dict]:
    return [{"label": _progress_label(k), "value": k} for k in metric_keys]


def _slider_marks(values: list[float]) -> dict:
    """Build slider tick marks for a discrete grid; label every value when ≤8,
    otherwise label first/last + a few interior ticks.
    """
    if not values:
        return {}
    style = {"fontSize": "9px", "color": COLORS["text_muted"]}
    if len(values) <= 8:
        return {float(v): {"label": f"{v:g}", "style": style} for v in values}
    indices = sorted({
        0, len(values) - 1,
        len(values) // 4, len(values) // 2, (3 * len(values)) // 4,
    })
    return {float(values[i]): {"label": f"{values[i]:g}", "style": style}
            for i in indices}


# ---------------------------------------------------------------------------
# Value formatters
# ---------------------------------------------------------------------------

def _human_axis_value(axis_key: str, raw_val) -> str:
    """Format a swept axis value for the slider read-out."""
    metric = METRIC_BY_KEY.get(axis_key)
    if metric is None:
        return str(raw_val)
    try:
        v = float(raw_val)
    except (TypeError, ValueError):
        return str(raw_val)
    unit = metric.unit
    suffix = f" {unit}" if unit else ""
    if metric.is_cold_path or not metric.log_scale:
        if v == int(v):
            return f"{int(v)}{suffix}"
        return f"{v:.3f}{suffix}"
    # Hot-path log axes have already been materialised by ``np.logspace``
    # — values are physical magnitudes, not exponents.
    if unit == "ns":
        if v >= 1e6:
            return f"{v/1e6:.2f} ms"
        if v >= 1e3:
            return f"{v/1e3:.2f} µs"
        return f"{v:.0f} ns"
    if unit == "Hz":
        if v >= 1e9:
            return f"{v/1e9:.2f} GHz"
        if v >= 1e6:
            return f"{v/1e6:.2f} MHz"
        return f"{v:.0f} Hz"
    return f"{v:.3g}"


def _fmt_fid(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v >= 0.9999:
        return "1.000"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.4f}"
