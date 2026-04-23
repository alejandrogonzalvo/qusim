"""Save / load session state for the DSE GUI.

A *session* captures every input control, the current view state, and (when
present) the most recent sweep result so the user can reopen exactly what they
were looking at. See ``docs/plans/2026-04-23-save-load-sessions-design.md``.

This module is intentionally Dash-free: ``collect_session`` and
``apply_session`` take plain dicts so they can be unit-tested without spinning
up an app.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def collect_session(
    controls: dict,
    view: dict,
    sweep_data: dict | None,
) -> dict:
    """Build the JSON-shaped session dict.

    ``controls`` and ``view`` are stored verbatim (the shape is frozen by the
    schema; the caller is responsible for building the right dict). ``sweep_data``
    is the full sweep result as stored in ``_SWEEP_CACHE``; pass ``None`` if no
    sweep has been run.
    """
    sweep_block: dict[str, Any]
    if sweep_data is None:
        sweep_block = {"present": False}
    else:
        sweep_block = {"present": True}
        for k in (
            "metric_keys", "xs", "ys", "zs", "axes", "shape",
            "grid", "facets", "facet_keys",
        ):
            if k in sweep_data:
                sweep_block[k] = sweep_data[k]

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "app": {"name": "qusim-dse"},
        "controls": controls,
        "view": view,
        "sweep": sweep_block,
    }


class SessionError(ValueError):
    """Raised when a session payload cannot be used."""


_REQUIRED_TOP_LEVEL = ("schema_version", "controls", "view", "sweep")


def validate(session: Any) -> None:
    """Raise ``SessionError`` if *session* is not a loadable session dict."""
    if not isinstance(session, dict):
        raise SessionError(f"session must be a dict, got {type(session).__name__}")
    for k in _REQUIRED_TOP_LEVEL:
        if k not in session:
            raise SessionError(f"session is missing required key: {k!r}")
    version = session["schema_version"]
    if version != SCHEMA_VERSION:
        raise SessionError(
            f"unsupported session schema version {version!r} "
            f"(this build reads version {SCHEMA_VERSION})"
        )


def dump(session: dict) -> bytes:
    """Serialize *session* to gzipped JSON bytes."""
    return gzip.compress(
        json.dumps(session, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        compresslevel=6,
    )


def load(raw: bytes | str) -> dict:
    """Parse a session file. Accepts gzipped bytes, raw JSON bytes, or JSON str."""
    if isinstance(raw, str):
        return json.loads(raw)
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


@dataclass
class SessionApply:
    """Result of :func:`apply_session`.

    ``controls`` and ``view`` are the filtered dicts ready to be fanned out to
    Dash Outputs. ``sweep_data`` is the sweep result (or ``None``). ``warnings``
    lists human-readable notes about any dropped fields; the Dash layer surfaces
    them in the error banner on a best-effort basis.

    The returned dicts are shallow copies: nested values may alias the loaded
    session. Treat nested values as read-only.
    """

    controls: dict
    view: dict
    sweep_data: dict | None
    warnings: list[str] = field(default_factory=list)


def apply_session(session: Any) -> SessionApply:
    """Validate *session* and return a :class:`SessionApply`."""
    # Local import avoids a load-time cycle: constants.py imports nothing here.
    from gui.constants import CAT_METRIC_BY_KEY, METRIC_BY_KEY

    validate(session)

    ctrls = dict(session["controls"])
    view = dict(session["view"])
    warnings: list[str] = []

    # Drop axes whose metric key is no longer known to this build.
    axes_in = ctrls.get("axes", [])
    known = set(METRIC_BY_KEY) | set(CAT_METRIC_BY_KEY)
    axes_out = []
    for ax in axes_in:
        k = ax.get("key")
        if k in known:
            axes_out.append(ax)
        else:
            warnings.append(
                f"dropped sweep axis with unknown metric key {k!r}"
            )
    ctrls["axes"] = axes_out
    ctrls["num_metrics"] = len(axes_out)

    sweep_block = session.get("sweep", {})
    sweep_data: dict | None
    if sweep_block.get("present"):
        sweep_data = {
            k: sweep_block[k]
            for k in (
                "metric_keys", "xs", "ys", "zs", "axes", "shape",
                "grid", "facets", "facet_keys",
            )
            if k in sweep_block
        }
    else:
        sweep_data = None

    return SessionApply(
        controls=ctrls, view=view, sweep_data=sweep_data, warnings=warnings,
    )


def build_controls_dict(
    *,
    num_metrics: int,
    dropdown_vals: list,
    slider_vals: list,
    checklist_vals: list,
    cfg_circuit_type: str,
    cfg_num_qubits: int,
    cfg_num_cores: int,
    cfg_topology: str,
    cfg_intracore_topology: str,
    cfg_placement: str,
    cfg_routing_algorithm: str,
    cfg_seed: int,
    cfg_dynamic_decoupling: list,
    cfg_max_cold: int | None,
    cfg_max_hot: int | None,
    cfg_max_workers: int | None,
    cfg_output_metric: str,
    cfg_threshold_enable: list,
    num_thresholds: int,
    threshold_values: list,
    threshold_colors: list,
    noise_values: dict,
    hot_reload: list,
) -> dict:
    """Assemble the schema-shaped ``controls`` dict from raw callback values."""
    axes = []
    for i in range(int(num_metrics or 0)):
        key = dropdown_vals[i]
        if key is None:
            continue
        axes.append({
            "key": key,
            "slider": list(slider_vals[i]) if slider_vals[i] is not None else None,
            "checklist": list(checklist_vals[i]) if checklist_vals[i] else None,
        })

    return {
        "num_metrics": len(axes),
        "axes": axes,
        "circuit": {
            "num_qubits": cfg_num_qubits,
            "num_cores": cfg_num_cores,
            "seed": cfg_seed,
            "circuit_type": cfg_circuit_type,
            "topology_type": cfg_topology,
            "intracore_topology": cfg_intracore_topology,
            "placement": cfg_placement,
            "routing_algorithm": cfg_routing_algorithm,
            "dynamic_decoupling": bool(cfg_dynamic_decoupling and "yes" in cfg_dynamic_decoupling),
        },
        "noise": dict(noise_values),
        "thresholds": {
            "output_metric": cfg_output_metric,
            "enable": bool(cfg_threshold_enable and "yes" in cfg_threshold_enable),
            "num_thresholds": int(num_thresholds or 3),
            "values": list(threshold_values),
            "colors": list(threshold_colors),
        },
        "performance": {
            "max_cold": cfg_max_cold,
            "max_hot": cfg_max_hot,
            "max_workers": cfg_max_workers,
        },
        "hot_reload": bool(hot_reload and "on" in hot_reload),
    }


def build_view_dict(
    view_type: str,
    frozen_axis: int,
    frozen_slider_value: float | None,
) -> dict:
    return {
        "view_type": view_type,
        "frozen_axis": frozen_axis,
        "frozen_slider_value": frozen_slider_value,
    }
