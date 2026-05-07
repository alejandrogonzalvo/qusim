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
import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _strip_for_save(facet: dict) -> dict:
    """Drop fields that aren't JSON-serializable from a facet dict.

    ``per_qubit_data["cells"]`` is keyed by tuple cell indices and holds
    numpy ndarrays — neither survives ``json.dumps``. Per-cell grids are
    also large (~MB-scale) and cheaply regenerable from a fresh sweep, so
    we omit them and keep only the lightweight axis metadata that the
    topology view needs to recreate the slider scaffolding on load.
    """
    out = dict(facet)
    pq = out.get("per_qubit_data")
    if isinstance(pq, dict) and "cells" in pq:
        pq = {k: v for k, v in pq.items() if k != "cells"}
        out["per_qubit_data"] = pq
    return out


def collect_session(
    controls: dict,
    view: dict,
    sweep_data: dict | None,
    name: str = "",
) -> dict:
    """Build the JSON-shaped session dict.

    ``controls`` and ``view`` are stored verbatim (the shape is frozen by the
    schema; the caller is responsible for building the right dict). ``sweep_data``
    is the full sweep result as stored in ``_SWEEP_CACHE``; pass ``None`` if no
    sweep has been run. ``name`` is the user-supplied session title; empty
    string is the "Untitled" default.
    """
    sweep_block: dict[str, Any]
    if sweep_data is None:
        sweep_block = {"present": False}
    else:
        sweep_block = {"present": True}
        for k in (
            "metric_keys", "xs", "ys", "zs", "axes", "shape",
            "grid", "facets", "facet_keys", "per_qubit_data",
        ):
            if k not in sweep_data:
                continue
            v = sweep_data[k]
            # Strip unserializable per-cell grids inside each facet so
            # faceted sweeps round-trip through ``json.dumps``.
            if k == "facets" and isinstance(v, list):
                v = [_strip_for_save(f) if isinstance(f, dict) else f for f in v]
            elif k == "per_qubit_data" and isinstance(v, dict):
                # Drop ``cells`` (heavy per-cell ndarrays, also non-JSON);
                # keep ``axis_keys`` / ``axis_values`` / ``shape`` /
                # ``cold_config`` / ``fixed_noise`` so the topology view
                # can rebuild the scrub-slider scaffolding after a load
                # and re-derive each cell on demand via run_cold/run_hot.
                v = {kk: vv for kk, vv in v.items() if kk != "cells"}
            sweep_block[k] = v

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": _utc_now_iso(),
        "app": {"name": "quadris-dse"},
        "name": name or "",
        "controls": controls,
        "view": view,
        "sweep": sweep_block,
    }


_FILENAME_SAFE_RE = re.compile(r"[^\w.-]+", flags=re.UNICODE)


def sanitize_filename(name: str) -> str:
    """Return a safe filename stem for a user-supplied session title.

    Replaces filesystem-unsafe characters and whitespace with hyphens, collapses
    runs of hyphens, and falls back to ``"quadris-session"`` when the result is
    empty.
    """
    cleaned = _FILENAME_SAFE_RE.sub("-", (name or "").strip())
    cleaned = cleaned.strip("-")
    # Collapse any run of hyphens to a single one.
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "quadris-session"


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
    name: str = ""
    warnings: list[str] = field(default_factory=list)


def apply_session(session: Any) -> SessionApply:
    """Validate *session* and return a :class:`SessionApply`."""
    # Local import avoids a load-time cycle: constants.py imports nothing here.
    from gui.constants import CAT_METRIC_BY_KEY, METRIC_BY_KEY

    validate(session)

    ctrls = dict(session["controls"])
    view = dict(session["view"])
    name = session.get("name", "") or ""
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
                "grid", "facets", "facet_keys", "per_qubit_data",
            )
            if k in sweep_block
        }
    else:
        sweep_data = None

    return SessionApply(
        controls=ctrls, view=view, sweep_data=sweep_data,
        name=name, warnings=warnings,
    )


def build_controls_dict(
    *,
    num_metrics: int,
    dropdown_vals: list,
    slider_vals: list,
    checklist_vals: list,
    cfg_circuit_type: str,
    cfg_qubits_per_core: int,
    cfg_num_cores: int,
    cfg_topology: str,
    cfg_intracore_topology: str,
    cfg_placement: str,
    cfg_routing_algorithm: str,
    cfg_seed: int,
    cfg_dynamic_decoupling: list,
    cfg_communication_qubits: int = 1,
    cfg_buffer_qubits: int = 1,
    cfg_num_logical_qubits: int | None = None,
    cfg_pin_axis: str = "cores",
    cfg_max_cold: int | None = None,
    cfg_max_hot: int | None = None,
    cfg_max_workers: int | None = None,
    cfg_output_metric: str = "overall_fidelity",
    cfg_view_mode: str = "absolute",
    cfg_threshold_enable: list | None = None,
    num_thresholds: int = 3,
    threshold_values: list | None = None,
    threshold_colors: list | None = None,
    noise_values: dict | None = None,
    hot_reload: list | None = None,
    fom_config: dict | None = None,
) -> dict:
    """Assemble the schema-shaped ``controls`` dict from raw callback values."""
    cfg_threshold_enable = cfg_threshold_enable or []
    threshold_values = threshold_values or []
    threshold_colors = threshold_colors or []
    noise_values = noise_values or {}
    hot_reload = hot_reload or []

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
            "qubits_per_core": int(cfg_qubits_per_core or 16),
            "num_cores": int(cfg_num_cores or 1),
            "pin_axis": cfg_pin_axis or "cores",
            "communication_qubits": int(cfg_communication_qubits or 1),
            "buffer_qubits": int(cfg_buffer_qubits or 1),
            "num_logical_qubits": int(
                cfg_num_logical_qubits if cfg_num_logical_qubits is not None
                else (cfg_qubits_per_core or 16)
            ),
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
            "view_mode": cfg_view_mode or "absolute",
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
        "fom": fom_config or None,
    }


def build_view_dict(
    view_type: str,
    frozen_axis: int,
    frozen_slider_value: float | None,
    pareto_x: str | None = None,
    pareto_y: str | None = None,
) -> dict:
    out: dict = {
        "view_type": view_type,
        "frozen_axis": frozen_axis,
        "frozen_slider_value": frozen_slider_value,
    }
    # Pareto-axis keys are only saved when an example explicitly sets
    # them (they aren't user-saved through the GUI yet).  Their absence
    # keeps the GUI defaults (``total_epr_pairs`` × ``overall_fidelity``).
    if pareto_x is not None:
        out["pareto_x"] = pareto_x
    if pareto_y is not None:
        out["pareto_y"] = pareto_y
    return out
