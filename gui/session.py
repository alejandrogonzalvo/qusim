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
