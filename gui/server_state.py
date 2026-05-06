"""In-memory server state for the Dash GUI.

Owns the long-lived singletons that survive between callbacks: the DSE
engine, sweep result cache, per-cell fidelity cache, per-user sweep
progress state, and the ``/api/progress`` Flask route. Kept in one module
so the rest of ``gui/`` can import what it needs without picking up the
rest of ``app.py``'s side effects.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

from gui.constants import CAT_METRIC_BY_KEY, METRIC_BY_KEY
from gui.dse_engine import DSEEngine, SweepProgress

# ---------------------------------------------------------------------------
# Engine + global sweep lock
# ---------------------------------------------------------------------------

_engine = DSEEngine()
sweep_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Server-side sweep cache
#
# The browser store (``sweep-result-store``) used to carry the full grid
# (~12 MB for a 50k-point 3D sweep). Every downstream callback re-parsed
# that blob on the browser main thread. Instead we keep the full grid
# server-side and only send the browser a small token + axes metadata;
# callbacks that need the grid fetch it back via ``_get_sweep``.
# ---------------------------------------------------------------------------

# Sized for ~3 concurrent users keeping their last few sweeps in memory.
_SWEEP_CACHE_MAX = 12
_SWEEP_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_sweep_token_counter = 0
_sweep_cache_lock = threading.Lock()

_session_load_counter = 0
_session_load_lock = threading.Lock()


def _next_session_hw(current_processed: int) -> int:
    """Return a value strictly greater than any plausible current sweep-dirty."""
    global _session_load_counter
    with _session_load_lock:
        _session_load_counter += 10000
        return (current_processed or 0) + _session_load_counter


def _store_sweep(data: dict) -> str:
    """Stash a full sweep_data dict server-side, return a short token."""
    global _sweep_token_counter
    with _sweep_cache_lock:
        _sweep_token_counter += 1
        token = f"sweep-{_sweep_token_counter}"
        _SWEEP_CACHE[token] = data
        while len(_SWEEP_CACHE) > _SWEEP_CACHE_MAX:
            _SWEEP_CACHE.popitem(last=False)
    return token


def _get_sweep(browser_store: dict | None) -> dict | None:
    """Look up the full sweep_data for a browser-side slim record."""
    if not isinstance(browser_store, dict):
        return None
    token = browser_store.get("token")
    if not token:
        return None
    with _sweep_cache_lock:
        return _SWEEP_CACHE.get(token)


def _slim_sweep_for_browser(data: dict) -> dict:
    """Strip the grid, keep axes + metric_keys + token for browser state.

    The axes are small (few hundred floats at most) and several callbacks
    read them to render UI knobs (frozen slider, shape hints) so we keep
    them inline.
    """
    slim: dict = {"token": _store_sweep(data), "metric_keys": data.get("metric_keys", [])}
    for k in ("xs", "ys", "zs", "axes", "shape"):
        if k in data:
            slim[k] = data[k]
    if "facet_keys" in data:
        slim["facet_keys"] = data["facet_keys"]
    # Light per-axis metadata for the topology-view sliders (axis keys + which
    # ones are cold).  The heavy ``cold_config``/``fixed_noise`` snapshots
    # stay server-side under the same token.
    pq = data.get("per_qubit_data")
    if isinstance(pq, dict):
        slim["per_qubit_meta"] = {
            "axis_keys": list(pq.get("axis_keys", [])),
            "axis_values": [list(v) for v in pq.get("axis_values", [])],
            "shape": list(pq.get("shape", [])),
        }
    return slim


# ---------------------------------------------------------------------------
# Per-cell fidelity cache
# ---------------------------------------------------------------------------

_PER_CELL_CACHE_MAX = 256
_per_cell_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_per_cell_cache_lock = threading.Lock()


def _per_cell_cache_key(
    sweep_store: dict | None, facet_idx: int | None, cell_idx: tuple,
) -> tuple | None:
    """Build a cache key from the sweep token + facet + cell index.

    Returns None when there is no token (e.g. no sweep loaded yet) so the
    caller can fall back to direct computation without caching.
    """
    if not isinstance(sweep_store, dict):
        return None
    token = sweep_store.get("token")
    if not token:
        return None
    return (token, facet_idx, tuple(cell_idx))


def _per_cell_cache_get(key: tuple) -> dict | None:
    with _per_cell_cache_lock:
        hit = _per_cell_cache.get(key)
        if hit is not None:
            _per_cell_cache.move_to_end(key)
        return hit


def _per_cell_cache_put(key: tuple, value: dict) -> None:
    with _per_cell_cache_lock:
        _per_cell_cache[key] = value
        _per_cell_cache.move_to_end(key)
        while len(_per_cell_cache) > _PER_CELL_CACHE_MAX:
            _per_cell_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Per-user sweep progress
#
# Sweeps are serialised by ``sweep_lock``, but multiple users may poll
# concurrently, so each user only sees the progress of the sweep they
# kicked off. The active sid for the current sweep is stored in a
# threadlocal that the sweep callback sets before running, so
# ``_update_progress`` (called deep inside the sweep machinery) can route
# writes to the right slot.
# ---------------------------------------------------------------------------

_sweep_progress_by_sid: dict[str, dict] = {}
_sweep_progress_lock = threading.Lock()
_sweep_progress_tls = threading.local()


def _progress_label(k: str) -> str:
    if k in METRIC_BY_KEY:
        return METRIC_BY_KEY[k].label
    if k in CAT_METRIC_BY_KEY:
        return CAT_METRIC_BY_KEY[k].label
    return k


def _set_progress(sid: str, payload: dict) -> None:
    with _sweep_progress_lock:
        if payload.get("running"):
            _sweep_progress_by_sid[sid] = payload
        else:
            _sweep_progress_by_sid.pop(sid, None)


def _get_progress(sid: str) -> dict:
    with _sweep_progress_lock:
        return _sweep_progress_by_sid.get(sid) or {"running": False}


def _update_progress(p: SweepProgress) -> None:
    """Progress callback passed into sweep methods."""
    sid = getattr(_sweep_progress_tls, "sid", None)
    if not sid:
        return
    _set_progress(sid, {
        "running": True,
        "completed": p.completed,
        "total": p.total,
        "percentage": p.percentage,
        "current_params": {
            _progress_label(k): v for k, v in p.current_params.items()
        },
        "cold_completed": p.cold_completed,
        "cold_total": p.cold_total,
        "phase": "sweeping",
    })


# ---------------------------------------------------------------------------
# Flask route registration
# ---------------------------------------------------------------------------

def register_routes(server: Any) -> None:
    """Attach the ``/api/progress`` route to a Flask server.

    Called from ``app.py`` once after ``app = dash.Dash(...)`` so the
    route is wired up before ``app.run``.
    """

    @server.route("/api/progress")
    def _api_progress():  # noqa: ANN202 — Flask handler
        from flask import request
        sid = request.args.get("sid", "")
        payload = _get_progress(sid) if sid else {"running": False}
        return json.dumps(payload), 200, {"Content-Type": "application/json"}
