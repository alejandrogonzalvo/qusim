# Save / load sessions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a **Save** and **Load** pair in the top bar that round-trips the complete exploratory state (every slider/input/dropdown, the current sweep result, and the view state) to a `*.qusim.json.gz` file so users can resume a session exactly as they left it, without re-sweeping.

**Architecture:**

* New `gui/session.py` — pure module with `collect_session`, `apply_session`, `dump`, `load`, and a schema validator. No Dash imports; Dash-facing code lives in `gui/app.py`.
* Two new Dash components in the top bar: `html.Button#save-btn` + `dcc.Download#session-download`, and `dcc.Upload#session-upload` (styled as a button). Load is atomic: one big callback fans out to every control, to `sweep-result-store`, to `main-plot.figure`, to `interp-grid-store`, and to the view-tab container.
* Auto-sweep is suppressed after load by (1) advancing both `sweep-dirty` and `sweep-processed` to a high-water mark and (2) re-syncing the clientside `window._sweepDirty` via a companion clientside callback.
* Session file format: gzipped JSON with `schema_version: 1`. Unknown top-level keys are preserved on round-trip for forward compat.

**Tech Stack:** Python 3, Dash, Plotly, dash-core-components, pytest, NumPy, stdlib `gzip` + `json` + `base64`.

**Workflow:** Working on branch `feature/sessions` inside the `.worktrees/sessions` worktree. Frequent commits — one per task.

---

### Task 1: Create `gui/session.py` with schema constants + `collect_session`

**Files:**
- Create: `gui/session.py`
- Create: `tests/test_session.py`

**Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
"""Tests for save/load sessions (serialization only; Dash wiring is separate)."""

import pytest


# ---------------------------------------------------------------------------
# collect_session: build a JSON-shaped dict from raw control/view/sweep values
# ---------------------------------------------------------------------------

class TestCollectSession:
    def _minimal_controls(self):
        return {
            "num_metrics": 3,
            "axes": [
                {"key": "t1", "slider": [4.0, 6.0], "checklist": None},
                {"key": "t2", "slider": [4.0, 6.0], "checklist": None},
                {"key": "two_gate_time", "slider": [1.0, 2.0], "checklist": None},
            ],
            "circuit": {
                "num_qubits": 16, "num_cores": 4, "seed": 42,
                "circuit_type": "qft", "topology_type": "ring",
                "intracore_topology": "all_to_all", "placement": "random",
                "routing_algorithm": "hqa_sabre", "dynamic_decoupling": False,
            },
            "noise": {"single_gate_error": 1e-4},
            "thresholds": {
                "output_metric": "overall_fidelity",
                "enable": True, "num_thresholds": 3,
                "values": [0.5, 0.7, 0.9, None, None],
                "colors": ["#ff0000", "#ffaa00", "#00ff00", None, None],
            },
            "performance": {"max_cold": None, "max_hot": None, "max_workers": None},
            "hot_reload": False,
        }

    def _minimal_view(self):
        return {"view_type": "isosurface", "frozen_axis": 2, "frozen_slider_value": 0.5}

    def test_includes_schema_version(self):
        from gui.session import collect_session, SCHEMA_VERSION
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert s["schema_version"] == SCHEMA_VERSION

    def test_includes_timestamp(self):
        from gui.session import collect_session
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert "saved_at" in s
        assert s["saved_at"].endswith("Z") or "+" in s["saved_at"]

    def test_sweep_absent_when_none(self):
        from gui.session import collect_session
        s = collect_session(self._minimal_controls(), self._minimal_view(), None)
        assert s["sweep"]["present"] is False
        assert "grid" not in s["sweep"]

    def test_sweep_present_with_data(self):
        from gui.session import collect_session
        sweep = {
            "metric_keys": ["t1", "t2"],
            "xs": [1.0, 2.0], "ys": [3.0, 4.0],
            "shape": [2, 2],
            "grid": [[{"overall_fidelity": 0.9}, {"overall_fidelity": 0.8}],
                     [{"overall_fidelity": 0.7}, {"overall_fidelity": 0.6}]],
        }
        s = collect_session(self._minimal_controls(), self._minimal_view(), sweep)
        assert s["sweep"]["present"] is True
        assert s["sweep"]["metric_keys"] == ["t1", "t2"]
        assert s["sweep"]["grid"] == sweep["grid"]

    def test_roundtrips_controls_verbatim(self):
        from gui.session import collect_session
        ctrls = self._minimal_controls()
        s = collect_session(ctrls, self._minimal_view(), None)
        assert s["controls"] == ctrls

    def test_roundtrips_view_verbatim(self):
        from gui.session import collect_session
        view = self._minimal_view()
        s = collect_session(self._minimal_controls(), view, None)
        assert s["view"] == view
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_session.py -v
```
Expected: all tests FAIL with `ModuleNotFoundError: No module named 'gui.session'`.

**Step 3: Write the implementation**

Create `gui/session.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_session.py -v
```
Expected: 6 passed.

**Step 5: Commit**

```bash
git add gui/session.py tests/test_session.py
git commit -m "feat(session): add collect_session + schema constants"
```

---

### Task 2: Add `dump` / `load` (gzip round-trip)

**Files:**
- Modify: `gui/session.py`
- Modify: `tests/test_session.py` (append)

**Step 1: Append failing tests**

Append to `tests/test_session.py`:

```python
# ---------------------------------------------------------------------------
# dump / load: gzipped-JSON round-trip
# ---------------------------------------------------------------------------

class TestDumpLoad:
    def _example(self):
        from gui.session import collect_session
        return collect_session(
            TestCollectSession()._minimal_controls(),
            TestCollectSession()._minimal_view(),
            None,
        )

    def test_dump_returns_gzip_bytes(self):
        from gui.session import dump
        raw = dump(self._example())
        assert isinstance(raw, bytes)
        # gzip magic: first two bytes
        assert raw[:2] == b"\x1f\x8b"

    def test_load_roundtrip_from_gzip_bytes(self):
        from gui.session import dump, load
        original = self._example()
        back = load(dump(original))
        assert back == original

    def test_load_accepts_plain_json_string(self):
        from gui.session import load
        import json as _json
        payload = {"schema_version": 1, "controls": {}, "view": {}, "sweep": {"present": False}, "saved_at": "x", "app": {"name": "qusim-dse"}}
        back = load(_json.dumps(payload))
        assert back == payload

    def test_load_accepts_plain_json_bytes(self):
        from gui.session import load
        import json as _json
        payload = {"schema_version": 1, "controls": {}, "view": {}, "sweep": {"present": False}, "saved_at": "x", "app": {"name": "qusim-dse"}}
        back = load(_json.dumps(payload).encode("utf-8"))
        assert back == payload
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_session.py::TestDumpLoad -v
```
Expected: FAIL with `ImportError: cannot import name 'dump'`.

**Step 3: Extend the implementation**

Append to `gui/session.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_session.py -v
```
Expected: 10 passed.

**Step 5: Commit**

```bash
git add gui/session.py tests/test_session.py
git commit -m "feat(session): add gzipped JSON dump/load with format sniffing"
```

---

### Task 3: Add `validate` (schema checks)

**Files:**
- Modify: `gui/session.py`
- Modify: `tests/test_session.py` (append)

**Step 1: Append failing tests**

Append to `tests/test_session.py`:

```python
# ---------------------------------------------------------------------------
# validate: schema_version + required top-level keys
# ---------------------------------------------------------------------------

class TestValidate:
    def _good(self):
        return {
            "schema_version": 1,
            "saved_at": "2026-04-23T00:00:00Z",
            "app": {"name": "qusim-dse"},
            "controls": {}, "view": {},
            "sweep": {"present": False},
        }

    def test_good_session_passes(self):
        from gui.session import validate
        validate(self._good())  # no exception

    def test_missing_schema_version_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        del bad["schema_version"]
        with pytest.raises(SessionError, match="schema_version"):
            validate(bad)

    def test_wrong_schema_version_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        bad["schema_version"] = 999
        with pytest.raises(SessionError, match="version"):
            validate(bad)

    def test_missing_controls_raises(self):
        from gui.session import validate, SessionError
        bad = self._good()
        del bad["controls"]
        with pytest.raises(SessionError, match="controls"):
            validate(bad)

    def test_non_dict_raises(self):
        from gui.session import validate, SessionError
        with pytest.raises(SessionError):
            validate([1, 2, 3])
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_session.py::TestValidate -v
```
Expected: 5 FAIL with `ImportError: cannot import name 'validate'`.

**Step 3: Extend the implementation**

Append to `gui/session.py` (before `dump`, after `SCHEMA_VERSION`):

```python
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
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_session.py -v
```
Expected: 15 passed.

**Step 5: Commit**

```bash
git add gui/session.py tests/test_session.py
git commit -m "feat(session): add validate with schema_version check"
```

---

### Task 4: Add `apply_session` (returns a structured result)

This task introduces the dataclass the Dash callback will consume. It does
**no** Dash work — it just produces typed fields from a validated session dict,
pruning any axes whose metric keys are no longer known.

**Files:**
- Modify: `gui/session.py`
- Modify: `tests/test_session.py` (append)

**Step 1: Append failing tests**

Append to `tests/test_session.py`:

```python
# ---------------------------------------------------------------------------
# apply_session: adapt a loaded dict into typed fields for the Dash layer
# ---------------------------------------------------------------------------

class TestApplySession:
    def _session(self, **overrides):
        ctrls = {
            "num_metrics": 2,
            "axes": [
                {"key": "t1", "slider": [4.0, 6.0], "checklist": None},
                {"key": "t2", "slider": [4.0, 6.0], "checklist": None},
            ],
            "circuit": {
                "num_qubits": 16, "num_cores": 4, "seed": 42,
                "circuit_type": "qft", "topology_type": "ring",
                "intracore_topology": "all_to_all", "placement": "random",
                "routing_algorithm": "hqa_sabre", "dynamic_decoupling": False,
            },
            "noise": {"single_gate_error": 1e-4},
            "thresholds": {
                "output_metric": "overall_fidelity",
                "enable": False, "num_thresholds": 3,
                "values": [None, None, None, None, None],
                "colors": [None, None, None, None, None],
            },
            "performance": {"max_cold": None, "max_hot": None, "max_workers": None},
            "hot_reload": False,
        }
        view = {"view_type": "heatmap", "frozen_axis": 2, "frozen_slider_value": 0.5}
        s = {
            "schema_version": 1,
            "saved_at": "x", "app": {"name": "qusim-dse"},
            "controls": ctrls, "view": view,
            "sweep": {"present": False},
        }
        s.update(overrides)
        return s

    def test_returns_dataclass_with_controls_view_sweep(self):
        from gui.session import apply_session
        out = apply_session(self._session())
        assert out.controls["num_metrics"] == 2
        assert out.view["view_type"] == "heatmap"
        assert out.sweep_data is None
        assert out.warnings == []

    def test_passes_through_known_metric_keys(self):
        from gui.session import apply_session
        out = apply_session(self._session())
        keys = [ax["key"] for ax in out.controls["axes"]]
        assert keys == ["t1", "t2"]

    def test_drops_axis_with_unknown_metric_key(self):
        from gui.session import apply_session
        s = self._session()
        s["controls"]["axes"].append({"key": "nope_not_real", "slider": [0, 1], "checklist": None})
        s["controls"]["num_metrics"] = 3
        out = apply_session(s)
        assert [ax["key"] for ax in out.controls["axes"]] == ["t1", "t2"]
        assert out.controls["num_metrics"] == 2
        assert any("nope_not_real" in w for w in out.warnings)

    def test_returns_sweep_data_when_present(self):
        from gui.session import apply_session
        s = self._session()
        s["sweep"] = {
            "present": True,
            "metric_keys": ["t1", "t2"],
            "xs": [1, 2], "ys": [3, 4], "shape": [2, 2],
            "grid": [[{"overall_fidelity": 0.9}, {"overall_fidelity": 0.8}],
                     [{"overall_fidelity": 0.7}, {"overall_fidelity": 0.6}]],
        }
        out = apply_session(s)
        assert out.sweep_data is not None
        assert out.sweep_data["metric_keys"] == ["t1", "t2"]
        assert out.sweep_data["grid"][0][0]["overall_fidelity"] == 0.9

    def test_invalid_session_raises(self):
        from gui.session import apply_session, SessionError
        with pytest.raises(SessionError):
            apply_session({"schema_version": 0})
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_session.py::TestApplySession -v
```
Expected: 5 FAIL with `ImportError: cannot import name 'apply_session'`.

**Step 3: Extend the implementation**

Append to `gui/session.py`:

```python
from dataclasses import dataclass, field


@dataclass
class SessionApply:
    """Result of :func:`apply_session`.

    ``controls`` and ``view`` are the filtered dicts ready to be fanned out to
    Dash Outputs. ``sweep_data`` is the sweep result (or ``None``). ``warnings``
    lists human-readable notes about any dropped fields; the Dash layer surfaces
    them in the error banner on a best-effort basis.
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
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_session.py -v
```
Expected: 20 passed.

**Step 5: Commit**

```bash
git add gui/session.py tests/test_session.py
git commit -m "feat(session): add apply_session + SessionApply dataclass"
```

---

### Task 5: Add `build_controls_dict` / `build_view_dict` helpers

These helpers bridge the flat tuple of Dash `State` values (which the save
callback will receive) into the schema-shaped `controls` / `view` dicts. They
live in `gui/session.py` so they can be unit-tested without Dash.

**Files:**
- Modify: `gui/session.py`
- Modify: `tests/test_session.py` (append)

**Step 1: Append failing tests**

Append to `tests/test_session.py`:

```python
# ---------------------------------------------------------------------------
# build_controls_dict: assemble schema-shaped dict from raw callback args
# ---------------------------------------------------------------------------

class TestBuildControlsDict:
    def _args(self, **overrides):
        # Match the order the save callback passes to build_controls_dict.
        args = {
            "num_metrics": 2,
            "dropdown_vals": ["t1", "t2", None, None, None, None, None, None, None, None, None, None],
            "slider_vals":   [[4.0, 6.0], [4.0, 6.0]] + [None] * 10,
            "checklist_vals": [[], []] + [None] * 10,
            "cfg_circuit_type": "qft",
            "cfg_num_qubits": 16, "cfg_num_cores": 4,
            "cfg_topology": "ring", "cfg_intracore_topology": "all_to_all",
            "cfg_placement": "random", "cfg_routing_algorithm": "hqa_sabre",
            "cfg_seed": 42, "cfg_dynamic_decoupling": [],
            "cfg_max_cold": None, "cfg_max_hot": None, "cfg_max_workers": None,
            "cfg_output_metric": "overall_fidelity",
            "cfg_threshold_enable": [], "num_thresholds": 3,
            "threshold_values": [0.5, 0.7, 0.9, None, None],
            "threshold_colors": ["#ff0000", "#ffaa00", "#00ff00", None, None],
            "noise_values": {"single_gate_error": 1e-4, "two_gate_error": 1e-3},
            "hot_reload": [],
        }
        args.update(overrides)
        return args

    def test_picks_up_active_axes(self):
        from gui.session import build_controls_dict
        ctrls = build_controls_dict(**self._args())
        assert ctrls["num_metrics"] == 2
        keys = [ax["key"] for ax in ctrls["axes"]]
        assert keys == ["t1", "t2"]
        assert ctrls["axes"][0]["slider"] == [4.0, 6.0]
        assert ctrls["axes"][0]["checklist"] is None

    def test_includes_all_circuit_keys(self):
        from gui.session import build_controls_dict
        ctrls = build_controls_dict(**self._args())
        for k in (
            "num_qubits", "num_cores", "seed",
            "circuit_type", "topology_type", "intracore_topology",
            "placement", "routing_algorithm", "dynamic_decoupling",
        ):
            assert k in ctrls["circuit"]

    def test_hot_reload_bool_from_checklist(self):
        from gui.session import build_controls_dict
        on = build_controls_dict(**self._args(hot_reload=["on"]))
        off = build_controls_dict(**self._args(hot_reload=[]))
        assert on["hot_reload"] is True
        assert off["hot_reload"] is False

    def test_threshold_enable_bool_from_checklist(self):
        from gui.session import build_controls_dict
        on = build_controls_dict(**self._args(cfg_threshold_enable=["yes"]))
        off = build_controls_dict(**self._args(cfg_threshold_enable=[]))
        assert on["thresholds"]["enable"] is True
        assert off["thresholds"]["enable"] is False


# ---------------------------------------------------------------------------
# build_view_dict
# ---------------------------------------------------------------------------

class TestBuildViewDict:
    def test_copies_the_three_fields(self):
        from gui.session import build_view_dict
        v = build_view_dict("frozen_heatmap", 1, 5.0)
        assert v == {"view_type": "frozen_heatmap", "frozen_axis": 1, "frozen_slider_value": 5.0}

    def test_none_slider_is_allowed(self):
        from gui.session import build_view_dict
        v = build_view_dict("isosurface", 2, None)
        assert v["frozen_slider_value"] is None
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_session.py::TestBuildControlsDict tests/test_session.py::TestBuildViewDict -v
```
Expected: 6 FAIL with `ImportError`.

**Step 3: Extend the implementation**

Append to `gui/session.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_session.py -v
```
Expected: 26 passed.

**Step 5: Commit**

```bash
git add gui/session.py tests/test_session.py
git commit -m "feat(session): add build_controls_dict / build_view_dict helpers"
```

---

### Task 6: Add Save / Load buttons to the top bar

This is a UI-only change. No wiring yet. Just the widgets, a new
`dcc.Download`, and a `dcc.Upload`.

**Files:**
- Modify: `gui/app.py` — `_topbar()` (lines ~237-305) and the store list (~596-605)

**Step 1: Add the widgets to `_topbar()`**

In `gui/app.py`, find the inner `html.Div` containing `hot-reload-toggle` and
`run-btn` (around lines 277-303). Replace that div's `children` list to
prepend two new buttons and an upload:

```python
                children=[
                    html.Button(
                        "Save",
                        id="save-btn",
                        n_clicks=0,
                        style={
                            "background": "transparent",
                            "border": f"1px solid {COLORS['border']}",
                            "color": COLORS["text_muted"],
                            "borderRadius": "6px",
                            "padding": "6px 14px",
                            "fontSize": "12px",
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Upload(
                        id="session-upload",
                        children=html.Span(
                            "Load",
                            style={
                                "border": f"1px solid {COLORS['border']}",
                                "color": COLORS["text_muted"],
                                "borderRadius": "6px",
                                "padding": "6px 14px",
                                "fontSize": "12px",
                                "cursor": "pointer",
                                "display": "inline-block",
                            },
                        ),
                        multiple=False,
                        accept=".gz,.json",
                        style_active={},
                    ),
                    dcc.Checklist(
                        id="hot-reload-toggle",
                        options=[{"label": " Hot reload", "value": "on"}],
                        value=["on"],
                        style={"fontSize": "12px", "color": COLORS["text_muted"]},
                        inputStyle={"marginRight": "4px"},
                    ),
                    html.Button(
                        "▶  Run",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            "background": COLORS["accent"],
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "6px 20px",
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "display": "none",
                        },
                    ),
                ],
```

**Step 2: Add the download store**

In `gui/app.py`, inside `app.layout`'s store list (around line 606), add:

```python
        dcc.Download(id="session-download"),
```

**Step 3: Verify app still starts**

Run:
```
python -c "import gui.app; print('ok')"
```
Expected: prints `ok` with no traceback. (Startup imports the layout.)

**Step 4: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): add Save/Load buttons + session-download to top bar"
```

---

### Task 7: Wire the Save callback

**Files:**
- Modify: `gui/app.py` — append a new callback near the CSV-export callback (after line ~1745).

**Step 1: Add the save callback**

At the end of the CSV-export section (after line ~1745 in the original file,
just before the `Clientside callbacks: sync threshold color swatches` header),
append:

```python
# ---------------------------------------------------------------------------
# Callback: save session → gzipped JSON download
# ---------------------------------------------------------------------------


@app.callback(
    Output("session-download", "data"),
    Input("save-btn", "n_clicks"),
    *_METRIC_DROPDOWN_STATES,
    *_METRIC_SLIDER_STATES,
    State("num-metrics-store", "data"),
    *_METRIC_CHECKLIST_STATES,
    State("cfg-circuit-type", "value"),
    State("cfg-num-qubits", "value"),
    State("cfg-num-cores", "value"),
    State("cfg-topology", "value"),
    State("cfg-intracore-topology", "value"),
    State("cfg-placement", "value"),
    State("cfg-routing-algorithm", "value"),
    State("cfg-seed", "value"),
    State("cfg-dynamic-decoupling", "value"),
    State("cfg-max-cold", "value"),
    State("cfg-max-hot", "value"),
    State("cfg-max-workers", "value"),
    State("cfg-output-metric", "value"),
    State("cfg-threshold-enable", "value"),
    *[State(f"cfg-threshold-{i}", "value") for i in range(5)],
    *[State(f"cfg-threshold-color-{i}", "value") for i in range(5)],
    State("num-thresholds-store", "data"),
    State("view-type-store", "data"),
    State("frozen-axis-store", "data"),
    State("frozen-slider", "value"),
    State("hot-reload-toggle", "value"),
    State("sweep-result-store", "data"),
    *_NOISE_SLIDER_STATES,
    prevent_initial_call=True,
)
def on_save_session(n_clicks, *all_args):
    if not n_clicks:
        return dash.no_update

    from gui.session import build_controls_dict, build_view_dict, collect_session, dump
    import time as _time

    idx = 0
    dropdown_vals = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    slider_vals   = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    num_metrics   = all_args[idx]; idx += 1
    checklist_vals = list(all_args[idx:idx + MAX_METRICS]); idx += MAX_METRICS
    cfg_circuit_type = all_args[idx]; idx += 1
    cfg_num_qubits = all_args[idx]; idx += 1
    cfg_num_cores = all_args[idx]; idx += 1
    cfg_topology = all_args[idx]; idx += 1
    cfg_intracore_topology = all_args[idx]; idx += 1
    cfg_placement = all_args[idx]; idx += 1
    cfg_routing_algorithm = all_args[idx]; idx += 1
    cfg_seed = all_args[idx]; idx += 1
    cfg_dd = all_args[idx]; idx += 1
    cfg_max_cold = all_args[idx]; idx += 1
    cfg_max_hot = all_args[idx]; idx += 1
    cfg_max_workers = all_args[idx]; idx += 1
    cfg_output_metric = all_args[idx]; idx += 1
    cfg_threshold_enable = all_args[idx]; idx += 1
    t_vals = list(all_args[idx:idx + 5]); idx += 5
    tc_vals = list(all_args[idx:idx + 5]); idx += 5
    num_thresholds = all_args[idx]; idx += 1
    view_type = all_args[idx]; idx += 1
    frozen_axis = all_args[idx]; idx += 1
    frozen_slider_value = all_args[idx]; idx += 1
    hot_reload = all_args[idx]; idx += 1
    sweep_store = all_args[idx]; idx += 1
    noise_slider_vals = list(all_args[idx:])

    noise_values: dict = {}
    for i, m in enumerate(SWEEPABLE_METRICS):
        v = noise_slider_vals[i]
        if v is None:
            continue
        noise_values[m.key] = _slider_to_value(v, m.log_scale)

    controls = build_controls_dict(
        num_metrics=num_metrics,
        dropdown_vals=dropdown_vals,
        slider_vals=slider_vals,
        checklist_vals=checklist_vals,
        cfg_circuit_type=cfg_circuit_type,
        cfg_num_qubits=cfg_num_qubits,
        cfg_num_cores=cfg_num_cores,
        cfg_topology=cfg_topology,
        cfg_intracore_topology=cfg_intracore_topology,
        cfg_placement=cfg_placement,
        cfg_routing_algorithm=cfg_routing_algorithm,
        cfg_seed=cfg_seed,
        cfg_dynamic_decoupling=cfg_dd,
        cfg_max_cold=cfg_max_cold,
        cfg_max_hot=cfg_max_hot,
        cfg_max_workers=cfg_max_workers,
        cfg_output_metric=cfg_output_metric,
        cfg_threshold_enable=cfg_threshold_enable,
        num_thresholds=num_thresholds,
        threshold_values=t_vals,
        threshold_colors=tc_vals,
        noise_values=noise_values,
        hot_reload=hot_reload,
    )
    view = build_view_dict(view_type, frozen_axis, frozen_slider_value)
    sweep_data = _get_sweep(sweep_store)

    session = collect_session(controls, view, sweep_data)
    raw = dump(session)

    fname = _time.strftime("qusim-session-%Y%m%d-%H%M%S.qusim.json.gz")
    # dcc.Send bytes directly via base64 data URL.
    import base64
    return dict(
        content=base64.b64encode(raw).decode("ascii"),
        filename=fname,
        base64=True,
        type="application/gzip",
    )
```

Note: `_NOISE_SLIDER_STATES` is already defined earlier in `app.py` (used by
`run_sweep`). Verify with:

```
grep -n "_NOISE_SLIDER_STATES" gui/app.py
```

If it's not defined as a module-level list, define it near `_METRIC_DROPDOWN_STATES`:

```python
_NOISE_SLIDER_STATES = [State(f"noise-{m.key}", "value") for m in SWEEPABLE_METRICS]
```

Add this line only if grep shows it's not already module-level.

**Step 2: Verify app starts**

Run:
```
python -c "import gui.app; print('ok')"
```
Expected: `ok`.

**Step 3: Manual smoke test**

```
poetry run python -m gui.app
```

In the browser at `http://localhost:8050`:
1. Wait for initial sweep.
2. Click **Save**.
3. Browser downloads `qusim-session-<timestamp>.qusim.json.gz`. Verify the file
   is non-empty and gunzip produces valid JSON:

```
gunzip -c ~/Downloads/qusim-session-*.qusim.json.gz | python -m json.tool | head -40
```

Stop the server with Ctrl-C.

**Step 4: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): wire Save button to write gzipped session JSON"
```

---

### Task 8: Add the Load callback (controls only, no sweep rehydration yet)

Split the Load into two tasks: first just set the controls and view, then in
Task 9 wire up sweep rehydration and auto-sweep suppression. This keeps the
blast radius small and lets us verify parts of the restore work in isolation.

**Files:**
- Modify: `gui/app.py` — append a new callback after the Save callback from Task 7.

**Step 1: Add the load callback skeleton**

Append to `gui/app.py`:

```python
# ---------------------------------------------------------------------------
# Callback: load session → rehydrate controls, view, and sweep result
# ---------------------------------------------------------------------------

_AXIS_OUTPUTS = (
    [Output(f"metric-dropdown-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
    + [Output(f"metric-slider-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
    + [Output(f"metric-checklist-{i}", "value", allow_duplicate=True) for i in range(MAX_METRICS)]
)

_NOISE_OUTPUTS = [Output(f"noise-{m.key}", "value", allow_duplicate=True) for m in SWEEPABLE_METRICS]

_CFG_OUTPUTS = [
    Output("cfg-circuit-type", "value", allow_duplicate=True),
    Output("cfg-num-qubits", "value", allow_duplicate=True),
    Output("cfg-num-cores", "value", allow_duplicate=True),
    Output("cfg-topology", "value", allow_duplicate=True),
    Output("cfg-intracore-topology", "value", allow_duplicate=True),
    Output("cfg-placement", "value", allow_duplicate=True),
    Output("cfg-routing-algorithm", "value", allow_duplicate=True),
    Output("cfg-seed", "value", allow_duplicate=True),
    Output("cfg-dynamic-decoupling", "value", allow_duplicate=True),
    Output("cfg-max-cold", "value", allow_duplicate=True),
    Output("cfg-max-hot", "value", allow_duplicate=True),
    Output("cfg-max-workers", "value", allow_duplicate=True),
    Output("cfg-output-metric", "value", allow_duplicate=True),
    Output("cfg-threshold-enable", "value", allow_duplicate=True),
]

_THRESH_OUTPUTS = (
    [Output(f"cfg-threshold-{i}", "value", allow_duplicate=True) for i in range(5)]
    + [Output(f"cfg-threshold-color-{i}", "value", allow_duplicate=True) for i in range(5)]
)


def _value_to_slider(val: float, log_scale: bool) -> float:
    """Inverse of ``_slider_to_value``."""
    import math
    if log_scale and val is not None and val > 0:
        return math.log10(val)
    return val


@app.callback(
    *_AXIS_OUTPUTS,
    *_CFG_OUTPUTS,
    *_THRESH_OUTPUTS,
    Output("num-metrics-store", "data", allow_duplicate=True),
    Output("num-thresholds-store", "data", allow_duplicate=True),
    Output("hot-reload-toggle", "value", allow_duplicate=True),
    Output("view-type-store", "data", allow_duplicate=True),
    Output("frozen-axis-store", "data", allow_duplicate=True),
    *_NOISE_OUTPUTS,
    Output("status-bar", "children", allow_duplicate=True),
    Output("error-banner", "children", allow_duplicate=True),
    Output("error-banner", "style", allow_duplicate=True),
    Input("session-upload", "contents"),
    State("session-upload", "filename"),
    prevent_initial_call=True,
)
def on_load_session(contents, filename):
    import base64
    from gui.session import load as session_load, apply_session, SessionError

    if contents is None:
        raise dash.exceptions.PreventUpdate

    # ``contents`` has shape 'data:<mime>;base64,<payload>'.
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        session = session_load(raw)
        result = apply_session(session)
    except (SessionError, ValueError, OSError) as exc:
        banner = _build_error_banner_children("Failed to load session", str(exc))
        return _load_error_return(banner)

    ctrls = result.controls
    view = result.view

    # Axes → MAX_METRICS slots. Pad unused slots with dash.no_update.
    dropdown_out = [dash.no_update] * MAX_METRICS
    slider_out   = [dash.no_update] * MAX_METRICS
    checklist_out = [dash.no_update] * MAX_METRICS
    for i, ax in enumerate(ctrls["axes"][:MAX_METRICS]):
        dropdown_out[i] = ax["key"]
        slider_out[i] = ax["slider"] if ax["slider"] is not None else dash.no_update
        checklist_out[i] = ax["checklist"] if ax["checklist"] is not None else []

    circuit = ctrls["circuit"]
    cfg_out = [
        circuit["circuit_type"],
        circuit["num_qubits"],
        circuit["num_cores"],
        circuit["topology_type"],
        circuit["intracore_topology"],
        circuit["placement"],
        circuit["routing_algorithm"],
        circuit["seed"],
        ["yes"] if circuit["dynamic_decoupling"] else [],
        ctrls["performance"]["max_cold"],
        ctrls["performance"]["max_hot"],
        ctrls["performance"]["max_workers"],
        ctrls["thresholds"]["output_metric"],
        ["yes"] if ctrls["thresholds"]["enable"] else [],
    ]

    t_vals = list(ctrls["thresholds"]["values"]) + [None] * 5
    tc_vals = list(ctrls["thresholds"]["colors"]) + [None] * 5
    thresh_out = t_vals[:5] + tc_vals[:5]

    noise = ctrls["noise"]
    noise_out = [
        _value_to_slider(noise.get(m.key), m.log_scale)
        if noise.get(m.key) is not None
        else dash.no_update
        for m in SWEEPABLE_METRICS
    ]

    msg = f"Loaded {filename or 'session'}"
    if result.warnings:
        msg += f"  ({len(result.warnings)} warning(s))"
    banner_children = []
    banner_style = _ERROR_BANNER_HIDDEN_STYLE
    if result.warnings:
        banner_children = _build_error_banner_children(
            "Session loaded with warnings",
            "\n".join(result.warnings),
        )
        banner_style = _error_banner_visible_style()

    return (
        *dropdown_out,
        *slider_out,
        *checklist_out,
        *cfg_out,
        *thresh_out,
        ctrls["num_metrics"],
        ctrls["thresholds"]["num_thresholds"],
        ["on"] if ctrls["hot_reload"] else [],
        view["view_type"],
        view["frozen_axis"],
        *noise_out,
        msg,
        banner_children,
        banner_style,
    )


def _load_error_return(banner_children):
    """Return tuple for on_load_session error path — everything else no-op."""
    outputs_total = (
        3 * MAX_METRICS   # dropdown + slider + checklist
        + len(_CFG_OUTPUTS)
        + 10              # threshold values + colors
        + 5               # num-metrics, num-thresholds, hot-reload, view-type, frozen-axis
        + len(SWEEPABLE_METRICS)
        + 3               # status-bar + banner children + banner style
    )
    stub = [dash.no_update] * outputs_total
    stub[-3] = "Load failed"
    stub[-2] = banner_children
    stub[-1] = _error_banner_visible_style()
    return tuple(stub)
```

**Step 2: Verify app still imports**

Run:
```
python -c "import gui.app; print('ok')"
```
Expected: `ok`.

**Step 3: Manual smoke test**

```
poetry run python -m gui.app
```

1. Change several controls (e.g. set seed to 99, change circuit to GHZ, move a
   slider). Click **Save**.
2. Then change the controls back to defaults. Click **Load** and pick the
   file you saved.
3. Verify every control you changed is restored. The plot will still reflect
   the old (cached) sweep — Task 9 fixes that.

Stop the server.

**Step 4: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): wire Load to rehydrate controls + view stores"
```

---

### Task 9: Rehydrate the sweep result + suppress auto-sweep on load

**Files:**
- Modify: `gui/app.py` — `on_load_session` (extend outputs, rebuild figure, fill `_SWEEP_CACHE`).
- Modify: `gui/app.py` — add companion clientside callback that writes `window._sweepDirty`.

**Step 1: Extend `on_load_session` with sweep-rehydration outputs**

Before the `@app.callback` decorator of `on_load_session`, add these additional
Outputs to the existing decorator (keep order; the function return tuple must
match):

```python
    Output("main-plot", "figure", allow_duplicate=True),
    Output("sweep-result-store", "data", allow_duplicate=True),
    Output("interp-grid-store", "data", allow_duplicate=True),
    Output("view-tab-container", "children", allow_duplicate=True),
    Output("frozen-slider-container", "style", allow_duplicate=True),
    Output("frozen-slider", "min", allow_duplicate=True),
    Output("frozen-slider", "max", allow_duplicate=True),
    Output("frozen-slider", "value", allow_duplicate=True),
    Output("sweep-dirty", "data", allow_duplicate=True),
    Output("sweep-processed", "data", allow_duplicate=True),
```

**Step 2: Extend the function body to produce figure + sweep rehydration**

Replace the final `return (...)` in `on_load_session` with:

```python
    # --- Rehydrate the sweep result (if present) -------------------------
    fig_out = dash.no_update
    sweep_store_out = dash.no_update
    interp_out = dash.no_update
    view_tab_out = dash.no_update
    frozen_style = {"display": "none"}
    frozen_min = dash.no_update
    frozen_max = dash.no_update
    frozen_val = dash.no_update

    sweep_data = result.sweep_data
    if sweep_data is not None:
        from gui.interpolation import (
            permute_sweep_for_frozen,
            sweep_to_interp_grid,
            frozen_slider_config,
            is_frozen_view,
        )
        ndim = len(sweep_data.get("metric_keys", []))
        out_key = ctrls["thresholds"]["output_metric"] or "overall_fidelity"

        # Apply same frozen-axis permutation used by the main sweep callback
        # so axis 2 is always the frozen one downstream.
        if ndim == 3 and "facets" not in sweep_data:
            sweep_data = permute_sweep_for_frozen(sweep_data, view["frozen_axis"] or 2)
            interp_out = sweep_to_interp_grid(sweep_data, out_key)
            fs_cfg = frozen_slider_config(sweep_data)
            if fs_cfg and is_frozen_view(view["view_type"]):
                frozen_style = {"padding": "4px 16px 8px"}
                frozen_min = fs_cfg["min"]
                frozen_max = fs_cfg["max"]
                frozen_val = view["frozen_slider_value"] if view["frozen_slider_value"] is not None else fs_cfg["default"]

        thresh_vals = [v for v in ctrls["thresholds"]["values"] if v is not None]
        thresh_colors = [
            c for i, c in enumerate(ctrls["thresholds"]["colors"])
            if ctrls["thresholds"]["values"][i] is not None
        ]
        thresh = thresh_vals if ctrls["thresholds"]["enable"] else None
        if view["view_type"] in ("isosurface", "scatter3d"):
            thresh = thresh_vals or None

        fig_out = build_figure(
            ndim, sweep_data, out_key,
            view_type=view["view_type"],
            thresholds=thresh,
            threshold_colors=thresh_colors or None,
        )
        sweep_store_out = _slim_sweep_for_browser(sweep_data)
        view_tab_out = make_view_tab_bar(ndim, view["view_type"])

    # --- Suppress auto-sweep: advance both counters past current high ---
    hw = (ctrls.get("_sweep_hw", 0) or 0) + 10000  # placeholder, see Step 3
```

(The `_sweep_hw` line is a placeholder; the real counter sync happens in
Step 3 below.)

**Step 3: Add a monotonic high-water mark counter + companion clientside callback**

At the top of `gui/app.py` near other module-level counters (after
`_sweep_token_counter`), add:

```python
_session_load_counter = 0
_session_load_lock = threading.Lock()


def _next_session_hw(current_processed: int) -> int:
    """Return a value strictly greater than any plausible current sweep-dirty."""
    global _session_load_counter
    with _session_load_lock:
        _session_load_counter += 10000
        return (current_processed or 0) + _session_load_counter
```

In `on_load_session`, replace the placeholder `hw = ...` line with:

```python
    hw = _next_session_hw(0)
```

Then extend the return tuple to include the ten new values at the end:

```python
    return (
        *dropdown_out, *slider_out, *checklist_out,
        *cfg_out, *thresh_out,
        ctrls["num_metrics"],
        ctrls["thresholds"]["num_thresholds"],
        ["on"] if ctrls["hot_reload"] else [],
        view["view_type"],
        view["frozen_axis"],
        *noise_out,
        msg, banner_children, banner_style,
        # sweep rehydration
        fig_out, sweep_store_out, interp_out, view_tab_out,
        frozen_style, frozen_min, frozen_max, frozen_val,
        hw, hw,  # sweep-dirty, sweep-processed
    )
```

Update `_load_error_return` to append 10 more `dash.no_update` entries so its
output count matches:

```python
def _load_error_return(banner_children):
    outputs_total = (
        3 * MAX_METRICS
        + len(_CFG_OUTPUTS)
        + 10
        + 5
        + len(SWEEPABLE_METRICS)
        + 3
        + 10  # sweep rehydration outputs
    )
    stub = [dash.no_update] * outputs_total
    # status + banner live at the same relative positions as before
    banner_idx = outputs_total - 10 - 1  # one slot before the sweep block
    stub[banner_idx - 1] = "Load failed"
    stub[banner_idx] = banner_children
    stub[banner_idx + 1] = _error_banner_visible_style()
    return tuple(stub)
```

Also add this clientside callback at the end of the file (after the existing
`toggle_frozen_slider_visibility`):

```python
# ---------------------------------------------------------------------------
# Clientside: after session load, resync window._sweepDirty so the next
# user-driven input increment starts above the post-load high-water mark.
# ---------------------------------------------------------------------------

app.clientside_callback(
    """function(dirty) {
        if (typeof dirty === 'number') {
            window._sweepDirty = dirty;
            window._lastProcessed = dirty;
            window._sweepPending = false;
        }
        return window.dash_clientside.no_update;
    }""",
    Output("sweep-trigger", "data", allow_duplicate=True),
    Input("sweep-processed", "data"),
    prevent_initial_call=True,
)
```

**Step 4: Verify app starts and the full test suite still passes**

Run:
```
python -c "import gui.app; print('ok')"
pytest tests/test_session.py tests/test_frozen_slider.py tests/test_interpolation.py -v
```
Expected: `ok`, then all tests green.

**Step 5: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): rehydrate sweep on load + suppress auto-sweep via hw counter"
```

---

### Task 10: End-to-end manual browser test

**Files:** none (verification only).

**Step 1: Start the dev server**

```
poetry run python -m gui.app
```

Wait for `qusim DSE GUI starting at http://localhost:8050`.

**Step 2: Golden-path test**

1. In the browser, wait for the initial sweep to complete.
2. Switch to the **Frozen Heat** view tab.
3. Drag the frozen slider to a distinctive, non-default position (e.g. the 75%
   mark). Note the plot's colour pattern.
4. Change:
   - Circuit type → GHZ
   - Seed → 99
   - 1Q Gate Error → drag the slider range to something non-default
   - A noise value (e.g. T1) → different value
   - Thresholds → enable and set the first value to 0.5
5. Wait for the hot-reload sweep to complete.
6. Click **Save**. Note the download filename.
7. Reload the browser (Ctrl-R). Defaults are restored.
8. Click **Load** → select the saved file.

Expected:
- Status bar: `Loaded qusim-session-<timestamp>.qusim.json.gz`.
- Circuit type dropdown shows GHZ.
- Seed input shows 99.
- 1Q Gate Error slider is at the range you set.
- Noise sliders match.
- Threshold toggle is on, first value = 0.5.
- View tab **Frozen Heat** is selected.
- Plot matches what you saw pre-save (same colour pattern, same axis labels).
- **No sweep runs** — status bar stays on `Loaded …`, no "Compiling…" spinner.

**Step 3: Error path test**

1. Create a junk file: `echo 'not a session' > /tmp/bad.qusim.json`.
2. Click **Load** → pick `/tmp/bad.qusim.json`.
3. Verify the error banner appears with a readable message. The rest of the
   UI remains unchanged.

**Step 4: Unknown-metric-key test**

1. `gunzip -c` a saved session to plain JSON, edit one axis `key` to a garbage
   value (e.g. `"key": "not_a_real_metric"`), gzip back up.
2. Load the edited file.
3. Verify: the banner shows a warning (`dropped sweep axis with unknown metric
   key 'not_a_real_metric'`), the remaining axes load normally.

**Step 5: Stop the server**

Ctrl-C in the terminal.

**Step 6: Optional polish commit**

If any stylistic adjustments are needed (button padding, upload hover state,
banner wording), commit as:

```bash
git add gui/app.py gui/assets/style.css
git commit -m "style(gui): polish session Save/Load buttons"
```

---

## Notes for the implementer

* **Output count must match return-tuple length.** The Load callback has a
  large fan-out; every `return (...)` (happy path *and* error path) must emit
  the same number of values as there are `Output(...)`s in the decorator. Use:

  ```
  grep -c "Output(" gui/app.py  # before + after each task
  ```

  as a cheap sanity check. Run the app and the browser console will show the
  exact mismatch if one slips through.

* **`allow_duplicate=True` is required** on every Output in `on_load_session`
  because another callback also writes to the same target. Dash will raise at
  app startup if you forget it.

* **`_slider_to_value` / `_value_to_slider`.** The noise sliders store
  log-space positions when `m.log_scale` is True. `run_sweep` already uses
  `_slider_to_value`; we need its inverse on load. Implementation is a couple
  of lines — see Task 8.

* **Don't trust the file extension.** The `load` function sniffs for gzip
  magic bytes, so a user who accidentally renames `.json.gz` → `.json` still
  loads correctly.

* **Sweep rehydration is best-effort.** If the sweep payload is structurally
  wrong (e.g. `xs` missing when the grid is 2D), `build_figure` will raise.
  We catch it and surface via the error banner — see the `except` block.

* **YAGNI:** No auto-save, no localStorage caching, no "recent sessions" list,
  no encryption. Save button → file. Load button → file. That's the whole
  feature.

* **References:**
  - Design doc: `docs/plans/2026-04-23-save-load-sessions-design.md`
  - Similar fan-out pattern: `run_sweep` in `gui/app.py` (lines 1026-1419)
  - Serialization reference: `gui/interpolation.py::sweep_to_interp_grid`
