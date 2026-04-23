# Save / load sessions

**Date:** 2026-04-23

## Problem

The DSE GUI has dozens of knobs — sweep axes, ranges, noise sliders, cold-path
config, thresholds, performance caps, view tab, frozen-axis selection. After a
long sweep produces an interesting plot there is no way to come back to it
later without manually recreating every control value *and* re-running the
sweep. Users lose reproducibility and waste minutes per comparison.

## Goal

Add a **Save** / **Load** pair in the top bar that round-trips the complete
exploratory state:

1. every input control (sliders, dropdowns, checklists, inputs, stores),
2. the most recent sweep result (so the plot renders without re-sweeping),
3. the view state (view tab, frozen axis, frozen slider position).

Loading a session produces the same plot the user saw when they saved, without
triggering a re-run.

## Non-goals

- Cloud sync, per-user accounts, or multi-session merges.
- Saving multiple sweeps per file (one saved sweep per session file).
- Exporting the session for another tool (that is what the existing CSV export
  is for).
- Backwards compatibility across breaking changes to `MetricDef` / cold-path
  config. Sessions carry a `schema_version`; mismatched versions refuse to load
  with a clear banner error rather than partially applying.

## What lives in "state"

Inventory taken from `gui/app.py`. Three buckets:

### A. Input controls (user-configurable)

| Location              | IDs                                                                                   |
|-----------------------|---------------------------------------------------------------------------------------|
| Left sidebar          | `num-metrics-store`; per-axis `metric-dropdown-{i}`, `metric-slider-{i}`, `metric-checklist-{i}` for `i ∈ [0, MAX_SWEEP_AXES)` |
| Right panel / Circuit | `cfg-num-qubits`, `cfg-num-cores`, `cfg-seed`, `cfg-circuit-type`, `cfg-topology`, `cfg-intracore-topology`, `cfg-placement`, `cfg-routing-algorithm`, `cfg-dynamic-decoupling` |
| Right panel / Noise   | `noise-{m.key}` for every `m` in `SWEEPABLE_METRICS`                                  |
| Right panel / Thresh. | `cfg-output-metric`, `cfg-threshold-enable`, `num-thresholds-store`, `cfg-threshold-{0..4}`, `cfg-threshold-color-{0..4}` |
| Right panel / Perf.   | `cfg-max-cold`, `cfg-max-hot`, `cfg-max-workers`                                      |
| Top bar               | `hot-reload-toggle`                                                                   |

### B. View state

- `view-type-store` — which tab (`isosurface`, `frozen_heatmap`, …).
- `frozen-axis-store` — which of the 3 axes is currently frozen.
- Current `frozen-slider.value` — optional; snapshot of where the user left it.

### C. Sweep result

The full `sweep_data` dict that currently lives in `_SWEEP_CACHE` server-side,
keyed by the token in `sweep-result-store`:

- `metric_keys`, `xs`, `ys`, `zs`, `axes`, `shape`
- `grid` — nested list of per-point metric dicts (main payload)
- `facets`, `facet_keys` when categorical axes are active

### What we deliberately DO NOT save

- **Rendered Plotly figures.** Reconstructed via `build_figure(sweep_data, …)`
  on load. Saving raw figure JSON bloats the file and couples the format to
  Plotly internals.
- `interp-grid-store`. Deterministically derived from `sweep_data` via
  `sweep_to_interp_grid`.
- Internal pipeline state: `sweep-dirty`, `sweep-processed`, `sweep-trigger`,
  `num-metrics-store` (redundant with the visible rows), `num-thresholds-store`
  (redundant with the threshold input count).

  Correction on the two "redundant" stores: they *are* saved, because the
  clientside "hide row" logic on the right panel depends on them and
  re-deriving them on load would duplicate logic from `toggle_metric_rows` and
  `toggle_threshold_rows`. Cheap to store, so we do.

## File format

`*.qusim.json.gz` — gzipped JSON. Rationale:

- JSON is human-inspectable and version-stable. The user's existing
  `.gitignore` already excludes `*.json.gz`, so saved sessions won't pollute
  commits by default.
- Gzip reclaims most of the ~10× redundancy in nested `grid` dicts. A typical
  3D sweep (12³ × ~10 metrics ≈ 140 KB uncompressed) lands around 20–30 KB
  gzipped.
- Pickle was rejected: faster and smaller, but breaks across Python versions
  and is a security footgun on untrusted input.

### Shape

```json
{
  "schema_version": 1,
  "saved_at": "2026-04-23T12:34:56Z",
  "app": { "name": "qusim-dse", "git_sha": "c978327" },

  "controls": {
    "num_metrics": 3,
    "axes": [
      { "key": "t1",            "slider": [4.0, 6.0], "checklist": null },
      { "key": "t2",            "slider": [4.0, 6.0], "checklist": null },
      { "key": "circuit_type",  "slider": null,       "checklist": ["qft", "ghz"] }
    ],
    "circuit": {
      "num_qubits": 16, "num_cores": 4, "seed": 42,
      "circuit_type": "qft", "topology_type": "ring",
      "intracore_topology": "all_to_all", "placement": "random",
      "routing_algorithm": "hqa_sabre", "dynamic_decoupling": false
    },
    "noise": { "single_gate_error": 1e-4, "two_gate_error": 1e-3, … },
    "thresholds": {
      "output_metric": "overall_fidelity",
      "enable": true,
      "num_thresholds": 3,
      "values":  [0.5, 0.7, 0.9, null, null],
      "colors":  ["#ff0000", "#ffaa00", "#00ff00", null, null]
    },
    "performance": { "max_cold": null, "max_hot": null, "max_workers": null },
    "hot_reload": false
  },

  "view": {
    "view_type": "frozen_heatmap",
    "frozen_axis": 1,
    "frozen_slider_value": 5.0
  },

  "sweep": {
    "present": true,
    "metric_keys": ["t1", "t2", "two_gate_time"],
    "xs": [...], "ys": [...], "zs": [...],
    "shape": [12, 12, 12],
    "axes": [...],
    "facet_keys": null,
    "grid": [[[ { "overall_fidelity": 0.87, ... }, ... ] ] ]
  }
}
```

If no sweep has been run yet at save time, `sweep.present = false` and the
grid/axes fields are omitted. Loading such a session sets controls only and
shows the empty-plot placeholder.

## Architecture

### UI (top bar)

```
  qusim  DSE Explorer   [ status bar ]   [Save] [Load]  [☑ Hot reload] [Run]
```

- **Save** button → `dcc.Download`. Filename defaults to
  `qusim-session-YYYYMMDD-HHMMSS.qusim.json.gz`.
- **Load** button is a `dcc.Upload` styled as a button. Accepts only
  `.qusim.json.gz` / `.qusim.json` (uncompressed fallback for debugging).
- On successful load, the status bar briefly reads `Loaded <filename>`.
- On schema / format error, the existing `#error-banner` lights up with the
  failure reason.

### Serialization module

`gui/session.py` (new). Pure-ish: no Dash imports, just dict-in / dict-out.

- `collect_session(controls, view, sweep_data) -> dict` — builds the JSON-shaped
  dict.
- `apply_session(session_dict) -> SessionApply` — returns a dataclass of the
  values the Dash callback needs to fan out to outputs, after schema check.
- `dump(session_dict) -> bytes` (gzipped).
- `load(raw_bytes | str) -> dict` — decompresses if gz-magic, else parses as
  JSON.
- Schema validator: version check + key presence check. Unknown keys are
  preserved on round-trip (forward-compat for additive future fields).

### Callbacks (Dash wiring)

**Save** (single callback):

```
Output: csv-download (reuse existing dcc.Download OR add a new dcc.Download
        `session-download` — prefer the new one to keep MIME types separate)
Inputs: save-btn.n_clicks
States: every control listed in bucket A, the view stores, sweep-result-store
Behaviour: read full sweep via _get_sweep, build session dict, gzip, return.
```

**Load** (single callback, large fan-out):

```
Inputs:  load-upload.contents
States:  load-upload.filename
Outputs: every control value in bucket A
         + view-type-store, frozen-axis-store, frozen-slider-container(style),
           frozen-slider(min/max/value)
         + sweep-result-store (new token)
         + main-plot.figure, interp-grid-store
         + view-tab-container.children (to reflect N-dim)
         + status-bar.children, error-banner.children/style
         + sweep-dirty (allow_duplicate) and sweep-processed — set equal to
           suppress auto-sweep after control restoration
Behaviour: b64-decode upload, gunzip, parse JSON, validate schema. Populate
           _SWEEP_CACHE with the grid, build the figure, compute interp_grid,
           compute frozen-slider config.
```

### Suppressing auto-sweep after load

Restoring control values will cause the clientside `_sweepDirty` increment
callback to fire repeatedly. We need to reach a steady state where
`dirty <= processed` so the JS poll gate never triggers `sweep-trigger`.

**Approach (chosen): bump both counters past the current max.**

The load callback:
1. Computes the new high-water mark `hw = (current sweep-processed) + 1000`
   (1000 chosen to exceed any plausible single-load input storm).
2. Writes `sweep-dirty := hw` *and* `sweep-processed := hw` via
   `allow_duplicate=True` on `sweep-dirty`.
3. Also writes `window._sweepDirty` via a companion clientside callback so the
   next user-driven increment picks up from the new baseline rather than from
   a stale 0.

This keeps the gate satisfied even while the input bump callback fires some
number of times in between — as long as the final `dirty` ends at ≤ `processed`
once the update burst settles, which holds because the clientside bump callback
does `return (window._sweepDirty || 0) + 1`, and we set `window._sweepDirty = hw`.

**Approach (rejected): disable hot-reload during load.**

Simpler to implement but (a) changes a user-visible toggle, (b) still leaves
the `sweep-dirty` counter in an inconsistent state for the next manual Run.

### Error handling

All errors (bad magic bytes, wrong version, corrupt JSON, missing required
field, unknown metric key on restore) fall through to the existing
`#error-banner` with a scoped message. The upload widget resets so the user
can retry without a full page reload.

**Unknown metric keys:** A saved session may reference a `MetricDef.key` that
no longer exists (e.g. the metric was renamed in code since the save). The
loader drops that axis with a warning in the banner and continues — the rest
of the session still applies. Loader's rule: *best-effort, warn loudly.*

## Trade-offs

- **JSON vs. pickle.** See above. Chose JSON for transparency and safety;
  accept ~2× size penalty and slower load for huge grids. Acceptable since
  sweeps are capped to a few tens of thousands of points.
- **Save rendered plot vs. rebuild.** Rebuilding is the canonical path used
  everywhere else (view tab switch, threshold change), so reusing it keeps
  the figure rendering logic single-source-of-truth.
- **Large fan-out load callback vs. many small ones.** One big callback is
  uglier but guarantees atomic restore — no partial state where the sliders
  have been reset but the plot hasn't. Small callbacks would interleave with
  the existing `sweep-dirty` cascade and produce flicker.
- **Schema migrations.** Not attempted in v1. `schema_version` is present so
  a future migration layer can be slotted in without breaking old files.

## Testing strategy

1. **Unit tests** on `gui/session.py` — serialize ↔ deserialize round-trip for
   representative control dicts (numeric-only axes, categorical axes, empty
   sweep), schema-version rejection, unknown-metric pruning.
2. **Integration tests** via a headless Dash instance — simulate a save,
   reload into a fresh app instance, assert all control values and
   sweep-result-store match. (Matches the pattern already used in
   `tests/test_frozen_slider.py` for UI-adjacent logic.)
3. **Manual browser test** — the only way to verify the suppress-auto-sweep
   approach works end-to-end. Script included in the plan.

## Out of scope (explicitly)

- Saving multiple sweeps per session.
- Encrypting session files.
- A "recent sessions" list in the UI.
- Auto-save to localStorage on every change.
