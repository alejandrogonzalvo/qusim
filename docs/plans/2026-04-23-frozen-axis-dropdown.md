# Frozen-axis Dropdown Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the static "Frozen axis" label in the 3D-sweep frozen-heat / frozen-contour views with a dropdown that lets the user pick which of the three axes is the frozen one. Dropdown opens upward; choice persists within session.

**Architecture:**

* New `dcc.Store#frozen-axis-store` (default `2`).
* New `dcc.Dropdown#frozen-axis-dropdown` replaces `html.Span#frozen-slider-label`. CSS class `dse-dropdown-up` flips its menu to open upward.
* New helper `permute_sweep_for_frozen(sweep_data, frozen_idx)` in `gui/interpolation.py`. It returns a sweep-data-shaped dict with `xs`, `ys`, `zs`, `grid`, `metric_keys` rearranged so the chosen axis sits at position 2. Downstream code (`build_figure`, `sweep_to_interp_grid`, `frozen_slice`, the JS `qusimInterp.frozenSlice`) keeps assuming axis 2 is frozen.
* Main sweep callback consumes `frozen-axis-store` as `State` and applies permutation before figure/interp build.
* New refresh callback fires on dropdown change and rebuilds figure + interp + slider from cached `sweep-result-store` (no resweep).

**Tech Stack:** Python 3, Dash, Plotly, dash-core-components, pytest, NumPy.

**Workflow:** Working directly on `main` (small, focused change). Frequent commits — one per task.

---

### Task 1: Add `permute_sweep_for_frozen` helper

**Files:**
- Modify: `gui/interpolation.py` (add new function near line 286, after `frozen_slider_config`)
- Test: `tests/test_frozen_slider.py` (append a new test class at end)

**Step 1: Write the failing tests**

Append to `tests/test_frozen_slider.py`:

```python
# ---------------------------------------------------------------------------
# permute_sweep_for_frozen: rearrange sweep_data so frozen axis is at index 2
# ---------------------------------------------------------------------------

class TestPermuteSweepForFrozen:
    def _make_3d_sweep(self):
        # value[i][j][k] = 100*i + 10*j + k  → uniquely identifies original cell
        xs = [1.0, 2.0]
        ys = [3.0, 4.0, 5.0]
        zs = [6.0, 7.0, 8.0, 9.0]
        grid = []
        for i in range(len(xs)):
            plane = []
            for j in range(len(ys)):
                row = []
                for k in range(len(zs)):
                    row.append({"overall_fidelity": 100*i + 10*j + k})
                plane.append(row)
            grid.append(plane)
        return {
            "metric_keys": ["t1", "t2", "two_gate_time"],
            "xs": xs, "ys": ys, "zs": zs,
            "grid": grid,
        }

    def test_frozen_idx_2_is_identity(self):
        from gui.interpolation import permute_sweep_for_frozen
        sweep = self._make_3d_sweep()
        out = permute_sweep_for_frozen(sweep, 2)
        assert out["xs"] == sweep["xs"]
        assert out["ys"] == sweep["ys"]
        assert out["zs"] == sweep["zs"]
        assert out["metric_keys"] == sweep["metric_keys"]
        assert out["grid"] == sweep["grid"]

    def test_frozen_idx_0_swaps_x_to_z(self):
        from gui.interpolation import permute_sweep_for_frozen
        sweep = self._make_3d_sweep()
        out = permute_sweep_for_frozen(sweep, 0)
        # Axis 0 (xs, t1) becomes the frozen axis (zs)
        assert out["zs"] == sweep["xs"]
        assert out["xs"] == sweep["ys"]
        assert out["ys"] == sweep["zs"]
        assert out["metric_keys"] == ["t2", "two_gate_time", "t1"]
        # Spot-check a value: original grid[1][2][3] = 123
        # After permutation, frozen axis = old i, free axes = (old j, old k)
        # New grid[i'][j'][k'] where (i', j', k') maps to (old j=i', old k=j', old i=k')
        new_grid = out["grid"]
        # new_grid[i'=2][j'=3][k'=1] should equal old grid[1][2][3] = 123
        assert new_grid[2][3][1]["overall_fidelity"] == 123

    def test_frozen_idx_1_swaps_y_to_z(self):
        from gui.interpolation import permute_sweep_for_frozen
        sweep = self._make_3d_sweep()
        out = permute_sweep_for_frozen(sweep, 1)
        # Axis 1 (ys, t2) becomes the frozen axis (zs)
        assert out["zs"] == sweep["ys"]
        assert out["xs"] == sweep["xs"]
        assert out["ys"] == sweep["zs"]
        assert out["metric_keys"] == ["t1", "two_gate_time", "t2"]
        # Original grid[1][2][3] = 123 → new (i'=1 [old i], j'=3 [old k], k'=2 [old j])
        new_grid = out["grid"]
        assert new_grid[1][3][2]["overall_fidelity"] == 123

    def test_returns_input_when_not_3d(self):
        from gui.interpolation import permute_sweep_for_frozen
        sweep = {"metric_keys": ["t1", "t2"], "xs": [1.0], "ys": [2.0], "grid": [[{"overall_fidelity": 0.5}]]}
        out = permute_sweep_for_frozen(sweep, 0)
        assert out is sweep  # unchanged passthrough
```

**Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_frozen_slider.py::TestPermuteSweepForFrozen -v
```
Expected: all 4 tests FAIL with `ImportError: cannot import name 'permute_sweep_for_frozen'`.

**Step 3: Write the implementation**

Add to `gui/interpolation.py` after `frozen_slider_config` (around line 287):

```python
def permute_sweep_for_frozen(sweep_data: dict, frozen_idx: int) -> dict:
    """Return a sweep_data dict rearranged so that ``frozen_idx`` ends at axis 2.

    Downstream consumers (build_figure, sweep_to_interp_grid, frozen_slice) all
    assume the frozen axis is the third one (``zs``). For 3D sweeps this helper
    permutes ``xs``, ``ys``, ``zs``, ``grid`` and ``metric_keys`` so that the
    user's chosen ``frozen_idx`` becomes axis 2 while the other two axes keep
    their original relative order at positions 0 and 1.

    Returns the input unchanged for non-3D sweeps or when ``frozen_idx == 2``.
    """
    metric_keys = sweep_data.get("metric_keys", [])
    if len(metric_keys) != 3 or frozen_idx == 2:
        return sweep_data

    if frozen_idx not in (0, 1):
        return sweep_data

    axes_data = [sweep_data["xs"], sweep_data["ys"], sweep_data["zs"]]
    free = [i for i in range(3) if i != frozen_idx]
    new_order = free + [frozen_idx]  # e.g. frozen=0 → [1, 2, 0]
    new_xs = list(axes_data[new_order[0]])
    new_ys = list(axes_data[new_order[1]])
    new_zs = list(axes_data[new_order[2]])
    new_keys = [metric_keys[i] for i in new_order]

    # Original grid is indexed as grid[i_x][i_y][i_z].
    # We want new_grid[i_x'][i_y'][i_z'] where the indices correspond to new_order.
    nx, ny, nz = len(sweep_data["xs"]), len(sweep_data["ys"]), len(sweep_data["zs"])
    sizes = [nx, ny, nz]
    new_sizes = [sizes[new_order[0]], sizes[new_order[1]], sizes[new_order[2]]]
    old_grid = sweep_data["grid"]

    new_grid = [
        [
            [None for _ in range(new_sizes[2])]
            for _ in range(new_sizes[1])
        ]
        for _ in range(new_sizes[0])
    ]
    # idx[d] = old-axis-d position, computed from new indices.
    for a in range(new_sizes[0]):
        for b in range(new_sizes[1]):
            for c in range(new_sizes[2]):
                old_idx = [0, 0, 0]
                old_idx[new_order[0]] = a
                old_idx[new_order[1]] = b
                old_idx[new_order[2]] = c
                new_grid[a][b][c] = old_grid[old_idx[0]][old_idx[1]][old_idx[2]]

    out = dict(sweep_data)
    out["xs"] = new_xs
    out["ys"] = new_ys
    out["zs"] = new_zs
    out["metric_keys"] = new_keys
    out["grid"] = new_grid
    return out
```

**Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_frozen_slider.py::TestPermuteSweepForFrozen -v
```
Expected: 4 passed.

Also re-run the full test file to catch regressions:
```
pytest tests/test_frozen_slider.py -v
```
Expected: all green.

**Step 5: Commit**

```bash
git add gui/interpolation.py tests/test_frozen_slider.py
git commit -m "feat(gui): add permute_sweep_for_frozen helper for 3D-sweep axis swap"
```

---

### Task 2: Add CSS for upward-opening dropdown

**Files:**
- Modify: `gui/assets/style.css` (append to end of file)

**Step 1: Add CSS rule**

Append to `gui/assets/style.css`:

```css

/* ── Dropdown that opens upward (used by the frozen-axis selector at the
       bottom of the heatmap so its menu does not run off-screen) ────── */
.dse-dropdown-up [class*="-menu"],
.dse-dropdown-up .Select-menu-outer {
    top: auto !important;
    bottom: 100% !important;
    margin-top: 0 !important;
    margin-bottom: 4px !important;
    box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.12) !important;
}
```

**Step 2: Commit**

```bash
git add gui/assets/style.css
git commit -m "style(gui): add dse-dropdown-up class to flip menu open direction"
```

(No automated test — visual verification happens in Task 5.)

---

### Task 3: Replace label with dropdown + add Store

**Files:**
- Modify: `gui/app.py` (lines 478-488 — the `html.Span#frozen-slider-label` block)
- Modify: `gui/app.py` (line ~595 — store list inside `app.layout`)

**Step 1: Replace the label span with a Dropdown**

In `gui/app.py`, find lines 479-488:

```python
                            html.Span(
                                id="frozen-slider-label",
                                children="Frozen axis",
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "600",
                                    "color": COLORS["text_muted"],
                                    "whiteSpace": "nowrap",
                                },
                            ),
```

Replace with:

```python
                            html.Div(
                                style={"width": "170px", "flexShrink": "0"},
                                children=dcc.Dropdown(
                                    id="frozen-axis-dropdown",
                                    className="dse-dropdown dse-dropdown-up",
                                    options=[],
                                    value=2,
                                    clearable=False,
                                    searchable=False,
                                    style={"fontSize": "11px"},
                                ),
                            ),
```

**Step 2: Add the store**

In `gui/app.py` after line 602 (the `interp-grid-store`), add:

```python
        dcc.Store(id="frozen-axis-store", data=2, storage_type="memory"),
```

**Step 3: Search-and-verify**

Run:
```
grep -n "frozen-slider-label" gui/app.py
```
Expected: only matches in callback Output declarations (lines around 1034) — those will be cleaned up in Task 4. The layout no longer references `frozen-slider-label`.

Run:
```
grep -n "frozen-axis-dropdown\|frozen-axis-store" gui/app.py
```
Expected: at least 2 hits — the dropdown definition and the store.

**Step 4: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): swap frozen-slider-label for frozen-axis-dropdown + store"
```

---

### Task 4: Wire main callback + add refresh callback

**Files:**
- Modify: `gui/app.py` — main sweep callback (around lines 1023-1387)
- Modify: `gui/app.py` — add new refresh callback after the existing `toggle_frozen_slider_visibility` (after line 1856)

**Step 1: Update the main sweep callback**

In `gui/app.py`:

a) Replace `Output("frozen-slider-label", "children")` (line 1034) with `Output("frozen-axis-dropdown", "options")` and add a second new output `Output("frozen-axis-dropdown", "value")`:

```python
    Output("frozen-axis-dropdown", "options"),
    Output("frozen-axis-dropdown", "value"),
```

b) Add `State("frozen-axis-store", "data")` to the `State` list of the main callback (place it next to other stores, e.g. after `State("hot-reload-toggle", "value")` on line 1040). Update the function signature accordingly: add a `frozen_axis_idx` parameter at the matching position.

c) Inside the callback body, locate the block at lines 1351-1360 (the `if ndim == 3 and "facets" not in sweep_data:` block). Modify it to apply permutation and produce dropdown options:

Before the `interp_grid = sweep_to_interp_grid(...)` line, add:

```python
        from gui.interpolation import permute_sweep_for_frozen
        from gui.constants import METRIC_BY_KEY

        frozen_dropdown_options = dash.no_update
        frozen_dropdown_value = dash.no_update

        if ndim == 3 and "facets" not in sweep_data:
            metric_keys_3d = sweep_data.get("metric_keys", [])
            # Validate stored frozen index against current metric set; reset to 2
            # if out of range.
            try:
                f_idx = int(frozen_axis_idx) if frozen_axis_idx is not None else 2
            except (TypeError, ValueError):
                f_idx = 2
            if f_idx not in (0, 1, 2):
                f_idx = 2

            permuted = permute_sweep_for_frozen(sweep_data, f_idx)
            interp_grid = sweep_to_interp_grid(permuted, out_key)
            fs_cfg = frozen_slider_config(permuted)
            if fs_cfg and is_frozen_view(view):
                frozen_style = {"padding": "4px 16px 8px"}
                frozen_min = fs_cfg["min"]
                frozen_max = fs_cfg["max"]

            # Always emit dropdown options for the 3 metrics
            frozen_dropdown_options = [
                {
                    "label": (METRIC_BY_KEY.get(mk).label if METRIC_BY_KEY.get(mk) else mk),
                    "value": i,
                }
                for i, mk in enumerate(metric_keys_3d)
            ]
            frozen_dropdown_value = f_idx
            # Use the permuted sweep_data so build_figure and the slim store
            # see the rearranged axes.
            sweep_data = permuted
```

Remove the now-redundant block that previously assigned `interp_grid` / `fs_cfg` / `frozen_label` (lines 1351-1360 in the original).

d) Replace the trailing `frozen_label,` in the return tuple (around line 1384) with `frozen_dropdown_options, frozen_dropdown_value,`.

e) In the `except` branch's return tuple (around line 1399-1410+), replace whatever previously held `frozen_label` (or `dash.no_update` for it) with two `dash.no_update` entries to match the new output count. Verify by counting outputs.

**Step 2: Drop the now-unused frozen_label local**

Remove the line `frozen_label = dash.no_update` (was around line 1349). Also delete the inner block lines 1358-1360 if they remain.

**Step 3: Add the refresh-on-dropdown-change callback**

After the existing `toggle_frozen_slider_visibility` callback (after line 1856 in the original file), append:

```python
# ---------------------------------------------------------------------------
# Callback: dropdown choice → update frozen-axis-store and rebuild figure
#           from the cached sweep result (no re-sweep).
# ---------------------------------------------------------------------------

@app.callback(
    Output("frozen-axis-store", "data"),
    Output("main-plot", "figure", allow_duplicate=True),
    Output("interp-grid-store", "data", allow_duplicate=True),
    Output("frozen-slider", "min", allow_duplicate=True),
    Output("frozen-slider", "max", allow_duplicate=True),
    Output("frozen-slider", "value", allow_duplicate=True),
    Input("frozen-axis-dropdown", "value"),
    State("sweep-result-store", "data"),
    State("view-type-store", "data"),
    State("cfg-output-metric", "value"),
    prevent_initial_call=True,
)
def on_frozen_axis_change(frozen_idx, sweep_data, view_type, output_key):
    if frozen_idx is None or sweep_data is None:
        raise dash.exceptions.PreventUpdate
    if len(sweep_data.get("metric_keys", [])) != 3:
        raise dash.exceptions.PreventUpdate
    try:
        f_idx = int(frozen_idx)
    except (TypeError, ValueError):
        raise dash.exceptions.PreventUpdate
    if f_idx not in (0, 1, 2):
        raise dash.exceptions.PreventUpdate

    from gui.interpolation import permute_sweep_for_frozen
    permuted = permute_sweep_for_frozen(sweep_data, f_idx)
    out_key = output_key or "overall_fidelity"
    interp_grid = sweep_to_interp_grid(permuted, out_key)
    fs_cfg = frozen_slider_config(permuted)
    fig = build_figure(3, permuted, out_key, view_type=view_type)

    new_min = fs_cfg["min"] if fs_cfg else dash.no_update
    new_max = fs_cfg["max"] if fs_cfg else dash.no_update
    new_val = fs_cfg["default"] if fs_cfg else dash.no_update
    return f_idx, fig, interp_grid, new_min, new_max, new_val
```

(Imports `sweep_to_interp_grid`, `frozen_slider_config`, `build_figure` should already exist at the top of the file — verify with grep before adding the callback.)

**Step 4: Run the existing test suite to catch regressions**

```
pytest tests/test_frozen_slider.py -v
```
Expected: green.

```
pytest tests/ -k "frozen or interpolation or sweep" -v
```
Expected: green.

**Step 5: Commit**

```bash
git add gui/app.py
git commit -m "feat(gui): wire frozen-axis dropdown to permute sweep + rebuild figure"
```

---

### Task 5: Manual browser test

**Step 1: Start the dev server**

In a separate terminal (or background):
```
poetry run python -m gui.app
```

Wait for `qusim DSE GUI starting at http://localhost:8050`.

**Step 2: Reproduce the original setup**

In the browser at `http://localhost:8050`:

1. Confirm there are 3 metrics on the left (T1, T2, 2Q Gate Time by default).
2. Click the **Run** button (or wait for hot-reload sweep).
3. Switch the view tab to **Frozen Heat**.

**Step 3: Verify dropdown UX**

a) **Replaces the label.** The bottom-left of the heatmap shows a dropdown (not the static text "Frozen axis"). Its current value is the 3rd metric ("2Q Gate Time") — this matches today's behaviour.

b) **Opens upward.** Click the dropdown. The menu appears *above* the closed input, not below — verify it does not extend off the bottom of the viewport.

c) **Switching axis.** Pick "T1 (Relaxation)" from the dropdown.
   - The X-axis label of the heatmap changes from T1 to T2.
   - The Y-axis label changes from T2 to 2Q Gate Time.
   - The slider range now spans T1's sweep range (≈ 1 µs – 10 ms in log10).
   - The slider tooltip shows a value in T1's range.
   - The heatmap re-renders without an extra "compiling…" overlay (no resweep).

d) **Picking the second metric.** Pick "T2 (Dephasing)" — verify X = T1, Y = 2Q Gate Time, slider is over T2.

e) **Persistence.** Without re-running the sweep, switch view tab to **Frozen Ctr** and back. Dropdown selection is remembered. Now click **Run** to re-execute the sweep with the same metric set. Dropdown selection is still remembered.

f) **Reset on metric change.** Change one of the left-side metric dropdowns (e.g., swap "2Q Gate Time" for "Photon Loss"). Run sweep. Dropdown options update; selection stays at the same index (2) because that index is still valid — but the metric label at index 2 is now the new metric.

**Step 4: Stop the server**

`Ctrl-C` in the terminal.

**Step 5: Commit any small fixups**

If any stylistic adjustments were needed (e.g. dropdown width or margins) commit them as a separate `style(gui): …` commit.

---

## Notes for the implementer

* **Output count must match return tuple length.** The main callback has many outputs — adding/removing one means every `return (...)` statement must change. Use grep for `return (` inside the callback to find all branches and update them consistently.
* **`COLORS["text_muted"]` may not exist** — verify before referencing. The original `Span` used it, but the new dropdown gets its colour from existing `.dse-dropdown` CSS rules.
* **`METRIC_BY_KEY` import.** It's already imported in `gui/app.py` (used elsewhere in the same callback) — verify before adding a duplicate `from gui.constants import METRIC_BY_KEY`.
* **Don't re-run the heavy sweep.** The whole point of the refresh callback is to avoid hitting the simulation backend. Confirm by watching the status bar — it should *not* update when the dropdown changes.
* **YAGNI:** No need for N-D (>3) support, no need for keyboard shortcuts, no need to sync the dropdown with the left-side metric ordering. Just the three-axis dropdown.
