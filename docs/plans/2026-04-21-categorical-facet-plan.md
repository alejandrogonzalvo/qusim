# Categorical Faceted Subplots — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow the researcher to compare qualitative (categorical) parameters by faceting the plot into a grid of subplots — one panel per categorical combination.

**Architecture:** Categorical variables (circuit_type, topology_type, intracore_topology, placement, routing_algorithm) move from the right panel into a new "Categorical" section in the left sidebar, each with a dropdown and "Compare" checkbox. When compare is active, `run_sweep` loops over the cartesian product of categorical values, runs one full cold-path sweep per combo, and passes the results to `build_figure` which uses `make_subplots` to lay them out in a grid.

**Tech Stack:** Python / Dash / Plotly / numpy. No engine or Rust changes.

**Design doc:** `docs/plans/2026-04-21-categorical-facet-design.md`

---

### Task 1: Registry — add `placement` to `CATEGORICAL_METRICS` and add `cold_config_key`

**Files:**
- Modify: `gui/constants.py:12-19` (CatMetricDef dataclass)
- Modify: `gui/constants.py:229-260` (CATEGORICAL_METRICS list)

**Step 1: Add `cold_config_key` field to `CatMetricDef`**

The `placement` dropdown maps to `placement_policy` in cold_config, not `placement`. Add a field so the facet loop knows which key to override.

Edit `gui/constants.py:12-19`:

```python
@dataclass
class CatMetricDef:
    key: str
    label: str
    options: list  # list of {"label": ..., "value": ...} dicts
    is_cold_path: bool
    description: str
    cold_config_key: str = ""  # key name in cold_config; defaults to self.key

    def __post_init__(self):
        if not self.cold_config_key:
            self.cold_config_key = self.key
```

**Step 2: Add `placement` entry to `CATEGORICAL_METRICS`**

Add after the `routing_algorithm` entry in `gui/constants.py:258`:

```python
    CatMetricDef(
        key="placement",
        label="Placement",
        options=PLACEMENT_OPTIONS,
        is_cold_path=True,
        description="Initial qubit placement policy (random / spectral)",
        cold_config_key="placement_policy",
    ),
```

**Step 3: Commit**

```bash
git add gui/constants.py
git commit -m "feat: add placement to CATEGORICAL_METRICS, add cold_config_key field"
```

---

### Task 2: Components — new `make_categorical_section()`, clean up old approach

**Files:**
- Modify: `gui/components.py:8-26` (imports)
- Modify: `gui/components.py:180-314` (make_metric_selector — remove categorical checklist, remove categorical options from dropdown)
- Modify: `gui/components.py:426-633` (make_fixed_config_panel — remove categorical dropdowns)
- Add new function: `make_categorical_section()`

**Step 1: Remove categorical options from `make_metric_selector` dropdown**

In `gui/components.py:206-209`, change the combined dropdown options back to numeric-only:

```python
    # Dropdown options: numeric metrics only (categoricals are in their own section)
    all_options = [{"label": nm.label, "value": nm.key} for nm in SWEEPABLE_METRICS]
```

**Step 2: Remove the categorical checklist container from `make_metric_selector`**

Remove `gui/components.py:280-301` (the `metric-checklist-container-{index}` div and its `dcc.Checklist`).

However, the checklist IDs (`metric-checklist-{index}`) may be referenced by callbacks. Check `app.py` for any references to `metric-checklist-` and remove/update those callbacks too.

**Step 3: Add `make_categorical_section()` function**

Add after `make_add_metric_button()` (around line 341):

```python
def make_categorical_section() -> html.Div:
    """Build the 'Categorical' section for the left sidebar.

    Each categorical parameter gets a dropdown + a 'Cmp' compare checkbox.
    """
    rows = []
    for cat in CATEGORICAL_METRICS:
        rows.append(
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginBottom": "8px",
                },
                children=[
                    html.Div(
                        cat.label,
                        style={
                            "fontSize": "11px",
                            "color": COLORS["text"],
                            "minWidth": "70px",
                            "flexShrink": "0",
                        },
                    ),
                    dcc.Dropdown(
                        id=f"cat-{cat.key}",
                        options=cat.options,
                        value=cat.options[0]["value"],
                        clearable=False,
                        className="dse-dropdown",
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    dcc.Checklist(
                        id=f"cat-compare-{cat.key}",
                        options=[{"label": "Cmp", "value": "on"}],
                        value=[],
                        style={
                            "fontSize": "10px",
                            "color": COLORS["text_muted"],
                            "flexShrink": "0",
                        },
                        inputStyle={"marginRight": "2px"},
                    ),
                ],
            )
        )

    return html.Div(
        children=[
            html.Div(
                "Categorical",
                style={
                    "fontSize": "11px",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": COLORS["accent"],
                    "marginBottom": "8px",
                    "paddingBottom": "6px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                    "marginTop": "12px",
                },
            ),
            *rows,
        ],
    )
```

**Step 4: Remove categorical dropdowns from `make_fixed_config_panel`**

In `gui/components.py`, remove these blocks from the `circuit_content` div (lines ~440-531):
- `cfg-circuit-type` dropdown and its label
- `cfg-topology` dropdown and its label
- `cfg-intracore-topology` dropdown and its label
- `cfg-placement` dropdown and its label
- `cfg-routing-algorithm` dropdown and its label

Keep: `cfg-row-num-qubits`, `cfg-row-num-cores`, `cfg-seed`, `cfg-dynamic-decoupling`, and the Sweep Budget section.

**Important:** The old IDs (`cfg-circuit-type`, `cfg-topology`, etc.) are used in `app.py` callbacks. The new IDs are `cat-circuit_type`, `cat-topology_type`, `cat-intracore_topology`, `cat-placement`, `cat-routing_algorithm`. All `app.py` references to the old IDs must be updated in Task 3.

**Step 5: Commit**

```bash
git add gui/components.py
git commit -m "feat: add make_categorical_section, remove categoricals from sweep axis dropdowns and right panel"
```

---

### Task 3: App layout — categorical section in left sidebar, rewire callback IDs

**Files:**
- Modify: `gui/app.py:32-42` (imports — add `make_categorical_section`)
- Modify: `gui/app.py:295-379` (`_left_sidebar` — add categorical section)
- Modify: `gui/app.py:938-980` (`run_sweep` callback States — replace old cfg-* IDs with cat-* IDs)
- Modify: `gui/app.py:906-935` (SIM_INPUTS — replace old cfg-* Inputs with cat-* Inputs)
- Modify: `gui/app.py:709-730` (noise row visibility callback — remove cfg-* references if needed)

**Step 1: Import `make_categorical_section` and `CATEGORICAL_METRICS`**

Add to `gui/app.py:32-42`:

```python
from gui.components import (
    ...,
    make_categorical_section,
)
from gui.constants import (
    ...,
    CATEGORICAL_METRICS,
)
```

**Step 2: Add categorical section to `_left_sidebar`**

In `gui/app.py:295-379`, add `make_categorical_section()` after the add/remove buttons and estimated-points div:

```python
            # Categorical section
            make_categorical_section(),
```

**Step 3: Replace old cfg-* callback IDs with cat-* IDs**

Global search-and-replace in `gui/app.py`:
- `State("cfg-circuit-type", "value")` → `State("cat-circuit_type", "value")`
- `State("cfg-topology", "value")` → `State("cat-topology_type", "value")`
- `State("cfg-intracore-topology", "value")` → `State("cat-intracore_topology", "value")`
- `State("cfg-placement", "value")` → `State("cat-placement", "value")`
- `State("cfg-routing-algorithm", "value")` → `State("cat-routing_algorithm", "value")`
- Same for any `Input(...)` references to these IDs in `_SIM_INPUTS`.

**Step 4: Add compare checkbox States to `run_sweep`**

Add 5 new States after the categorical dropdown States:

```python
*[State(f"cat-compare-{cat.key}", "value") for cat in CATEGORICAL_METRICS],
```

And unpack them inside `run_sweep`:

```python
compare_vals = all_args[idx:idx + len(CATEGORICAL_METRICS)]; idx += len(CATEGORICAL_METRICS)
```

**Step 5: Update `cold_config` assembly in `run_sweep`**

Change `run_sweep` lines ~1065-1074 to read from the new categorical dropdown IDs. The categorical values are now in variables named by their cat key. Build `cold_config` using `CatMetricDef.cold_config_key`:

```python
cold_config = {
    "num_qubits": int(num_qubits or 16),
    "num_cores": int(num_cores or 4),
    "seed": int(seed or 42),
}
# Add categorical values
for i, cat in enumerate(CATEGORICAL_METRICS):
    cold_config[cat.cold_config_key] = cat_dropdown_vals[i] or cat.options[0]["value"]
```

**Step 6: Verify app starts without errors**

```bash
cd /home/agonhid/dev/qusim && python -c "from gui.app import app; print('OK')"
```

Expected: `OK` (no import errors)

**Step 7: Commit**

```bash
git add gui/app.py
git commit -m "feat: wire categorical section into left sidebar, rewire callback IDs"
```

---

### Task 4: Facet loop in `run_sweep`

**Files:**
- Modify: `gui/app.py` — `run_sweep` function (around line 983)

**Step 1: Write `_build_facet_combos` helper**

Add as a module-level function in `gui/app.py`:

```python
def _build_facet_combos(
    compare_vals: list,
    cat_dropdown_vals: list,
) -> tuple[list[dict], list[str]]:
    """Build cartesian product of active compare toggles.

    Returns (facet_combos, facet_keys).
    facet_combos: list of dicts mapping cold_config_key → value.
    facet_keys: list of CatMetricDef.key for active compares.

    When no compare is active, returns ([{}], []).
    """
    from itertools import product as cart_product

    active_cats = []
    facet_keys = []
    for i, cat in enumerate(CATEGORICAL_METRICS):
        if compare_vals[i] and "on" in compare_vals[i]:
            active_cats.append([
                (cat.cold_config_key, opt["value"], cat.key, opt["label"])
                for opt in cat.options
            ])
            facet_keys.append(cat.key)

    if not active_cats:
        return [{}], []

    combos = []
    for combo in cart_product(*active_cats):
        d = {}
        label = {}
        for cold_key, value, cat_key, human_label in combo:
            d[cold_key] = value
            label[cat_key] = human_label
        combos.append({"overrides": d, "label": label})
    return combos, facet_keys
```

**Step 2: Add facet loop into `run_sweep`**

After building `cold_config` and `fixed_noise`, replace the existing sweep dispatch with:

```python
facet_combos, facet_keys = _build_facet_combos(compare_vals, cat_dropdown_vals)
num_facets = len(facet_combos)
is_faceted = num_facets > 1

if is_faceted:
    facets = []
    for fi, combo in enumerate(facet_combos):
        facet_cold = {**cold_config, **combo["overrides"]}
        facet_cached = _engine.run_cold(**facet_cold, noise=fixed_noise)

        # Run the same sweep for this facet
        facet_sd = _run_single_sweep(
            facet_cached, active, fixed_noise, facet_cold,
            max_cold, max_hot,
        )
        facet_sd["label"] = combo["label"]
        facets.append(facet_sd)

    sweep_data = {
        "metric_keys": [k for k, _ in active],
        "facets": facets,
        "facet_keys": facet_keys,
    }
else:
    sweep_data = _run_single_sweep(
        cached, active, fixed_noise, cold_config,
        max_cold, max_hot,
    )
```

**Step 3: Extract `_run_single_sweep` helper**

Refactor the existing 1D/2D/3D/ND dispatch into a helper function:

```python
def _run_single_sweep(cached, active, fixed_noise, cold_config, max_cold, max_hot):
    """Run a single (non-faceted) sweep. Returns sweep_data dict."""
    sweep_data = {"metric_keys": [k for k, _ in active]}
    if len(active) == 1:
        k0, r0 = active[0]
        xs, results = _engine.sweep_1d(...)
        sweep_data["xs"] = xs.tolist()
        sweep_data["grid"] = [_result_to_dict(r) for r in results]
    elif len(active) == 2:
        ...  # existing 2D code
    elif len(active) == 3:
        ...  # existing 3D code
    else:
        ...  # existing ND code
    return sweep_data
```

This is a direct extract of the existing sweep dispatch code (~lines 1080-1150), with no logic changes.

**Step 4: Update progress reporting for facets**

Wrap the progress callback to include facet info:

```python
def _facet_progress(fi, num_facets, base_callback):
    def wrapper(p):
        p_total = p.total * num_facets
        p_completed = fi * p.total + p.completed
        base_callback(SweepProgress(
            completed=p_completed, total=p_total,
            current_params=p.current_params,
            cold_completed=p.cold_completed,
            cold_total=p.cold_total,
        ))
    return wrapper
```

**Step 5: Commit**

```bash
git add gui/app.py
git commit -m "feat: add facet loop in run_sweep, extract _run_single_sweep helper"
```

---

### Task 5: Write test for `_build_facet_combos`

**Files:**
- Create: `tests/test_faceting.py`

**Step 1: Write tests**

```python
"""Tests for categorical faceting logic."""

import pytest
from gui.app import _build_facet_combos
from gui.constants import CATEGORICAL_METRICS


def test_no_compares_returns_single_empty():
    """No compare toggles → single empty combo, no regression."""
    compare_vals = [[] for _ in CATEGORICAL_METRICS]
    cat_vals = [cat.options[0]["value"] for cat in CATEGORICAL_METRICS]
    combos, keys = _build_facet_combos(compare_vals, cat_vals)
    assert combos == [{}]
    assert keys == []


def test_single_compare_returns_all_options():
    """One compare toggle → one combo per option value."""
    compare_vals = [[] for _ in CATEGORICAL_METRICS]
    cat_vals = [cat.options[0]["value"] for cat in CATEGORICAL_METRICS]
    # Enable compare for routing_algorithm (index 3 in CATEGORICAL_METRICS)
    ra_idx = next(i for i, c in enumerate(CATEGORICAL_METRICS) if c.key == "routing_algorithm")
    compare_vals[ra_idx] = ["on"]

    combos, keys = _build_facet_combos(compare_vals, cat_vals)
    ra_cat = CATEGORICAL_METRICS[ra_idx]
    assert keys == ["routing_algorithm"]
    assert len(combos) == len(ra_cat.options)
    # Each combo overrides the cold_config_key
    for combo, opt in zip(combos, ra_cat.options):
        assert combo["overrides"][ra_cat.cold_config_key] == opt["value"]
        assert combo["label"]["routing_algorithm"] == opt["label"]


def test_two_compares_cartesian_product():
    """Two compare toggles → cartesian product."""
    compare_vals = [[] for _ in CATEGORICAL_METRICS]
    cat_vals = [cat.options[0]["value"] for cat in CATEGORICAL_METRICS]
    ra_idx = next(i for i, c in enumerate(CATEGORICAL_METRICS) if c.key == "routing_algorithm")
    tp_idx = next(i for i, c in enumerate(CATEGORICAL_METRICS) if c.key == "topology_type")
    compare_vals[ra_idx] = ["on"]
    compare_vals[tp_idx] = ["on"]

    combos, keys = _build_facet_combos(compare_vals, cat_vals)
    ra_cat = CATEGORICAL_METRICS[ra_idx]
    tp_cat = CATEGORICAL_METRICS[tp_idx]
    expected = len(ra_cat.options) * len(tp_cat.options)
    assert len(combos) == expected
    assert set(keys) == {"routing_algorithm", "topology_type"}


def test_placement_uses_cold_config_key():
    """placement maps to placement_policy in cold_config."""
    compare_vals = [[] for _ in CATEGORICAL_METRICS]
    cat_vals = [cat.options[0]["value"] for cat in CATEGORICAL_METRICS]
    pl_idx = next(i for i, c in enumerate(CATEGORICAL_METRICS) if c.key == "placement")
    compare_vals[pl_idx] = ["on"]

    combos, keys = _build_facet_combos(compare_vals, cat_vals)
    assert keys == ["placement"]
    for combo in combos:
        assert "placement_policy" in combo["overrides"]
```

**Step 2: Run tests**

```bash
cd /home/agonhid/dev/qusim && python -m pytest tests/test_faceting.py -v
```

Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add tests/test_faceting.py
git commit -m "test: add unit tests for _build_facet_combos"
```

---

### Task 6: Plotting — `_build_faceted_figure` and faceted CSV

**Files:**
- Modify: `gui/plotting.py:1300-1445` (`build_figure`)
- Add: `gui/plotting.py` — new `_build_faceted_figure()` function
- Modify: `gui/plotting.py:1284-1297` (`sweep_to_csv`)

**Step 1: Write `_facet_grid_shape` helper**

```python
def _facet_grid_shape(facet_keys: list[str], facets: list[dict]) -> tuple[int, int, list[str]]:
    """Compute (rows, cols, titles) for a faceted subplot grid.

    1 facet dim: 1 row × N cols.
    2 facet dims: first key → rows, second → cols.
    3+ facet dims: product of all but last → rows, last → cols.
    """
    if len(facet_keys) == 0:
        return 1, 1, [""]

    # Collect unique values per facet key, in order of first appearance
    unique_per_key: dict[str, list[str]] = {k: [] for k in facet_keys}
    for f in facets:
        for k in facet_keys:
            v = f["label"].get(k, "")
            if v not in unique_per_key[k]:
                unique_per_key[k].append(v)

    if len(facet_keys) == 1:
        vals = unique_per_key[facet_keys[0]]
        return 1, len(vals), vals

    # 2+ keys: last key → columns, product of rest → rows
    col_key = facet_keys[-1]
    col_vals = unique_per_key[col_key]
    row_keys = facet_keys[:-1]

    from itertools import product as cart_product
    row_combos = list(cart_product(*(unique_per_key[k] for k in row_keys)))

    titles = []
    for rc in row_combos:
        row_label = " / ".join(rc)
        for cv in col_vals:
            titles.append(f"{row_label} / {cv}")

    return len(row_combos), len(col_vals), titles
```

**Step 2: Write `_build_faceted_figure`**

```python
def _build_faceted_figure(
    num_metrics: int,
    sweep_data: dict,
    output_key: str,
    view_type: str | None = None,
    thresholds: list[float] | None = None,
    threshold_colors: list[str] | None = None,
    frozen_z: float | None = None,
) -> go.Figure:
    """Build a grid of subplots, one per facet combination."""
    from plotly.subplots import make_subplots

    facets = sweep_data["facets"]
    facet_keys = sweep_data["facet_keys"]
    rows, cols, titles = _facet_grid_shape(facet_keys, facets)

    # Determine subplot type from view_type
    is_3d = view_type in ("scatter3d", "isosurface")
    spec_type = {"type": "scene"} if is_3d else {}
    specs = [[spec_type for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        specs=specs if is_3d else None,
        shared_yaxes=not is_3d,
        horizontal_spacing=0.06,
        vertical_spacing=0.08 if rows > 1 else 0.05,
    )

    # Compute global zmin/zmax for shared colorscale
    global_zmin, global_zmax = _compute_global_z_range(facets, output_key)

    for fi, facet in enumerate(facets):
        r = fi // cols + 1
        c = fi % cols + 1

        # Build a single-facet figure using the existing per-view builders
        single_fig = build_figure(
            num_metrics=num_metrics,
            sweep_data=facet,
            output_key=output_key,
            view_type=view_type,
            thresholds=thresholds,
            threshold_colors=threshold_colors,
            frozen_z=frozen_z,
        )

        # Transfer traces into the subplot
        for trace in single_fig.data:
            if is_3d:
                scene_name = f"scene{fi + 1}" if fi > 0 else "scene"
                trace.scene = scene_name
            fig.add_trace(trace, row=r, col=c)

    fig.update_layout(
        **_LAYOUT_BASE,
        showlegend=False,
        height=max(400, 350 * rows),
    )
    fig.update_layout(uirevision="keep")
    return fig
```

**Step 3: Wire faceted path into `build_figure`**

At the top of `build_figure` (line ~1310), add:

```python
    if "facets" in sweep_data:
        return _build_faceted_figure(
            num_metrics, sweep_data, output_key,
            view_type, thresholds, threshold_colors, frozen_z,
        )
```

**Step 4: Update `sweep_to_csv` for faceted data**

```python
def sweep_to_csv(sweep_data: dict) -> str:
    if "facets" in sweep_data:
        return _faceted_sweep_to_csv(sweep_data)
    # ... existing code unchanged ...


def _faceted_sweep_to_csv(sweep_data: dict) -> str:
    """CSV export with extra columns for facet keys."""
    facet_keys = sweep_data["facet_keys"]
    lines = []
    header_written = False

    for facet in sweep_data["facets"]:
        metric_keys, available_outputs, rows = _flatten_sweep_to_table(facet)
        if not header_written:
            col_names = [k for k in facet_keys]
            for k in metric_keys:
                m = METRIC_BY_KEY.get(k)
                col_names.append(m.label if m else k)
            for k in available_outputs:
                col_names.append(_OUTPUT_LABELS.get(k, k))
            lines.append(",".join(col_names))
            header_written = True

        label_vals = [facet["label"].get(k, "") for k in facet_keys]
        for row in rows:
            line_parts = [str(v) for v in label_vals]
            line_parts.extend(f"{v}" for v in row)
            lines.append(",".join(line_parts))

    return "\n".join(lines)
```

**Step 5: Commit**

```bash
git add gui/plotting.py
git commit -m "feat: add _build_faceted_figure and faceted CSV export"
```

---

### Task 7: Write test for faceted plotting

**Files:**
- Modify: `tests/test_plotting_views.py`

**Step 1: Add test for faceted 1D figure**

```python
class TestFacetedFigure:
    def test_faceted_1d_creates_subplots(self, sweep_1d_data):
        """Faceted 1D sweep → one subplot per facet."""
        xs, results = sweep_1d_data
        base = {
            "metric_keys": ["single_gate_error"],
            "xs": xs.tolist(),
            "grid": results,
        }
        sweep_data = {
            "metric_keys": ["single_gate_error"],
            "facets": [
                {**base, "label": {"routing_algorithm": "HQA + Sabre"}},
                {**base, "label": {"routing_algorithm": "TeleSABRE"}},
            ],
            "facet_keys": ["routing_algorithm"],
        }
        fig = build_figure(1, sweep_data, "overall_fidelity", view_type="line")
        assert isinstance(fig, go.Figure)
        # Should have traces from both facets
        assert len(fig.data) >= 2

    def test_faceted_csv_has_facet_columns(self, sweep_1d_data):
        """CSV export includes facet key columns."""
        xs, results = sweep_1d_data
        base = {
            "metric_keys": ["single_gate_error"],
            "xs": xs.tolist(),
            "grid": results,
        }
        sweep_data = {
            "metric_keys": ["single_gate_error"],
            "facets": [
                {**base, "label": {"routing_algorithm": "HQA + Sabre"}},
                {**base, "label": {"routing_algorithm": "TeleSABRE"}},
            ],
            "facet_keys": ["routing_algorithm"],
        }
        csv = sweep_to_csv(sweep_data)
        header = csv.split("\n")[0]
        assert "routing_algorithm" in header
```

**Step 2: Run tests**

```bash
cd /home/agonhid/dev/qusim && python -m pytest tests/test_plotting_views.py::TestFacetedFigure -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_plotting_views.py
git commit -m "test: add faceted figure and CSV export tests"
```

---

### Task 8: Budget display and estimated-points update

**Files:**
- Modify: `gui/app.py` — `update_budget_warning` callback and `estimated-points` rendering

**Step 1: Update estimated-points to show facet multiplier**

Find the callback that updates `estimated-points` (or the inline computation in `run_sweep`). When facets are active, display:

```
~900 points × 6 facets = 5,400 total
```

This requires reading the compare checkboxes in the budget estimation callback. Add them as Inputs/States to the relevant callback.

**Step 2: Commit**

```bash
git add gui/app.py
git commit -m "feat: show facet multiplier in estimated-points display"
```

---

### Task 9: Sweep store and downstream callbacks

**Files:**
- Modify: `gui/app.py` — `_slim_sweep_for_browser`, `_store_sweep`, and downstream plot-update callbacks

**Step 1: Update `_slim_sweep_for_browser` to handle faceted data**

The slim record needs `facet_keys` and facet count for downstream callbacks (view tab bar, frozen slider, etc.):

```python
def _slim_sweep_for_browser(data: dict) -> dict:
    slim: dict = {"token": _store_sweep(data), "metric_keys": data.get("metric_keys", [])}
    for k in ("xs", "ys", "zs", "axes", "shape"):
        if k in data:
            slim[k] = data[k]
    if "facet_keys" in data:
        slim["facet_keys"] = data["facet_keys"]
        slim["num_facets"] = len(data.get("facets", []))
    return slim
```

**Step 2: Ensure `build_figure` in the plot-update callback uses full sweep_data from server cache**

The plot-update callback already calls `_get_sweep(sweep_store)` to get the full data. No change needed — the full data includes `facets` and `facet_keys`, which `build_figure` will detect.

**Step 3: Commit**

```bash
git add gui/app.py
git commit -m "feat: update slim sweep record and downstream callbacks for faceting"
```

---

### Task 10: Run full test suite and manual verification

**Step 1: Run all tests**

```bash
cd /home/agonhid/dev/qusim && python -m pytest tests/ -v --timeout=120
```

Expected: All pass, including existing tests (no regressions).

**Step 2: Launch the app and test manually**

```bash
cd /home/agonhid/dev/qusim && python gui/app.py
```

Test scenarios:
1. No compare toggles → sweep works as before (unfaceted)
2. Enable compare on routing_algorithm → 1×2 faceted plot
3. Enable compare on routing_algorithm + topology_type → 2×3 grid
4. Switch between 1D, 2D, 3D sweeps with facets active
5. Check CSV export has facet columns
6. Check analysis views (parallel, slices, importance, pareto, correlation) facet correctly

**Step 3: Commit any fixes**

```bash
git add -A && git commit -m "fix: address issues found during manual testing"
```
