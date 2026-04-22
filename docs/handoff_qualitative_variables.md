# Handoff: Qualitative Variables in the DSE UI

## Goal

Allow the user to sweep over **qualitative (categorical) parameters** on the
DSE axes, not just numeric ones.  Today, parameters like `circuit_type`,
`topology_type`, `intracore_topology`, and `routing_algorithm` are fixed
dropdowns in the right panel.  After this feature, the user can put any of
them on a sweep axis and compare fidelity/time across, e.g.,
`ring` vs `all_to_all` vs `linear` topology in a single heatmap.

---

## Current architecture (what exists today)

### `gui/constants.py`
- `SWEEPABLE_METRICS` — list of `MetricDef` dataclasses, **all numeric**.
  Each has `slider_min/max`, `log_scale`, `num_steps`, and crucially
  `is_cold_path: bool` (True only for `num_qubits` and `num_cores`).
- `METRIC_BY_KEY` — dict keyed by `MetricDef.key`.
- Categorical options already exist as separate lists:
  `CIRCUIT_TYPES`, `TOPOLOGY_TYPES`, `INTRACORE_TOPOLOGY_TYPES`,
  `ROUTING_ALGORITHM_OPTIONS`, `PLACEMENT_OPTIONS`.

### `gui/dse_engine.py`
- `DSEEngine.COLD_PATH_KEYS = frozenset({"num_qubits", "num_cores"})` — the
  set of keys that require a full cold compilation when they change.
- `DSEEngine.INTEGER_KEYS` — keys that need `int()` coercion.
- `_eval_point(cold_config, noise, swept)`:
  - If a swept key is in `COLD_PATH_KEYS` → put in `cfg` (triggers new cold run).
  - Otherwise → put in `hot_noise` (hot-path re-evaluation, cheap).
- `_metric_values(metric_key, low, high, n)` → builds the numeric axis array
  using `METRIC_BY_KEY[metric_key]`.  **Does not support categorical axes.**
- `_has_cold(*keys)` → checks whether any swept key needs a cold compilation.
- `_points_per_axis / _points_budget` → budget computation, cold vs hot.

### `gui/components.py`
- `make_metric_selector(idx, default_key)` → builds one sweep-axis row with:
  - A `dcc.Dropdown` (`id={"type":"metric-key","index":idx}`) for metric key.
  - A `dcc.RangeSlider` (`id={"type":"metric-slider","index":idx}`) for
    `[low, high]` range.
  - Slider marks + tooltip via `_log_marks` / `_linear_marks`.
- `make_fixed_config_panel()` → right-panel dropdowns: `cfg-circuit-type`,
  `cfg-topology`, `cfg-intracore-topology`, `cfg-placement`,
  `cfg-routing-algorithm`.

### `gui/app.py`
- Pattern-match callbacks on `{"type":"metric-key","index":ALL}` drive slider
  updates when the user changes which metric is on an axis.
- `run_sweep` callback collects axis keys + slider `[low, high]` values and
  calls `engine.sweep_nd(...)`.
- `cold_config` dict is assembled from the right-panel dropdowns; all
  categorical values come from there.

---

## What needs to change

### 1. `MetricDef` / registry (`constants.py`)

Add a `CatMetricDef` (or extend `MetricDef`) for categorical parameters:

```python
@dataclass
class CatMetricDef:
    key: str
    label: str
    options: list[dict]   # same format as dcc.Dropdown options
    is_cold_path: bool
    description: str
```

Add entries for every sweepable categorical:

```python
CATEGORICAL_METRICS: list[CatMetricDef] = [
    CatMetricDef(
        key="circuit_type",
        label="Circuit Type",
        options=CIRCUIT_TYPES,
        is_cold_path=True,
        description="Quantum circuit benchmark (QFT, GHZ, Random)",
    ),
    CatMetricDef(
        key="topology_type",
        label="Inter-core Topology",
        options=TOPOLOGY_TYPES,
        is_cold_path=True,
        description="Inter-core connectivity (ring / all-to-all / linear)",
    ),
    CatMetricDef(
        key="intracore_topology",
        label="Intra-core Topology",
        options=INTRACORE_TOPOLOGY_TYPES,
        is_cold_path=True,
        description="On-chip qubit connectivity within each core",
    ),
    CatMetricDef(
        key="routing_algorithm",
        label="Routing Algorithm",
        options=ROUTING_ALGORITHM_OPTIONS,
        is_cold_path=True,
        description="Routing algorithm (HQA+Sabre / TeleSABRE)",
    ),
]

CAT_METRIC_BY_KEY = {m.key: m for m in CATEGORICAL_METRICS}
```

All categorical params are `is_cold_path=True` (changing them requires a new
cold compilation).

### 2. `DSEEngine` (`dse_engine.py`)

**`COLD_PATH_KEYS`** — extend to include all categorical keys:

```python
COLD_PATH_KEYS = frozenset({
    "num_qubits", "num_cores",
    "circuit_type", "topology_type", "intracore_topology",
    "routing_algorithm",
})
```

**`_eval_point`** — categorical swept values go into `cfg`, not `hot_noise`
(already works if `COLD_PATH_KEYS` is extended, because the branch is
`if k in self.COLD_PATH_KEYS: cfg[k] = v`).  Drop the `int(v)` coercion for
non-integer keys:

```python
for k, v in swept.items():
    if k in self.COLD_PATH_KEYS:
        cfg[k] = int(v) if k in self.INTEGER_KEYS else v
    else:
        hot_noise[k] = v
```

**`_metric_values`** — add categorical branch:

```python
def _metric_values(self, metric_key, low, high, n):
    if metric_key in CAT_METRIC_BY_KEY:
        # low/high unused; return the full list of option values
        return np.array([o["value"] for o in CAT_METRIC_BY_KEY[metric_key].options])
    m = METRIC_BY_KEY[metric_key]
    ...  # existing numeric logic
```

**`_has_cold`** — already works (categorical keys will be in `COLD_PATH_KEYS`).

**`_points_per_axis` / budget** — categorical axes have a fixed number of
values (their options list length), so they should not be subject to the
normal budget-based point count.  Pass `n_override` when calling
`_metric_values` if the axis is categorical and clamp the budget computation
to treat categorical axes as fixed-size.

### 3. `make_metric_selector` (`components.py`)

The current selector shows a `RangeSlider`.  Categorical axes need a
**multi-select `Checklist`** (or multi-value `Dropdown`) instead of a slider.

Option A — replace slider with a checklist when a categorical key is selected:
```
[Dropdown: metric key] → [Checklist: option1 ✓ option2 ✓ option3]
```

Option B — always show both slider and checklist containers, toggle
`display: none` via a callback based on whether the selected key is numeric or
categorical.

**Recommended approach (B)**: keep the current `RangeSlider` container and
add a sibling `dcc.Checklist` container with `id={"type":"metric-checklist","index":idx}`.
A callback on `{"type":"metric-key","index":ALL}` shows the right one.

The checklist values (list of selected option values) replace `[low, high]`
as the axis spec for categorical dimensions.

### 4. `run_sweep` callback (`app.py`)

Currently reads `[low, high]` from `metric-slider`.  For categorical axes,
read selected values from `metric-checklist` instead.  The sweep engine just
needs the list of axis values — the abstraction already works because
`sweep_nd` accepts pre-built value arrays.

**X/Y axis labels for categorical axes** — the plot's axis tick labels need to
show the human-readable option label (e.g., "Ring") not the value key
("ring").  Pass an `axis_tick_labels` dict to `build_figure`.

---

## Key data-flow change (summary)

```
Before:  axis spec = (metric_key, low_float, high_float)
After:   axis spec = (metric_key, low_float, high_float) | (cat_key, [val1, val2, ...])
```

`sweep_nd` already takes pre-built `value_arrays` per axis — the only change
needed there is to allow string arrays.  The engine's inner loop calls
`_eval_point(cold_config, noise, swept)` where `swept[k] = v`; for categorical
values `v` is already a string, which routes to `cfg[k] = v` (no coercion).

---

## Files to touch

| File | Change |
|------|--------|
| `gui/constants.py` | Add `CatMetricDef`, `CATEGORICAL_METRICS`, `CAT_METRIC_BY_KEY` |
| `gui/dse_engine.py` | Extend `COLD_PATH_KEYS`; fix `_eval_point` coercion; categorical branch in `_metric_values`; budget ignores cat axis size |
| `gui/components.py` | `make_metric_selector`: add hidden checklist alongside slider; toggle via callback |
| `gui/app.py` | `run_sweep`: read checklist values for cat axes; axis label mapping for plots |
| `gui/plotting.py` | Accept string tick labels on axes (if not already) |

No Rust/Python library changes needed — all qualitative params are already
accepted by `run_cold` as string kwargs.

---

## Existing behaviour to preserve

- Numeric axes: slider UI, log/linear scale, budget-capped point count — **unchanged**.
- `num_qubits` / `num_cores` cold-path budgeting — **unchanged**.
- Hot-path noise sweeps — **unchanged**.
- TeleSABRE routing when `routing_algorithm="telesabre"` — **unchanged**.
