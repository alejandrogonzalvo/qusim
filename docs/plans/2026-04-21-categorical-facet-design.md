# Design: Categorical Variables as Faceted Subplots

**Date:** 2026-04-21
**Branch:** tele-sabre

## Goal

Allow the researcher to compare qualitative (categorical) parameters visually
by faceting the plot into a grid of subplots — one panel per categorical
combination. Numeric sweep axes stay on the plot axes; categoricals split the
plot into panels.

Example: a T1 × T2 heatmap with "Compare" enabled on `routing_algorithm`
(2 options) and `topology_type` (3 options) produces a 2×3 grid of heatmaps,
each with the same color scale, so the researcher instantly sees which
routing/topology combination dominates.

## Approach

**Faceting in `run_sweep` (Approach A).** The engine stays numeric-only. The
`run_sweep` callback detects active compare toggles, builds the cartesian
product of categorical values, runs one full sweep per combination, and passes
all results to the plotting layer which arranges them as subplots.

This was chosen over engine-level faceting (Approach B) because each
categorical value requires a fresh cold compilation anyway — there is no
shared state between facets that the engine could exploit. A separate
`gui/faceting.py` wrapper (Approach C) was rejected as unnecessary
indirection for what is essentially a loop.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Multiple compare toggles active simultaneously | Cartesian product grid (e.g., 2×3) |
| 1D presentation | Side-by-side subplots (not overlaid lines) |
| 3D presentation | Facet into subplot grid with per-panel `scene` |
| Analysis views (Parallel, Slices, Importance, Pareto, Correlation) | Faceted subplots, same grid layout as sweep views |

## UI Changes

### Left Sidebar — New "Categorical" Section

Below the numeric sweep axes and add/remove buttons, a new section appears:

```
┌─ SWEEP AXES ─────────────────────┐
│  Metric 1: [T1 ▾]  [===slider===]│
│  Metric 2: [T2 ▾]  [===slider===]│
│  + Add axis    − Remove           │
│  ~1,800 points                    │
├─ CATEGORICAL ─────────────────────┤
│  Circuit type    [QFT ▾]  ☐ Cmp  │
│  Inter-core topo [Ring ▾] ☐ Cmp  │
│  Intra-core topo [A2A ▾]  ☐ Cmp  │
│  Placement       [Rand ▾] ☐ Cmp  │
│  Routing         [HQA ▾]  ☐ Cmp  │
└───────────────────────────────────┘
```

Each row: label + dropdown + "Cmp" checkbox.

- **Compare OFF** (default): dropdown selects a single value fed into
  `cold_config`. Current behavior unchanged.
- **Compare ON**: dropdown grayed out, all options for that categorical
  participate. Plot facets.

The categorical dropdowns move out of the right panel's Circuit tab into the
left sidebar. The right panel retains: Qubits slider, Cores slider, Seed,
Dynamic Decoupling, Sweep Budget, Noise tab, Thresholds tab.

Dropdown IDs stay the same (`cfg-circuit-type`, `cfg-topology`, etc.) to
minimize callback rewiring.

## Sweep Execution

### Facet loop in `run_sweep`

1. Read the 5 compare checkboxes.
2. For each active compare, collect the full options list from `constants.py`.
3. Build the cartesian product → `facet_combos`.
4. When no toggles are active: `facet_combos = [{}]` — single sweep, zero
   regression.
5. For each combo:
   - Merge into `cold_config` (overwriting dropdown values).
   - `_engine.run_cold(...)` — fresh cold compilation.
   - Run sweep (1D/2D/3D/ND) as today.
   - Collect `(label_dict, sweep_data)`.

### Progress / budget

- Progress reports total across all facets.
- Estimated-points display: `points_per_facet × num_facets`.

### Sweep data structure

```python
# Unfaceted (backward-compatible):
sweep_data = {"metric_keys": [...], "xs": [...], "grid": [...]}

# Faceted:
sweep_data = {
    "metric_keys": [...],
    "facets": [
        {"label": {"routing_algorithm": "HQA + Sabre", ...},
         "xs": [...], "grid": [...]},
        ...
    ],
    "facet_keys": ["routing_algorithm", "topology_type"],
}
```

When `"facets"` is absent, all downstream code behaves as today.

## Plotting

### Subplot grid

- **1 facet dimension**: 1 row × N columns.
- **2 facet dimensions**: first key → rows, second → columns.
- **3+ facet dimensions**: rows = product of all but last key,
  columns = last key.

Each panel gets a subplot title (e.g., "HQA+Sabre / Ring").

### Per-view behavior

| View type | Subplot spec |
|-----------|-------------|
| 1D line | `make_subplots(rows, cols)` — xy |
| 2D heatmap/contour | xy, shared colorscale across panels |
| 3D scatter/isosurface | `specs=[[{"type": "scene"}, ...]]` |
| Frozen heatmap/contour | Same as 2D, frozen slider applies to all panels |
| Analysis views | Same grid, each panel from its facet's data |

### Color scale synchronization

Heatmaps/contours: global `zmin`/`zmax` across all facets.
1D line: shared Y-axis range.

### Threshold overlays

Drawn identically in every panel. No logic change, just repeated per subplot.

### CSV export

Extra columns per facet key:

```
routing_algorithm, topology_type, t1, t2, overall_fidelity, ...
HQA + Sabre,       Ring,         1e4, 5e4, 0.8731, ...
```

## Files to Touch

| File | Change |
|------|--------|
| `gui/constants.py` | `CatParamDef` dataclass, `CATEGORICAL_PARAMS` registry |
| `gui/components.py` | `make_categorical_section()`, remove categoricals from `make_fixed_config_panel` |
| `gui/app.py` | Categoricals in `_left_sidebar()`, compare States in `run_sweep`, facet loop, facet-aware progress/budget |
| `gui/plotting.py` | `_build_faceted_figure()`, shared colorscale, subplot titles, faceted CSV |

### Files NOT touched

- `gui/dse_engine.py` — engine stays numeric-only
- `gui/interpolation.py` — called per-panel in a loop
- Rust/Python library — no changes
