# qusim DSE GUI — View Implementation Plan

## Overview

This document specifies every visualization view the DSE GUI should support. Views are organized into three tiers: **sweep views** (parameter-space plots driven by the axis/range system), **analysis views** (computed from sweep results, no re-simulation needed), and **introspection views** (per-design-point deep dives into a single simulation result).

Each view entry defines: the research question it answers, the Plotly trace type, the data contract from the backend, interaction behaviors, and implementation notes.

---

## Tier 1 — Sweep views

These are the core plots driven by the left-sidebar axis system. The user selects 1–3 parameters to sweep, sets ranges, and the backend computes a grid of `qusim.map_circuit()` calls. The output metric (fidelity by default) is always the dependent variable.

### 1.1 Line sweep (1 axis)

**Question**: "How does fidelity change as I vary `num_cores` from 2 to 12?"

**Plotly trace**: `scatter` with `mode: 'lines+markers'`

**Data shape**: Two 1D arrays — `x_values: float[]` (swept parameter) and `y_values: float[]` (output metric).

**Behavior**:
- Primary line shows the selected output metric (default: `overall_fidelity`).
- A dropdown or toggle switches the metric to `total_epr_pairs`, `total_teleportations`, or `circuit_depth_after_routing`. The line redraws from cached data — no re-simulation.
- When the user enables "multi-seed mode" (e.g. seeds 0–4), the backend runs N seeds per grid point. The plot shows the mean line with a shaded `±1σ` band using `fill: 'tonexty'`. This is critical because spectral clustering placement is stochastic.
- Optional: overlay a second metric on a secondary y-axis (right side). E.g. fidelity on the left, EPR pairs on the right, both as lines. Plotly supports this via `yaxis: 'y2'`.

**Interactions**:
- Hover shows exact values in a tooltip: parameter value, metric value, seed count.
- Click a point to open the Introspection view (Tier 3) for that specific configuration.
- Draggable horizontal threshold line. The user sets a target fidelity (e.g. 0.9) and the plot shades the region below it in red. The x-intercepts are labeled — these are the feasibility boundaries.

---

### 1.2 Contour heatmap (2 axes)

**Question**: "Where in the `num_cores × gate_error` space does fidelity exceed 0.9?"

**Plotly traces**: `heatmap` + `contour` overlay

**Data shape**: `x_values: float[]`, `y_values: float[]`, `z_grid: float[][]` (2D matrix, shape `[len(y), len(x)]`).

**Why contour, not just a color grid**: A flat heatmap forces the researcher to interpolate values by color, which is imprecise and fails entirely for colorblind users. Contour lines draw explicit iso-fidelity boundaries — the researcher can immediately see "fidelity = 0.9 runs along this curve." This is the single most impactful improvement over a basic heatmap.

**Behavior**:
- The heatmap provides the color fill. The contour overlay draws labeled iso-lines at regular intervals (auto) or at user-specified thresholds.
- A threshold slider draws one bold contour at the target fidelity. Everything below is dimmed or hatched as "infeasible."
- Metric switching (same as 1.1): toggle between fidelity / EPR pairs / teleportations. Redraws from cache.
- Log-scale toggle per axis. Essential for error rates (1e-5 to 1e-1).

**Interactions**:
- **Interactive crosshair slices**: Hovering on the heatmap shows two synchronized 1D slice plots — a horizontal slice (fixing y, varying x) and a vertical slice (fixing x, varying y). These appear as small panels adjacent to the heatmap or as overlays on the axes. This lets the researcher mentally "cut" through the surface without switching to a 1D view. Implementation: use Plotly's `onHover` event to read the cursor position, extract the corresponding row/column from `z_grid`, and update two secondary `scatter` traces in side panels.
- Click a cell to open the Introspection view for that configuration.
- Zoom/pan with Plotly's built-in tools. On zoom, the grid resolution should ideally increase (re-fetch denser data from backend for the visible region). If this is too complex, just interpolate.

**Colorscale**: Default to a sequential diverging scale centered on the threshold. E.g. blue (high fidelity) → white (threshold) → red (low fidelity). Let the user pick from: `Viridis`, `Plasma`, `RdYlBu`, `Inferno`. Provide a "colorblind-safe" toggle that restricts to perceptually uniform colormaps.

---

### 1.3 Isosurface (3 axes)

**Question**: "In a 3-parameter space, where is the boundary where fidelity = 0.9?"

**Plotly trace**: `isosurface` (preferred) or `volume` with slicing

**Data shape**: `x_values: float[]`, `y_values: float[]`, `z_values: float[]`, `fidelity_volume: float[][][]` (3D tensor, shape `[len(z), len(y), len(x)]`).

**Why isosurface instead of 3D scatter**: A point cloud in 3D is the most common DSE visualization failure. You can't judge depth, density, or structure from colored dots. The isosurface renders a shell at a specific fidelity level — the researcher's real question is "where does fidelity = X?" and the isosurface *is* that boundary. The surface can be semi-transparent, allowing nested surfaces at multiple thresholds (e.g. 0.9 and 0.7).

**Behavior**:
- A slider sets the iso-level. Dragging it reshapes the surface in real-time.
- Multiple iso-levels can be active simultaneously with decreasing opacity (outermost = lowest fidelity, most transparent).
- Alternative mode: **Orthogonal slice planes**. Three draggable cutting planes (one per axis) that show 2D heatmap cross-sections through the volume. The user drags the X-slice plane along the x-axis and sees a YZ heatmap at that x-value. This is often more readable than isosurfaces for researchers unfamiliar with volumetric rendering.

**Interactions**:
- Rotate/zoom with Plotly's 3D orbit controls.
- Camera angle is preserved across data updates (store `scene.camera` and restore it after `Plotly.react()`).
- Click on the surface to see the exact configuration and open Introspection view.
- Toggle between isosurface mode and slice-plane mode.

**Fallback for large grids**: If the 3D grid exceeds ~30×30×30 = 27,000 points, use marching cubes on the backend to extract the isosurface mesh and send triangle vertices directly instead of the full volume. This keeps the frontend fast.

---

## Tier 2 — Analysis views

These are computed from the sweep results after the simulation grid completes. They answer higher-level research questions and do not require re-running simulations. The frontend computes these from the cached result data, or the backend computes them on request.

### 2.1 Parallel coordinates

**Question**: "Across all parameters at once, which configurations give fidelity > 0.8?"

**Plotly trace**: `parcoords`

**Data shape**: A flat table where each row is one design point from the sweep. Columns are all swept + static parameters plus all output metrics. Shape: `[num_grid_points, num_params + num_metrics]`.

**Why this is essential**: In quantum architecture DSE you typically care about 5+ parameters (num_cores, qubits_per_core, gate_error_1q, gate_error_2q, T1, topology, placement). Sweep plots show at most 3 at a time. Parallel coordinates show all dimensions simultaneously. Each parameter gets a vertical axis; each design point is a polyline connecting its values across all axes. Lines are colored by fidelity.

**Behavior**:
- Each axis supports **brushing**: the user drags a range on any axis to filter. E.g. brush the fidelity axis to show only > 0.8, and the surviving polylines reveal which parameter combinations achieve that target.
- Multiple brushes can be active simultaneously (conjunctive filtering).
- Axes are reorderable by drag — placing correlated parameters adjacent reveals patterns.
- Color encoding: lines colored by the output metric value (continuous colorscale).

**Interactions**:
- Brush ranges are bidirectionally linked: adjusting a brush on the parallel coordinates view should update any other visible plot (e.g. highlight the corresponding region in a contour heatmap). This is "brushing and linking."
- Click a single polyline to open the Introspection view for that design point.
- Export the filtered subset as CSV.

**Data note**: This view works best when combining data from multiple sweeps. If the user has run several sweep configurations, concatenating them into one parallel coordinates plot gives a richer picture.

---

### 2.2 Slice plot (marginal effects)

**Question**: "Holding everything else at defaults, what is the isolated effect of each parameter on fidelity?"

**Plotly trace**: Grid of `scatter` subplots

**Data shape**: For each parameter, a 1D sweep holding all others at their default/center values. This can be extracted from the full grid data by slicing, or computed as a lightweight separate run.

**Why this matters**: The sweep heatmap shows joint effects of 2 parameters. The slice plot shows marginal effects of each parameter independently, making it easy to compare sensitivities side-by-side. This is Optuna's `plot_slice`.

**Layout**: A grid of small subplots (2 or 3 columns). One subplot per swept parameter. X-axis = parameter value, Y-axis = fidelity. All subplots share the same Y-axis scale for visual comparison.

**Behavior**:
- Each subplot shows a scatter of points (if sampling) or a line (if extracted from the grid).
- A vertical dashed line marks the current "default" value of each parameter.
- Subplots are sorted by visual impact (steepest slopes first) or alphabetically, togglable.

**Interactions**:
- Hover shows exact values.
- Click on a subplot to "promote" that parameter to the main 1D sweep view for deeper exploration.

---

### 2.3 Parameter importance (sensitivity analysis)

**Question**: "Which hardware parameter has the biggest impact on fidelity?"

**Plotly trace**: `bar` (horizontal)

**Data shape**: A ranked list of `(parameter_name, importance_score)` pairs.

**Computation**: Computed from the sweep grid data using one of:
- **Variance-based (Sobol indices)**: Decompose total fidelity variance into first-order contributions per parameter. The `SALib` Python library computes this efficiently from a grid.
- **Simpler alternative (fANOVA)**: functional ANOVA decomposition, which Optuna uses internally. Fits a random forest to the grid data and extracts feature importances.
- **Simplest fallback**: For each parameter, compute the range of fidelity across its sweep domain (max - min while averaging over others). Rank by range. Fast, no library needed, gives a reasonable first approximation.

**Behavior**:
- Horizontal bar chart, one bar per parameter, sorted by importance.
- Color encodes direction: if increasing the parameter improves fidelity, the bar is blue (positive); if it hurts fidelity, the bar is red (negative). This is the "sign" of the marginal effect at the current operating point.
- A toggle switches between first-order effects (individual) and total effects (including interactions).

**Why reviewers ask for this**: When publishing DSE results, the first question is "what matters?" This chart answers it. It also guides the researcher toward which parameters are worth sweeping more finely.

---

### 2.4 Pareto front (multi-objective)

**Question**: "Show me the tradeoff between fidelity and EPR pair consumption"

**Plotly trace**: `scatter` (2D) or `scatter3d` (3D)

**Data shape**: The same flat table as parallel coordinates, but projected to 2 or 3 output metrics.

**Why this matters**: In multi-core quantum architectures, high fidelity often requires more EPR pairs (teleportation is expensive). The researcher needs to see the tradeoff frontier — the set of designs where you can't improve one metric without worsening another.

**Behavior**:
- All design points plotted as dots. Dominated points (where another design is better in ALL metrics) are dimmed/gray.
- The Pareto frontier is drawn as a connected line (2D) or surface (3D) in a highlighted color.
- Hover shows the full configuration of each point.
- Axis selectors let the user choose which 2 or 3 metrics to plot. Default: fidelity vs EPR pairs.

**Interactions**:
- Click a Pareto-optimal point to open the Introspection view.
- Toggle "show dominated points" on/off for cleaner view.
- Brushing: select a region on the Pareto front and see which parameter ranges those designs fall in (linked to parallel coordinates if both views are open).

**Computation**: Pareto dominance is computed client-side from the grid results. For N points with M objectives, a simple O(N²M) pairwise comparison is fast enough for grids up to ~10,000 points.

---

### 2.5 Correlation matrix

**Question**: "Which parameters are correlated with each other or with the output metrics?"

**Plotly trace**: `heatmap` (annotated)

**Data shape**: A square matrix of Pearson or Spearman correlation coefficients between all parameters and all output metrics.

**Why useful**: Reveals unexpected parameter interactions. E.g. if `num_cores` and `T1` are jointly correlated with fidelity in a way that neither is alone, the researcher knows to investigate that interaction. Also useful for validating the experimental design — perfectly correlated parameters indicate redundancy.

**Behavior**:
- Square heatmap with parameter/metric names on both axes.
- Cells are annotated with the correlation coefficient (2 decimal places).
- Colorscale: diverging (blue for positive, red for negative, white for zero).
- Upper triangle can show Pearson, lower triangle Spearman (or just one).

**Interactions**:
- Click a cell to switch to a 2D contour view of that specific parameter pair.

---

## Tier 3 — Introspection views

These render data from a single `QusimResult` — one specific design point. The user reaches them by clicking on a point in any Tier 1 or Tier 2 plot. They show the internal details of how that particular circuit execution performs.

### 3.1 Per-qubit fidelity timeline

**Question**: "For this specific configuration, how does each qubit's fidelity evolve across circuit layers?"

**Plotly trace**: `heatmap` (x = layer index, y = qubit index, color = fidelity)

**Data shape**: `operational_fidelity_grid: float[][]` from `QusimResult`, shape `[num_layers, num_qubits]`.

**Behavior**:
- Rows are qubits, columns are circuit layers (time steps). Color is the fidelity at each point.
- Click a single qubit row to see its fidelity trace as a 1D line (equivalent to `result.get_qubit_fidelity_over_time(qubit_index)`).
- Overlay the teleportation events as markers — vertical lines or icons at the layers where a qubit was teleported between cores. This shows causality: fidelity drops correlate with teleportation events.

**Decomposed view**: Side-by-side or stacked heatmaps for the three fidelity components:
- Algorithmic fidelity (gate errors only)
- Routing overhead (SWAPs and teleportations)
- Coherence decay (T1/T2)

This lets the researcher see which noise source dominates at each point in the circuit.

---

### 3.2 Core placement map

**Question**: "How are qubits distributed across cores, and how does the mapping evolve?"

**Plotly trace**: Custom visualization (animated Sankey or alluvial diagram)

**Data shape**: `placements: int[][]` from `QusimResult`, shape `[num_layers + 1, num_qubits]`. Each entry is the core index where that qubit lives at that layer.

**Behavior**:
- An animated or scrubbable view showing qubits as colored dots inside core boxes, rearranging as the user steps through circuit layers.
- Teleportation events are highlighted as arrows between cores.
- Static summary: a Sankey-style flow showing qubit migration patterns across the full circuit. Thick flows mean many qubits move between those cores; thin flows mean few.

**Simpler alternative**: A table/heatmap where rows are qubits, columns are layers, and the cell color is the core index. Teleportation events appear as color transitions in a row.

---

### 3.3 Fidelity decomposition bar

**Question**: "For this design point, how much fidelity is lost to gates vs routing vs decoherence?"

**Plotly trace**: `bar` (stacked or grouped)

**Data shape**: Aggregate values extracted from the fidelity grids. For each component: `1 - product(component_grid)` gives the total fidelity loss from that source.

**Behavior**:
- A single stacked bar (or waterfall chart) showing: starting fidelity (1.0) → algorithmic loss → routing loss → coherence loss → final fidelity.
- A waterfall is more intuitive: the bar starts at 1.0 and successive blocks subtract each loss component, ending at the overall fidelity.
- When comparing two design points, show them side by side.

---

## Cross-cutting features

These apply across all views.

### Brushing and linking

When multiple views are visible simultaneously (e.g. parallel coordinates + contour heatmap in a split layout), selecting/brushing in one view highlights the corresponding data in all others. Implementation: a shared Zustand store holds the current selection (a set of design point indices). Each view component subscribes to this store and applies visual highlighting (brighter color, larger marker, etc.) to selected points.

### Threshold overlay

A global fidelity threshold (default: 0.9, adjustable). In every view, regions/points below this threshold are visually marked as infeasible:
- Line plots: shaded red region below the threshold line.
- Heatmaps: hatched or dimmed cells.
- Parallel coordinates: grayed-out polylines.
- Pareto front: points in the infeasible region are crossed out.

### Metric switching

A global dropdown (or per-view dropdown) selects the output metric to visualize. Options: `overall_fidelity` (default), `total_epr_pairs`, `total_teleportations`, `circuit_depth`. Switching redraws all views from cached data — never re-runs the simulation.

### Export

Every view supports:
- **PNG/SVG** export of the current plot (Plotly's built-in `toImage`).
- **CSV** export of the underlying data table.
- **LaTeX snippet** for the plot configuration (axis labels, parameter ranges) to paste directly into a paper.

### Seed aggregation

A toggle switches between "single seed" (fast, deterministic with `seed=42`) and "multi-seed" (runs N seeds, default N=5). In multi-seed mode:
- Line plots show mean ± σ bands.
- Heatmaps show mean values, with a toggle to show the std-dev heatmap.
- Parallel coordinates color by mean fidelity.
- The introspection view shows one seed at a time with a seed selector.

---

## View layout and navigation

The GUI should support two modes:

### Single-view mode (default)
One plot fills the center panel. A tab bar or dropdown at the top of the plot area selects which view is active. The left sidebar (sweep axes) and right sidebar (static config) always flank the plot.

### Multi-view mode
The center panel splits into 2 or 4 tiles (2×1 or 2×2). Each tile can show a different view of the same data. Brushing and linking works across tiles. This is essential for the parallel-coordinates-plus-heatmap workflow.

### View selector

A horizontal tab bar above the plot area with icons:

```
[Line] [Contour] [3D] | [Parallel] [Slices] [Importance] [Pareto] [Corr.] | [Qubit timeline] [Placement] [Decomposition]
 ── sweep ──            ── analysis ──────────────────────────────────────    ── introspection (per-point) ──────────────
```

Tier 3 (introspection) tabs are grayed out until the user clicks a specific design point in a Tier 1 or 2 view.

---

## Backend data contract

All views consume data from the same backend response. The WebSocket `result` message should include everything needed for all views:

```typescript
interface DSEResult {
  // Grid metadata
  axes: { name: string; values: number[]; log_scale: boolean }[];
  static_config: Record<string, number | string>;
  seeds: number[];

  // Per-grid-point results (flat arrays, row-major order matching the meshgrid)
  overall_fidelity: number[];       // length = product of axis steps × num_seeds
  total_epr_pairs: number[];
  total_teleportations: number[];

  // Shape info for reshaping flat arrays into N-d grids
  shape: number[];                  // e.g. [10, 15] for a 2-axis sweep
  num_seeds: number;

  // Per-point introspection data (only populated for the currently inspected point)
  inspected_point?: {
    config: Record<string, number | string>;
    operational_fidelity_grid: number[][];   // [num_layers, num_qubits]
    coherence_fidelity_grid: number[][];
    placements: number[][];                  // [num_layers+1, num_qubits]
    teleportation_events: { layer: number; qubit: number; from_core: number; to_core: number }[];
  };
}
```

Tier 2 analysis views (parallel coordinates, importance, Pareto, correlation) are computed client-side from the flat arrays. Tier 3 introspection data is fetched on-demand when the user clicks a design point — a separate WebSocket message requests the detailed `QusimResult` for that specific configuration.

---

## Implementation phases

| Phase | Views | Effort |
|-------|-------|--------|
| 1 | Line sweep (1.1) + contour heatmap (1.2) + metric switching | Core loop. ~3 days. |
| 2 | Parallel coordinates (2.1) + slice plot (2.2) + multi-view layout | The researcher's daily driver. ~3 days. |
| 3 | Isosurface (1.3) + Pareto front (2.4) + threshold overlay | Harder visualization. ~3 days. |
| 4 | Parameter importance (2.3) + correlation matrix (2.5) + seed aggregation | Analysis tooling. ~2 days. |
| 5 | Per-qubit timeline (3.1) + fidelity decomposition (3.3) + placement map (3.2) | Introspection layer. ~3 days. |
| 6 | Brushing-and-linking + export + LaTeX snippets | Polish. ~2 days. |
