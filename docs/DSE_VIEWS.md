# DSE GUI — View Catalog

Catalog of all plot types available in the qusim Design Space Exploration GUI, organized by the research question each answers.

---

## Sweep Plots

Visualize raw simulation results across 1, 2, or 3 swept hardware parameters.

### 01 — 1D Line Sweep

> *"How does fidelity change as I vary T1 from 10⁴ to 10⁶ ns?"*

Standard x-y line plot with spline-smoothed curves and fill-to-zero shading. Overlay multiple output metrics (fidelity, EPR pairs, circuit time) by switching the Y-axis dropdown. Axes auto-switch to log scale when the swept parameter uses log spacing.

| Detail | Value |
|--------|-------|
| Plot type | `scatter + fill` |
| Dimensionality | 1D sweep |
| View key | `line` |
| Sweep points | 60 |

**Enhancements:** threshold overlay (horizontal dashed lines with red-shaded infeasible region), unified hover with 4-decimal precision.

---

### 02 — 2D Heatmap

> *"Where in the T1 × T2 space does fidelity drop below 0.5?"*

Flat color grid mapping two swept parameters to an output metric via a continuous grayscale colorscale. Quick overview of the design space, but no iso-level detail.

| Detail | Value |
|--------|-------|
| Plot type | `heatmap` |
| Dimensionality | 2D sweep |
| View key | `heatmap` |
| Sweep points | 30 × 30 |
| Colorscale | Grayscale (#2B2B2B → #F0F0F0) |

**Enhancements:** hover tooltips (X, Y coordinates + output value at 4-decimal precision).

---

### 03 — 2D Contour

> *"Show me exact iso-fidelity boundaries in the gate error × teleport error space."*

Heatmap with labeled contour lines overlaid, showing exact iso-level boundaries. This is the default 2D view because contours let the researcher read precise fidelity thresholds without guessing from color gradients.

| Detail | Value |
|--------|-------|
| Plot type | `contour + heatmap` |
| Dimensionality | 2D sweep |
| View key | `contour` |
| Sweep points | 30 × 30 |
| Colorscale | Diverging 5-point (red → yellow → blue) |

**Enhancements:** threshold overlays rendered as bold colored iso-lines with labels, hover tooltips.

---

### 04 — 3D Scatter

> *"In a 3-parameter space, where do the high-fidelity configurations cluster?"*

3D point cloud with each design point colored by the output metric. Full rotation, zoom, and pan. Falls back to this view when the grid is too sparse for isosurface rendering.

| Detail | Value |
|--------|-------|
| Plot type | `scatter3d` |
| Dimensionality | 3D sweep |
| View key | `scatter3d` |
| Sweep points | 12 × 12 × 12 |
| Point size | 3.5px, opacity 0.85 |
| Colorscale | Grayscale |

**Enhancements:** threshold overlays as larger marker groups (6px) with contrasting colors, 3D scene manipulation, hover with all coordinates + fidelity.

---

### 05 — 3D Isosurface

> *"In a 3-parameter space, show me the shell where fidelity = 0.9."*

Volumetric surface rendering at user-defined iso-levels. The researcher sets threshold values and the surfaces reshape to show where fidelity crosses each boundary. This is the default 3D view.

| Detail | Value |
|--------|-------|
| Plot type | `isosurface` |
| Dimensionality | 3D sweep |
| View key | `isosurface` |
| Sweep points | 12 × 12 × 12 |
| Minimum grid | 3 × 3 × 3 (27 points); falls back to scatter3d if insufficient |
| Colorscale | Monochromatic per surface |

**Enhancements:** multiple threshold levels rendered as separate isosurface shells with increasing opacity, 3D scene manipulation.

---

### 06 — Frozen Heatmap

> *"Fix the third parameter at a specific value and show me a 2D slice of the 3D space."*

2D heatmap slice through a 3D sweep volume. A slider controls the frozen parameter value, and the plot updates instantly via client-side JavaScript interpolation — no server round-trip.

| Detail | Value |
|--------|-------|
| Plot type | `frozen heatmap` |
| Dimensionality | 3D sweep (2D slice) |
| View key | `frozen_heatmap` |
| Interpolation | Trilinear 3D → bilinear 2D (JS) |
| Colorscale | Grayscale |

**Enhancements:** zero-latency slider updates, hover tooltips.

---

### 07 — Frozen Contour

> *"Same 2D slice, but with iso-fidelity contour lines."*

Same frozen-slider concept as Frozen Heatmap, but rendered as a contour plot with labeled iso-lines. Supports threshold overlays.

| Detail | Value |
|--------|-------|
| Plot type | `frozen contour` |
| Dimensionality | 3D sweep (2D slice) |
| View key | `frozen_contour` |
| Interpolation | Trilinear 3D → bilinear 2D (JS) |
| Colorscale | Diverging 5-point |

**Enhancements:** threshold overlays as colored iso-lines, zero-latency slider updates.

---

## Analysis Plots

Dimensionality-agnostic views that work on any 1D/2D/3D sweep by flattening the grid to a table of (parameters → outputs). These answer the questions researchers spend most of their time on.

### 08 — Parallel Coordinates

> *"Across all parameters simultaneously, which configurations give fidelity > 0.8?"*

Every parameter and output metric gets a vertical axis. Each design point is a polyline connecting its values across all axes. Lines are colored by the selected output metric. The researcher brushes (drags a range) on any axis to filter — e.g., drag on the fidelity axis to show only > 0.8, then visually see which parameter ranges those solutions cluster in.

| Detail | Value |
|--------|-------|
| Plot type | `parcoords` |
| View key | `parallel` |
| Coloring | Diverging 5-point colorscale by output metric |
| Interactivity | Range brushing on each axis |

---

### 09 — Slice Plot (Marginal Effect)

> *"Holding everything else constant, what is the isolated effect of T1 on fidelity?"*

One subplot per swept parameter arranged in a grid (max 3 columns). In each subplot, X = parameter value, Y = output metric, with all other parameters fixed at their median values. Shows marginal sensitivity so you can compare per-parameter effects at a glance.

| Detail | Value |
|--------|-------|
| Plot type | `scatter subplots` |
| View key | `slices` |
| Layout | Grid, max 3 columns |
| Data | Other params fixed at median |

---

### 10 — Parameter Importance (Sensitivity)

> *"Which hardware parameter has the biggest impact on fidelity?"*

Horizontal bar chart ranking parameters by their contribution to output variance. Importance is computed as max(output) − min(output) across each parameter's range. Bars are sorted ascending so the most important parameter is visually prominent at the top.

| Detail | Value |
|--------|-------|
| Plot type | `bar` |
| View key | `importance` |
| Metric | Range-based importance (max − min) |
| Sorting | Ascending (most important at top) |

---

### 11 — Pareto Front (Multi-Objective)

> *"Show me the tradeoff between fidelity and EPR pair consumption."*

Scatter plot with Overall Fidelity on the Y-axis and Total EPR Pairs on the X-axis. The Pareto frontier is highlighted with a connecting line; dominated points are dimmed in gray. Hover shows full configuration for each point.

| Detail | Value |
|--------|-------|
| Plot type | `scatter + line` |
| View key | `pareto` |
| X-axis | Total EPR Pairs (cost) |
| Y-axis | Overall Fidelity (quality) |
| Frontier color | Blue (#4575b4) |
| Dominated color | Gray |

**Enhancements:** threshold overlays as horizontal dashed lines.

---

### 12 — Correlation Matrix

> *"Which parameters and outputs are statistically related?"*

Annotated heatmap showing Spearman rank correlations between all parameter–output pairs. Each cell displays the correlation coefficient (−1.0 to +1.0). Quickly reveals which parameters drive which outputs and whether any parameters are redundant.

| Detail | Value |
|--------|-------|
| Plot type | `annotated heatmap` |
| View key | `correlation` |
| Metric | Spearman rank correlation |
| Colorscale | Diverging (−1 to +1, centered at white) |
| Annotations | Coefficient values in each cell |

---

## Enhancement Techniques

These features apply across multiple plot types.

| Technique | Description | Available in |
|-----------|-------------|-------------|
| **Threshold overlay** | Up to 5 configurable iso-levels. Rendered as dashed lines (1D), colored iso-lines (2D contour), marker groups (3D scatter), or surface shells (isosurface). Default levels: 0.3, 0.6, 0.9. Colors: 5-point palette from red to blue. | Line, Contour, Scatter3D, Isosurface, Frozen Contour, Pareto |
| **Client-side interpolation** | Trilinear → bilinear JavaScript interpolation for frozen slider views. Slider drag updates the plot with zero server latency. | Frozen Heatmap, Frozen Contour |
| **Hover tooltips** | Unified hover mode with custom templates. Coordinates at 3 significant figures, output values at 4 decimal places. | All views |
| **Axis log scaling** | Automatic log scale when the swept parameter has `log_scale=True`. | All sweep views |
| **Metric switching** | Toggle the output metric (Y-axis or color axis) between 6 options without re-running the sweep. | All views |
| **Range brushing** | Click-drag on any axis to filter design points interactively. | Parallel Coordinates |
| **PNG export** | Download via Plotly modebar at 1200×800px, 2× scale. | All views |
| **CSV export** | Export raw sweep data to CSV. | All views |

---

## Output Metrics

Any view can display these output metrics on the Y-axis, color axis, or as the analysis target:

| Metric | Key | Range |
|--------|-----|-------|
| Overall Fidelity | `overall_fidelity` | 0–1 |
| Algorithmic Fidelity | `algorithmic_fidelity` | 0–1 |
| Routing Fidelity | `routing_fidelity` | 0–1 |
| Coherence Fidelity | `coherence_fidelity` | 0–1 |
| Circuit Time | `total_circuit_time_ns` | ns |
| Total EPR Pairs | `total_epr_pairs` | count |

---

## Sweepable Parameters

These hardware parameters can be placed on any sweep axis (up to 3 simultaneously):

| Parameter | Key | Scale | Unit |
|-----------|-----|-------|------|
| 1Q Gate Error | `single_gate_error` | Log | — |
| 2Q Gate Error | `two_gate_error` | Log | — |
| Teleport Error/Hop | `teleportation_error_per_hop` | Log | — |
| T1 (Relaxation) | `t1` | Log | ns |
| T2 (Dephasing) | `t2` | Log | ns |
| 1Q Gate Time | `single_gate_time` | Log | ns |
| 2Q Gate Time | `two_gate_time` | Log | ns |
| Teleport Time/Hop | `teleportation_time_per_hop` | Log | ns |
| Readout Mitigation | `readout_mitigation_factor` | Linear | — |

---

## Interactivity Matrix

| Feature | 1D Line | 2D Heat | 2D Contour | 3D Scatter | 3D Iso | Frozen | Analysis |
|---------|---------|---------|------------|------------|--------|--------|----------|
| Hover tooltips | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Thresholds | ✓ | — | ✓ | ✓ | ✓ | Contour | Pareto |
| Zoom / Pan | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3D Rotation | — | — | — | ✓ | ✓ | — | — |
| Range brushing | — | — | — | — | — | — | Parallel |
| Frozen slider | — | — | — | — | — | ✓ | — |
| PNG export | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CSV export | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
