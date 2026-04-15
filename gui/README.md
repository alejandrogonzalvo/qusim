# qusim DSE GUI

Interactive Design Space Explorer for multi-core quantum architectures. Sweep hardware parameters (gate errors, coherence times, teleportation costs) and visualize their impact on circuit fidelity in real time.

## Quick start

```bash
cd /path/to/qusim
pip install -r gui/requirements.txt
python gui/app.py
# Open http://localhost:8050
```

On startup the app auto-runs a 3D sweep (T1 x T2 x 2Q Gate Time) and renders an isosurface view.

## Architecture

```
gui/
├── app.py           # Dash application, layout, callbacks
├── components.py    # UI component factories (sidebar, config panel, tab bar)
├── plotting.py      # Plotly figure builders for all view types
├── constants.py     # Parameter registry, view tab definitions, defaults
├── dse_engine.py    # Two-stage DSE engine (cold path + hot path)
├── assets/style.css # Global CSS (light theme)
├── requirements.txt # Python dependencies
└── __init__.py
```

**Two-stage execution model:**
- **Cold path** — circuit transpilation + HQA mapping + SABRE routing (~1-10s). Cached when circuit/topology don't change.
- **Hot path** — fidelity estimation reusing cached placements (<1ms per point). Used during sweeps.

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Topbar (logo, status, Run button)                          │
├──────────┬────────────────────────────────┬─────────────────┤
│  Left    │  View Tab Bar         [CSV]    │  CONFIGURATION  │
│  sidebar │  [Scatter][Iso] | [Par][Slices]│  ┌────┬───┬───┐ │
│  (sweep  ├────────────────────────────────┤  │Circ│Noi│Thr│ │
│   axes,  │                                │  ├────┴───┴───┤ │
│   range  │  Main Plot                     │  │ tab content │ │
│   slider)│                                │  │             │ │
└──────────┴────────────────────────────────┴─────────────────┘
```

## Features

### Implemented

#### Sweep views (Tier 1)

| View | Tab | Axes | Description |
|------|-----|------|-------------|
| Line plot | `Line` | 1D | Scatter with spline interpolation and fill. Log-scale x-axis for error rates. |
| Heatmap | `Heatmap` | 2D | Color grid with grayscale colorscale. Log-scale axis support. |
| Contour heatmap | `Contour` | 2D | Heatmap + labeled iso-line overlay (diverging RdYlBu colorscale). Default for 2D sweeps. |
| 3D scatter | `Scatter` | 3D | Scatter3d with points colored by fidelity. Threshold iso-levels highlighted as larger colored boundary markers. |
| Isosurface | `Isosurface` | 3D | One isosurface shell per threshold level (default 3: 0.3, 0.6, 0.9). Each with user-configurable color. Default 3D view. |

#### Analysis views (Tier 2)

| View | Tab | Description |
|------|-----|-------------|
| Parallel coordinates | `Parallel` | All parameters + output metrics on parallel axes. Lines colored by selected output. Axis brushing for filtering. Works on any sweep dimensionality. |
| Slice plot | `Slices` | Grid of subplots showing marginal effect of each swept parameter. 1D slices through center of grid holding others at defaults. Shared Y-axis scale. |
| Parameter importance | `Importance` | Horizontal bar chart ranked by range-based sensitivity (max-min of output across each parameter's sweep domain). |
| Pareto front | `Pareto` | Scatter plot of fidelity vs EPR pairs. Pareto-optimal points highlighted with connected line; dominated points dimmed. |
| Correlation matrix | `Corr.` | Annotated heatmap of Spearman rank correlations between all swept parameters and output metrics. Diverging blue-white-red colorscale. |

#### Cross-cutting features

| Feature | Description |
|---------|-------------|
| Multi-threshold overlay | Up to 5 iso-level thresholds (3 defaults: 0.3, 0.6, 0.9). Each has a color swatch + hex input. Isosurface/scatter3d always use them; other views show them when "Show on non-3D views" is checked. Renders as dashed lines (1D/pareto), bold contours (2D), boundary markers (scatter3d), or isosurface shells (isosurface). |
| CSV export | Download button above the plot exports the full sweep data as a CSV file. |
| PNG/SVG export | Built-in Plotly toolbar export at 2x resolution (1200x800). |
| Tabbed config panel | Right panel split into Circuit / Noise / Thresholds tabs to avoid scrolling. |
| Auto-run on startup | Default sweep (T1 x T2 x 2Q Gate Time, 1 core, full range) runs automatically and shows isosurface view. |

#### Configuration

- **Left sidebar**: Up to 3 sweep axes with dropdown (parameter selector) + range slider. Defaults: T1, T2, 2Q Gate Time at full range.
- **Right panel (tabbed)**:
  - **Circuit tab**: Circuit type (QFT, GHZ, Random), qubits (4-80), cores (1-16, default 1), topology (ring, all-to-all, linear), placement (random, spectral clustering), seed, dynamic decoupling toggle
  - **Noise tab**: 9 sweepable hardware parameters with log/linear sliders. Swept parameters auto-hidden.
  - **Thresholds tab**: Output metric selector, up to 5 iso-levels with color picker + numeric input

#### Sweep engine

- 1D sweeps: 60 points
- 2D sweeps: 30x30 grid
- 3D sweeps: 12x12x12 grid
- Metric switching redraws from cache (no re-sweep)

### Planned (not yet implemented)

#### Cross-cutting features (remaining)

| Feature | Description | Plan ref |
|---------|-------------|----------|
| **Multi-view layout** | 2x1 or 2x2 split of the center panel. Different views of same data side by side. | §Layout |
| **Brushing & linking** | Selection in one view highlights corresponding data in all others. Requires multi-view. | §Brushing |
| **Seed aggregation** | Multi-seed mode with mean +/- sigma bands on line plots, std-dev heatmaps. Requires backend changes. | §Seeds |
| **LaTeX snippet export** | Export plot configuration as LaTeX code for paper figures. | §Export |

#### Tier 3 — Introspection views (per-design-point)

| View | Description | Plan ref |
|------|-------------|----------|
| **Per-qubit fidelity timeline** (3.1) | Heatmap of qubit fidelity over circuit layers. Click-to-inspect single qubit. | §3.1 |
| **Core placement map** (3.2) | Animated or scrubbable qubit-to-core mapping across layers. | §3.2 |
| **Fidelity decomposition** (3.3) | Waterfall chart: 1.0 -> algorithmic loss -> routing loss -> coherence loss -> final. | §3.3 |

**Note**: Tier 3 views require backend changes to expose per-point `QusimResult` data.

## Sweepable parameters

| Key | Label | Scale | Default |
|-----|-------|-------|---------|
| `single_gate_error` | 1Q Gate Error | log | 1e-4 |
| `two_gate_error` | 2Q Gate Error | log | 1e-3 |
| `teleportation_error_per_hop` | Teleport Error/Hop | log | 1e-2 |
| `t1` | T1 (Relaxation) | log | 100,000 ns |
| `t2` | T2 (Dephasing) | log | 50,000 ns |
| `single_gate_time` | 1Q Gate Time | log | 20 ns |
| `two_gate_time` | 2Q Gate Time | log | 100 ns |
| `teleportation_time_per_hop` | Teleport Time/Hop | log | 1,000 ns |
| `readout_mitigation_factor` | Readout Mitigation | linear | 0.0 |

## Tests

```bash
# Run all GUI view tests
python -m pytest tests/test_plotting_views.py -v

# Run full test suite
python -m pytest tests/ -v
```

Current coverage: 104 tests (92 view tests + 12 core tests), all passing.

## Implementation plan

The full view specification lives in `DSE_VIEW_IMPLEMENTATION_PLAN.md` at the project root. It covers:
- Data contracts and Plotly trace types for every view
- Interaction behaviors (hover, click-to-inspect, brushing)
- Backend data shape requirements
- Implementation phases and effort estimates
