# qusim DSE GUI

Interactive Design Space Explorer for multi-core quantum architectures. Sweep hardware parameters (gate errors, coherence times, teleportation costs) and visualize their impact on circuit fidelity in real time.

## Quick start

```bash
cd /path/to/qusim
pip install -r gui/requirements.txt
python gui/app.py
# Open http://localhost:8050
```

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
┌─────────────────────────────────────────────────────────┐
│  Topbar (logo, status, Run button)                      │
├──────────┬──────────────────────────────┬───────────────┤
│  Left    │  View Tab Bar                │  Right panel  │
│  sidebar │  [Heatmap] [Contour] | [Par] │  (fixed       │
│  (sweep  ├──────────────────────────────┤   config,     │
│   axes,  │                              │   noise,      │
│   range  │  Main Plot                   │   output      │
│   slider)│                              │   metric)     │
└──────────┴──────────────────────────────┴───────────────┘
```

## Features

### Implemented

#### Sweep views (Tier 1)

| View | Tab | Axes | Description |
|------|-----|------|-------------|
| Line plot | `Line` | 1D | Scatter with spline interpolation and fill. Log-scale x-axis for error rates. |
| Heatmap | `Heatmap` | 2D | Color grid with grayscale colorscale. Log-scale axis support. |
| Contour heatmap | `Contour` | 2D | Heatmap + labeled iso-line overlay (diverging RdYlBu colorscale). Default for 2D sweeps. |
| 3D scatter | `Scatter` | 3D | Scatter3d with points colored by fidelity. |
| Isosurface | `Isosurface` | 3D | Volumetric iso-level surfaces with adjustable opacity. Falls back to scatter3d on sparse grids (<3x3x3). |

#### Analysis views (Tier 2)

| View | Tab | Description |
|------|-----|-------------|
| Parallel coordinates | `Parallel` | All parameters + output metrics on parallel axes. Lines colored by selected output. Axis brushing for filtering. Works on any sweep dimensionality. |
| Slice plot | `Slices` | Grid of subplots showing marginal effect of each swept parameter. 1D slices through center of grid holding others at defaults. Shared Y-axis scale. |
| Parameter importance | `Importance` | Horizontal bar chart ranked by range-based sensitivity (max−min of output across each parameter's sweep domain). |
| Pareto front | `Pareto` | Scatter plot of fidelity vs EPR pairs. Pareto-optimal points highlighted with connected line; dominated points dimmed. |
| Correlation matrix | `Corr.` | Annotated heatmap of Spearman rank correlations between all swept parameters and output metrics. Diverging blue-white-red colorscale. |

#### View system

- **Tab bar** above the plot with sweep-specific tabs + analysis tabs separated by `|`
- Tabs auto-select based on sweep dimensionality (1D→Line, 2D→Contour, 3D→Scatter)
- Switching tabs re-renders from cached data — no re-simulation
- View state persisted in `dcc.Store`

#### Configuration

- **Left sidebar**: Up to 3 sweep axes with dropdown (parameter selector) + range slider
- **Right panel**: Circuit type (QFT, GHZ, Random), qubits (4-80), cores (1-16), topology (ring, all-to-all, linear), placement (random, spectral clustering), seed, dynamic decoupling toggle
- **Noise panel**: 9 sweepable hardware parameters with log/linear sliders. Swept parameters auto-hidden from the fixed config panel.
- **Output metric selector**: Overall fidelity, algorithmic/routing/coherence fidelity, circuit time, EPR pairs

#### Sweep engine

- 1D sweeps: 60 points
- 2D sweeps: 30x30 grid
- 3D sweeps: 12x12x12 grid
- Metric switching redraws from cache (no re-sweep)

### Planned (not yet implemented)

#### Tier 3 — Introspection views (per-design-point)

| View | Description | Plan ref |
|------|-------------|----------|
| **Per-qubit fidelity timeline** (3.1) | Heatmap of qubit fidelity over circuit layers. Click-to-inspect single qubit. | §3.1 |
| **Core placement map** (3.2) | Animated or scrubbable qubit-to-core mapping across layers. | §3.2 |
| **Fidelity decomposition** (3.3) | Waterfall chart: 1.0 → algorithmic loss → routing loss → coherence loss → final. | §3.3 |

#### Cross-cutting features

| Feature | Description | Plan ref |
|---------|-------------|----------|
| **Threshold overlay** | Global fidelity cutoff line (default 0.9). Regions below shaded/dimmed in all views. | §Threshold |
| **Multi-view layout** | 2x1 or 2x2 split of the center panel. Different views of same data side by side. | §Layout |
| **Export** | PNG/SVG export (built-in), CSV data export, LaTeX snippet for paper figures. | §Export |
| **Brushing & linking** | Selection in one view highlights corresponding data in all others. | §Brushing |
| **Seed aggregation** | Multi-seed mode with mean ± σ bands on line plots, std-dev heatmaps. | §Seeds |

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

Current coverage: 83 tests (71 view tests + 12 core tests), all passing.

## Implementation plan

The full view specification lives in `DSE_VIEW_IMPLEMENTATION_PLAN.md` at the project root. It covers:
- Data contracts and Plotly trace types for every view
- Interaction behaviors (hover, click-to-inspect, brushing)
- Backend data shape requirements
- Implementation phases and effort estimates
