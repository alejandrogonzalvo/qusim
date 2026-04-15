# Agent Handoff — DSE GUI View System

## Context

You are continuing implementation of the qusim DSE GUI view system. This is a Dash + Plotly web application for Design Space Exploration of multi-core quantum architectures.

## Before you start

Read these files in order:

1. `gui/README.md` — Architecture, implemented features, planned features, parameter registry
2. `DSE_VIEW_IMPLEMENTATION_PLAN.md` — Full specification for every view (data shapes, Plotly traces, interactions, colorscales)
3. `gui/plotting.py` — All plot functions live here. Follow the existing patterns.
4. `gui/constants.py` — `ANALYSIS_TABS` list is where new analysis views get registered
5. `gui/components.py` — Tabbed right panel (Circuit/Noise/Thresholds), view tab bar, metric selectors
6. `gui/app.py` — `build_figure()` dispatches by `view_type`; callbacks for sweep, replot, tab switching, CSV export, threshold sync
7. `tests/test_plotting_views.py` — All view tests. Follow the existing test patterns.

## What's done

- **Tier 1 complete**: Line (1D), Heatmap (2D), Contour (2D), Scatter3d (3D), Isosurface (3D)
- **Tier 2 complete**: Parallel Coordinates, Slice Plot, Parameter Importance, Pareto Front, Correlation Matrix
- **Multi-threshold system**: Up to 5 iso-levels (defaults: 0.3, 0.6, 0.9) with per-threshold color pickers. Isosurface renders one shell per level. Scatter3d highlights boundary points. Line/contour/pareto show threshold lines when checkbox enabled.
- **Cross-cutting**: CSV export, threshold overlay, tabbed config panel (Circuit/Noise/Thresholds), auto-run on startup
- **Defaults**: T1 x T2 x 2Q Gate Time sweep at full range, 1 core, isosurface view
- **UI polish**: Darkened section titles with separator lines, tabbed right panel to avoid scrolling
- **104 tests passing** (92 view + 12 core), branch: `dse-ui`

## Next tasks — pick from these options

### Option A: Cross-cutting features (no backend changes needed)

#### A.1 — Multi-view layout
- Split center panel into 2x1 or 2x2 tiles
- Each tile shows a different view of the same sweep data
- Add a layout toggle button near the view tab bar
- See §Layout in `DSE_VIEW_IMPLEMENTATION_PLAN.md`

#### A.2 — Brushing & linking (requires A.1)
- Selection in one tile highlights corresponding data in all others
- Shared selection store in `dcc.Store` (set of design point indices)
- Each view applies visual highlighting to selected points

#### A.3 — LaTeX snippet export
- Add a "LaTeX" button next to CSV
- Generates axis labels, parameter ranges, and figure caption as a LaTeX snippet
- Copy to clipboard or download as .tex file

### Option B: Tier 3 — Introspection views (requires backend changes)

These need the DSE engine to expose per-point `QusimResult` data (operational fidelity grid, placements, teleportation events). The backend change is: when a user clicks a design point, fetch the full `QusimResult` for that configuration.

#### B.1 — Per-qubit fidelity timeline (§3.1)
- Heatmap: x = layer index, y = qubit index, color = fidelity
- Click a qubit row to see its 1D fidelity trace
- Overlay teleportation events as markers

#### B.2 — Fidelity decomposition bar (§3.3)
- Waterfall chart: 1.0 → algorithmic loss → routing loss → coherence loss → final
- Side-by-side comparison of two design points

#### B.3 — Core placement map (§3.2)
- Heatmap: rows = qubits, columns = layers, color = core index
- Teleportation events appear as color transitions

### Option C: Seed aggregation (requires backend changes)

- Multi-seed mode toggle (N seeds, default N=5)
- Backend runs N seeds per grid point
- Line plots: mean ± σ bands
- Heatmaps: mean values with std-dev toggle
- Parallel coordinates: color by mean fidelity

## TDD workflow (mandatory)

For each view/feature:
1. **Write tests first** in `tests/test_plotting_views.py` — import the new function, test trace types, data shapes, routing
2. **Run tests** — confirm they fail (RED)
3. **Commit** the failing tests
4. **Implement** — add function to `plotting.py`, register in `ANALYSIS_TABS` if needed, handle in `build_figure()`
5. **Run tests** — confirm they pass (GREEN)
6. **Commit** the implementation
7. **Run full suite** — `.venv/bin/python -m pytest tests/ -v` — all must pass before moving on

## Patterns to follow

- Plot functions take `sweep_data: dict` (the store format) + `output_key: str`
- Analysis views use `_flatten_sweep_to_table()` to get a flat table from any-dimension grid
- Threshold-aware functions accept `thresholds: list[float] | None` and `threshold_colors: list[str] | None`
- Colorscales: use the diverging `[#d73027, #fc8d59, #fee08b, #91bfdb, #4575b4]` for data views
- Layout: spread `{**_LAYOUT_BASE, "margin": dict(...)}` to avoid duplicate kwarg errors
- Test fixtures: reuse `sweep_data_store_1d`, `sweep_data_store_2d`, `sweep_data_store_3d`
- The `build_figure` dispatcher checks analysis view types before the dimension-specific sweep views
- Right panel uses `dcc.Tabs` with 3 tabs; new config sections go into existing tabs or a new tab
- Color pickers use `html.Div` swatch + `dcc.Input(type="text")` for hex values (Dash `dcc.Input` doesn't support `type="color"`)
- Clientside callbacks sync swatch background to hex input value

## Do not

- Do not modify the DSE engine (`dse_engine.py`) unless implementing Tier 3 or seed aggregation
- Do not change existing test assertions — only add new tests
- Do not add external dependencies (scipy, scikit-learn, etc.) — use numpy only
- Do not use `html.Input` — Dash's `html` module doesn't have it. Use `dcc.Input` or `html.Div` with styling
