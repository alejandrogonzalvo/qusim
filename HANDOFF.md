# Agent Handoff — DSE GUI View System

## Context

You are continuing implementation of the qusim DSE GUI view system. This is a Dash + Plotly web application for Design Space Exploration of multi-core quantum architectures.

## Before you start

Read these files in order:

1. `gui/README.md` — Architecture, implemented features, planned features, parameter registry
2. `DSE_VIEW_IMPLEMENTATION_PLAN.md` — Full specification for every view (data shapes, Plotly traces, interactions, colorscales)
3. `gui/plotting.py` — All plot functions live here. Follow the existing patterns.
4. `gui/constants.py` — `ANALYSIS_TABS` list is where new analysis views get registered
5. `gui/components.py` — `make_view_tab_bar()` renders the tab bar; analysis tabs appear after a `|` separator
6. `gui/app.py` — `build_figure()` dispatches by `view_type`; `on_view_tab_click()` handles tab switching
7. `tests/test_plotting_views.py` — All view tests. Follow the existing test patterns.

## What's done

- **Phase 1 complete**: Line (1D), Heatmap (2D), Contour (2D), Scatter3d (3D), Isosurface (3D)
- **Phase 2 complete**: Parallel Coordinates (2.1), Slice Plot (2.2), Parameter Importance (2.3), Pareto Front (2.4), Correlation Matrix (2.5)
- **View tab system**: Working with sweep tabs + analysis tabs, auto-select by dimensionality, state in `dcc.Store`
- **83 tests passing** (71 view + 12 core)

## Next task: Tier 3 — Introspection views or cross-cutting features

See `DSE_VIEW_IMPLEMENTATION_PLAN.md` for full specifications. Options:

### Tier 3 — Introspection views (per-design-point deep dives)
- §3.1 Per-qubit fidelity timeline (heatmap of qubit fidelity over circuit layers)
- §3.2 Core placement map (animated qubit-to-core mapping)
- §3.3 Fidelity decomposition bar (waterfall: 1.0 → losses → final)
- **Note**: These require backend changes to expose per-point `QusimResult` data

### Cross-cutting features
- Threshold overlay (global fidelity cutoff line, dim infeasible regions)
- Multi-view layout (2x1 or 2x2 split of center panel)
- Export (CSV data export, LaTeX snippet)
- Brushing & linking (selection sync across views)
- Seed aggregation (multi-seed mode with mean ± σ bands)

## TDD workflow (mandatory)

For each view:
1. **Write tests first** in `tests/test_plotting_views.py` — import the new function, test trace types, data shapes, routing
2. **Run tests** — confirm they fail (RED)
3. **Implement** — add `plot_xxx()` to `plotting.py`, add to `ANALYSIS_TABS`, handle in `build_figure()`
4. **Run tests** — confirm they pass (GREEN)
5. **Run full suite** — `.venv/bin/python -m pytest tests/ -v` — all must pass before moving on

## Patterns to follow

- Plot functions take `sweep_data: dict` (the store format) + `output_key: str`
- Analysis views use `_flatten_sweep_to_table()` to get a flat table from any-dimension grid
- Colorscales: use the diverging `[#d73027, #fc8d59, #fee08b, #91bfdb, #4575b4]` for data views
- Layout: spread `{**_LAYOUT_BASE, "margin": dict(...)}` to avoid duplicate kwarg errors
- Test fixtures: reuse `sweep_data_store_1d`, `sweep_data_store_2d`, `sweep_data_store_3d`
- The `build_figure` dispatcher checks analysis view types before the dimension-specific sweep views

## Do not

- Do not modify the DSE engine (`dse_engine.py`) — all analysis views compute from cached sweep data
- Do not change existing test assertions — only add new tests
- Do not add external dependencies (scipy, scikit-learn, etc.) — use numpy only
- Do not implement Tier 3 (introspection) views yet — those need backend changes
