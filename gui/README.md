# qusim DSE GUI

Interactive Design Space Explorer for multi-core quantum
architectures. Sweep up to six hardware / architectural parameters,
view fidelity surfaces, Pareto frontiers, parameter importance, and
custom Figures of Merit — all rendered in your browser.

## Quick start

```bash
# From the repo root
pip install -e ".[gui]"      # core + Dash + Cytoscape
qusim-dse                    # → http://127.0.0.1:8050

# Or the legacy way (still works):
python gui/app.py
```

`qusim-dse` is a console script wired to `gui.app:main`; both routes
hit the same Dash app. Override the bind address with
`QUSIM_HOST=0.0.0.0 QUSIM_PORT=8080 qusim-dse`. On startup the app
auto-runs a 3-D sweep (T1 × T2 × 2Q gate time) and shows an
isosurface.

## Where the GUI ends and the library begins

The Dash app is a *thin* orchestrator. Every piece of "real work"
(sweep generation, parameter registry, FoM evaluation, Pareto math,
sweep flattening) lives in the `qusim.dse` and `qusim.analysis`
library packages. The GUI consumes those libraries the same way a
user script would.

```
gui/
├── app.py            # Dash callbacks, layout, state stores. Imports
│                       qusim.dse.DSEEngine + qusim.analysis.FomConfig.
├── components.py     # Sidebar / topbar / right-panel widget factories
├── plotting.py       # Plotly figure builders — Plotly is the only
│                       GUI-specific dep here. Calls into
│                       qusim.dse.flatten + qusim.analysis.pareto
│                       for the math.
├── session.py        # Save / load full UI state (Dash-free, testable)
├── examples.py       # Canned DSE sessions for the Examples dropdown
├── derivatives.py    # Numeric ∂/∂x for the Elasticity / |∇F| views
├── interpolation.py  # Trilinear → bilinear interp for Frozen views
├── constants.py      # GUI-only presentation knobs (VIEW_TABS,
│                       VIEW_MODES). The data registry it used to own
│                       now lives in qusim.dse.axes; this file
│                       re-exports it for back-compat.
├── assets/style.css  # Light-theme stylesheet
├── requirements.txt  # Legacy pin file; pyproject [gui] extras are canonical
├── __init__.py
│
│ # Backwards-compat shims (re-export from qusim library):
├── dse_engine.py     # ⇒ qusim.dse.engine
└── fom.py            # ⇒ qusim.analysis.fom
```

The shims exist so older code (`from gui.dse_engine import DSEEngine`,
test fixtures patching `gui.dse_engine.X`) keeps working. New code
should import from `qusim.dse` / `qusim.analysis` directly.

## Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Topbar  qusim-dse · Examples ▾ · status · [▶ Run] · save / load   │
├──────────┬───────────────────────────────────┬───────────────────┤
│ Sweep    │ View tabs   [Line|Heat|3D|Par|Pareto|Merit|…]  [CSV]   │
│ axes     ├───────────────────────────────────┤   Configuration    │
│ (1–6)    │                                   │   ┌─Circ─Topo─Noi─┐│
│ + range  │           main plot               │   │ tab content    ││
│ sliders  │                                   │   │                ││
│          │                                   │   │                ││
│          ├───────────────────────────────────┤   │                ││
│          │ frozen-axis slider (3-D views)    │   │                ││
└──────────┴───────────────────────────────────┴───────────────────┘
```

## Views

The sweep dimensionality determines which views are offered. See
[`../docs/DSE_VIEWS.md`](../docs/DSE_VIEWS.md) for the catalogue with
research-question framing and Plotly trace details.

### Sweep views (driven by axis dimensionality)

| Dim | Tabs |
|---|---|
| 1-D | `Line` |
| 2-D | `Heatmap` |
| 3-D | `Scatter` · `Isosurface` (default) · `Frozen Heat` |
| ≥4-D | analysis tabs only (`Parallel`, `Slices`, `Importance`, …) |

### Analysis views (work at any dimensionality)

`Parallel` · `Slices` · `Importance` · `Pareto` · `Corr.` ·
`Elasticity` · `Merit` · `Topology`

Each view has a *view mode* dropdown — `Absolute`, `|∇F| (gradient
magnitude)`, `Elasticity`, `d²F/dx²`, `∂²F/∂x∂y` — that transforms
the underlying scalar field. Modes that don't make sense at the
current dimensionality are filtered out automatically.

## Sweepable parameters

20 axes across noise, hardware, and architecture. Defined in
[`qusim.dse.axes`](../python/qusim/dse/axes.py) (`SWEEPABLE_METRICS`)
plus a categorical layer (`CATEGORICAL_METRICS`) for non-numeric axes.

Numeric noise (hot-path; cheap):

`single_gate_error`, `two_gate_error`, `epr_error_per_hop`,
`measurement_error`, `t1`, `t2`, `single_gate_time`, `two_gate_time`,
`epr_time_per_hop`, `measurement_time`, `readout_mitigation_factor`,
`classical_link_width`, `classical_clock_freq_hz`,
`classical_routing_cycles`.

Architectural (cold-path; one cold compile per unique value):

`qubits` (alias: physical = logical), `num_qubits`,
`num_logical_qubits`, `num_cores`, `communication_qubits`,
`buffer_qubits`.

Categorical (cold-path):

`circuit_type` · `topology_type` · `intracore_topology` ·
`routing_algorithm` · `placement`.

## Two-stage execution model

| Path | Cost | Triggered by |
|---|---|---|
| **Cold** | 1–10 s per cell | any cold-path key changes |
| **Hot**  | <1 ms per cell | only noise / hot-path keys change |

Cold compilations are cached and shared across sweep cells with the
same structural footprint; hot evaluations are batched through a
single Rust call. Empirical RAM model schedules cold workers across a
forkserver pool so concurrent compiles never exceed `MemAvailable`.

## Sweep budgets

The default budgets (in [`qusim.dse.axes`](../python/qusim/dse/axes.py)):

| Axes | Hot-path total | Cold-path total |
|---|---|---|
| 1-D  | 60 points     | 15 |
| 2-D  | 30 × 30 = 900 | 8 × 8 = 64 |
| 3-D  | 12³ = 1 728   | 5³ = 125 |
| ≥4-D | `MAX_TOTAL_POINTS_HOT = 5_000` total budget | `MAX_COLD_COMPILATIONS = 64` total |

The hot budget is further clamped by `_max_hot_points_for_memory()`
to leave 3 GB headroom for the OS / browser / UI; the cold budget by
`_mem_budget_mb()` (the larger of 1 GB or 30 % of total RAM).

## Sessions

The full UI state — circuit + topology + active axes + slider ranges
+ FoM + view mode + frozen-axis values — round-trips through a JSON
blob via `gui/session.py`. The Examples dropdown ships canned
sessions in [`gui/examples.py`](examples.py) (`generate_example_sessions.py`
in `scripts/` regenerates them).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Coverage focuses on the GUI-bound code:

- `tests/test_plotting_views.py` — every plot builder
- `tests/test_plotting_merit.py` — Merit / Pareto / FoM rendering
- `tests/test_faceting.py` — categorical-axis facet layouts
- `tests/test_frozen_slider.py` — client-side trilinear interpolation
- `tests/test_nd_sweep.py`, `tests/test_sweep_progress.py`,
  `tests/test_parallel_sweep.py` — engine paths

## See also

- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — layered diagram.
- [`../docs/DSE_VIEWS.md`](../docs/DSE_VIEWS.md) — view catalogue.
- [`../python/qusim/dse/README.md`](../python/qusim/dse/README.md) — DSE library reference.
- [`../python/qusim/analysis/README.md`](../python/qusim/analysis/README.md) — FoM + Pareto reference.
