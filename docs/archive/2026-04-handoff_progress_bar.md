# Agent Handoff — DSE Progress Bar

> **Archived (2026-05).** Sweep progress reporting has shipped — see
> `SweepProgress` in `qusim.dse.results` and the `progress_callback`
> kwarg on `DSEEngine.sweep_*`. File paths and "current state"
> sections below are point-in-time; do not treat them as authoritative.

## Your task

Add a loading/progress bar to the DSE GUI that shows sweep progress when the computation takes more than a trivial amount of time. This happens when hot reload is not possible — i.e. when cold-path parameters (num_qubits, num_cores) are being swept, or when the Run button is clicked for the first time.

## Architecture overview

The app is a Dash + Plotly web application. Key files:

| File | Role |
|------|------|
| `gui/app.py` | Dash callbacks, layout, sweep orchestration |
| `gui/components.py` | UI component factories |
| `gui/dse_engine.py` | Two-stage sweep engine (cold path + hot path) |
| `gui/constants.py` | Parameter definitions, sweep point counts |
| `gui/assets/style.css` | Stylesheet |

## When to show the progress bar

The sweep callback is `run_sweep()` in `gui/app.py` (line ~733). It runs synchronously:

1. **Cold path** (`_engine.run_cold(...)`) — builds circuit, runs HQA + SABRE routing. Takes 0.1–10s per configuration. Cached when structural params don't change.
2. **Sweep loop** (`_engine.sweep_1d/2d/3d(...)`) — evaluates many design points.

**Hot-path-only sweeps** (noise params like T1, gate error) are sub-second and don't need a progress bar — each point is <1ms.

**Cold-path sweeps** are slow. When `num_qubits` or `num_cores` is on a sweep axis (`MetricDef.is_cold_path == True`), every sweep point triggers a full re-mapping. A 15-point 1D qubit sweep can take 30+ seconds. These are the sweeps that need the bar.

You can detect this with `DSEEngine._has_cold(*metric_keys)` or by checking `METRIC_BY_KEY[key].is_cold_path` for each active sweep axis.

## Progress callback — already wired

The `SweepProgress` dataclass and `progress_callback` parameter are already implemented in `gui/dse_engine.py`:

```python
@dataclass
class SweepProgress:
    completed: int           # points finished so far
    total: int               # total points in sweep
    current_params: dict     # e.g. {"num_cores": 4.0}

    @property
    def percentage(self) -> float:  # 0.0–100.0
```

All three sweep methods accept `progress_callback: Callable[[SweepProgress], None] | None`:
- `sweep_1d(...)` — line 387
- `sweep_2d(...)` — line 416
- `sweep_3d(...)` — line 454

The callback fires after every single design point. The `app.py` caller currently passes `None` (no callback). Your job is to wire it up.

## Sweep point counts

From `gui/constants.py`:

| Dimension | Hot-path points | Cold-path points |
|-----------|----------------|-----------------|
| 1D | 60 | 15 |
| 2D | 30×30 = 900 | 8×8 = 64 |
| 3D | 12³ = 1728 | 5³ = 125 |

Cold-path integers (cores 1–8, qubits 4–32) are de-duplicated after rounding, so actual counts may be lower.

## The challenge: Dash is synchronous

Dash callbacks are blocking — you can't update the UI mid-callback. Options to solve this:

### Option 1: Background callback + `dcc.Interval` polling (recommended)

Dash supports [long callbacks](https://dash.plotly.com/long-callbacks) via `dash.long_callback` or background workers. Alternatively:

1. Store progress in a shared object (e.g. a module-level `SweepProgress` or a `dcc.Store` polled by `dcc.Interval`).
2. The `progress_callback` writes to this shared state on each iteration.
3. A separate `dcc.Interval` callback (e.g. every 500ms) reads the shared state and updates the progress bar UI.
4. When the sweep finishes, the main callback returns results as normal and clears the progress state.

The `dcc.Interval` component already exists at `id="sweep-poll"` (line ~470 of `app.py`) firing every 300ms. You could reuse it or add a dedicated one.

### Option 2: `dash.long_callback` with `set_progress`

Dash's `@dash.long_callback` decorator supports a `set_progress` argument that updates an `Output` mid-computation. This is the cleanest Dash-native approach but requires a `diskcache` or `celery` backend.

## UI placement

The progress bar should appear in the center panel, overlaying or replacing the plot area while the sweep runs. Current layout structure in `app.py`:

```
header-bar (contains status-bar text at id="status-bar")
├── left sidebar (metric selectors)
├── center panel
│   ├── view tab bar
│   ├── main-plot (id="main-plot", the dcc.Graph)
│   └── frozen slider (id="frozen-slider-container")
└── right sidebar (config tabs)
```

The `status-bar` (line 192) currently shows text like `"Cold path: cached ✓ | Sweep (2D, 900 pts): 0.42s | Total: 0.5s"`. You could either:

- Add a `dcc.Progress` or styled `html.Div` progress bar below the status text
- Overlay a progress bar on top of `main-plot` while sweep is running
- Replace the plot with a centered progress indicator during computation

## Existing status bar output

The sweep callback returns `status` as the 3rd output (line 696: `Output("status-bar", "children")`). This is set once at the end of the sweep (line 881–884). During the sweep the UI is frozen.

## What to implement

1. **Progress bar component** — add to the layout (components.py or inline in app.py). A horizontal bar with percentage text. Hidden by default.
2. **Shared progress state** — module-level variable or `dcc.Store` that the `progress_callback` writes to.
3. **Wire the callback** — in `run_sweep()`, pass a `progress_callback` to the sweep methods when cold-path params are active.
4. **Polling callback** — reads the shared state and updates the progress bar. Fire every 300–500ms. Show/hide the bar based on whether a sweep is running.
5. **Hide on completion** — when the sweep finishes and the main callback returns, hide the progress bar and show the plot.

## Thread safety

The `run_sweep` callback holds `sweep_lock` (a `threading.Lock`, line ~772). The polling callback runs in a different thread. The shared progress state must be thread-safe — use a `threading.Lock` or an atomic-friendly dataclass.

## Styling

Follow the existing theme from `gui/components.py`:

```python
COLORS = {
    "bg": "#FFFFFF",
    "surface": "#F5F5F5",
    "surface2": "#EBEBEB",
    "border": "#D4D4D4",
    "accent": "#2B2B2B",
    "accent2": "#555555",
    "text": "#2B2B2B",
    "text_muted": "#888888",
}
```

The bar itself should use `accent` (#2B2B2B) on a `surface2` (#EBEBEB) track. Keep it minimal — no animations, just a width transition.

## Files you will modify

- `gui/app.py` — add polling callback, wire progress_callback, add progress bar to layout
- `gui/components.py` — optional, if you extract the progress bar into a factory function

## Files you should NOT modify

- `gui/dse_engine.py` — progress callback is already wired, don't change the engine
- `gui/plotting.py` — no plot changes needed
- `gui/constants.py` — no constant changes needed

## TDD workflow (mandatory)

1. Write tests first — verify progress bar appears/hides, polling callback returns correct state
2. Run tests — confirm they fail (RED)
3. Implement
4. Run tests — confirm they pass (GREEN)
5. Run full suite: `.venv/bin/python -m pytest tests/ -v`

## Do not

- Do not add external dependencies beyond what Dash provides
- Do not change the sweep engine or plotting logic
- Do not break existing tests (174 passing)
- Do not use `html.Input` — Dash's `html` module doesn't have it, use `dcc.Input`
