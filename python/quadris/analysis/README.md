# `quadris.analysis`

Analysis primitives over sweep results: a safe expression evaluator
for user-defined Figures of Merit, and Pareto-frontier helpers.

These are the algorithmic backbones of the GUI's Merit and Pareto
views, but they work on any `quadris.dse.SweepResult` (or its
``.to_sweep_data()`` form), so custom Python scripts get the same
machinery as the Dash app.

## Quick start

```python
from quadris.dse import DSEEngine, NOISE_DEFAULTS
from quadris.analysis import FomConfig, compute_for_sweep, pareto_front

engine = DSEEngine()
cached = engine.run_cold(
    circuit_type="qft", num_qubits=24, num_cores=4,
    topology_type="ring", intracore_topology="all_to_all",
    placement_policy="spectral", seed=0,
    communication_qubits=2, buffer_qubits=1,
)

xs, ys, grid = engine.sweep_2d(
    cached=cached,
    metric_key1="t1",              low1=4.0, high1=6.5,
    metric_key2="two_gate_error",  low2=-5.0, high2=-2.0,
    fixed_noise=dict(NOISE_DEFAULTS),
)
sweep_data = {
    "metric_keys": ["t1", "two_gate_error"],
    "xs": list(xs), "ys": list(ys), "grid": grid,
}

fom = FomConfig(
    name="Fidelity yield",
    intermediates=(("infidelity", "max(1 - overall_fidelity, 1e-12)"),),
    numerator="log(1 / infidelity)",
    denominator="max(total_epr_pairs, 1)",
)
result = compute_for_sweep(sweep_data, fom)
print(result.values.shape, result.error)
```

## Public surface

```python
from quadris.analysis import (
    # Figure of Merit
    FomConfig, FomResult, FomBreakdown, FomError,
    PRESETS, PRESET_OPTIONS, DEFAULT_FOM, SAFE_FUNCS,
    evaluate, compute_for_sweep, compute_breakdown, primitives_for_sweep,

    # Pareto
    pareto_front_mask, pareto_front,
)
```

## Figure of Merit

A `FomConfig` is `numerator / denominator` where each side is an
arithmetic expression over *primitives*:

- swept input axes (e.g. `t1`, `num_cores`, `two_gate_error`)
- simulation output keys (e.g. `overall_fidelity`,
  `total_epr_pairs`, `total_circuit_time_ns`)
- named **intermediates** that previous expressions defined

Expressions go through a strict AST whitelist:

- `+ - * / // % **`, unary `+ -`
- numeric constants
- function calls limited to `log` (alias `ln`), `log2`, `log10`,
  `exp`, `sqrt`, `abs`, `min`, `max`, `pow`, `clip`
- no attribute access, subscripting, lambdas, comprehensions, imports,
  `__builtins__` access, or arbitrary names

Any disallowed construct raises `FomError` from `_validate`. Failures
during `compute_for_sweep` are surfaced via `FomResult.error` rather
than raised — so the GUI can show them without trapping exceptions.

Built-in presets in `PRESETS`:

| Preset | Numerator | Denominator |
|---|---|---|
| `fidelity`                 | `overall_fidelity` | `1` |
| `fidelity_per_epr`         | `overall_fidelity` | `max(total_epr_pairs, 1)` |
| `fidelity_per_time`        | `overall_fidelity` | `max(total_circuit_time_ns, 1)` |
| `fidelity_over_cost`       | `overall_fidelity` | `cost`  ← intermediate `cost = max(total_epr_pairs + 1e-9 * total_circuit_time_ns, 1)` |
| `neg_log_infidelity_per_epr` | `log(1 / max(1 - overall_fidelity, 1e-12))` | `max(total_epr_pairs, 1)` |

## Pareto frontier

`pareto_front_mask(num, den)` returns a boolean mask over a flat
sweep, selecting the points that are Pareto-optimal under
*maximise* `num` and *minimise* `den`. O(N²) — fine for the
≤4 096-point sweeps the engine produces.

`pareto_front(sweep, objective_x, objective_y)` is the higher-level
convenience: takes a `SweepResult` (or sweep dict), looks up the
optimisation direction for each output metric in
`PARETO_METRIC_ORIENTATION`, and returns
`{"x": …, "y": …, "mask": …, "axes": {<axis_key>: …}}` ready for
plotting.

```python
front = pareto_front(sr, objective_x="total_epr_pairs",
                          objective_y="overall_fidelity")
# front["mask"] selects Pareto-optimal points.
# front["axes"]["num_cores"] gives the axis value at each point —
# handy for hover labels in custom plots.
```

## Module layout

| Module | Concern |
|---|---|
| `fom.py`    | Expression validator (`_validate`), evaluator (`evaluate`, `compute_for_sweep`, `compute_breakdown`), config dataclass (`FomConfig`), presets |
| `pareto.py` | `pareto_front_mask` (low-level), `pareto_front` (sweep-aware) |

Both modules are pure-numpy — no Plotly / Dash dependency. They're
safe to import in any environment that has `quadris` installed.

## See also

- `quadris.dse` — sweep engine + parameter registry ([`../dse/README.md`](../dse/README.md)).
- `examples/dse_fom_heatmap.py` — Figure-of-Merit example end-to-end.
- `examples/dse_2d_pareto.py` — Pareto frontier example end-to-end.
