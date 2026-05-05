# `qusim.dse`

Design Space Exploration toolkit. Build N-dimensional sweeps over
circuit, topology, and noise parameters; cache the expensive
structural pass; re-evaluate noise configurations cheaply from the
cached mapping.

This module is GUI-agnostic — every public name here is consumed
identically by the Dash app and by user scripts / notebooks.

## Quick start

```python
from qusim.dse import DSEEngine, NOISE_DEFAULTS

engine = DSEEngine()

# 1) Cold path — once. Builds the circuit, the topology, runs the
#    routing backend (HQA+SABRE by default), caches a CachedMapping.
cached = engine.run_cold(
    circuit_type="qft", num_qubits=32, num_cores=4,
    topology_type="ring", intracore_topology="all_to_all",
    placement_policy="spectral", seed=0,
    communication_qubits=2, buffer_qubits=1,
)
print(f"{cached.cold_time_s:.2f}s; {cached.total_teleportations} teleports")

# 2) Hot path — many. Each call is a single Rust round-trip.
xs, results = engine.sweep_1d(
    cached=cached, metric_key="two_gate_error",
    low=-5.0, high=-1.0, fixed_noise=dict(NOISE_DEFAULTS),
)
```

## Public surface

```python
from qusim.dse import (
    # Engine
    DSEEngine, SweepResult, SweepProgress, CachedMapping,

    # Parameter registry
    MetricDef, CatMetricDef,
    SWEEPABLE_METRICS, METRIC_BY_KEY,
    CATEGORICAL_METRICS, CAT_METRIC_BY_KEY,
    OUTPUT_METRICS, OUTPUT_METRIC_LABEL,
    PARETO_METRIC_ORIENTATION, FIDELITY_METRICS,
    NOISE_DEFAULTS, DEFAULT_SWEEP_AXES, MAX_SWEEP_AXES,

    # Topology helpers
    inter_core_neighbors, clamp_k_for_topology, clamp_b_for_topology,
    max_data_slots, total_reserved_slots,

    # Tabular access
    flatten_sweep_to_table,
)
```

## Module layout

The implementation is split into focused submodules so each file has
one reason to change:

| Module | Concern | Lines |
|---|---|---|
| `axes.py`     | Parameter registry (`MetricDef`, `SWEEPABLE_METRICS`, `NOISE_DEFAULTS`, `OUTPUT_METRICS`, `PARETO_METRIC_ORIENTATION`) | ~400 |
| `circuits.py` | Qiskit circuit builders + transpile pass (qft, ghz, random, custom QASM) | ~50 |
| `topology.py` | Inter/intra-core graph, slot layout, distance matrix, K/B clamps | ~570 |
| `noise.py`    | `_merge_noise`, derived teleportation cost, gate-array builder | ~120 |
| `results.py`  | `CachedMapping`, `SweepResult`, `SweepProgress`, `_RESULT_DTYPE`, row helpers | ~215 |
| `config.py`   | Alias expansion, `_clamp_cfg_comm_and_logical`, `_resolve_cell_cold_cfg` | ~95 |
| `flatten.py`  | `flatten_sweep_to_table` — sweep result → numpy column table | ~155 |
| `memory.py`   | RAM ceilings, cold-MB estimate, thread-pool capping | ~155 |
| `sweep.py`    | `_compile_one`, parallel cold pool, `_eval_cold_batch` | ~295 |
| `engine.py`   | `DSEEngine` façade — `run_cold` / `run_hot` / `sweep_nd` orchestration | ~750 |
| `backends/`   | Routing-algorithm strategies | |
| ` ├─ base.py`        | `Backend` protocol | ~40 |
| ` ├─ hqa_sabre.py`   | HQA + SABRE backend | ~115 |
| ` └─ telesabre.py`   | TeleSABRE backend (C library wrapper, DAG-layer remap) | ~290 |

## Cold path vs. hot path

DSE separates two cost regimes:

- **Cold path** (`run_cold`) — circuit + topology build, routing
  backend dispatch, structural caching. ~1–10 s per call. Required
  when any *cold-path* key changes
  (`circuit_type / num_qubits / num_cores / topology_type /
  intracore_topology / placement_policy / routing_algorithm /
  communication_qubits / buffer_qubits / num_logical_qubits`).
- **Hot path** (`run_hot`, `run_hot_batch`) — fidelity re-estimation
  using cached `gs_sparse + placements + sparse_swaps + distance_matrix`.
  <1 ms per point; thousands per Rust call via `run_hot_batch`.

`sweep_nd` switches between the two automatically based on which
metrics are on the axes.

## Backends

Routing dispatch is via `qusim.dse.backends`. Each backend implements
the `Backend` protocol — one ``compile(cold_cfg, noise, key) ->
CachedMapping`` method.

| Backend name | Module | Notes |
|---|---|---|
| `hqa_sabre` (default) | `backends/hqa_sabre.py` | HQA initial mapping + SABRE swap insertion via `qusim.map_circuit` |
| `telesabre`           | `backends/telesabre.py` | TeleSABRE C library via `qusim.rust_core.telesabre_map_and_estimate`; remaps placements/swaps from gate-step space to DAG-layer space |

Adding a new backend is one file plus an entry in
`backends/__init__.py:_BACKENDS`. See `backends/base.py` for the
protocol contract.

## Sweep axes

The unified entry point is `sweep_nd(cached, sweep_axes, fixed_noise,
cold_config=None, ...)`. Each entry in `sweep_axes` is a tuple:

- `(metric_key, low, high)` for **numeric** axes — endpoints are
  `log10` exponents when `MetricDef.log_scale=True`, raw values
  otherwise. Integer-typed metrics (`num_qubits`, `num_cores`, …) are
  rounded and de-duplicated.
- `(metric_key, [v1, v2, …])` for **categorical** axes
  (`circuit_type`, `topology_type`, `intracore_topology`,
  `routing_algorithm`, `placement`).

Per-axis point counts are derived from a split budget model:
`MAX_COLD_COMPILATIONS = 64` for cold-path axes,
`MAX_TOTAL_POINTS_HOT = 5_000` for hot. Both are clamped further by an
empirical RAM model — exceeding `MemAvailable` would OOM-kill the
process.

The 1-D / 2-D / 3-D variants (`sweep_1d`, `sweep_2d`, `sweep_3d`) are
thin wrappers around `sweep_nd` that return the legacy
`(xs[, ys[, zs]], grid)` tuple instead of a `SweepResult`.

## SweepResult

`SweepResult` carries the full N-D structured grid plus axis metadata:

```python
sr = engine.sweep_nd(...)
sr.shape          # (len(axis_0), len(axis_1), …)
sr.metric_keys    # ['num_cores', 'two_gate_error', …]
sr.axes           # [array_for_axis_0, array_for_axis_1, …]
sr.grid           # numpy structured array, dtype=_RESULT_DTYPE
sr.total_points   # product of shape

# Convert to the dict form the GUI / FoM / Pareto layers consume:
sd = sr.to_sweep_data()
```

For a 1-D / 2-D / 3-D sweep, ``to_sweep_data()`` returns the legacy
``{xs[, ys[, zs]], grid}`` shape; for ≥4-D it keeps the structured
grid in place to avoid the allocation that would OOM the process at
sweep sizes typical of the GUI.

## Parameter registry

```python
from qusim.dse import SWEEPABLE_METRICS, METRIC_BY_KEY

for m in SWEEPABLE_METRICS[:3]:
    print(m.key, m.label, "log" if m.log_scale else "lin",
          "cold" if m.is_cold_path else "hot")
# single_gate_error  1Q Gate Error  log  hot
# two_gate_error     2Q Gate Error  log  hot
# epr_error_per_hop  EPR Error/Hop  log  hot
```

Every entry has a `slider_min`/`slider_max` (the GUI uses these for
slider styling — script users typically just supply their own
endpoints), a `log_scale` flag, a `unit` string, and an
`is_cold_path` boolean. The full set is in
[`axes.py`](axes.py).

## Performance & RAM safety

`sweep_nd` calls into two RAM-aware safety nets:

- `_max_hot_points_for_memory()` reads `/proc/meminfo` and clamps
  `max_hot` to whatever fits in `MemAvailable - _RESERVED_RAM_MB_HOT`.
  An oversized request raises early with a "Memory cap" message
  instead of silently OOM-killing mid-sweep.
- `_parallel_cold_sweep` schedules cold compilations across a
  forkserver pool so concurrent workers never exceed
  `_mem_budget_mb()`. Each worker is also capped via `RLIMIT_AS` to
  its fair share of the budget — runaway allocations raise
  `MemoryError` inside the worker rather than OOM-killing the host.

`memory.py` also caps every per-process thread pool
(`OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, …) to 1 at import time.
Workers use process-level parallelism already, so library-internal
threading would just oversubscribe the CPU and stall the scheduler.

## See also

- `qusim.analysis` — FoM evaluator + Pareto frontier ([`../analysis/README.md`](../analysis/README.md)).
- `examples/` — three end-to-end scripts using only the library API.
- `docs/ARCHITECTURE.md` — layered diagram of the whole stack.
