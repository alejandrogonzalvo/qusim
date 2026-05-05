# Quadrature project — `quadris` technical report

> **Naming.** The package is referred to as **`quadris`** throughout
> this document. The name is provisional; the codebase currently
> exposes it as `qusim` and the rename will land before the deliverable
> is finalised.

## 0. Executive summary

`quadris` is a multi-core quantum architecture simulator with a
built-in Design Space Exploration (DSE) toolkit. The system has three
clean layers — a Rust core (PyO3-bound) that owns every hot loop, a
GUI-agnostic Python library (`quadris`, `quadris.dse`,
`quadris.analysis`) that drives both interactive exploration and
headless scripts, and an optional Dash app (`quadris-dse`) that puts a
UI on top.

```mermaid
graph TD
  subgraph GUI ["Dash GUI (optional)"]
    APP[quadris-dse: app, callbacks, plots]
  end
  subgraph LIB ["Python library"]
    QU[quadris<br/>map_circuit, QusimResult]
    DSE[quadris.dse<br/>DSEEngine, SweepResult, axes]
    ANA[quadris.analysis<br/>FoM, Pareto]
  end
  subgraph RUST ["Rust core (PyO3)"]
    RC[map_and_estimate,<br/>estimate_hardware_fidelity[_batch],<br/>telesabre_map_and_estimate]
  end
  APP --> DSE --> QU --> RC
  APP --> ANA --> DSE
```

The simulator delivers (i) per-circuit fidelity prediction with a
three-channel noise model (algorithmic + routing + coherence),
(ii) sub-millisecond hot-path re-evaluation reusing a cached HQA
mapping, (iii) N-D parameter sweeps with budget-aware scheduling,
(iv) a custom-FoM evaluator and Pareto-frontier helper for
multi-objective design analysis, and (v) two routing back-ends
(HQA + SABRE; TeleSABRE) selected per call. Measured on the
benchmarking host below, the cold path scales from 0.26 s at 8
logical qubits to 1.44 s at 128 (4-core ring, default noise); a
batched hot-path call sustains ~17 µs per design point at scale; and
the parallel cold pool reaches 2.7× speed-up on 4 workers for cold
sweeps. The Rust core's predictions correlate with IBM-Q-mitigated
fidelity at Pearson r = +0.84 across 93 reference circuits using only
uniform noise parameters.

---

## 1. Functional capabilities

### 1.1 Single-circuit mapping

`quadris.map_circuit` accepts a transpiled Qiskit `QuantumCircuit`, a
`CouplingMap`, a `core_mapping` (physical qubit → core index), and a
noise dict; runs HQA initial placement followed by SABRE swap
insertion; and returns a `QusimResult` with overall, algorithmic,
routing, and coherence fidelities, plus per-layer × per-qubit grids
of each component, and aggregate counters (teleportations, swaps,
EPR pairs, total circuit time).

```python
from quadris import map_circuit
from quadris.hqa.placement import InitialPlacement
result = map_circuit(circuit=transp, full_coupling_map=cmap,
                     core_mapping=mapping, seed=42,
                     initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
                     single_gate_error=1e-4, two_gate_error=1e-3,
                     t1=100_000.0)
result.overall_fidelity, result.total_epr_pairs
```

### 1.2 Hot-path fidelity re-estimation

After one cold mapping, `quadris.estimate_fidelity_from_cache` and
its batched sibling `..._batch` re-evaluate fidelity for any new
noise dict in microseconds. The structural data (sparse interaction
tensor, placements grid, distance matrix, sparse-swaps grid, gate
arrays) is parsed once and reused; only the floating-point error/
timing parameters change per call. This is the engine behind every
DSE noise sweep.

### 1.3 Design Space Exploration — inputs, outputs, sweep model

The DSE engine (`quadris.dse.DSEEngine`) coordinates cold mapping
and hot evaluation across a multi-dimensional parameter grid.

**Logical-first parameterization.** The user pins exactly one of
`num_cores` or `qubits_per_core` (via `pin_axis`); the unpinned axis
is *deduced* so the chip always fits `num_logical_qubits` even as
the user sweeps comm/buffer overhead. `num_qubits = num_cores ·
qubits_per_core` is therefore a **derived output**, not an input —
the architecture absorbs comm/buffer reservations rather than
shrinking the algorithm.

| Input axis | Role | Constancy in a sweep |
|---|---|---|
| `num_logical_qubits` | algorithm size (L) | constant per sweep |
| `num_cores` ∨ `qubits_per_core` | one is pinned | the pinned axis can be swept; the other is derived |
| `communication_qubits` (K), `buffer_qubits` (B) | per-group EPR + buffer reservation, B ≤ K | per-cell |
| `topology_type`, `intracore_topology` | inter-/intra-core graph | per-cell |
| `routing_algorithm` ∈ {`hqa_sabre`, `telesabre`} | back-end selection | per-cell |
| `placement_policy` ∈ {random, spectral} | HQA initial placement | per-cell |
| `seed`, `circuit_type`, `custom_qasm` | circuit / RNG | per-cell |
| noise dict (T1, T2, gate errors, EPR error, …) | per-cell hot path | per-cell |

**Outputs per cell** (`SweepResult.grid`, structured numpy array):
overall, algorithmic, routing, coherence, readout fidelity;
`total_circuit_time_ns`, `total_epr_pairs`, `total_swaps`,
`total_teleportations`, `total_network_distance`. Optional per-cell
per-qubit fidelity grids when `keep_per_qubit_grids=True`.

**Sweep entry points.** `sweep_nd(sweep_axes, fixed_noise,
cold_config, ...)` is the canonical N-D driver; `sweep_1d`,
`sweep_2d`, `sweep_3d` are wrappers that return the legacy
`(xs[, ys[, zs]], grid)` tuple. Axis specs accept `(metric_key,
low, high)` for numeric axes (endpoints are log10 exponents when the
metric is log-scaled) and `(metric_key, [v1, v2, ...])` for
categorical axes.

### 1.4 Custom Figures of Merit

`quadris.analysis.FomConfig` holds an arbitrary expression
`numerator / denominator` with named intermediates. Expressions are
parsed against a strict AST whitelist (arithmetic +
`log/log2/log10/exp/sqrt/abs/min/max/pow/clip` only; no attribute
access, subscripting, lambdas, or comprehensions). `compute_for_sweep`
returns a vectorised result over the entire flattened sweep.

### 1.5 Pareto frontier

`quadris.analysis.pareto_front(sweep, objective_x, objective_y)`
returns Pareto-optimal mask + axis values, automatically resolving
each output's max/min orientation from `PARETO_METRIC_ORIENTATION`.
A lower-level `pareto_front_mask(num, den)` operates on raw arrays.

### 1.6 Routing back-end selection

Two back-ends are registered:

- **HQA + SABRE** (`routing_algorithm="hqa_sabre"`, default) — the
  reference pipeline. HQA produces a layer-wise core assignment;
  SABRE inserts intra-core SWAPs against the per-core coupling map.
- **TeleSABRE** (`"telesabre"`) — a unified router (vendored C
  library, FFI'd through Rust) that handles intra-core SWAPs and
  inter-core teleportations in a single pass. Comparison: §5.6.

### 1.7 Interactive GUI

The `quadris-dse` Dash application exposes the entire engine through
a browser UI: up to six sweep axes, 13 view types (line, heatmap,
3-D scatter / isosurface / frozen-heat slice, plus parallel
coordinates, slices, importance, Pareto, correlation, elasticity,
merit, and topology overlays), a session save/load mechanism, and
canned example sessions. Auto-runs a 3-D sweep on startup.

---

## 2. System architecture

### 2.1 Three-layer model

```
┌────────────────────────────────────────────────┐
│  Dash GUI (gui/) — quadris-dse                  │   optional
└──────────────────┬──────────────────────────────┘
                   │ imports
┌──────────────────▼──────────────────────────────┐
│  Python library (python/quadris/)               │
│   quadris      — single-circuit API             │
│   quadris.dse  — DSEEngine, sweeps, registry    │
│   quadris.analysis — FoM, Pareto                │
└──────────────────┬──────────────────────────────┘
                   │ PyO3 / numpy zero-copy
┌──────────────────▼──────────────────────────────┐
│  Rust core (src/)                               │
│   hqa, noise, routing, telesabre, python_api    │
└──────────────────────────────────────────────────┘
```

### 2.2 Module map

| Directory | Responsibility |
|---|---|
| `src/hqa/` | Hungarian Qubit Assignment + lookahead + sparse interaction tensor |
| `src/noise/` | Three-channel fidelity model (algorithmic, routing, coherence) |
| `src/routing/` | SABRE swap-insertion glue; teleportation event log |
| `src/telesabre/` | Rust wrapper around the vendored TeleSABRE C library (`csrc/telesabre/`) |
| `src/python_api.rs` | PyO3 entry points (4 functions) |
| `python/quadris/` | Single-circuit API (`map_circuit`, `QusimResult`, etc.) |
| `python/quadris/dse/` | `DSEEngine` façade + leaf modules: `axes`, `circuits`, `topology`, `noise`, `results`, `config`, `flatten`, `memory`, `sweep`, `backends/{base,hqa_sabre,telesabre}` |
| `python/quadris/analysis/` | `FomConfig`, `evaluate`, `pareto_front` |
| `gui/` | Dash app, plots, components, sessions |
| `tests/` | Python test suite + benchmark scripts |
| `examples/` | End-to-end library examples |

### 2.3 Two-stage execution model

| Stage | Cost | Triggered when |
|---|---|---|
| Cold path (`run_cold`) | ~0.3 – 1.5 s on QFT, 16 – 128 logical qubits | any cold-path key changes (circuit, topology, cores/qpc, K, B, placement, routing algorithm) |
| Hot path (`run_hot`, `run_hot_batch`) | ~13 – 170 µs per cell single-call; ~17 µs/cell at batch ≥ 100 | only noise / hot-path keys change |

Sweep cells with the same cold key share one cold compilation; hot
evaluations are batched through a single Rust call.

### 2.4 Public API surface (one-line summary)

| Name | What it does |
|---|---|
| `quadris.map_circuit(circuit, full_coupling_map, core_mapping, ..., noise...)` | HQA + SABRE → `QusimResult` |
| `quadris.telesabre_map_circuit(circuit_path, device_json, config_json, ...)` | TeleSABRE C library → `QusimResult` |
| `quadris.estimate_fidelity_from_cache(gs_sparse, placements, dist, swaps, ...)` | Single hot-path call |
| `quadris.estimate_fidelity_from_cache_batch(..., noise_dicts: list)` | Vectorised batch call |
| `quadris.dse.DSEEngine().run_cold(...)` → `CachedMapping` | Cold mapping with cache |
| `DSEEngine.run_hot(cached, noise)` → `dict` | Re-estimate from cache |
| `DSEEngine.sweep_nd(cached, sweep_axes, fixed_noise, cold_config=...)` → `SweepResult` | N-D sweep |
| `quadris.analysis.compute_for_sweep(sweep, FomConfig)` → `FomResult` | FoM evaluation |
| `quadris.analysis.pareto_front(sweep, x_key, y_key)` → `dict` | Pareto frontier |

---

## 3. Algorithms & computational complexity

Notation: **N** = qubits, **L** = circuit layers (DAG depth), **K** =
cores, **E** = two-qubit gates per layer (≤ N/2), **P** = layer
qubit pairs whose endpoints sit on different cores (conflicts).

### 3.1 HQA initial placement, lookahead, Hungarian matching

Per layer, HQA (i) projects the future onto the current layer via a
spatio-temporal lookahead, (ii) detects conflicting qubit pairs,
(iii) resolves core-balance parity, and (iv) solves an
assignment problem to pick the cheapest remap.

```
for ℓ in 0..L:
    look = sum over t in [ℓ, ℓ+H): exp(-t/σ) · interaction_tensor[t]
    conflicts = pairs (q1,q2) in look where core(q1) ≠ core(q2)
    balance free-slot parity by swapping movable qubits between cores
    assignment = Kuhn–Munkres over (conflict pairs × cores), cost = look-weighted hops
    apply assignment → placements[ℓ+1]
```

| Step | Cost per layer |
|---|---|
| Lookahead (truncated horizon **H** = 20) | Θ(H · E) = Θ(E) |
| Active set + movable check | Θ(E + N) |
| Conflict detection (sorted edges) | Θ(E log E) |
| Per-pair core-likelihood | Θ(P · N) |
| Hungarian assignment | Θ(max(P, K)³) |
| Partition validation | Θ(E) |

**Total**: **Θ(L · (P · N + max(P, K)³))**. The truncated horizon
turns the original O(L) lookahead sweep per layer into O(1), since
contributions beyond ~20 layers are below 10⁻⁶ for σ = 1.

### 3.2 Initial-placement policies

Two strategies are implemented:

- **Random**: balanced random partition. Θ(N).
- **Spectral clustering**: build a weighted interaction graph,
  compute the **K** smallest non-trivial Laplacian eigenvectors,
  k-means in eigenspace. Θ(N · k_iter + |E| · iter_eig); empirically
  yields lower routing cost than random by ≈ 30 % on QFT.

### 3.3 SABRE intra-core swap insertion

Run by the `MultiCoreOrchestrator`: per HQA core slice, slice the
DAG, hand it to Qiskit's `SabreSwap` against the local coupling
map, and capture the exact `(layer, q1, q2)` SWAPs SABRE inserts.
Cost is dominated by Qiskit's transpiler pass — typically Θ(L · N)
amortised on the dense circuits we benchmark.

### 3.4 TeleSABRE — unified routing back-end

TeleSABRE is a C library (vendored under `csrc/telesabre/`)
implementing a single-pass router that handles intra-core SWAPs and
inter-core teleportations together. The Rust crate
(`src/telesabre/`) wraps the FFI; the Python backend
(`backends/telesabre.py`) marshals the QASM circuit + a generated
device JSON into temp files, calls `telesabre_map_and_estimate`, and
re-maps the returned placements + sparse-swaps from gate-step space
into Qiskit DAG-layer space so the downstream noise model sees a
consistent row count regardless of back-end.

### 3.5 Noise model — three channels

Total per-qubit fidelity at the final layer is the product of three
disjoint factors:

**Algorithmic** — depolarising error per native gate:

$$F_\text{algo} = \prod_{g \in \mathcal{C}} (1 - \epsilon_g)$$

**Routing** — SABRE SWAPs and inter-core teleportations:

$$F_\text{SWAP} = (1 - 3\epsilon_\text{2Q})^{n_\text{SWAP}}, \quad
  F_\text{tele} = (1 - \epsilon_\text{hop})^{\text{distance}}$$

**Coherence** — exponential T1/T2 decay over per-qubit busy time:

$$F_\text{coh} = \exp\!\left(-\frac{t_\text{idle}}{T_1}\right) \cdot
                 \exp\!\left(-\frac{t_\text{idle}}{T_2}\right)$$

The Rust pass is Θ(L · N) per estimate (one sweep through the
sparse interaction tensor, layer by layer). Every event is recorded
on a shared per-qubit busy timeline so coherence sees the same idle
windows that routing decisions create.

### 3.6 Teleportation cost decomposition

The bundled `teleportation_error_per_hop` and
`teleportation_time_per_hop` are derived from constituents matching
the depolarising-channel paper's protocol cost: 1 EPR generation +
4 two-qubit gates (1 preprocess CNOT, 3 buffer-SWAP CNOTs) + 2
single-qubit gates (Hadamard preprocess, X/Z correction) + 1 mid-
circuit measurement.

$$F_\text{hop} \approx \sqrt{1-\epsilon_\text{EPR}}\;\;
                       (1 - \tfrac{2}{3}\epsilon_\text{2Q})^4\;\;
                       (1-\epsilon_\text{1Q})^2\;\;
                       (1-\epsilon_\text{meas})$$

The (2/3)·ε factor on each CNOT is the per-qubit marginal of a d=4
depolarising channel under the η-model (`src/noise/eta_cnot_update`).

### 3.7 Sweep-axis count derivation (split-budget model)

Given N axes, the engine splits the budget: categorical axes have
fixed counts (length of selected option list); numeric **cold-path**
axes share a `MAX_COLD_COMPILATIONS = 64` budget; **hot** axes share
the remaining `MAX_TOTAL_POINTS_HOT = 5 000` budget. Per-axis count
≈ ⌊budget^{1/n_axes}⌋, clamped to ≥ `MIN_POINTS_PER_AXIS = 3` and
further capped by an empirical RAM model (§5.2). The hot ceiling
keeps a 3 GB headroom for OS + browser; the cold ceiling keeps the
larger of 1 GB or 30 % of total RAM free.

### 3.8 Pareto frontier and FoM evaluation

**Pareto** — pairwise mask `pareto_front_mask(num, den)`: a point i
is dominated iff ∃ j with `num[j] ≥ num[i] ∧ den[j] ≤ den[i]`, with
at least one inequality strict. Implementation: O(N²) numpy
broadcast. Acceptable for ≤ 4 096-point sweeps the engine produces.

**FoM** — expressions are parsed once, validated against the AST
whitelist, then evaluated vectorised on numpy columns for the entire
flattened sweep. Per-point cost is a few fused BLAS ops; whole-sweep
cost is dominated by the AST `_validate` walk (Θ(|expr|)).

---

## 4. Implementation overview

The Rust crate is split into single-concern modules: `hqa/`
(interaction-tensor + Kuhn-Munkres + lookahead), `noise/` (the three
channels), `routing/` (SABRE event capture and teleportation event
log), `telesabre/` (FFI wrapper). All four expose Python-visible
work via `python_api.rs`'s four PyO3 entry points
(`map_and_estimate`, `estimate_hardware_fidelity`,
`estimate_hardware_fidelity_batch`, `telesabre_map_and_estimate`).
NumPy arrays cross the boundary zero-copy.

The Python library `python/quadris/` re-exports each entry point
with type-annotated wrappers (`map_circuit`, `telesabre_map_circuit`,
`estimate_fidelity_from_cache[_batch]`) and adds the
`MultiCoreOrchestrator` SABRE pass. `quadris.dse` is a focused
package (≤ 800 lines per file): the DSE façade `engine.py` delegates
parameter registry, topology / slot-layout, noise merging, result
containers, config normalisation, and parallel cold scheduling to
focused leaf modules. Routing dispatch is via a `Backend` strategy
(`backends/{base,hqa_sabre,telesabre}.py`) — adding a third routing
algorithm is one new file plus an entry in `backends/__init__.py`.

Sweep orchestration (`sweep.py`) provides a single `_compile_one`
helper used by both the foreground `run_cold` and the multi-process
worker `_eval_cold_batch`, eliminating drift between the two cold
paths. Workers are forkserver-spawned, address-space-capped via
`RLIMIT_AS`, and recycled every four jobs to flush Qiskit / BLAS
caches. Per-process thread pools (OpenMP, OpenBLAS, Rayon) are
pinned to 1 thread at module import — workers parallelise at the
process level only.

---

## 5. Performance & benchmarks

**Test bench.**  AMD Ryzen 7 9700X (8 physical cores, 16 hardware
threads, base 3.8 GHz) · 64 GB DDR5 · Ubuntu 24.04.4 LTS, kernel
6.17.0-22 · Python 3.12.3 · Qiskit 2.x. All numbers come from
`tests/bench_quadrature_report.py`; supplementary scripts referenced
where indicated.

### 5.1 Cold-path latency vs. logical qubits

QFT-L on a ring of 4 cores (2 cores at L = 8), K = B = 1, spectral
placement, default noise. Median of three sub-process runs.

| L | num_cores | qubits/core | cold latency (s) | teleportations | intra-core SWAPs |
|--:|--:|--:|--:|--:|--:|
| 8   | 2 | 8  | **0.26** | 24    | 17    |
| 16  | 4 | 8  | **0.27** | 95    | 57    |
| 32  | 4 | 12 | **0.31** | 351   | 210   |
| 64  | 4 | 20 | **0.50** | 1 340 | 747   |
| 128 | 4 | 36 | **1.44** | 4 040 | 2 928 |

Latency is sub-linear in N up to ~32 logical qubits because Hungarian
matching dominates only when the number of conflicting pairs P is
non-trivial. Past L = 64, growth tracks Θ(L · P · N) as predicted by
§3.1.

### 5.2 Cold-path peak resident-set size vs. logical qubits

| L | peak RSS (MiB) |
|--:|--:|
| 8   | 146 |
| 16  | 148 |
| 32  | 158 |
| 64  | 221 |
| 128 | 632 |

The Python interpreter + Qiskit + numpy floor sits around ~145 MiB;
above ~64 logical qubits, the routed-circuit DAG and SABRE state
become dominant. The DSE engine consults this empirical curve
(`_EMPIRICAL_COLD_MB`) when scheduling its parallel cold pool — the
sum of estimated worker peaks is constrained to the available
budget, with single oversized jobs allowed to run alone.

### 5.3 Hot-path single-call latency

QFT-L on a ring of 2/4 cores; one `run_hot` per noise dict;
median / min / max over 100 reps.

| L | num_cores | qubits/core | median (µs) | min (µs) | max (µs) |
|--:|--:|--:|--:|--:|--:|
| 16 | 2 | 12 | **13.5**  | 13.4  | 24.3  |
| 32 | 4 | 12 | **41.9**  | 41.3  | 81.5  |
| 64 | 4 | 20 | **172.5** | 170.9 | 239.3 |

Hot-path latency tracks Θ(L · N) per the §3.5 complexity (single
sweep through the sparse tensor + per-qubit timeline updates).

### 5.4 Hot-path batched throughput

QFT-32 on a 4-core ring, varying `run_hot_batch` size; median of 5
reps. Per-cell amortised cost includes one Rust→Python crossing
plus the per-cell post-processing.

| batch size | wall time | per-cell amortised |
|--:|--:|--:|
| 1     | 0.0 ms  | 42.3 µs |
| 10    | 0.2 ms  | 18.9 µs |
| 100   | 1.7 ms  | 16.9 µs |
| 1 000 | 16.7 ms | **16.7 µs** |

Batching delivers a 2.5× per-cell speed-up at scale; once batch ≥
100 the FFI call's fixed cost is amortised away. This is what makes
multi-thousand-cell hot sweeps tractable.

### 5.5 Parallel cold-pool scaling

Sweep over `(num_cores ∈ [2, 10], two_gate_error)` — ten distinct
cold compilations + 9 hot points each (4 995 cells total), QFT-16,
default noise.

| max_workers | wall time (s) | speed-up |
|--:|--:|--:|
| 1 | 1.60 | 1.00× |
| 2 | 1.09 | 1.47× |
| 4 | 0.59 | 2.72× |
| 8 | 0.63 | 2.52× |

Speed-up plateaus at 4 workers because the empirical RAM-aware
scheduler caps concurrent cold compilations once estimated peak RSS
exceeds the per-host budget; on this 64 GB host that ceiling is
hit when 4 cold compiles overlap. The slight regression at 8
workers is the additional process-pool overhead with no extra
compute capacity.

### 5.6 Mapping quality vs. HQA paper / IBM Q reference

Ran 93 reference circuits from arXiv:2503.06693v2 through `quadris`
under uniform IBM-Q-typical noise (single-gate ε = 2.3 × 10⁻⁴,
two-gate ε = 8.2 × 10⁻³, T₁ = 120 µs, T₂ = 100 µs); compared to the
paper's mitigated IBM-Q hardware fidelity column.

| Statistic | Value |
|---|---|
| Matched circuits | **93** |
| Mean (quadris − IBM Q) | **+0.260** |
| Mean abs error | 0.269 |
| RMS error | 0.341 |
| Pearson r | **+0.84** |
| IBM Q range | [0.001, 1.001] |
| `quadris` range | [0.028, 0.999] |

The +0.26 bias reflects the use of *uniform* error rates; the paper
reports that per-qubit / per-pair calibration data is required to
hit the absolute IBM-Q numbers within ~5 %. The Pearson correlation
of +0.84 demonstrates that with no calibration, the *relative*
fidelity ranking across very different circuits is preserved.
Per-qubit and per-pair calibration is already wired through the
Python API; the next benchmark with real backend `properties()` data
is expected to close most of the bias.

### 5.7 TeleSABRE vs. HQA + SABRE

QFT and GHZ circuits, 4-core ring, K = B = 1, default noise.

| Circuit | L | back-end | cold (s) | teleports | swaps | overall fid |
|---|--:|---|--:|--:|--:|--:|
| QFT | 16 | hqa_sabre  | 0.28 | 95  | 57  | 0.003 |
| QFT | 16 | telesabre  | 0.08 | 107 | **37** | **0.422** |
| QFT | 32 | hqa_sabre  | 0.06 | 351 | 210 | 0.000 |
| QFT | 32 | telesabre  | 0.36 | 172 | 105 | **0.046** |
| GHZ | 16 | hqa_sabre  | 0.01 | 18  | 2   | 0.043 |
| GHZ | 16 | telesabre  | 0.01 | 19  | 14  | **0.410** |
| GHZ | 32 | hqa_sabre  | 0.05 | 34  | 3   | 0.000 |
| GHZ | 32 | telesabre  | 0.02 | 35  | 20  | **0.052** |

TeleSABRE inserts dramatically fewer SABRE-style intra-core SWAPs on
QFT (37 vs 57 at L = 16; 105 vs 210 at L = 32) and produces a
non-trivial fidelity at the noise level where HQA+SABRE collapses to
zero. The SWAP-cost saving is the dominant driver — each SWAP is
charged as 3 × CNOT under the noise model, so 105 fewer SWAPs at L
= 32 saves 315 CNOTs of error budget. Cold-time ratio is workload-
dependent: TeleSABRE wins on QFT-16 (0.08 s vs 0.28 s), HQA+SABRE
wins on QFT-32 (0.06 s vs 0.36 s).

### 5.8 Sweep-grid cell memory

Per-cell storage of 14 float64 output fields, measured on a 4 096-
cell grid.

| Layout | Bytes / cell | Total |
|---|--:|--:|
| structured numpy array (`_RESULT_DTYPE`) | **112** | 448 KiB |
| legacy list-of-dicts | 1 627 | 6 508 KiB |

The structured-array layout used since the recent engine refactor
saves **14.5×** memory per cell. At a 5 000-cell sweep this is the
difference between 0.5 MiB and 8 MiB held alongside the plot for
its lifetime; at the 50 000-cell ceiling, the difference is 5 MiB
vs. 78 MiB — enough to matter on a host running the GUI alongside a
browser and the OS.

---

## 6. GUI overview

The Dash app (`gui/app.py`) is a thin orchestrator over the library
— every numerical operation it performs is a call into
`quadris.dse` or `quadris.analysis`. Layout: left sidebar (sweep
axes + range sliders, up to 6), centre panel (plot + view-tab bar),
right sidebar (circuit / topology / noise / threshold / FoM tabs),
top bar (Run, Examples dropdown, Save/Load session). View catalogue:

| Tier | Views |
|---|---|
| Sweep (1-D / 2-D / 3-D) | line, heatmap, scatter3d, isosurface, frozen-heat slice |
| Analysis (any dim) | parallel coords, slices, importance, Pareto, correlation, elasticity, merit, topology |

Sessions round-trip the full UI state through a JSON blob, including
custom FoM expressions and frozen-axis slider values. Examples
dropdown ships canned 128-qubit DSE sessions for first-time users.

---

## 7. Validation & testing

The Python test suite covers the engine, plot builders, sweep
orchestration, and end-to-end smoke flows (229 focused tests in
`test_nd_sweep`, `test_sweep_progress`, `test_plotting_views`,
`test_plotting_merit`, `test_faceting`, `test_frozen_slider`).
Property-style tests check (a) memory-cap RuntimeError when a sweep
exceeds the RAM ceiling, (b) progress-callback invariants under 1-D
through 5-D sweeps, (c) backwards compatibility of the legacy
`sweep_1d/2d/3d` shape against `sweep_nd`, and (d) parallel
scheduler determinism. The Rust crate has its own `cargo test` unit
suite for HQA, the noise channels, and the teleportation event log;
`cargo bench` covers HQA hot loops. Mapping-quality validation is
the §5.6 reference comparison against the published IBM Q dataset.

---

## Appendix A — File index

| Symbol / capability | Location |
|---|---|
| `map_circuit`, `QusimResult` | `python/quadris/__init__.py` |
| `DSEEngine`, `SweepResult`, `SweepProgress` | `python/quadris/dse/engine.py` (façade) |
| Parameter registry (axes, NOISE_DEFAULTS) | `python/quadris/dse/axes.py` |
| Pinned-axis architecture resolver | `python/quadris/dse/config.py` |
| Topology + slot layout | `python/quadris/dse/topology.py` |
| Noise merging + tele-cost derivation | `python/quadris/dse/noise.py` |
| Routing back-ends | `python/quadris/dse/backends/` |
| Sweep / parallel pool | `python/quadris/dse/sweep.py` |
| FoM evaluator | `python/quadris/analysis/fom.py` |
| Pareto helpers | `python/quadris/analysis/pareto.py` |
| HQA Rust impl | `src/hqa/` |
| Noise model Rust impl | `src/noise/mod.rs` |
| TeleSABRE C library | `csrc/telesabre/` |
| TeleSABRE Rust wrapper | `src/telesabre/` |
| PyO3 entry points | `src/python_api.rs` |
| Dash app | `gui/app.py` |
| Plot builders | `gui/plotting.py` |
| Benchmark runner | `tests/bench_quadrature_report.py` |

## Appendix B — Public API signatures

```python
# quadris (top level)
def map_circuit(circuit, full_coupling_map, core_mapping, *, seed,
                initial_placement, single_gate_error, two_gate_error,
                t1, t2, ...) -> QusimResult
def telesabre_map_circuit(circuit_path, device_json_path,
                          config_json_path, ...) -> QusimResult
def estimate_fidelity_from_cache(gs_sparse, placements, distance_matrix,
                                 sparse_swaps, gate_error_arr,
                                 gate_time_arr, ...) -> dict
def estimate_fidelity_from_cache_batch(..., noise_dicts: list[dict]
                                       ) -> list[dict]

# quadris.dse
class DSEEngine:
    def run_cold(circuit_type, num_logical_qubits, num_cores,
                 qubits_per_core, topology_type, ..., pin_axis,
                 routing_algorithm, communication_qubits,
                 buffer_qubits, noise=None, ...) -> CachedMapping
    def run_hot(cached, noise: dict) -> dict
    def run_hot_batch(cached, noise_dicts: list[dict],
                      keep_grids=False) -> list[dict]
    def sweep_nd(cached, sweep_axes, fixed_noise, cold_config=None,
                 progress_callback=None, parallel=False,
                 max_workers=None, max_cold=None, max_hot=None,
                 keep_per_qubit_grids=False) -> SweepResult

@dataclass
class SweepResult:
    metric_keys: list[str]
    axes: list[np.ndarray]
    grid: np.ndarray  # structured dtype with one float per output
    per_qubit_data: dict | None
    @property
    def shape(self) -> tuple[int, ...]
    def to_sweep_data(self) -> dict

# quadris.analysis
@dataclass
class FomConfig:
    name: str
    numerator: str
    denominator: str
    intermediates: tuple[tuple[str, str], ...]

def compute_for_sweep(sweep_data: dict, config: FomConfig) -> FomResult
def pareto_front(sweep, objective_x: str, objective_y: str) -> dict
def pareto_front_mask(num: np.ndarray, den: np.ndarray) -> np.ndarray
```

## Appendix C — Glossary

- **HQA** — Hungarian Qubit Assignment (Escofet *et al.*, arXiv:2309.12182).
- **SABRE** — Swap-Based Bidirectional Heuristic Search (Li *et al.*, ASPLOS 2019), the Qiskit-bundled router for intra-core SWAP insertion.
- **TeleSABRE** — Variant of SABRE that handles intra-core SWAPs and inter-core teleportations in a single pass; vendored as a C library.
- **EPR pair** — Bell pair shared between two cores, consumed once per teleportation hop.
- **TREX** — T-REx readout-error mitigation (`readout_mitigation_factor` in the noise dict).
- **FoM** — Figure of Merit, a user-defined `numerator / denominator` expression over swept axes + outputs.
- **DSE** — Design Space Exploration: parameter sweeps over architectural / noise / circuit choices.
- **Cold path / hot path** — full mapping (~seconds) vs. cached re-estimation (~µs).
- **K, B** — communication qubits per group (K) and buffer qubits per group (B); `B ≤ K` per group rule.
- **Pinned axis** — the architectural axis (`num_cores` or `qubits_per_core`) the user holds fixed; the unpinned one is derived.
