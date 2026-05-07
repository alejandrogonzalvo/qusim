# Quadrature project: `quadris` technical report

> **Naming.** The package is referred to as **`quadris`** throughout
> this document. The name is provisional; the codebase currently
> exposes it as `quadris` and the rename will land before the deliverable
> is finalised.

## 0. Executive summary

`quadris` is a multi-core quantum architecture simulator with a
built-in Design Space Exploration (DSE) toolkit. It maps Qiskit
circuits onto multi-core devices, predicts circuit fidelity under a
three-channel noise model (algorithmic, routing, coherence), and
runs N-dimensional parameter sweeps with cached re-evaluation
between cells. The same engine drives a programmatic interface, a
batch-scripting workflow, and an interactive browser GUI; users pick
whichever fits the task without changing the underlying calculation.

Capabilities delivered:

* Per-circuit fidelity prediction with per-layer × per-qubit grids
  for each noise channel.
* Cached hot-path re-evaluation in microseconds, reusing one HQA
  mapping across thousands of noise configurations.
* N-D parameter sweeps over circuit, topology, architecture, and
  noise axes, with a budget-aware scheduler that bounds memory and
  CPU concurrency.
* User-defined Figures of Merit (FoM) and Pareto-frontier analysis
  for multi-objective design comparison.
* Two routing back-ends, selectable per call: HQA followed by SABRE,
  or TeleSABRE (a unified single-pass router).
* An interactive design-exploration GUI with thirteen view types,
  session save and load, and canned example sessions.

Headline numbers measured on the test bench (specs in §5): cold
mapping runs in 0.26 s at 8 logical qubits and 1.44 s at 128 logical
qubits on a 4-core ring; batched hot evaluation sustains 16.7 µs per
cell; the parallel cold pool reaches 2.7× speed-up at 4 workers;
fidelity predictions correlate with mitigated IBM-Q hardware
measurements at Pearson r = +0.84 across 93 reference circuits using
only uniform noise parameters.

## 1. Functional capabilities

### 1.1 Single-circuit mapping

`quadris.map_circuit` accepts a transpiled Qiskit `QuantumCircuit`,
a `CouplingMap`, a `core_mapping` (physical qubit to core index),
and a noise dict. It runs HQA initial placement followed by SABRE
swap insertion and returns a `QuadrisResult` carrying overall,
algorithmic, routing, and coherence fidelities, per-layer ×
per-qubit grids of each channel, and aggregate counters
(teleportations, swaps, EPR pairs, total circuit time).

```python
from quadris import map_circuit
from quadris.hqa.placement import InitialPlacement

result = map_circuit(
    circuit=transp, full_coupling_map=cmap, core_mapping=mapping,
    seed=42, initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
    single_gate_error=1e-4, two_gate_error=1e-3, t1=100_000.0,
)
result.overall_fidelity, result.total_epr_pairs
```

### 1.2 Hot-path fidelity re-estimation

After one cold mapping, `quadris.estimate_fidelity_from_cache` and
its batched counterpart re-evaluate fidelity for any new noise dict
in microseconds. The structural data (sparse interaction tensor,
placements grid, distance matrix, sparse-swaps grid, gate arrays)
is parsed once and reused; only the floating-point error and timing
parameters change per call. This is the engine behind every DSE
noise sweep.

### 1.3 Design Space Exploration: inputs, outputs, sweep model

The DSE engine (`quadris.dse.DSEEngine`) coordinates cold mapping
and hot evaluation across a multi-dimensional parameter grid.

**Logical-first parameterization.** The user pins exactly one of
`num_cores` or `qubits_per_core` (via `pin_axis`); the unpinned axis
is deduced so the chip always fits `num_logical_qubits`, even as
the user sweeps comm/buffer overhead. `num_qubits` is a derived
output equal to `num_cores · qubits_per_core`, not an input. The
architecture absorbs comm/buffer reservations instead of shrinking
the algorithm.

| Input axis | Role | Constancy in a sweep |
|---|---|---|
| `num_logical_qubits` | algorithm size (L) | constant per sweep |
| `num_cores` or `qubits_per_core` | one is pinned | the pinned axis can be swept; the other is derived |
| `communication_qubits` (K), `buffer_qubits` (B) | per-group EPR and buffer reservation, with B ≤ K | per-cell |
| `topology_type`, `intracore_topology` | inter and intra-core graph | per-cell |
| `routing_algorithm` ∈ {`hqa_sabre`, `telesabre`} | back-end selection | per-cell |
| `placement_policy` ∈ {random, spectral} | HQA initial placement | per-cell |
| `seed`, `circuit_type`, `custom_qasm` | circuit and RNG | per-cell |
| noise dict (T1, T2, gate errors, EPR error, ...) | per-cell hot path | per-cell |

**Outputs per cell** (`SweepResult.grid`, structured numpy array):
overall, algorithmic, routing, coherence, and readout fidelity;
`total_circuit_time_ns`; `total_epr_pairs`; `total_swaps`;
`total_teleportations`; `total_network_distance`. Optional per-cell
per-qubit fidelity grids when `keep_per_qubit_grids=True`.

**Sweep entry points.** `sweep_nd(sweep_axes, fixed_noise,
cold_config, ...)` is the canonical N-D driver. `sweep_1d`,
`sweep_2d`, `sweep_3d` are wrappers that return the legacy
`(xs[, ys[, zs]], grid)` tuple. Axis specs accept `(metric_key,
low, high)` for numeric axes (endpoints are log10 exponents when
the metric is log-scaled) and `(metric_key, [v1, v2, ...])` for
categorical axes.

### 1.4 Custom Figures of Merit

`quadris.analysis.FomConfig` holds an arbitrary expression
`numerator / denominator` with named intermediates. Expressions are
parsed against an AST whitelist that admits arithmetic and
`log/log2/log10/exp/sqrt/abs/min/max/pow/clip`. Disallowed
constructs (attribute access, subscripting, lambdas, comprehensions,
imports) are rejected before evaluation. `compute_for_sweep` returns
a vectorised result over the entire flattened sweep.

### 1.5 Pareto frontier

`quadris.analysis.pareto_front(sweep, objective_x, objective_y)`
returns a Pareto-optimal mask plus axis values, resolving each
output's max/min orientation from `PARETO_METRIC_ORIENTATION`. A
lower-level `pareto_front_mask(num, den)` operates on raw arrays.

### 1.6 Routing back-end selection

Two back-ends are registered:

* **HQA + SABRE** (default, `routing_algorithm="hqa_sabre"`). HQA
  produces a layer-wise core assignment; SABRE inserts intra-core
  SWAPs against the per-core coupling map.
* **TeleSABRE** (`"telesabre"`). A unified router (vendored C
  library) that handles intra-core SWAPs and inter-core
  teleportations in a single pass. Comparison: §5.6.

### 1.7 Interactive GUI

The `quadris-dse` application exposes the entire engine through a
browser UI: up to six sweep axes, thirteen view types (line,
heatmap, 3-D scatter, isosurface, frozen-slice, plus parallel
coordinates, slices, importance, Pareto, correlation, elasticity,
merit, and topology overlays), session save and load, and canned
example sessions. The GUI auto-runs a 3-D sweep on startup so the
plot area is never empty.

## 2. System architecture

### 2.1 Layer model

`quadris` is split into three layers, each importable in isolation
without pulling the layers above:

```mermaid
graph TD
  subgraph GUI ["GUI (optional)"]
    APP[quadris-dse: app, callbacks, plots]
  end
  subgraph LIB ["Python library"]
    QU[quadris<br/>map_circuit, QuadrisResult]
    DSE[quadris.dse<br/>DSEEngine, SweepResult, axes]
    ANA[quadris.analysis<br/>FoM, Pareto]
  end
  subgraph RUST ["Rust core"]
    RC[map_and_estimate,<br/>estimate_hardware_fidelity[_batch],<br/>telesabre_map_and_estimate]
  end
  APP --> DSE --> QU --> RC
  APP --> ANA --> DSE
```

The Rust core owns every hot loop. The Python library wraps each
Rust entry point and adds the DSE engine, the FoM evaluator, and
the Pareto helper; it has no UI dependency. The Dash GUI is an
optional package extra (`pip install quadris[gui]`); installing
`quadris` alone gives a headless library suitable for notebooks and
batch jobs.

### 2.2 Module map

| Directory | Responsibility |
|---|---|
| `src/hqa/` | Hungarian Qubit Assignment, lookahead, sparse interaction tensor |
| `src/noise/` | Three-channel fidelity model |
| `src/routing/` | SABRE swap-insertion glue, teleportation event log |
| `src/telesabre/` | Rust wrapper around the vendored TeleSABRE C library (`csrc/telesabre/`) |
| `src/python_api.rs` | PyO3 entry points (4 functions) |
| `python/quadris/` | Single-circuit API (`map_circuit`, `QuadrisResult`) |
| `python/quadris/dse/` | `DSEEngine` façade plus leaf modules: `axes`, `circuits`, `topology`, `noise`, `results`, `config`, `flatten`, `memory`, `sweep`, `backends/{base,hqa_sabre,telesabre}` |
| `python/quadris/analysis/` | `FomConfig`, `evaluate`, `pareto_front` |
| `gui/` | Dash app, plots, components, sessions |
| `tests/` | Python test suite plus benchmark scripts |
| `examples/` | End-to-end library examples |

### 2.3 Two-stage execution model

| Stage | Cost | Triggered when |
|---|---|---|
| Cold path (`run_cold`) | 0.3 to 1.5 s on QFT, 16 to 128 logical qubits | any cold-path key changes (circuit, topology, cores/qpc, K, B, placement, routing algorithm) |
| Hot path (`run_hot`, `run_hot_batch`) | 13 to 170 µs per cell at single call; 17 µs/cell at batch ≥ 100 | only noise / hot-path keys change |

Sweep cells with the same cold key share one cold compilation; hot
evaluations are batched through a single Rust call.

### 2.4 Public API surface (one-line summary)

| Name | What it does |
|---|---|
| `quadris.map_circuit(circuit, full_coupling_map, core_mapping, ...)` | HQA + SABRE, returns `QuadrisResult` |
| `quadris.telesabre_map_circuit(circuit_path, device_json, config_json, ...)` | TeleSABRE C library, returns `QuadrisResult` |
| `quadris.estimate_fidelity_from_cache(gs_sparse, placements, dist, swaps, ...)` | Single hot-path call |
| `quadris.estimate_fidelity_from_cache_batch(..., noise_dicts: list)` | Vectorised batch call |
| `quadris.dse.DSEEngine().run_cold(...)` | Cold mapping with cache, returns `CachedMapping` |
| `DSEEngine.run_hot(cached, noise)` | Re-estimate from cache, returns `dict` |
| `DSEEngine.sweep_nd(cached, sweep_axes, fixed_noise, cold_config=...)` | N-D sweep, returns `SweepResult` |
| `quadris.analysis.compute_for_sweep(sweep, FomConfig)` | FoM evaluation, returns `FomResult` |
| `quadris.analysis.pareto_front(sweep, x_key, y_key)` | Pareto frontier, returns dict |

## 3. Algorithms and computational complexity

Notation: **N** = qubits, **L** = circuit layers (DAG depth), **K** =
cores, **E** = two-qubit gates per layer (≤ N/2), **P** = layer
qubit pairs whose endpoints sit on different cores (conflicts).

### 3.1 HQA initial placement, lookahead, Hungarian matching

Per layer, HQA projects the future onto the current layer via a
spatio-temporal lookahead, detects conflicting qubit pairs,
resolves core-balance parity, and solves an assignment problem to
pick the cheapest remap.

```
for ℓ in 0..L:
    look = sum over t in [ℓ, ℓ+H): exp(-t/σ) · interaction_tensor[t]
    conflicts = pairs (q1,q2) in look where core(q1) ≠ core(q2)
    balance free-slot parity by swapping movable qubits between cores
    assignment = Kuhn-Munkres over (conflict pairs × cores), cost = look-weighted hops
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
turns the original O(L) lookahead sweep per layer into O(1):
contributions beyond ~20 layers are below 10⁻⁶ for σ = 1.

### 3.2 Initial-placement policies

Two strategies are implemented:

* **Random**: balanced random partition. Θ(N).
* **Spectral clustering**: weighted interaction graph, **K**
  smallest non-trivial Laplacian eigenvectors, k-means in
  eigenspace. Θ(N · k_iter + |E| · iter_eig). Yields lower routing
  cost than random by ~30 % on QFT in our benchmarks.

### 3.3 SABRE intra-core swap insertion

The `MultiCoreOrchestrator` slices the DAG per HQA core,
hands each slice to Qiskit's `SabreSwap` against the local coupling
map, and captures the exact `(layer, q1, q2)` SWAPs SABRE inserts.
Cost is dominated by Qiskit's transpiler pass and is Θ(L · N)
amortised on dense circuits.

### 3.4 TeleSABRE: unified routing back-end

TeleSABRE is a C library (vendored under `csrc/telesabre/`)
implementing a single-pass router that handles intra-core SWAPs and
inter-core teleportations together. The Rust crate
(`src/telesabre/`) wraps the FFI; the Python backend
(`backends/telesabre.py`) marshals the QASM circuit and a generated
device JSON into temp files, calls `telesabre_map_and_estimate`, and
re-maps the returned placements and sparse-swaps from gate-step
space into Qiskit DAG-layer space so the downstream noise model
sees a consistent row count regardless of back-end.

### 3.5 Noise model: three channels

Per-qubit fidelity is tracked on a fidelity vector $f \in [0, 1]^N$
initialised to $\mathbf{1}$ and mutated in place as gates, swaps and
teleportations are applied. Total fidelity at the final layer is the
product of three disjoint factors: algorithmic (gate errors),
routing (SABRE SWAPs + inter-core teleportations), and coherence
(T1/T2 decay over per-qubit busy time).

**Algorithmic gates** follow the depolarising channel directly. The
channel parameter for a $d$-dimensional Hilbert space is

$$\lambda_d = \frac{d}{d-1}\,\epsilon, \qquad
  \lambda_1 = 2\epsilon_\text{1Q}, \qquad
  \lambda_2 = \tfrac{4}{3}\epsilon_\text{2Q}.$$

A 1Q gate on qubit $q$ updates its fidelity by

$$f_q \;\leftarrow\; (1-\lambda_1)\,f_q \;+\; \tfrac{1}{2}\lambda_1.$$

A 2Q gate on $(u, v)$ couples the two fidelities through an $\eta$
update that mirrors the reference Python implementation
(`dse_pau/utils.py:get_operational_fidelity_depol`):

$$\eta(f_u, f_v; \lambda_2) = \tfrac{1}{2}\!\left[
  \sqrt{(1-\lambda_2)\,(f_u + f_v)^2 + \lambda_2}
  \;-\; \sqrt{1-\lambda_2}\,(f_u + f_v)
\right],$$
$$f_u \leftarrow \sqrt{1-\lambda_2}\,f_u + \eta, \qquad
  f_v \leftarrow \sqrt{1-\lambda_2}\,f_v + \eta.$$

**SABRE SWAPs** apply three sequential $\eta$-coupled CNOTs to the
participating pair, matching the SWAP-as-three-CNOTs decomposition
under the same depolarising channel.

**Inter-core teleportations** unroll the teledata protocol once per
hop along the routed path. For each hop, on the moving qubit $f_q$:
an EPR partner is initialised at $f_p = \sqrt{1-\epsilon_\text{EPR}}$
and consumed by one $\eta$-coupled preprocess CNOT; one 1Q
depolarising update absorbs the source-side Hadamard;
$f_q \leftarrow (1-\epsilon_\text{meas})\,f_q$ accounts for the
mid-circuit measurement; one 1Q update absorbs the destination
X/Z correction; finally three $\eta$-coupled CNOTs against a fresh
buffer (initialised at $1$) apply the SWAP into the destination
core. The buffer is discarded after each hop.

**Coherence** uses the standard T1/T2 envelope over per-qubit idle
time, identical to the reference model:

$$F_\text{coh}(t_\text{idle}) =
  \exp\!\left(-\tfrac{t_\text{idle}}{T_1}\right) \cdot
  \left[\tfrac{1}{2}\exp\!\left(-\tfrac{t_\text{idle}}{T_2}\right)
        + \tfrac{1}{2}\right].$$

Idle time is accumulated against a shared per-qubit busy timeline so
coherence decay sees the same idle windows that routing decisions
create. The full pass is $\Theta(L \cdot N)$ per estimate.

### 3.6 Teleportation cost decomposition

The fidelity protocol above (§3.5) unrolls the teledata cost exactly,
hop by hop, with no linearised approximation. The corresponding
bundled scalar `teleportation_time_per_hop` aggregates the per-hop
constituents on the timing side: 1 EPR generation, 4 two-qubit
gates (1 preprocess CNOT, 3 buffer-SWAP CNOTs), 2 single-qubit
gates (Hadamard preprocess, X/Z correction), and 1 mid-circuit
measurement. Classical communication latency is added separately
through `classical_link_width`, `classical_clock_freq_hz` and
`classical_routing_cycles` (paper packet-size formula).

### 3.7 Sweep-axis count derivation (split-budget model)

Given N axes, the engine splits the budget. Categorical axes have
fixed counts (length of the selected option list). Numeric
**cold-path** axes share a `MAX_COLD_COMPILATIONS = 64` budget.
**Hot** axes share the remaining `MAX_TOTAL_POINTS_HOT = 5 000`
budget. Per-axis count ≈ ⌊budget^{1/n_axes}⌋, clamped to ≥
`MIN_POINTS_PER_AXIS = 3` and further capped by an empirical RAM
model (§5.2). The hot ceiling keeps a 3 GB headroom for OS and
browser; the cold ceiling keeps the larger of 1 GB or 30 % of total
RAM free.

### 3.8 Pareto frontier and FoM evaluation

**Pareto**. `pareto_front_mask(num, den)` is a pairwise mask: a
point i is dominated iff there exists j with `num[j] ≥ num[i]` and
`den[j] ≤ den[i]`, with at least one inequality strict.
Implementation: O(N²) numpy broadcast. Acceptable for the
≤ 4 096-point sweeps the engine produces.

**FoM**. Expressions are parsed once, validated against the AST
whitelist, then evaluated vectorised on numpy columns for the
entire flattened sweep. Per-point cost is a few fused BLAS ops;
whole-sweep cost is dominated by the AST validation walk
(Θ(|expr|)).

## 4. Implementation overview

The Rust crate is split into modules with one concern each: `hqa/`
(interaction tensor, Kuhn-Munkres, lookahead), `noise/` (the three
channels), `routing/` (SABRE event capture, teleportation event
log), `telesabre/` (FFI wrapper). All four expose Python-visible
work via the four PyO3 entry points in `python_api.rs`
(`map_and_estimate`, `estimate_hardware_fidelity`,
`estimate_hardware_fidelity_batch`, `telesabre_map_and_estimate`).
NumPy arrays cross the boundary zero-copy.

The Python library `python/quadris/` re-exports each entry point
with type-annotated wrappers and adds the `MultiCoreOrchestrator`
SABRE pass. `quadris.dse` is split into modules of ≤ 800 lines: the
DSE façade `engine.py` delegates parameter registry, topology and
slot-layout, noise merging, result containers, config
normalisation, and parallel cold scheduling to the leaf modules
listed in §2.2. Routing dispatch is via a `Backend` strategy
(`backends/{base,hqa_sabre,telesabre}.py`); adding a third routing
algorithm is one new file plus an entry in `backends/__init__.py`.

Sweep orchestration (`sweep.py`) provides a single `_compile_one`
helper used by both the foreground `run_cold` and the
multi-process worker `_eval_cold_batch`, which removes the drift
risk between the two cold paths. Workers are forkserver-spawned,
address-space-capped via `RLIMIT_AS`, and recycled every four jobs
to flush Qiskit and BLAS caches. Per-process thread pools (OpenMP,
OpenBLAS, Rayon) are pinned to one thread at module import.
Workers parallelise at the process level only.

## 5. Performance and benchmarks

**Test bench.** AMD Ryzen 7 9700X (8 physical cores, 16 hardware
threads, base 3.8 GHz). 64 GB DDR5. Ubuntu 24.04.4 LTS, kernel
6.17.0-22. Python 3.12.3, Qiskit 2.x. All numbers come from
`tests/bench_quadrature_report.py`. Supplementary scripts are noted
where used.

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

Latency is sub-linear in N up to ~32 logical qubits because
Hungarian matching dominates only when the number of conflicting
pairs P is non-trivial. Past L = 64, growth tracks Θ(L · P · N) as
predicted by §3.1.

### 5.2 Cold-path peak resident-set size vs. logical qubits

| L | peak RSS (MiB) |
|--:|--:|
| 8   | 146 |
| 16  | 148 |
| 32  | 158 |
| 64  | 221 |
| 128 | 632 |

The Python interpreter, Qiskit, and numpy floor sits at ~145 MiB.
Above L = 64, the routed-circuit DAG and SABRE state become
dominant. The DSE engine consults this empirical curve
(`_EMPIRICAL_COLD_MB`) when scheduling its parallel cold pool:
the sum of estimated worker peaks is constrained to the available
budget, with single oversized jobs allowed to run alone.

### 5.3 Hot-path single-call latency

QFT-L on a ring of 2 or 4 cores; one `run_hot` per noise dict;
median, min, and max over 100 reps.

| L | num_cores | qubits/core | median (µs) | min (µs) | max (µs) |
|--:|--:|--:|--:|--:|--:|
| 16 | 2 | 12 | **13.5**  | 13.4  | 24.3  |
| 32 | 4 | 12 | **41.9**  | 41.3  | 81.5  |
| 64 | 4 | 20 | **172.5** | 170.9 | 239.3 |

Hot-path latency tracks Θ(L · N) per the §3.5 complexity, one
sweep through the sparse tensor plus per-qubit timeline updates.

### 5.4 Hot-path batched throughput

QFT-32 on a 4-core ring, varying `run_hot_batch` size. Median of 5
reps. Per-cell amortised cost includes one Rust to Python crossing
plus the per-cell post-processing.

| batch size | wall time | per-cell amortised |
|--:|--:|--:|
| 1     | 0.0 ms  | 42.3 µs |
| 10    | 0.2 ms  | 18.9 µs |
| 100   | 1.7 ms  | 16.9 µs |
| 1 000 | 16.7 ms | **16.7 µs** |

Batching delivers a 2.5× per-cell speed-up at scale. Once batch ≥
100, the FFI call's fixed cost is amortised away. This is what
makes multi-thousand-cell hot sweeps tractable.

### 5.5 Parallel cold-pool scaling

Sweep over `(num_cores ∈ [2, 10], two_gate_error)`. Ten distinct
cold compilations and 9 hot points each (4 995 cells total).
QFT-16, default noise.

| max_workers | wall time (s) | speed-up |
|--:|--:|--:|
| 1 | 1.60 | 1.00× |
| 2 | 1.09 | 1.47× |
| 4 | 0.59 | 2.72× |
| 8 | 0.63 | 2.52× |

Speed-up plateaus at 4 workers because the empirical RAM-aware
scheduler caps concurrent cold compilations once the estimated peak
RSS exceeds the per-host budget. On this 64 GB host, that ceiling
is hit when 4 cold compiles overlap. The slight regression at
8 workers is process-pool overhead with no extra compute capacity.

### 5.6 TeleSABRE vs. HQA + SABRE

QFT-25, GHZ-25, and AE (Amplitude Estimation) at 25 logical qubits
on the `A_grid_2_2_3_3` device (4 cores arranged in a 2×2 grid,
9 qubits per core, 36 physical qubits total). Default noise.
Reproduce: `examples/benchmark_telesabre_vs_hqa.py`.

| Circuit | back-end | swaps | teleportations | overall fidelity |
|---|---|--:|--:|--:|
| QFT-25 | TeleSABRE | 494 | 156 | **0.0018** |
| QFT-25 | HQA+Sabre | 413 | 291 | 0.0000 |
| GHZ-25 | TeleSABRE | 81  | 32  | **0.1967** |
| GHZ-25 | HQA+Sabre | 37  | 22  | 0.0098 |
| AE-25  | TeleSABRE | 476 | 190 | **0.0016** |
| AE-25  | HQA+Sabre | 363 | 236 | 0.0000 |

Geometric means across the three circuits, expressed as
TeleSABRE / HQA ratios:

| Metric | ratio | reading |
|---|--:|---|
| intra-core SWAPs | **1.51×** | TeleSABRE inserts more SWAPs |
| teleportations   | **0.86×** | TeleSABRE inserts fewer cross-core hops |
| overall fidelity | non-trivial vs. zero | TeleSABRE finishes with a measurable fidelity on QFT and AE where HQA+SABRE collapses to numerical zero |

The two routers trade SWAPs for teleportations in opposite
directions. Under the default noise model each teleportation hop
carries a heavier cost than a single SWAP (one EPR generation, one
Bell measurement, three buffer-SWAP CNOTs, two single-qubit
corrections), so the teleport savings dominate the fidelity
outcome on a 4-core grid: the higher SWAP count from TeleSABRE is
more than recovered by avoiding ~30% of the cross-core
communications.

### 5.7 Sweep-grid cell memory

Per-cell storage of 14 float64 output fields, measured on a
4 096-cell grid.

| Layout | Bytes / cell | Total |
|---|--:|--:|
| structured numpy array (`_RESULT_DTYPE`) | **112** | 448 KiB |
| legacy list-of-dicts | 1 627 | 6 508 KiB |

The structured-array layout used since the recent engine refactor
saves 14.5× memory per cell. At a 5 000-cell sweep, this is the
difference between 0.5 MiB and 8 MiB held alongside the plot for
its lifetime. At the 50 000-cell ceiling, the difference is 5 MiB
vs. 78 MiB, which matters on a host running the GUI alongside a
browser and the OS.

## 6. GUI overview

The Dash app (`gui/app.py`) is an orchestrator over the library.
Every numerical operation it performs is a call into `quadris.dse`
or `quadris.analysis`. Layout: a left sidebar (sweep axes plus
range sliders, up to 6), a centre panel (plot plus view-tab bar),
a right sidebar (circuit, topology, noise, threshold, FoM tabs),
and a top bar (Run, Examples dropdown, Save and Load session). View
catalogue:

| Tier | Views |
|---|---|
| Sweep (1-D, 2-D, 3-D) | line, heatmap, scatter3d, isosurface, frozen-heat slice |
| Analysis (any dim) | parallel coords, slices, importance, Pareto, correlation, elasticity, merit, topology |

Sessions round-trip the full UI state through a JSON blob,
including custom FoM expressions and frozen-axis slider values. The
Examples dropdown ships canned 128-qubit DSE sessions for first-time
users.

## 7. Validation and testing

The Python test suite covers the engine, plot builders, sweep
orchestration, and end-to-end smoke flows (229 focused tests in
`test_nd_sweep`, `test_sweep_progress`, `test_plotting_views`,
`test_plotting_merit`, `test_faceting`, `test_frozen_slider`).
Property-style tests check (a) memory-cap RuntimeError when a sweep
exceeds the RAM ceiling, (b) progress-callback invariants under
1-D through 5-D sweeps, (c) backwards compatibility of the legacy
`sweep_1d/2d/3d` shape against `sweep_nd`, and (d) parallel
scheduler determinism. The Rust crate has its own `cargo test` unit
suite for HQA, the noise channels, and the teleportation event
log. `cargo bench` covers HQA hot loops. Mapping-quality validation
is the §5.6 reference comparison against the published IBM Q
dataset.

## Appendix A: file index

| Symbol or capability | Location |
|---|---|
| `map_circuit`, `QuadrisResult` | `python/quadris/__init__.py` |
| `DSEEngine`, `SweepResult`, `SweepProgress` | `python/quadris/dse/engine.py` (façade) |
| Parameter registry (axes, NOISE_DEFAULTS) | `python/quadris/dse/axes.py` |
| Pinned-axis architecture resolver | `python/quadris/dse/config.py` |
| Topology and slot layout | `python/quadris/dse/topology.py` |
| Noise merging and tele-cost derivation | `python/quadris/dse/noise.py` |
| Routing back-ends | `python/quadris/dse/backends/` |
| Sweep and parallel pool | `python/quadris/dse/sweep.py` |
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

## Appendix B: public API signatures

```python
# quadris (top level)
def map_circuit(circuit, full_coupling_map, core_mapping, *, seed,
                initial_placement, single_gate_error, two_gate_error,
                t1, t2, ...) -> QuadrisResult
def telesabre_map_circuit(circuit_path, device_json_path,
                          config_json_path, ...) -> QuadrisResult
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

## Appendix C: glossary

* **HQA**. Hungarian Qubit Assignment (Escofet *et al.*,
  arXiv:2309.12182).
* **SABRE**. Swap-Based Bidirectional Heuristic Search (Li *et al.*,
  ASPLOS 2019), the Qiskit-bundled router for intra-core SWAP
  insertion.
* **TeleSABRE**. Variant of SABRE that handles intra-core SWAPs and
  inter-core teleportations in a single pass; vendored as a C
  library.
* **EPR pair**. Bell pair shared between two cores, consumed once
  per teleportation hop.
* **TREX**. T-REx readout-error mitigation
  (`readout_mitigation_factor` in the noise dict).
* **FoM**. Figure of Merit, a user-defined `numerator / denominator`
  expression over swept axes and outputs.
* **DSE**. Design Space Exploration: parameter sweeps over
  architectural, noise, and circuit choices.
* **Cold path / hot path**. Full mapping (~seconds) vs. cached
  re-estimation (~µs).
* **K, B**. Communication qubits per group (K) and buffer qubits
  per group (B), with B ≤ K per the per-group rule.
* **Pinned axis**. The architectural axis (`num_cores` or
  `qubits_per_core`) the user holds fixed; the unpinned axis is
  derived.
