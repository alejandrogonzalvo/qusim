# Logical-first DSE parameterization

**Date:** 2026-05-05
**Status:** approved

## Problem

Today's DSE engine treats `num_qubits` (physical device size) as the
primary input. When the user sweeps comm/buffer qubits, the engine
silently shrinks `num_logical_qubits` (the algorithm size) to fit the
remaining data slots. A QFT-100 sweep over `communication_qubits`
quietly becomes a QFT-99/QFT-98/... — different circuit per cell —
making sweeps incomparable.

The user wants the **circuit to be sacred** during a sweep. The
architecture (cores, qubits-per-core, hence physical qubit count)
should adapt to absorb the changing comm/buffer overhead.

## New parameterization

### Inputs (user-set, sweepable)

- `num_logical_qubits` — algorithm size (auto-derived from QASM upload)
- `qubits_per_core` (qpc) — slots per core (uniform across cores)
- `num_cores` — only when pin = cores
- `communication_qubits` (K) — comm qubits per inter-core group
- `buffer_qubits` (B) — buffer qubits per group
- `intra_topology`, `inter_topology`, `routing_algorithm`

### Pin toggle (Q3 = (a))

A lock icon on each of `num_cores` and `qubits_per_core`. Exactly one
is pinned at a time. Pinning cores → qpc is derived; pinning qpc →
cores is derived. The unpinned axis is greyed-out and shows its
derived value live.

**Default = cores pinned** (always feasible; familiar mental model).

### Derived (output, read-only)

- `derived_num_cores` (when qpc-pinned)
- `derived_qubits_per_core` (when cores-pinned)
- `num_qubits = num_cores · qubits_per_core` (always)
- `idle_reserved_qubits` — comm slots that exist but host no edge
  (corner/edge cores in non-uniform inter-topologies)

All four become first-class FoM/Pareto axis options.

### Removed

- `num_qubits` slider and sweep axis
- `qubits` sweep alias (use `num_logical_qubits` instead)
- The clamping of `num_logical_qubits` to data-slot capacity in
  `_clamp_cfg_comm_and_logical` (replaced by deduction)

## Reservation rule

Every core reserves the **same** `G_max(num_cores, inter_topology) ·
(K + B)` slots, where `G_max` is the worst-case neighbour count
across the chip (4 for grid, 2 for ring with `nc ≥ 3`, etc.).
Cores with `G(c) < G_max` carry idle/unused comm slots. Trade-off:
visual + structural uniformity at the cost of waste at corners.

`data_per_core = qpc − G_max · (K + B)` (uniform across cores).

## Deduction

**Pin = cores, derive qpc:**
```
qpc = ceil(logical / nc) + G_max(nc, topo) · (K + B)
```
Always feasible.

**Pin = qpc, derive nc:** smallest `nc ≥ 1` such that
```
nc · (qpc − G_max(nc, topo) · (K + B)) ≥ logical
```
For ring/linear/grid the LHS is monotone in nc; loop nc = 1, 2, …
until satisfied. For all-to-all the LHS is `nc·qpc − nc(nc−1)(K+B)`,
a downward parabola; check up to nc ≈ qpc / (K+B); if the peak still
< logical, mark **infeasible**.

`nc = 1` is always tried first; if logical ≤ qpc, the answer is 1
and K/B contribute nothing (single-core).

## Infeasibility handling (G1, G5)

- Where the constraint is local to the slider (B > K), block it on
  the slider directly.
- Where the constraint requires deduction (no nc fits, qpc < K+B+1,
  intra-core layout cannot place K+B on a side), the cell renders
  **white/NaN** in the heat map.
- Sweep-axis space shows a banner: *"X of Y cells skipped
  (infeasible)"*. Colorbar auto-excludes NaN from min/max.
- Single-cell run with infeasible inputs → error popup, run-button
  disabled. Hot-reload silently no-ops.

## GUI changes

**Topology tab:**
- Remove "Physical qubits" slider
- New "Qubits per core" slider
- Lock icons on `Cores` and `Qubits per core`; click to flip pin
- Derived axis renders as a read-only badge with the live computed
  value
- Small badge below both: `→ N physical qubits, M idle reserved`

**Circuit tab:**
- When custom QASM is uploaded, the `num_logical_qubits` sweep axis
  is greyed out in the metric dropdown ("fixed by uploaded circuit")

**Sweep axis dropdown:**
- The unpinned axis (cores in qpc-mode, qpc in cores-mode) is
  filtered out of the sweep options
- `num_qubits` and `qubits` are removed entirely

## Engine changes

### `topology.py`
- New: `g_max(num_cores, inter_topology) -> int`
- New: `deduce_num_cores(logical, qpc, K, B, inter_topology) -> int | None`
- New: `deduce_qubits_per_core(logical, num_cores, K, B, inter_topology) -> int`
- Rewrite `_build_topology` to use uniform `G_max · (K+B)`
  reservation per core (no per-core variation)
- Drop `clamp_k_for_topology`, `clamp_b_for_topology`,
  `max_data_slots` (callers move to deduction)

### `config.py`
- Replace `_clamp_cfg_comm_and_logical` with
  `_resolve_architecture(cfg)`:
  - Reads `pin_axis`, the pinned value, and the inputs
  - Computes the derived axis via the deduction
  - Sets `num_qubits = nc · qpc` on the cfg
  - Returns infeasible flag for callers to render NaN
- Drop `_expand_qubits_alias` (no more `qubits` alias)

### `axes.py`
- Remove `qubits` and `num_qubits` `MetricDef`s
- Add `qubits_per_core` `MetricDef`
- Keep `num_cores` (only sweepable when pinned)

### `engine.py`
- Update `COLD_PATH_KEYS` to drop `num_qubits`, `qubits`; add
  `qubits_per_core`, `pin_axis`
- `_eval_point` and `_eval_cold_batch` call `_resolve_architecture`
  before `run_cold`
- `_group_nq` computes `num_qubits` from swept dict
  (`nc · qpc`) for memory sizing
- `run_cold` infeasible cells return a sentinel `CachedMapping` with
  NaN fidelity (or just raise; the sweep grid catches and writes NaN)

### `results.py`
- Add scalar metrics `derived_num_cores`, `derived_qubits_per_core`,
  `num_qubits`, `idle_reserved_qubits` to `_RESULT_DTYPE` and
  `_RESULT_SCALAR_KEYS`

### `_build_telesabre_device_json`
- Same uniform G_max reservation rule

## Examples

Existing example session JSONs reference the old schema. Per G6,
they will be deleted and rebuilt for the new schema. New examples
should showcase:
- QFT-100 sweep over `communication_qubits` with cores pinned
  (showing qpc growing) — the headline use case
- Comm × buffer 2D sweep with B > K cells white
- Cores sweep at fixed logical (recovering today's "compare 2- vs
  4-core" workflow)

## Tests

- Engine: `deduce_num_cores` cases (ring, all-to-all, grid;
  feasibility cliff)
- Engine: `deduce_qubits_per_core` always feasible
- Engine: QFT-100 sweep over K from 1 to 5 with cores pinned at 4
  preserves `num_logical_qubits=100` in every cell
- Engine: 2D K×B sweep produces NaN for B > K cells
- Engine: TeleSABRE device JSON has correct uniform reservation

## Out of scope

- Renaming `num_qubits` storage in `CachedMapping.config_key`
  (keep as derived field; no migration story needed since examples
  are rebuilt)
- Heterogeneous core sizes (rejected by G2)
