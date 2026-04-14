# qusim Roadmap

Feature roadmap for the qusim fidelity simulator. Items are grouped by priority based on researcher impact and implementation effort.

Status legend: **done** | **in progress** | planned

---

## High Priority

### TeleSABRE integration — planned

The current two-stage pipeline (HQA global placement → SABRE intra-core routing) treats inter-core teleportations and intra-core SWAPs as independent passes. TeleSABRE is a unified router that natively handles both local SWAP insertion and inter-core teleportation in a single pass, potentially reducing total communication overhead compared to the decoupled approach.

Integration would replace the `MultiCoreOrchestrator` with TeleSABRE as the routing backend, while keeping the HQA placement table as the initial assignment fed into the router. The Rust fidelity engine would remain unchanged — it already accepts a generic sparse SWAP/teleportation event list.

**Effort:** Medium — requires adapting TeleSABRE's output event format to the existing sparse tensor interface.

### Per-qubit and per-pair gate error rates — **done**

Real backends have wildly varying gate errors across qubits. IBM Q calibration data provides per-qubit `sx` errors and per-pair `cx` errors.

- `single_gate_error_per_qubit: Optional[np.ndarray]` — shape `(n_qubits,)`, falls back to scalar
- `two_gate_error_per_pair: Optional[dict[tuple[int,int], float]]` — keyed by `(q1, q2)`, falls back to scalar for missing pairs

Implemented in `src/noise/mod.rs` with `ArchitectureParams.single_gate_error_for(q)` and `two_gate_error_for(u, v)` lookup methods. Both `process_computational_gates` and `process_sabre_swaps` use the per-qubit/per-pair values. Fully wired through the Python API.

### Readout / measurement error

The current model has no readout error parameter. This is a significant noise source on real hardware (1-5% per qubit on IBM Q).

- Add `readout_error: float` (scalar, uniform) to `ArchitectureParams`
- Add `readout_error_per_qubit: Optional[Vec<f64>>` for per-qubit readout errors
- Apply as a multiplicative factor on measured qubits at the final layer
- Expose in `map_circuit()` Python API

**Effort:** Low — single new noise term applied once at end of circuit.

### Architecture specification object

Users currently hand-build `CouplingMap` + `core_mapping` + distance matrices manually (see `examples/dse_qft_cores.py`). This is error-prone and verbose. A structured `Architecture` object would make DSE sweeps trivial.

```python
@dataclass
class CoreSpec:
    num_qubits: int
    intra_core_topology: str  # "ring", "grid", "all-to-all", "heavy-hex"
    t1: float | np.ndarray    # scalar or per-qubit
    t2: float | np.ndarray

@dataclass
class Architecture:
    cores: list[CoreSpec]
    inter_core_topology: str  # "ring", "star", "all-to-all", "mesh"
    inter_core_link_error: float
    inter_core_link_time: float
```

This would generate `CouplingMap`, `core_mapping`, and `distance_matrix` automatically from a declarative spec, and support heterogeneous cores (different qubit counts, topologies, and error characteristics per core).

**Effort:** Medium — Python-side sugar over existing Rust engine.

---

## Medium Priority


### Per-qubit final fidelity in output

Users manually index into grids to get per-qubit fidelity:
```python
result.algorithmic_fidelity_grid[-1, q] * result.routing_fidelity_grid[-1, q] * result.coherence_fidelity_grid[-1, q]
```

Add a convenience `per_qubit_fidelity: np.ndarray` (shape `(n_qubits,)`) to `QusimResult` that combines all three channels at the final layer.

**Effort:** Low — computed from existing grids.

### Backend calibration loader

The benchmark script manually hardcodes `T1_NS = 120_000`, `SINGLE_GATE_ERROR = 2.3e-4`, etc. A factory that pulls calibration data from IBM Q backends would bridge simulation and hardware.

```python
arch = Architecture.from_ibm_backend(backend)
result = map_circuit(circuit, arch)
```

Requires `qiskit-ibm-runtime` as an optional dependency. Would use `backend.properties()` to pull per-qubit T1/T2 and per-gate error rates.

**Effort:** Medium — parsing IBM Q calibration data into `ArchitectureParams`.

### Expose routing event log to Python

The `TeleportationEvent` list (timeslice, qubit, src_core, dst_core, distance) exists in Rust (`src/routing.rs`) but only aggregate stats are surfaced to Python. Exposing the full event log would help researchers analyze routing patterns and bottlenecks.

**Effort:** Low — data already computed in Rust, just needs PyO3 serialization.

### Fidelity breakdown by noise source

Researchers want to answer "what's killing my fidelity?" A per-qubit breakdown showing the relative contribution of gate errors vs. coherence vs. routing would enable targeted architecture optimization.

Could be derived from existing grids but a dedicated summary would save repeated work:
```python
result.fidelity_breakdown  # DataFrame or dict: {qubit: {algo: 0.99, routing: 0.95, coherence: 0.98}}
```

**Effort:** Low — computed from existing grid data.

---

## Low Priority

### Non-Qiskit circuit input

The Rust engine already works on a sparse `(layer, q1, q2, weight)` tensor. Exposing a lower-level entry point that accepts this directly would allow integration with Cirq, Pennylane, or raw OpenQASM without Qiskit.

- Expose `estimate_hardware_fidelity` with a raw numpy tensor interface
- Add `from_qasm(qasm_str)` convenience for OpenQASM 2/3 input

**Effort:** Medium — the Rust side is ready; needs Python convenience wrappers.

### Batch circuit API

Researchers often sweep over parameterized circuits (variational ansatze with different depths/entanglement patterns). A batch API would avoid repeated Python→Rust crossing overhead.

```python
results = map_circuits(circuits, architecture)  # returns list[QusimResult]
```

**Effort:** Low-Medium — loop in Rust instead of Python.

### JSON/CSV export

Results can't be serialized for downstream analysis pipelines. Adding `result.to_dict()`, `result.to_json()`, and `result.to_csv()` methods would enable integration with pandas, experiment tracking tools, etc.

**Effort:** Low — serialization of existing dataclass fields.

### EPR generation schedule

For resource estimation, knowing *when* EPR pairs are consumed (not just the total count) matters for quantum network capacity planning. The per-timeslice teleportation data exists but isn't framed as an EPR schedule.

**Effort:** Low — reshape existing `teleportations_per_slice` data.

---

## Completed

| Feature | Date | Commits |
|---------|------|---------|
| Depolarizing channel model (paper Algorithm 1) | 2026-03-24 | `528a06e` |
| Per-layer multiplicative coherence decay (paper Eq. 41) | 2026-03-24 | `07fee1b`, `56c1c59` |
| Benchmark against paper results | 2026-03-24 | `864f9e3`, `1103f19` |
| Per-qubit T1/T2 support | 2026-03-24 | in `ArchitectureParams` |
| Per-qubit single-gate and per-pair two-gate errors | 2026-04-06 | `7c67491`, `68d501c`, `a7c714a` |
| Gate-type differentiation (native 5D tensors + zero penalty defaults) | 2026-04-08 | Pending Commit |
