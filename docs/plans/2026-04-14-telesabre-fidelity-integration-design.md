# TeleSABRE Fidelity Integration Design

**Date:** 2026-04-14  
**Status:** Approved

## Goal

Return a full `QusimResult` with accurate fidelity fields from TeleSABRE, matching the interface of `map_circuit()`, so both routing algorithms can be compared apples-to-apples.

## Approach: Compact event log in `result_t`

Rather than reimplementing TeleSABRE in Rust or using approximate count-based fidelity, we add a minimal event log to the C library's `result_t`. The Rust side replays the log to reconstruct the `placements` array and `sparse_swaps` list that the existing fidelity engine (`estimate_fidelity`) already expects.

---

## Section 1 — C changes (`alejandrogonzalvo/telesabre` fork)

### New structs in `telesabre.h`

```c
typedef struct {
    int vqubit;     // virtual qubit being teleported
    int from_core;  // source core (before apply)
    int to_core;    // destination core (after apply)
    int gate_idx;   // number of circuit gates applied so far
} ts_teleport_event_t;

typedef struct {
    int vqubit1;   // first virtual qubit in SWAP
    int vqubit2;   // second virtual qubit in SWAP
    int gate_idx;  // number of circuit gates applied so far
} ts_swap_event_t;
```

### Extended `result_t`

```c
typedef struct result {
    int num_teledata;
    int num_telegate;
    int num_swaps;
    int depth;
    int num_deadlocks;
    bool success;

    // Event log (new)
    int *initial_virt_to_core;            // [num_vqubits]
    int  num_vqubits;
    ts_teleport_event_t *teleport_events;
    int  num_teleport_events;
    ts_swap_event_t     *swap_events;
    int  num_swap_events;
} result_t;
```

### Hook point: `telesabre_apply_candidate_op` (`telesabre.c`)

- **Before** `layout_apply_teleport`: resolve source virtual qubit and both cores from the layout; append a `ts_teleport_event_t` with `gate_idx = ts->num_applied_gates`.
- **Before** `layout_apply_swap`: resolve both virtual qubits; append a `ts_swap_event_t`.

### Initialisation & cleanup

- `telesabre_init`: records `initial_virt_to_core` snapshot from the starting layout.
- New `result_free(result_t *r)`: frees the three heap-allocated arrays. Called by the Rust `Drop` path after data is copied out.

**Estimated diff:** ~120 lines across `telesabre.h` and `telesabre.c`.

---

## Section 2 — Rust FFI + reconstruction (`qusim`)

### `src/telesabre/ffi.rs`

Mirror structs with `#[repr(C)]`:

```rust
#[repr(C)] pub struct TeleportEventT {
    pub vqubit: c_int, pub from_core: c_int, pub to_core: c_int, pub gate_idx: c_int,
}
#[repr(C)] pub struct SwapEventT {
    pub vqubit1: c_int, pub vqubit2: c_int, pub gate_idx: c_int,
}
#[repr(C)] pub struct ResultT {
    pub num_teledata: c_int, pub num_telegate: c_int, pub num_swaps: c_int,
    pub depth: c_int, pub num_deadlocks: c_int, pub success: bool,
    pub initial_virt_to_core: *mut c_int, pub num_vqubits: c_int,
    pub teleport_events: *mut TeleportEventT, pub num_teleport_events: c_int,
    pub swap_events: *mut SwapEventT,         pub num_swap_events: c_int,
}
```

New extern declaration: `pub fn result_free(result: *mut ResultT)`.

### `src/telesabre/mod.rs`

`TeleSabreResult` gains:

```rust
pub initial_virt_to_core: Vec<i32>,
pub teleport_events: Vec<(i32, i32, i32, i32)>,  // (vqubit, from_core, to_core, gate_idx)
pub swap_events: Vec<(i32, i32, i32)>,             // (vq1, vq2, gate_idx)
```

### `src/telesabre/reconstruct.rs` (new)

`reconstruct_placements(initial, teleport_events, gs_sparse_edge_list, num_layers, num_qubits) -> Array2<i32>`:

1. Fill column 0 from `initial_virt_to_core`.
2. Build a cumulative gate-count-per-layer map from `gs_sparse` (mirrors `_qiskit_circ_to_sparse_list` counting logic, excluding virtual gates).
3. For each teleport event, convert `gate_idx` → nearest DAG layer via the map.
4. Propagate placement forward: copy previous layer, then update the teleported qubit's core.

`sparse_swaps` is built directly from `swap_events` as `[[gate_idx_as_layer, vq1, vq2], ...]` — the format `estimate_hardware_fidelity` already consumes.

---

## Section 3 — Python API

### `src/python_api.rs`

New `#[pyfunction]` `telesabre_map_and_estimate`:

- **Inputs:** `circuit_path`, `device_path`, `config_path` (strings) + same hardware keyword args as `map_and_estimate`.
- **Flow:** `TeleSabre::new()` → `run()` → `reconstruct_placements()` → `estimate_fidelity()`.
- **Output:** same dict shape as `map_and_estimate` (fidelity grids, counts, etc.). Registered in `rust_core`.

### `python/qusim/__init__.py`

New `telesabre_map_circuit()` function:

```python
def telesabre_map_circuit(
    circuit_path: str | Path,
    device_json_path: str | Path,
    config_json_path: str | Path,
    # same hardware keyword args as map_circuit...
) -> QusimResult:
```

Takes file paths rather than a `QuantumCircuit` object — TeleSABRE reads QASM directly; round-tripping through Qiskit serialization is avoided. Returns a fully populated `QusimResult`.

### `examples/benchmark_telesabre_vs_hqa.py`

- `run_telesabre()` replaced by a call to `telesabre_map_circuit()`.
- `estimate_telesabre_fidelity()` helper removed.
- Both algorithms pull `overall_fidelity` from their respective `QusimResult`. The "estimated" label is dropped from the fidelity panel.

---

## File change summary

| Repo | Files touched |
|------|--------------|
| `alejandrogonzalvo/telesabre` | `src/telesabre.h`, `src/telesabre.c` |
| `qusim` | `csrc/telesabre/` (sync from fork), `src/telesabre/ffi.rs`, `src/telesabre/mod.rs`, `src/telesabre/reconstruct.rs` (new), `src/python_api.rs`, `python/qusim/__init__.py`, `examples/benchmark_telesabre_vs_hqa.py` |

## Out of scope

- Accepting `QuantumCircuit` objects in `telesabre_map_circuit` (requires QASM serialization).
- Per-layer fidelity grids for TeleSABRE (placement reconstruction gives layer-level data but not verified against HQA grid accuracy).
- Porting TeleSABRE to Rust.
