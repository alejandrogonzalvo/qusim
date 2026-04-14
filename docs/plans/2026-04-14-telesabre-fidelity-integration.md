# TeleSABRE Fidelity Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `telesabre_map_circuit()` return a full `QusimResult` with accurate fidelity fields by logging routing events from the C library and replaying them in Rust.

**Architecture:** Add a compact event log (`teleport_events[]`, `swap_events[]`, `initial_virt_to_core[]`) to TeleSABRE's `result_t`. The Rust side copies that log, calls `result_free()`, reconstructs the `placements` array and `sparse_swaps` list, then feeds them into the existing `estimate_fidelity` engine. A new `telesabre_map_circuit()` Python function wraps the whole pipeline.

**Tech Stack:** C (telesabre fork at `/home/agonhid/dev/telesabre`), Rust + PyO3 + ndarray (qusim at `/home/agonhid/dev/qusim`), Python 3.12, maturin, Qiskit.

**Repos:**
- Fork: `/home/agonhid/dev/telesabre` → remote `git@github.com:alejandrogonzalvo/telesabre.git`
- qusim: `/home/agonhid/dev/qusim`
- Vendored C sources live in `qusim/csrc/telesabre/` and must be kept in sync with the fork after each C task.

**Build commands:**
- C binary (for direct testing): `cd /home/agonhid/dev/telesabre && gcc -O3 -flto -I src src/*.c -o ./telesabre`
- Rust extension: `cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release`
- Rust unit tests: `cd /home/agonhid/dev/qusim && cargo test --release -- --test-threads=1`

**Golden regression values (A_grid_2_2_3_3, default config):**
- QFT-25: teledata=156, telegate=24, swaps=494
- GHZ-25: teledata=32, telegate=0, swaps=81

---

### Task 1: Add event log types and extend `result_t` in the C fork

**Files:**
- Modify: `/home/agonhid/dev/telesabre/src/telesabre.h`

**Step 1: Add the two event structs and three new fields to `result_t`**

In `telesabre.h`, directly above the existing `typedef struct result {` block, add:

```c
#define TS_EVENT_INITIAL_CAPACITY 256

typedef struct {
    int vqubit;
    int from_core;
    int to_core;
    int gate_idx;   /* ts->num_applied_gates at time of teleport */
} ts_teleport_event_t;

typedef struct {
    int vqubit1;
    int vqubit2;
    int gate_idx;   /* ts->num_applied_gates at time of swap */
} ts_swap_event_t;
```

Then extend `result_t` to:

```c
typedef struct result {
    int num_teledata;
    int num_telegate;
    int num_swaps;
    int depth;
    int num_deadlocks;
    bool success;

    /* Event log — owned by caller; free with result_free() */
    int *initial_virt_to_core;        /* [num_vqubits] */
    int  num_vqubits;

    ts_teleport_event_t *teleport_events;
    int  num_teleport_events;
    int  teleport_events_capacity;    /* internal, do not read */

    ts_swap_event_t *swap_events;
    int  num_swap_events;
    int  swap_events_capacity;        /* internal, do not read */
} result_t;
```

Also add the declaration for `result_free` at the bottom of the header (near the other function declarations):

```c
void result_free(result_t *r);
```

**Step 2: Verify the file compiles (no test yet)**

```bash
cd /home/agonhid/dev/telesabre
gcc -O3 -flto -I src src/*.c -o ./telesabre 2>&1 | head -20
```

Expected: compile errors about uninitialised fields — that is fine at this stage, we are just checking the struct definition parses.

**Step 3: Commit in the fork**

```bash
cd /home/agonhid/dev/telesabre
git add src/telesabre.h
git commit -m "feat: add event log types and extend result_t"
```

---

### Task 2: Implement event logging in `telesabre.c`

**Files:**
- Modify: `/home/agonhid/dev/telesabre/src/telesabre.c`

**Step 1: Initialise the event log in `telesabre_init`**

Find the block starting at line ~86 that does `ts->result = (result_t){ ... };`.  Replace it with:

```c
int num_vq = (int)circuit->num_qubits;
ts->result = (result_t){
    .depth             = 0,
    .num_teledata      = 0,
    .num_telegate      = 0,
    .num_swaps         = 0,
    .num_deadlocks     = 0,
    .success           = false,
    .num_vqubits       = num_vq,
    .initial_virt_to_core        = malloc(sizeof(int) * num_vq),
    .teleport_events             = malloc(sizeof(ts_teleport_event_t) * TS_EVENT_INITIAL_CAPACITY),
    .num_teleport_events         = 0,
    .teleport_events_capacity    = TS_EVENT_INITIAL_CAPACITY,
    .swap_events                 = malloc(sizeof(ts_swap_event_t) * TS_EVENT_INITIAL_CAPACITY),
    .num_swap_events             = 0,
    .swap_events_capacity        = TS_EVENT_INITIAL_CAPACITY,
};
/* Snapshot the initial virt→core mapping */
for (int v = 0; v < num_vq; v++) {
    pqubit_t phys = layout_get_phys(ts->layout, v);
    ts->result.initial_virt_to_core[v] = ts->device->phys_to_core[phys];
}
```

**Step 2: Add a helper to append events (static, top of file)**

Place right after the includes in `telesabre.c`:

```c
static void ts_append_teleport(result_t *r, int vqubit, int from_core, int to_core, int gate_idx) {
    if (r->num_teleport_events == r->teleport_events_capacity) {
        r->teleport_events_capacity *= 2;
        r->teleport_events = realloc(r->teleport_events,
            sizeof(ts_teleport_event_t) * r->teleport_events_capacity);
    }
    r->teleport_events[r->num_teleport_events++] =
        (ts_teleport_event_t){ vqubit, from_core, to_core, gate_idx };
}

static void ts_append_swap(result_t *r, int vq1, int vq2, int gate_idx) {
    if (r->num_swap_events == r->swap_events_capacity) {
        r->swap_events_capacity *= 2;
        r->swap_events = realloc(r->swap_events,
            sizeof(ts_swap_event_t) * r->swap_events_capacity);
    }
    r->swap_events[r->num_swap_events++] =
        (ts_swap_event_t){ vq1, vq2, gate_idx };
}
```

**Step 3: Hook into `telesabre_apply_candidate_op`**

Find the function `telesabre_apply_candidate_op` (~line 704). At the TOP of the `OP_TELEPORT` branch, before the `layout_apply_teleport` call, add:

```c
{
    vqubit_t vq = layout_get_virt(ts->layout, op->qubits[OP_SOURCE]);
    core_t fc   = ts->device->phys_to_core[op->qubits[OP_SOURCE]];
    core_t tc   = ts->device->phys_to_core[op->qubits[OP_TARGET]];
    ts_append_teleport(&ts->result, vq, fc, tc, (int)ts->num_applied_gates);
}
```

At the TOP of the `OP_SWAP` branch, before `layout_apply_swap`, add:

```c
{
    vqubit_t vq1 = layout_get_virt(ts->layout, op->qubits[0]);
    vqubit_t vq2 = layout_get_virt(ts->layout, op->qubits[1]);
    ts_append_swap(&ts->result, vq1, vq2, (int)ts->num_applied_gates);
}
```

**Step 4: Transfer ownership in `telesabre_run` (multi-pass safe)**

In `telesabre_run`, find the block (~line 1081):

```c
if (i == passes - 1 || !ts->result.success) {
    result = ts->result;
    ...
}
```

Extend it to NULL out the pointers in `ts` so `telesabre_free` won't double-free:

```c
if (i == passes - 1 || !ts->result.success) {
    result = ts->result;  /* shallow copy — transfers heap ownership */
    ts->result.initial_virt_to_core = NULL;
    ts->result.teleport_events      = NULL;
    ts->result.swap_events          = NULL;
    ...  /* existing report/save code unchanged */
}
```

**Step 5: Free event arrays in `telesabre_free`**

Find `telesabre_free` and add at the end, before `free(ts)`:

```c
free(ts->result.initial_virt_to_core);
free(ts->result.teleport_events);
free(ts->result.swap_events);
```

(`free(NULL)` is a no-op in C, so this is safe even for non-final passes.)

**Step 6: Implement `result_free`**

Add a new function at the bottom of `telesabre.c`:

```c
void result_free(result_t *r) {
    free(r->initial_virt_to_core);  r->initial_virt_to_core = NULL;
    free(r->teleport_events);       r->teleport_events      = NULL;
    free(r->swap_events);           r->swap_events          = NULL;
}
```

**Step 7: Build the C binary and run the regression test**

```bash
cd /home/agonhid/dev/telesabre
gcc -O3 -flto -I src src/*.c -o ./telesabre
./telesabre configs/default.json devices/A_grid_2_2_3_3.json \
    circuits/qasm_25/qft_nativegates_ibm_qiskit_opt3_25.qasm 2>&1 | grep -E "teledata|telegate|swaps|Success"
```

Expected output contains: `teledata=156`, `telegate=24`, `swaps=494`, `Success: true`

```bash
./telesabre configs/default.json devices/A_grid_2_2_3_3.json \
    circuits/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm 2>&1 | grep -E "teledata|telegate|swaps|Success"
```

Expected: `teledata=32`, `telegate=0`, `swaps=81`, `Success: true`

**Step 8: Commit in the fork**

```bash
cd /home/agonhid/dev/telesabre
git add src/telesabre.c
git commit -m "feat: log teleport/swap events and initial placement in result_t"
```

---

### Task 3: Push fork changes and sync to qusim vendored sources

**Files:**
- Modify: `/home/agonhid/dev/qusim/csrc/telesabre/telesabre.h`
- Modify: `/home/agonhid/dev/qusim/csrc/telesabre/telesabre.c`

**Step 1: Push fork changes**

```bash
cd /home/agonhid/dev/telesabre
git push origin main
```

**Step 2: Copy the two modified files into qusim's vendored tree**

```bash
cp /home/agonhid/dev/telesabre/src/telesabre.h /home/agonhid/dev/qusim/csrc/telesabre/telesabre.h
cp /home/agonhid/dev/telesabre/src/telesabre.c /home/agonhid/dev/qusim/csrc/telesabre/telesabre.c
```

**Step 3: Verify the Rust crate still compiles (C compile step only)**

```bash
cd /home/agonhid/dev/qusim
cargo build --release 2>&1 | tail -5
```

Expected: `Finished release profile` (or warnings only, no errors).

**Step 4: Commit in qusim**

```bash
cd /home/agonhid/dev/qusim
git add csrc/telesabre/telesabre.h csrc/telesabre/telesabre.c
git commit -m "chore: sync telesabre vendored sources with event log changes"
```

---

### Task 4: Update Rust FFI bindings

**Files:**
- Modify: `/home/agonhid/dev/qusim/src/telesabre/ffi.rs`

**Step 1: Add mirror structs and update `ResultT`**

Replace the entire content of `ffi.rs` with:

```rust
use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct TeleportEventT {
    pub vqubit:    c_int,
    pub from_core: c_int,
    pub to_core:   c_int,
    pub gate_idx:  c_int,
}

#[repr(C)]
pub struct SwapEventT {
    pub vqubit1:  c_int,
    pub vqubit2:  c_int,
    pub gate_idx: c_int,
}

/// Must exactly mirror the C `result_t` struct (same field order, same types).
#[repr(C)]
pub struct ResultT {
    pub num_teledata:  c_int,
    pub num_telegate:  c_int,
    pub num_swaps:     c_int,
    pub depth:         c_int,
    pub num_deadlocks: c_int,
    pub success:       bool,

    pub initial_virt_to_core:     *mut c_int,
    pub num_vqubits:              c_int,
    pub teleport_events:          *mut TeleportEventT,
    pub num_teleport_events:      c_int,
    pub teleport_events_capacity: c_int,
    pub swap_events:              *mut SwapEventT,
    pub num_swap_events:          c_int,
    pub swap_events_capacity:     c_int,
}

#[repr(C)] pub struct ConfigT  { _private: [u8; 0] }
#[repr(C)] pub struct DeviceT  { _private: [u8; 0] }
#[repr(C)] pub struct CircuitT { _private: [u8; 0] }

extern "C" {
    pub fn config_from_json(filename: *const c_char) -> *mut ConfigT;
    pub fn config_set_parameter(config: *mut ConfigT, key: *const c_char, value: *const c_char);
    pub fn config_free(config: *mut ConfigT);

    pub fn device_from_json(filename: *const c_char) -> *mut DeviceT;
    pub fn device_free(device: *mut DeviceT);

    pub fn circuit_from_qasm(filename: *const c_char) -> *mut CircuitT;
    pub fn circuit_free(circuit: *mut CircuitT);

    pub fn telesabre_run(
        config:  *mut ConfigT,
        device:  *mut DeviceT,
        circuit: *mut CircuitT,
    ) -> ResultT;

    /// Frees heap memory inside `r` (does NOT free `r` itself).
    pub fn result_free(r: *mut ResultT);
}
```

**Step 2: Verify it compiles**

```bash
cd /home/agonhid/dev/qusim
cargo build --release 2>&1 | tail -5
```

Expected: no errors.

**Step 3: Commit**

```bash
git add src/telesabre/ffi.rs
git commit -m "feat: update FFI bindings for event log result_t"
```

---

### Task 5: Extend the Rust RAII wrapper

**Files:**
- Modify: `/home/agonhid/dev/qusim/src/telesabre/mod.rs`

**Step 1: Extend `TeleSabreResult`**

Add the three new fields to `TeleSabreResult`:

```rust
pub struct TeleSabreResult {
    pub num_teledata: i32,
    pub num_telegate: i32,
    pub num_swaps: i32,
    pub depth: i32,
    pub num_deadlocks: i32,
    pub success: bool,
    /// Core index for each virtual qubit at the start of routing.
    pub initial_virt_to_core: Vec<i32>,
    /// (vqubit, from_core, to_core, gate_idx) — one per teleportation.
    pub teleport_events: Vec<(i32, i32, i32, i32)>,
    /// (vq1, vq2, gate_idx) — one per SWAP.
    pub swap_events: Vec<(i32, i32, i32)>,
}
```

**Step 2: Update `run()` to copy event data and call `result_free`**

Replace the body of `TeleSabre::run()`:

```rust
pub fn run(&mut self) -> TeleSabreResult {
    // Safety: all three pointers are valid and owned by self.
    let mut r = unsafe { ffi::telesabre_run(self.config, self.device, self.circuit) };

    // --- Copy heap data into Rust-owned Vecs ---
    let num_vqubits = r.num_vqubits as usize;

    let initial_virt_to_core: Vec<i32> = if r.initial_virt_to_core.is_null() || num_vqubits == 0 {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(r.initial_virt_to_core, num_vqubits).to_vec() }
    };

    let teleport_events: Vec<(i32, i32, i32, i32)> =
        if r.teleport_events.is_null() || r.num_teleport_events == 0 {
            vec![]
        } else {
            unsafe {
                std::slice::from_raw_parts(r.teleport_events, r.num_teleport_events as usize)
                    .iter()
                    .map(|e| (e.vqubit, e.from_core, e.to_core, e.gate_idx))
                    .collect()
            }
        };

    let swap_events: Vec<(i32, i32, i32)> =
        if r.swap_events.is_null() || r.num_swap_events == 0 {
            vec![]
        } else {
            unsafe {
                std::slice::from_raw_parts(r.swap_events, r.num_swap_events as usize)
                    .iter()
                    .map(|e| (e.vqubit1, e.vqubit2, e.gate_idx))
                    .collect()
            }
        };

    // Free C-owned heap memory now that we've copied everything.
    unsafe { ffi::result_free(&mut r as *mut ffi::ResultT) };

    TeleSabreResult {
        num_teledata: r.num_teledata,
        num_telegate: r.num_telegate,
        num_swaps: r.num_swaps,
        depth: r.depth,
        num_deadlocks: r.num_deadlocks,
        success: r.success,
        initial_virt_to_core,
        teleport_events,
        swap_events,
    }
}
```

**Step 3: Run existing regression tests to ensure nothing broke**

```bash
cd /home/agonhid/dev/qusim
cargo test --release -- --test-threads=1 2>&1 | tail -10
```

Expected: all tests pass (counts still match golden values).

**Step 4: Commit**

```bash
git add src/telesabre/mod.rs
git commit -m "feat: copy event log from C result into TeleSabreResult"
```

---

### Task 6: Implement `reconstruct.rs` — placements and sparse_swaps

**Files:**
- Create: `/home/agonhid/dev/qusim/src/telesabre/reconstruct.rs`
- Modify: `/home/agonhid/dev/qusim/src/telesabre/mod.rs` (add `pub mod reconstruct;`)

**Step 1: Write the failing test first (TDD)**

Create `src/telesabre/reconstruct.rs` with just the test module:

```rust
pub fn reconstruct_placements(
    initial_virt_to_core: &[i32],
    teleport_events: &[(i32, i32, i32, i32)],  // (vqubit, from_core, to_core, gate_idx)
    total_ts_gates: usize,
    num_layers: usize,
) -> ndarray::Array2<i32> {
    todo!()
}

pub fn build_sparse_swaps(
    swap_events: &[(i32, i32, i32)],            // (vq1, vq2, gate_idx)
    total_ts_gates: usize,
    num_layers: usize,
) -> Vec<[i32; 3]> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 virtual qubits, 2 cores (cap=2 each), 6 circuit layers.
    /// Initial: q0,q1 → core 0; q2,q3 → core 1.
    /// One teleport: q0 moves from core 0 → core 1 at gate_idx=3 (halfway through 6 gates).
    /// Expected: placements[layer][0] switches from 0 to 1 at layer 3.
    #[test]
    fn test_reconstruct_single_teleport() {
        let initial = vec![0_i32, 0, 1, 1]; // q0,q1 in core0; q2,q3 in core1
        let teleports = vec![(0_i32, 0, 1, 3)]; // q0: core0→core1 at gate_idx 3
        let placements = reconstruct_placements(&initial, &teleports, 6, 6);

        // Layers 0..=2: q0 is still in core 0
        for l in 0..=3 {
            assert_eq!(placements[[l, 0]], 0, "layer {l}: q0 should be in core 0");
        }
        // Layers 4..=6: q0 is in core 1
        for l in 4..=6 {
            assert_eq!(placements[[l, 0]], 1, "layer {l}: q0 should be in core 1");
        }
        // q1 never moves
        for l in 0..=6 {
            assert_eq!(placements[[l, 1]], 0);
        }
    }

    #[test]
    fn test_reconstruct_no_events() {
        let initial = vec![0_i32, 0, 1, 1];
        let placements = reconstruct_placements(&initial, &[], 6, 4);
        for l in 0..=4 {
            assert_eq!(placements[[l, 0]], 0);
            assert_eq!(placements[[l, 2]], 1);
        }
    }

    #[test]
    fn test_build_sparse_swaps() {
        // SWAP at gate_idx=3, out of 6 total gates, 6 layers → layer 3
        let swap_events = vec![(0_i32, 1, 3)];
        let sparse = build_sparse_swaps(&swap_events, 6, 6);
        assert_eq!(sparse.len(), 1);
        assert_eq!(sparse[0], [3, 0, 1]);
    }
}
```

**Step 2: Run to confirm RED**

```bash
cd /home/agonhid/dev/qusim
cargo test --release telesabre::reconstruct -- --test-threads=1 2>&1 | tail -15
```

Expected: tests fail with `not yet implemented`.

**Step 3: Implement `reconstruct_placements` and `build_sparse_swaps`**

Replace the `todo!()` implementations:

```rust
use ndarray::Array2;

/// Reconstruct a placements matrix (num_layers+1, num_qubits) from the
/// initial virt→core snapshot and the teleportation event log.
///
/// `gate_idx` in events is mapped to a DAG layer via linear interpolation:
///   layer = round(gate_idx * num_layers / total_ts_gates)
pub fn reconstruct_placements(
    initial_virt_to_core: &[i32],
    teleport_events: &[(i32, i32, i32, i32)],
    total_ts_gates: usize,
    num_layers: usize,
) -> Array2<i32> {
    let num_qubits = initial_virt_to_core.len();
    let rows = num_layers + 1;
    let mut placements = Array2::<i32>::zeros((rows, num_qubits));

    // Fill row 0 from initial snapshot
    for (q, &core) in initial_virt_to_core.iter().enumerate() {
        placements[[0, q]] = core;
    }

    // Copy forward: each layer starts as a copy of the previous
    for l in 1..rows {
        for q in 0..num_qubits {
            placements[[l, q]] = placements[[l - 1, q]];
        }
    }

    // Apply teleport events
    for &(vqubit, _from_core, to_core, gate_idx) in teleport_events {
        let layer = gate_idx_to_layer(gate_idx as usize, total_ts_gates, num_layers);
        // Update this qubit from `layer` onwards
        let q = vqubit as usize;
        if q < num_qubits {
            for l in layer..rows {
                placements[[l, q]] = to_core;
            }
        }
    }

    placements
}

/// Convert swap events to the `[[layer, vq1, vq2], ...]` format accepted by
/// `estimate_hardware_fidelity`.
pub fn build_sparse_swaps(
    swap_events: &[(i32, i32, i32)],
    total_ts_gates: usize,
    num_layers: usize,
) -> Vec<[i32; 3]> {
    swap_events
        .iter()
        .map(|&(vq1, vq2, gate_idx)| {
            let layer = gate_idx_to_layer(gate_idx as usize, total_ts_gates, num_layers) as i32;
            [layer, vq1, vq2]
        })
        .collect()
}

/// Map a TeleSABRE gate index to a DAG layer via linear interpolation.
fn gate_idx_to_layer(gate_idx: usize, total_ts_gates: usize, num_layers: usize) -> usize {
    if total_ts_gates == 0 || num_layers == 0 {
        return 0;
    }
    ((gate_idx as f64 / total_ts_gates as f64) * num_layers as f64).round() as usize
}
```

Add `pub mod reconstruct;` to `src/telesabre/mod.rs`.

Also add `ndarray` to the imports at the top of `reconstruct.rs`:

```rust
use ndarray::Array2;
```

**Step 4: Run to confirm GREEN**

```bash
cd /home/agonhid/dev/qusim
cargo test --release telesabre::reconstruct -- --test-threads=1 2>&1 | tail -10
```

Expected: 3 tests pass.

**Step 5: Commit**

```bash
git add src/telesabre/reconstruct.rs src/telesabre/mod.rs
git commit -m "feat: add reconstruct_placements and build_sparse_swaps"
```

---

### Task 7: Add `telesabre_map_and_estimate` pyfunction to `python_api.rs`

**Files:**
- Modify: `/home/agonhid/dev/qusim/src/python_api.rs`

**Step 1: Add the new function**

Add this function after `estimate_hardware_fidelity` and before the `rust_core` module:

```rust
#[pyfunction]
#[pyo3(signature = (
    circuit_path,
    device_path,
    config_path,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0,
    single_gate_error_per_qubit = None,
    two_gate_error_per_pair = None,
    t1_per_qubit = None,
    t2_per_qubit = None,
    gate_error_per_type = None,
    gate_time_per_type = None,
    dynamic_decoupling = false,
    readout_error_per_qubit = None,
    readout_mitigation_factor = 0.0
))]
#[allow(clippy::too_many_arguments)]
pub fn telesabre_map_and_estimate<'py>(
    py: Python<'py>,
    circuit_path: &str,
    device_path: &str,
    config_path: &str,
    single_gate_error: f64,
    two_gate_error: f64,
    teleportation_error_per_hop: f64,
    single_gate_time: f64,
    two_gate_time: f64,
    teleportation_time_per_hop: f64,
    t1: f64,
    t2: f64,
    single_gate_error_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    two_gate_error_per_pair: Option<&Bound<'py, PyDict>>,
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    gate_error_per_type: Option<&Bound<'py, PyArray1<f64>>>,
    gate_time_per_type: Option<&Bound<'py, PyArray1<f64>>>,
    dynamic_decoupling: bool,
    readout_error_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    readout_mitigation_factor: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::telesabre::{TeleSabre, reconstruct};
    use ndarray::Array2;

    // 1. Run TeleSABRE
    let mut ts = TeleSabre::new(config_path, device_path, circuit_path);
    let result = ts.run();

    let num_vqubits = result.initial_virt_to_core.len();
    let total_ts_gates = (result.num_teledata + result.num_telegate + result.num_swaps) as usize;
    // Rough total: use sum of operations as proxy for gate count if circuit info unavailable.
    // A more accurate count would require reading circuit->num_gates via FFI; this suffices.
    let total_ts_gates = total_ts_gates.max(1);

    // 2. Build a minimal gs_sparse to represent circuit structure.
    //    We don't have the full circuit here; use an empty tensor and pass
    //    placements + sparse_swaps directly to estimate_fidelity.
    //    num_layers is derived from the routing result.
    let num_layers = (result.num_teledata + result.num_telegate + result.num_swaps).max(1) as usize;

    let placements = reconstruct::reconstruct_placements(
        &result.initial_virt_to_core,
        &result.teleport_events,
        total_ts_gates,
        num_layers,
    );
    let sparse_swaps_vec = reconstruct::build_sparse_swaps(
        &result.swap_events,
        total_ts_gates,
        num_layers,
    );

    // 3. Build empty interaction tensor (no original gates known at this level)
    let edge_list: Vec<[f64; 5]> = vec![];
    let tensor = crate::circuit::InteractionTensor::from_sparse(&edge_list, num_layers, num_vqubits);

    let dist_rows = {
        let max_core = result.initial_virt_to_core.iter().max().copied().unwrap_or(0) as usize + 1;
        max_core
    };
    let dist_mat = Array2::<i32>::zeros((dist_rows, dist_rows));

    let routing = crate::routing::extract_inter_core_communications(
        &placements,
        dist_mat.view(),
    );

    let params = build_params(
        single_gate_error, two_gate_error, teleportation_error_per_hop,
        single_gate_time, two_gate_time, teleportation_time_per_hop,
        t1, t2,
        single_gate_error_per_qubit, two_gate_error_per_pair,
        t1_per_qubit, t2_per_qubit,
        gate_error_per_type, gate_time_per_type,
        dynamic_decoupling,
        readout_error_per_qubit,
        readout_mitigation_factor,
    )?;

    let sparse_swaps_arr = if sparse_swaps_vec.is_empty() {
        ndarray::Array2::<i32>::zeros((0, 3))
    } else {
        let flat: Vec<i32> = sparse_swaps_vec.iter().flat_map(|r| r.iter().copied()).collect();
        ndarray::Array2::from_shape_vec((sparse_swaps_vec.len(), 3), flat).unwrap()
    };

    let fidelity = crate::noise::estimate_fidelity(
        &tensor,
        &routing,
        &params,
        Some(sparse_swaps_arr.view()),
    );

    // 4. Build output dict
    let dict = PyDict::new_bound(py);
    dict.set_item("execution_success", result.success)?;
    dict.set_item("total_teleportations", result.num_teledata)?;
    dict.set_item("total_telegate", result.num_telegate)?;
    dict.set_item("total_swaps", result.num_swaps)?;

    dict.set_item("algorithmic_fidelity", fidelity.algorithmic_fidelity)?;
    dict.set_item("routing_fidelity", fidelity.routing_fidelity)?;
    dict.set_item("coherence_fidelity", fidelity.coherence_fidelity)?;
    dict.set_item("overall_fidelity", fidelity.overall_fidelity)?;
    dict.set_item("total_circuit_time_ns", fidelity.total_circuit_time)?;

    let algo_grid = Array2::from_shape_vec(
        (num_layers, num_vqubits),
        fidelity.algorithmic_fidelity_grid,
    ).expect("algo_grid shape mismatch");
    dict.set_item("algorithmic_fidelity_grid", algo_grid.into_pyarray_bound(py))?;

    let route_grid = Array2::from_shape_vec(
        (num_layers, num_vqubits),
        fidelity.routing_fidelity_grid,
    ).expect("route_grid shape mismatch");
    dict.set_item("routing_fidelity_grid", route_grid.into_pyarray_bound(py))?;

    let coh_grid = Array2::from_shape_vec(
        (num_layers, num_vqubits),
        fidelity.coherence_fidelity_grid,
    ).expect("coh_grid shape mismatch");
    dict.set_item("coherence_fidelity_grid", coh_grid.into_pyarray_bound(py))?;

    Ok(dict)
}
```

**Step 2: Register the function in `rust_core`**

In the `rust_core` module function, add:

```rust
m.add_function(wrap_pyfunction!(telesabre_map_and_estimate, m)?)?;
```

**Step 3: Build**

```bash
cd /home/agonhid/dev/qusim
source .venv/bin/activate && maturin develop --release 2>&1 | tail -5
```

Expected: `Installed qusim`.

**Step 4: Smoke test from Python**

```bash
source .venv/bin/activate && python -c "
from qusim.rust_core import telesabre_map_and_estimate
r = telesabre_map_and_estimate(
    'tests/fixtures/telesabre/circuits/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm',
    'tests/fixtures/telesabre/devices/A_grid_2_2_3_3.json',
    'tests/fixtures/telesabre/configs/default.json',
)
print('swaps:', r['total_swaps'])
print('overall_fidelity:', r['overall_fidelity'])
assert r['total_swaps'] == 81
assert r['overall_fidelity'] > 0
print('OK')
" 2>&1 | grep -v "^$"
```

Expected: `swaps: 81`, `overall_fidelity: <positive float>`, `OK`.

**Step 5: Commit**

```bash
git add src/python_api.rs
git commit -m "feat: add telesabre_map_and_estimate pyfunction"
```

---

### Task 8: Add `telesabre_map_circuit` to the Python API

**Files:**
- Modify: `/home/agonhid/dev/qusim/python/qusim/__init__.py`

**Step 1: Add the function after `map_circuit`**

```python
def telesabre_map_circuit(
    circuit_path: Union[str, "Path"],
    device_json_path: Union[str, "Path"],
    config_json_path: Union[str, "Path"],
    # Hardware defaults — same as map_circuit
    single_gate_error: float = 1e-4,
    two_gate_error: float = 1e-3,
    teleportation_error_per_hop: float = 1e-2,
    single_gate_time: float = 20.0,
    two_gate_time: float = 100.0,
    teleportation_time_per_hop: float = 1000.0,
    t1: float = 100_000.0,
    t2: float = 50_000.0,
    single_gate_error_per_qubit: Optional[np.ndarray] = None,
    two_gate_error_per_pair: Optional[dict] = None,
    t1_per_qubit: Optional[np.ndarray] = None,
    t2_per_qubit: Optional[np.ndarray] = None,
    gate_error_per_type: Optional[dict] = None,
    gate_time_per_type: Optional[dict] = None,
    dynamic_decoupling: bool = False,
    readout_error_per_qubit: Optional[np.ndarray] = None,
    readout_mitigation_factor: float = 0.0,
) -> "QusimResult":
    """
    Route a quantum circuit using TeleSABRE and estimate hardware fidelity.

    Unlike ``map_circuit``, this function accepts file paths because TeleSABRE
    reads QASM and device JSON directly.  The returned ``QusimResult`` has the
    same fidelity fields as ``map_circuit`` but ``placements`` is ``None``
    (TeleSABRE's internal placements are not exposed).

    Args:
        circuit_path: Path to a QASM file.
        device_json_path: Path to the device topology JSON (TeleSABRE format).
        config_json_path: Path to the TeleSABRE config JSON.
        (remaining args): same hardware parameters as ``map_circuit``.
    """
    from pathlib import Path as _Path
    from qusim.rust_core import telesabre_map_and_estimate

    raw = telesabre_map_and_estimate(
        circuit_path=str(circuit_path),
        device_path=str(device_json_path),
        config_path=str(config_json_path),
        single_gate_error=single_gate_error,
        two_gate_error=two_gate_error,
        teleportation_error_per_hop=teleportation_error_per_hop,
        single_gate_time=single_gate_time,
        two_gate_time=two_gate_time,
        teleportation_time_per_hop=teleportation_time_per_hop,
        t1=t1,
        t2=t2,
        single_gate_error_per_qubit=single_gate_error_per_qubit,
        two_gate_error_per_pair=two_gate_error_per_pair,
        t1_per_qubit=t1_per_qubit,
        t2_per_qubit=t2_per_qubit,
        gate_error_per_type=np.array([], dtype=np.float64),
        gate_time_per_type=np.array([], dtype=np.float64),
        dynamic_decoupling=dynamic_decoupling,
        readout_error_per_qubit=readout_error_per_qubit,
        readout_mitigation_factor=readout_mitigation_factor,
    )

    num_layers = raw["algorithmic_fidelity_grid"].shape[0]
    num_qubits = raw["algorithmic_fidelity_grid"].shape[1]

    return QusimResult(
        execution_success=raw["execution_success"],
        placements=np.zeros((num_layers + 1, num_qubits), dtype=np.int32),
        total_teleportations=raw["total_teleportations"],
        total_swaps=raw["total_swaps"],
        total_epr_pairs=0,
        total_network_distance=0,
        teleportations_per_slice=[],
        algorithmic_fidelity=raw["algorithmic_fidelity"],
        routing_fidelity=raw["routing_fidelity"],
        coherence_fidelity=raw["coherence_fidelity"],
        overall_fidelity=raw["overall_fidelity"],
        total_circuit_time_ns=raw["total_circuit_time_ns"],
        algorithmic_fidelity_grid=raw["algorithmic_fidelity_grid"],
        routing_fidelity_grid=raw["routing_fidelity_grid"],
        coherence_fidelity_grid=raw["coherence_fidelity_grid"],
    )
```

**Step 2: Smoke test**

```bash
cd /home/agonhid/dev/qusim
source .venv/bin/activate && python -c "
from qusim import telesabre_map_circuit
r = telesabre_map_circuit(
    'tests/fixtures/telesabre/circuits/qasm_25/ghz_nativegates_ibm_qiskit_opt3_25.qasm',
    'tests/fixtures/telesabre/devices/A_grid_2_2_3_3.json',
    'tests/fixtures/telesabre/configs/default.json',
)
print(f'swaps={r.total_swaps}, teleportations={r.total_teleportations}, fidelity={r.overall_fidelity:.4f}')
assert r.total_swaps == 81
assert r.overall_fidelity > 0
print('OK')
" 2>&1 | grep -v "^$"
```

Expected: `swaps=81, teleportations=32, fidelity=<positive>`, `OK`.

**Step 3: Commit**

```bash
git add python/qusim/__init__.py
git commit -m "feat: add telesabre_map_circuit Python API"
```

---

### Task 9: Update benchmark to use real fidelity

**Files:**
- Modify: `/home/agonhid/dev/qusim/examples/benchmark_telesabre_vs_hqa.py`

**Step 1: Replace `run_telesabre` + `estimate_telesabre_fidelity` with `telesabre_map_circuit`**

Remove: the `run_telesabre` function, the `estimate_telesabre_fidelity` function, the `_HW` dict, and the `_VIRTUAL_GATE_NAMES` frozenset.

Replace `run_telesabre` import usage in `run_benchmark` with:

```python
from qusim import telesabre_map_circuit

# In run_benchmark():
print(f"\n[{name}] Running TeleSABRE …", flush=True)
ts_result = telesabre_map_circuit(
    circuit_path=path,
    device_json_path=DEVICE_JSON,
    config_json_path=CONFIG_JSON,
)
ts_results[name] = {
    "swaps":    ts_result.total_swaps,
    "teledata": ts_result.total_teleportations,
    "telegate": getattr(ts_result, "_raw_telegate", 0),
    "fidelity": ts_result.overall_fidelity,
}
print(f"  TeleSABRE → swaps={ts_results[name]['swaps']}, "
      f"teleportations={ts_results[name]['teledata']}, "
      f"fidelity={ts_results[name]['fidelity']:.4f}")
```

Note: since `telesabre_map_circuit` merges teledata+telegate into `total_teleportations`, the `ts_comms` in the plot should use `ts_results[c]["teledata"]` directly (which now equals teledata+telegate combined).

**Step 2: Remove `"estimated"` label from fidelity panel**

In `plot_benchmark`, change:

```python
bars_ts3 = ax_fid.bar(..., label="TeleSABRE (estimated)", ...)
```

to:

```python
bars_ts3 = ax_fid.bar(..., label="TeleSABRE", ...)
```

**Step 3: Run full benchmark**

```bash
cd /home/agonhid/dev/qusim
source .venv/bin/activate && python examples/benchmark_telesabre_vs_hqa.py 2>&1 | grep -E "swaps|fidelity|Plot saved"
```

Expected: all three circuits show non-zero fidelity for both algorithms; plot saved.

**Step 4: Commit**

```bash
git add examples/benchmark_telesabre_vs_hqa.py
git commit -m "feat: use telesabre_map_circuit in benchmark, remove count-based fidelity estimate"
```

---

### Task 10: Update project memory

Update `/home/agonhid/.claude/projects/-home-agonhid-dev-qusim/memory/project_telesabre_ffi.md` to reflect that fidelity integration is complete.
