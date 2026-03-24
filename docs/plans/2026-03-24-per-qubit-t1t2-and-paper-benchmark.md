# Per-Qubit T1/T2 & Paper Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-qubit T1/T2 support, fix the decoherence formula, and benchmark qusim against the paper's results.

**Architecture:** Optional `Vec<f64>` fields on `ArchitectureParams` for per-qubit T1/T2, threaded through Rust→PyO3→Python. A benchmark script loads the paper's QASM circuits and compares qusim output against stored fidelity data.

**Tech Stack:** Rust (ndarray, pyo3, numpy crate), Python (qiskit, matplotlib, numpy), maturin build system.

**Build commands:**
- Rust tests: `cargo test`
- Build Python extension: `cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release`
- Python tests: `cd /home/agonhid/dev/qusim && source .venv/bin/activate && python -m pytest dse_pau/test_benchmarks.py -v`

**Design doc:** `docs/plans/2026-03-24-per-qubit-t1t2-and-paper-benchmark-design.md`

---

### Task 1: Fix decoherence formula in noise/mod.rs

**Files:**
- Modify: `src/noise/mod.rs:84-88`

**Step 1: Fix the formula**

In `src/noise/mod.rs`, change the `decoherence_fidelity` function from:

```rust
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (-idle_time / t2).exp()
}
```

to:

```rust
/// Idle-qubit decoherence from T1 relaxation and T2 dephasing (paper Eq. 41).
#[inline]
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (0.5 * (-idle_time / t2).exp() + 0.5)
}
```

**Step 2: Run Rust tests**

Run: `cargo test`
Expected: All existing tests pass. The formula change makes coherence higher (less aggressive), so all `> 0` and `<= 1` assertions still hold. The `zero_error_gives_perfect_operational_fidelity` test uses `t1: f64::INFINITY, t2: f64::INFINITY` which gives `1.0 * (0.5 * 1.0 + 0.5) = 1.0` — still passes.

**Step 3: Commit**

```bash
git add src/noise/mod.rs
git commit -m "fix: correct T1/T2 decoherence formula to match paper Eq. 41"
```

---

### Task 2: Add per-qubit T1/T2 fields to ArchitectureParams

**Files:**
- Modify: `src/noise/mod.rs:8-33` (struct + Default impl)
- Modify: `src/noise/mod.rs:354-370` (update_busy_and_coherence)

**Step 1: Add fields to the struct**

In `src/noise/mod.rs`, add two optional fields to `ArchitectureParams` after `t2`:

```rust
pub struct ArchitectureParams {
    pub single_gate_error: f64,
    pub two_gate_error: f64,
    pub teleportation_error_per_hop: f64,
    pub single_gate_time: f64,
    pub two_gate_time: f64,
    pub teleportation_time_per_hop: f64,
    pub t1: f64,
    pub t2: f64,
    // TODO: Add per-qubit single_gate_error and per-qubit-pair two_gate_error arrays
    /// Per-qubit T1 relaxation times in nanoseconds. Falls back to scalar `t1` if None.
    pub t1_per_qubit: Option<Vec<f64>>,
    /// Per-qubit T2 dephasing times in nanoseconds. Falls back to scalar `t2` if None.
    pub t2_per_qubit: Option<Vec<f64>>,
}
```

Update the `Default` impl to set both to `None`:

```rust
impl Default for ArchitectureParams {
    fn default() -> Self {
        Self {
            single_gate_error: 1e-4,
            two_gate_error: 1e-3,
            teleportation_error_per_hop: 1e-2,
            single_gate_time: 20.0,
            two_gate_time: 100.0,
            teleportation_time_per_hop: 1000.0,
            t1: 100_000.0,
            t2: 50_000.0,
            t1_per_qubit: None,
            t2_per_qubit: None,
        }
    }
}
```

**Step 2: Update update_busy_and_coherence to use per-qubit values**

In the `update_busy_and_coherence` function, replace:

```rust
layer_coh_grid[q] = decoherence_fidelity(idle_time, params.t1, params.t2);
```

with:

```rust
let q_t1 = params.t1_per_qubit.as_ref().map_or(params.t1, |v| v[q]);
let q_t2 = params.t2_per_qubit.as_ref().map_or(params.t2, |v| v[q]);
layer_coh_grid[q] = decoherence_fidelity(idle_time, q_t1, q_t2);
```

**Step 3: Fix existing test struct literals**

In `src/noise/mod.rs` tests, the `zero_error_gives_perfect_operational_fidelity` test constructs `ArchitectureParams` with `..Default::default()` — this already handles the new fields. No test changes needed.

**Step 4: Run Rust tests**

Run: `cargo test`
Expected: All pass — `None` falls back to scalar, no behavior change.

**Step 5: Commit**

```bash
git add src/noise/mod.rs
git commit -m "feat: add optional per-qubit T1/T2 to ArchitectureParams"
```

---

### Task 3: Add per-qubit T1/T2 Rust test

**Files:**
- Modify: `src/noise/mod.rs` (add test at end of `mod tests` block, before the closing `}`)

**Step 1: Write the test**

Add this test inside the existing `mod tests` block in `src/noise/mod.rs`:

```rust
#[test]
fn per_qubit_t1t2_differs_from_uniform() {
    let path = Path::new("dse_pau/test_vectors.json");
    let data = fs::read_to_string(path).unwrap();
    let parsed: Value = serde_json::from_str(&data).unwrap();
    let tc = &parsed["hqa_test_qft_25_all_to_all"];

    let num_qubits = tc["num_virtual_qubits"].as_u64().unwrap() as usize;
    let num_cores = tc["num_cores"].as_u64().unwrap() as usize;
    let num_layers = tc["num_layers"].as_u64().unwrap() as usize;

    let gs_sparse: Vec<[f64; 4]> = serde_json::from_value(tc["gs_sparse"].clone()).unwrap();
    let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_qubits);

    let initial_partition: Vec<i32> =
        serde_json::from_value(tc["input_initial_partition"].clone()).unwrap();
    let mut placements = Array2::<i32>::zeros((num_layers + 1, num_qubits));
    for (q, &val) in initial_partition.iter().enumerate() {
        placements[[0, q]] = val;
    }

    let core_caps: Vec<usize> =
        serde_json::from_value(tc["input_core_capacities"].clone()).unwrap();
    let dist_vecs: Vec<Vec<i32>> =
        serde_json::from_value(tc["input_distance_matrix"].clone()).unwrap();
    let dist_flat: Vec<i32> = dist_vecs.into_iter().flatten().collect();
    let dist_array = Array2::from_shape_vec((num_cores, num_cores), dist_flat).unwrap();

    let result = hqa_mapping(
        &tensor,
        placements,
        num_cores,
        &core_caps,
        dist_array.view(),
    );
    let routing = extract_inter_core_communications(&result, dist_array.view());

    // Uniform T1/T2
    let uniform_params = ArchitectureParams::default();
    let uniform_report = estimate_fidelity(&tensor, &routing, &uniform_params, None);

    // Per-qubit: give qubit 0 much worse T1/T2
    let mut t1_vec = vec![100_000.0; num_qubits];
    let mut t2_vec = vec![50_000.0; num_qubits];
    t1_vec[0] = 10_000.0; // 10x worse
    t2_vec[0] = 5_000.0;

    let per_qubit_params = ArchitectureParams {
        t1_per_qubit: Some(t1_vec),
        t2_per_qubit: Some(t2_vec),
        ..Default::default()
    };
    let per_qubit_report = estimate_fidelity(&tensor, &routing, &per_qubit_params, None);

    // Per-qubit with one bad qubit should have worse coherence
    assert!(
        per_qubit_report.coherence_fidelity < uniform_report.coherence_fidelity,
        "per-qubit with one degraded qubit should have worse coherence: {} vs {}",
        per_qubit_report.coherence_fidelity,
        uniform_report.coherence_fidelity
    );

    // Algorithmic fidelity should be identical (not affected by T1/T2)
    assert!(
        (per_qubit_report.algorithmic_fidelity - uniform_report.algorithmic_fidelity).abs() < 1e-12,
        "algorithmic fidelity should be unaffected by T1/T2 changes"
    );
}
```

**Step 2: Run the test**

Run: `cargo test per_qubit_t1t2_differs_from_uniform`
Expected: PASS

**Step 3: Commit**

```bash
git add src/noise/mod.rs
git commit -m "test: verify per-qubit T1/T2 produces different coherence from uniform"
```

---

### Task 4: Thread per-qubit T1/T2 through Python API (Rust side)

**Files:**
- Modify: `src/python_api.rs:14-28` (map_and_estimate signature)
- Modify: `src/python_api.rs:107-116` (map_and_estimate params construction)
- Modify: `src/python_api.rs:166-179` (estimate_hardware_fidelity signature)
- Modify: `src/python_api.rs:229-238` (estimate_hardware_fidelity params construction)

**Step 1: Update map_and_estimate signature and params**

Add `use numpy::PyArray1;` import (already imported at line 2).

Update the `#[pyo3(signature = (...))]` to add the two new optional params after `t2`:

```rust
#[pyo3(signature = (
    gs_sparse,
    initial_partition,
    num_cores,
    core_capacities,
    distance_matrix,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0,
    t1_per_qubit = None,
    t2_per_qubit = None
))]
```

Add matching function parameters after `t2: f64`:

```rust
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
```

Update the `ArchitectureParams` construction (around line 107):

```rust
let params = ArchitectureParams {
    single_gate_error,
    two_gate_error,
    teleportation_error_per_hop,
    single_gate_time,
    two_gate_time,
    teleportation_time_per_hop,
    t1,
    t2,
    t1_per_qubit: t1_per_qubit.map(|a| a.readonly().as_array().to_vec()),
    t2_per_qubit: t2_per_qubit.map(|a| a.readonly().as_array().to_vec()),
};
```

**Step 2: Update estimate_hardware_fidelity identically**

Same changes to `estimate_hardware_fidelity`:

Signature addition:

```rust
#[pyo3(signature = (
    gs_sparse,
    placements,
    distance_matrix,
    sparse_swaps,
    single_gate_error = 1e-4,
    two_gate_error = 1e-3,
    teleportation_error_per_hop = 1e-2,
    single_gate_time = 20.0,
    two_gate_time = 100.0,
    teleportation_time_per_hop = 1000.0,
    t1 = 100_000.0,
    t2 = 50_000.0,
    t1_per_qubit = None,
    t2_per_qubit = None
))]
```

Function parameters after `t2: f64`:

```rust
    t1_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
    t2_per_qubit: Option<&Bound<'py, PyArray1<f64>>>,
```

Params construction (around line 229):

```rust
let params = ArchitectureParams {
    single_gate_error,
    two_gate_error,
    teleportation_error_per_hop,
    single_gate_time,
    two_gate_time,
    teleportation_time_per_hop,
    t1,
    t2,
    t1_per_qubit: t1_per_qubit.map(|a| a.readonly().as_array().to_vec()),
    t2_per_qubit: t2_per_qubit.map(|a| a.readonly().as_array().to_vec()),
};
```

**Step 3: Run Rust tests**

Run: `cargo test`
Expected: All pass (no functional change to test code paths).

**Step 4: Commit**

```bash
git add src/python_api.rs
git commit -m "feat: expose per-qubit T1/T2 arrays in Python API bindings"
```

---

### Task 5: Thread per-qubit T1/T2 through Python wrapper

**Files:**
- Modify: `python/qusim/__init__.py:122-137` (map_circuit signature)
- Modify: `python/qusim/__init__.py:208-222` (map_and_estimate call)
- Modify: `python/qusim/__init__.py:244-257` (estimate_hardware_fidelity call)

**Step 1: Add kwargs to map_circuit**

Add two new optional parameters after `t2`:

```python
def map_circuit(
    circuit: qiskit.QuantumCircuit,
    full_coupling_map: qiskit.transpiler.CouplingMap,
    core_mapping: dict[int, int],
    seed: Optional[int] = None,
    initial_placement: InitialPlacement = InitialPlacement.RANDOM,
    # Hardware defaults
    single_gate_error: float = 1e-4,
    two_gate_error: float = 1e-3,
    teleportation_error_per_hop: float = 1e-2,
    single_gate_time: float = 20.0,
    two_gate_time: float = 100.0,
    teleportation_time_per_hop: float = 1000.0,
    t1: float = 100_000.0,
    t2: float = 50_000.0,
    t1_per_qubit: Optional[np.ndarray] = None,
    t2_per_qubit: Optional[np.ndarray] = None,
) -> QusimResult:
```

**Step 2: Pass through to both Rust calls**

In the `map_and_estimate()` call (around line 208), add:

```python
    raw_dict = map_and_estimate(
        gs_sparse=gs_sparse,
        initial_partition=initial_partition,
        num_cores=num_cores,
        core_capacities=core_caps,
        distance_matrix=dist_mat,
        single_gate_error=single_gate_error,
        two_gate_error=two_gate_error,
        teleportation_error_per_hop=teleportation_error_per_hop,
        single_gate_time=single_gate_time,
        two_gate_time=two_gate_time,
        teleportation_time_per_hop=teleportation_time_per_hop,
        t1=t1,
        t2=t2,
        t1_per_qubit=t1_per_qubit,
        t2_per_qubit=t2_per_qubit,
    )
```

In the `estimate_hardware_fidelity()` call (around line 244), add:

```python
    fidelity_dict = estimate_hardware_fidelity(
        gs_sparse=gs_sparse,
        placements=placements,
        distance_matrix=dist_mat,
        sparse_swaps=sparse_swaps_arr,
        single_gate_error=single_gate_error,
        two_gate_error=two_gate_error,
        teleportation_error_per_hop=teleportation_error_per_hop,
        single_gate_time=single_gate_time,
        two_gate_time=two_gate_time,
        teleportation_time_per_hop=teleportation_time_per_hop,
        t1=t1,
        t2=t2,
        t1_per_qubit=t1_per_qubit,
        t2_per_qubit=t2_per_qubit,
    )
```

**Step 3: Rebuild and smoke test**

Run:
```bash
cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release
```
Expected: Build succeeds.

Run:
```bash
cd /home/agonhid/dev/qusim && source .venv/bin/activate && python -c "from qusim import map_circuit; print('import OK')"
```
Expected: `import OK`

**Step 4: Commit**

```bash
git add python/qusim/__init__.py
git commit -m "feat: add per-qubit T1/T2 kwargs to map_circuit Python API"
```

---

### Task 6: Benchmark script

**Files:**
- Create: `examples/benchmark_against_paper.py`

**Step 1: Write the benchmark script**

Create `examples/benchmark_against_paper.py`:

```python
"""
Benchmark qusim fidelity predictions against the paper's results
(arXiv:2503.06693v2) and real IBM Q hardware measurements.

Loads the paper's QASM circuits, runs them through qusim in single-core
all-to-all mode, and plots the comparison.

TODO: When IBM Q access is available, use backend.properties() to pull
exact per-qubit T1/T2 and per-gate error rates for precise reproduction.
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import qiskit
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from qusim import map_circuit, InitialPlacement

# ---------------------------------------------------------------------------
# Configuration — typical IBM Q error rates
# ---------------------------------------------------------------------------
SINGLE_GATE_ERROR = 2.3e-4
TWO_GATE_ERROR = 8.2e-3
T1_NS = 120_000.0   # 120 μs
T2_NS = 116_000.0   # 116 μs
SINGLE_GATE_TIME = 35.0     # ns, typical IBM sx
TWO_GATE_TIME = 660.0       # ns, typical IBM cx
TELEPORTATION_ERROR = 0.0   # single core, no teleportation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER_REPO = os.path.join(os.path.dirname(__file__), '..', '..', 'Analytic-Model-of-Fidelity-under-Depolarizing-Noise')
CIRCUITS_DIR = os.path.join(PAPER_REPO, 'circuits')
DATA_FILE = os.path.join(PAPER_REPO, 'data', 'data_1.csv')


def load_paper_data(data_file):
    """Load the paper's pre-computed fidelity results from data_1.csv."""
    with open(data_file, 'r') as f:
        data = list(csv.reader(f, delimiter=','))

    filenames = data[0]
    num_qubits = [int(q) for q in data[1]]
    num_gates = [int(g) for g in data[3]]

    fid_esp = [float(f) for f in data[6]]

    fid_min_depol = [float(eval(f)[0]) for f in data[10]]
    fid_max_depol = [float(eval(f)[1]) for f in data[10]]
    fid_avg_depol = [(fid_min_depol[i] + fid_max_depol[i]) / 2 for i in range(len(fid_min_depol))]
    depol_error = [(fid_max_depol[i] - fid_min_depol[i]) / 2 for i in range(len(fid_min_depol))]

    sr_mit_IBMQ = [float(f) for f in data[13]]

    return {
        'filenames': filenames,
        'num_qubits': num_qubits,
        'num_gates': num_gates,
        'fid_esp': fid_esp,
        'fid_avg_depol': fid_avg_depol,
        'depol_error': depol_error,
        'sr_mit_IBMQ': sr_mit_IBMQ,
    }


def build_single_core_setup(num_qubits):
    """Create a single-core all-to-all setup for benchmarking without routing overhead."""
    edges = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append((i, j))
            edges.append((j, i))
    coupling_map = CouplingMap(edges)
    core_mapping = {i: 0 for i in range(num_qubits)}
    return coupling_map, core_mapping


def run_qusim_on_circuit(qasm_path, num_qubits):
    """Load a QASM circuit and estimate fidelity with qusim."""
    circ = qiskit.QuantumCircuit.from_qasm_file(qasm_path)

    # Mirror circuit (circ^-1 & circ) to match the paper's benchmarking methodology
    circ_inv = circ.copy()
    circ_inv.remove_final_measurements()
    circ_inv = circ_inv.inverse()
    circ_inv.barrier()
    rev_circ = circ_inv & circ

    # Transpile to basis gates
    transp = qiskit.transpile(
        rev_circ,
        basis_gates=['x', 'cx', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'measure'],
        optimization_level=0,
    )

    n = transp.num_qubits
    coupling_map, core_mapping = build_single_core_setup(n)

    result = map_circuit(
        circuit=transp,
        full_coupling_map=coupling_map,
        core_mapping=core_mapping,
        seed=42,
        initial_placement=InitialPlacement.RANDOM,
        single_gate_error=SINGLE_GATE_ERROR,
        two_gate_error=TWO_GATE_ERROR,
        teleportation_error_per_hop=TELEPORTATION_ERROR,
        single_gate_time=SINGLE_GATE_TIME,
        two_gate_time=TWO_GATE_TIME,
        teleportation_time_per_hop=1000.0,
        t1=T1_NS,
        t2=T2_NS,
    )

    return result.overall_fidelity


def main():
    paper_data = load_paper_data(DATA_FILE)
    filenames = paper_data['filenames']
    num_gates = paper_data['num_gates']

    # Run qusim on each circuit
    qusim_fidelities = []
    valid_indices = []

    for i, fname in enumerate(filenames):
        qasm_path = os.path.join(CIRCUITS_DIR, fname)
        if not os.path.exists(qasm_path):
            print(f"  SKIP (file not found): {fname}")
            qusim_fidelities.append(None)
            continue

        try:
            fid = run_qusim_on_circuit(qasm_path, paper_data['num_qubits'][i])
            qusim_fidelities.append(fid)
            valid_indices.append(i)
            print(f"  [{i+1}/{len(filenames)}] {fname}: qusim={fid:.4f}, IBM={paper_data['sr_mit_IBMQ'][i]:.4f}")
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            qusim_fidelities.append(None)

    # Filter to valid results
    v_gates = [num_gates[i] for i in valid_indices]
    v_qusim = [qusim_fidelities[i] for i in valid_indices]
    v_esp = [paper_data['fid_esp'][i] for i in valid_indices]
    v_depol = [paper_data['fid_avg_depol'][i] for i in valid_indices]
    v_depol_err = [paper_data['depol_error'][i] for i in valid_indices]
    v_ibmq = [paper_data['sr_mit_IBMQ'][i] for i in valid_indices]
    v_nq = [paper_data['num_qubits'][i] for i in valid_indices]

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    scatter_size = 20
    ticks_size = 15
    label_size = 20
    fig, ax = plt.subplots(figsize=(20, 7))

    ax.scatter(v_gates, v_ibmq, label='IBM Q (mitigated)', color='tab:purple', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, v_esp, label='ESP', color='tab:orange', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.errorbar(v_gates, v_depol, yerr=v_depol_err, fmt='none', c='gray', alpha=0.5, zorder=-1)
    ax.scatter(v_gates, v_depol, label='Paper (depol.)', color='tab:green', alpha=0.75,
               s=[scatter_size * q for q in v_nq])
    ax.scatter(v_gates, v_qusim, label='qusim', color='tab:red', alpha=0.75,
               s=[scatter_size * q for q in v_nq], marker='x')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Gates', fontsize=label_size)
    ax.set_ylabel('Fidelity', fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=ticks_size)
    ax.legend(fontsize=ticks_size)

    # Inset boxplot: fidelity differences vs IBM Q
    y_bottom = 0.1
    y_top = ax.get_ylim()[1] * 0.65
    axins = ax.inset_axes([13, y_bottom, 45, y_top - y_bottom], transform=ax.transData)
    axins.set_facecolor('snow')

    esp_diff = [v_esp[i] - v_ibmq[i] for i in range(len(v_ibmq))]
    depol_diff = [v_depol[i] - v_ibmq[i] for i in range(len(v_ibmq))]
    qusim_diff = [v_qusim[i] - v_ibmq[i] for i in range(len(v_ibmq))]

    colors = ['tab:orange', 'tab:green', 'tab:red']
    box = axins.boxplot(
        [esp_diff, depol_diff, qusim_diff],
        positions=[1, 2, 3],
        labels=['ESP', 'Paper', 'qusim'],
        patch_artist=True,
        widths=0.45,
    )
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
    for median in box['medians']:
        median.set_color('black')

    axins.tick_params(axis='both', which='major', labelsize=ticks_size * 0.85)
    axins.tick_params(axis='x', which='major', labelsize=ticks_size * 0.85, rotation=45)
    axins.set_ylabel('Fidelity\nDifference', fontsize=label_size * 0.85, labelpad=-3)
    axins.axhline(y=0, color='gray', linestyle='--', zorder=-10)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_vs_paper.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
```

**Step 2: Rebuild the extension and run**

Run:
```bash
cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release && python examples/benchmark_against_paper.py
```
Expected: Script runs, prints per-circuit fidelity comparisons, saves `examples/benchmark_vs_paper.png`.

**Step 3: Commit**

```bash
git add examples/benchmark_against_paper.py
git commit -m "feat: add benchmark script comparing qusim against paper results"
```

---

### Task 7: Review and validate results

**Step 1: Inspect the generated plot**

Open `examples/benchmark_vs_paper.png` and verify:
- qusim predictions are plotted (red x markers)
- All four series (IBM Q, ESP, Paper, qusim) are visible
- Inset boxplot shows the fidelity difference distributions
- qusim results are in the plausible range (between 0 and 1, correlated with gate count)

**Step 2: Assess the gap**

Compare the qusim boxplot median against ESP and Paper medians. Since qusim currently uses ESP-style multiplicative gate errors (not the paper's depolarizing channel), qusim should track close to ESP. Differences come from:
- The fixed T1/T2 formula (improvement)
- Uniform vs per-gate error rates (limitation)
- Single-core all-to-all means zero routing overhead

Document observations as a comment at the top of the benchmark script.
