# Per-Qubit T1/T2 Support & Paper Benchmark Validation

**Date**: 2026-03-24
**Status**: Approved

## Motivation

qusim's noise model has two issues when compared against the analytic fidelity model in [arXiv:2503.06693v2](https://arxiv.org/html/2503.06693v2):

1. **Wrong T1/T2 formula** in `noise/mod.rs`: uses `exp(-t/T1) * exp(-t/T2)` instead of the paper's Equation (41): `exp(-t/T1) * (0.5 * exp(-t/T2) + 0.5)`. The correct formula exists in `qubit.rs` but not in the main estimator.
2. **No per-qubit T1/T2**: qusim applies uniform T1/T2 to all qubits, while real hardware has different coherence times per qubit.

Before implementing the paper's full depolarizing channel model (Algorithm 1), we need a baseline benchmark comparing qusim's current predictions against the paper's results and real IBM Q hardware data.

## Design

### 1. Per-qubit T1/T2 in Rust (`noise/mod.rs`)

Add optional per-qubit override arrays to `ArchitectureParams`:

```rust
pub struct ArchitectureParams {
    // ... existing scalar fields unchanged ...
    pub t1: f64,
    pub t2: f64,
    pub t1_per_qubit: Option<Vec<f64>>,  // NEW
    pub t2_per_qubit: Option<Vec<f64>>,  // NEW
}
```

`Default` sets both to `None`. In `update_busy_and_coherence`, look up per-qubit values when available, fall back to scalar otherwise:

```rust
let q_t1 = params.t1_per_qubit.as_ref().map_or(params.t1, |v| v[q]);
let q_t2 = params.t2_per_qubit.as_ref().map_or(params.t2, |v| v[q]);
```

### 2. Fix decoherence formula (`noise/mod.rs`)

```rust
// Before (wrong):
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (-idle_time / t2).exp()
}

// After (matches paper Eq. 41):
fn decoherence_fidelity(idle_time: f64, t1: f64, t2: f64) -> f64 {
    (-idle_time / t1).exp() * (0.5 * (-idle_time / t2).exp() + 0.5)
}
```

### 3. Python API changes

**Rust (`python_api.rs`)**: Both `map_and_estimate` and `estimate_hardware_fidelity` get optional `t1_per_qubit` and `t2_per_qubit` numpy array params.

**Python (`__init__.py`)**: `map_circuit()` gets matching optional kwargs:

```python
t1_per_qubit: Optional[np.ndarray] = None,
t2_per_qubit: Optional[np.ndarray] = None,
```

Fully backward compatible — `None` falls back to scalar values.

### 4. Benchmark script (`examples/benchmark_against_paper.py`)

Loads the paper's 93 QASM circuits and `data_1.csv`, runs each through qusim in single-core all-to-all mode with uniform error rates (single: ~2.3e-4, two: ~8.2e-3, T1: 120us, T2: 116us), and generates a comparison plot mirroring the paper's Figure 1:

- x-axis: number of gates (log scale)
- y-axis: fidelity
- Series: IBM Q measured, ESP, Paper's depolarizing model, qusim
- Inset boxplot of fidelity differences vs IBM Q

### 5. Future TODOs (not in this scope)

- Per-qubit single-gate error arrays (`single_gate_error_per_qubit`)
- Per-qubit-pair two-gate error maps (`two_gate_error_per_pair`)
- IBM Q calibration data integration via `backend.properties()`
- Full depolarizing channel model with eta entanglement correction (Algorithm 1)
