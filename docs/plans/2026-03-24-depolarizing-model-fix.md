# Depolarizing Model & Coherence Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace qusim's ESP gate error model with the paper's depolarizing channel (Algorithm 1) and fix the coherence model to use per-layer multiplicative decay.

**Architecture:** Two changes in `src/noise/mod.rs`: (1) `process_computational_gates` switches from `F *= (1-ε)` to depolarizing channel with η correction, (2) `update_busy_and_coherence` switches from cumulative absolute idle time to per-layer multiplicative decay. Overall algorithmic fidelity becomes `∏ F_q` from the grid instead of scalar accumulation.

**Tech Stack:** Rust (ndarray, pyo3), maturin build, Python benchmark script for validation.

**Build commands:**
- Rust tests: `cargo test`
- Build Python extension: `cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release`
- Benchmark: `cd /home/agonhid/dev/qusim && source .venv/bin/activate && python examples/benchmark_against_paper.py`

**Design doc:** `docs/plans/2026-03-24-depolarizing-model-fix-design.md`

---

### Task 1: Add depolarization_lambda helper and update gate error model

**Files:**
- Modify: `src/noise/mod.rs:75-83` (replace `gate_fidelity` with `depolarization_lambda`)
- Modify: `src/noise/mod.rs:288-312` (`process_computational_gates`)

**Step 1: Replace the gate_fidelity helper**

In `src/noise/mod.rs`, replace the `gate_fidelity` function (lines 79-83) with:

```rust
/// Convert a gate error rate to the depolarization parameter λ (paper Eq. 39).
/// d = 2 for single-qubit gates, d = 4 for two-qubit gates.
#[inline]
fn depolarization_lambda(gate_error: f64, d: f64) -> f64 {
    d * gate_error / (d - 1.0)
}
```

**Step 2: Rewrite process_computational_gates**

Replace the body of `process_computational_gates` (lines 288-312) with the depolarizing channel model. The function no longer returns a scalar layer fidelity — it returns `()` since fidelity is tracked in the per-qubit grid. However, to minimize changes to the call site, we still return an f64 but compute it from the grid change.

Replace the full function with:

```rust
#[inline]
fn process_computational_gates(
    gates: &[(usize, usize, f64)],
    params: &ArchitectureParams,
    layer_busy_time: &mut [f64],
    layer_algo_grid: &mut [f64],
) -> f64 {
    let mut layer_algo_fidelity = 1.0;
    for &(u, v, _) in gates {
        if u == v {
            // Single-qubit gate: F_q = (1-λ)·F_q + λ/d, with d=2 (paper Algorithm 1)
            layer_busy_time[u] += params.single_gate_time;
            let lam = depolarization_lambda(params.single_gate_error, 2.0);
            let f_before = layer_algo_grid[u];
            layer_algo_grid[u] = (1.0 - lam) * layer_algo_grid[u] + lam / 2.0;
            if f_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[u] / f_before;
            }
        } else {
            // Two-qubit gate: η correction (paper Eq. 25, Algorithm 1)
            layer_busy_time[u] += params.two_gate_time;
            layer_busy_time[v] += params.two_gate_time;
            let lam = depolarization_lambda(params.two_gate_error, 4.0);
            let f1 = layer_algo_grid[u];
            let f2 = layer_algo_grid[v];
            let f1_before = f1;
            let f2_before = f2;

            let sqrt_1_lam = (1.0 - lam).sqrt();
            let eta = 0.5 * (((1.0 - lam) * (f1 + f2).powi(2) + lam).sqrt()
                           - sqrt_1_lam * (f1 + f2));

            layer_algo_grid[u] = sqrt_1_lam * f1 + eta;
            layer_algo_grid[v] = sqrt_1_lam * f2 + eta;

            if f1_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[u] / f1_before;
            }
            if f2_before > 0.0 {
                layer_algo_fidelity *= layer_algo_grid[v] / f2_before;
            }
        }
    }
    layer_algo_fidelity
}
```

**Step 3: Update overall algorithmic fidelity computation**

In `estimate_fidelity` (around line 230-240), replace the algorithmic fidelity computation. Change:

```rust
    FidelityReport {
        algorithmic_fidelity: overall_algorithmic,
```

to:

```rust
    // Overall algorithmic fidelity = product of per-qubit fidelities at final layer (paper Eq. 40)
    let mut overall_algorithmic_from_grid = 1.0_f64;
    if num_layers > 0 {
        for q in 0..num_qubits {
            overall_algorithmic_from_grid *= algorithmic_fidelity_grid[(num_layers - 1) * num_qubits + q];
        }
    }

    FidelityReport {
        algorithmic_fidelity: overall_algorithmic_from_grid,
```

Also update `overall_fidelity` in the same struct to use `overall_algorithmic_from_grid`:

```rust
        overall_fidelity: overall_algorithmic_from_grid * overall_routing * global_coherence,
```

The `overall_algorithmic` scalar accumulator variable (line 135) and `overall_algorithmic *= layer_algorithmic_fidelity` (line 206) can be left in place — they become unused but harmless. Or remove them if you prefer.

**Step 4: Run tests**

Run: `cargo test`
Expected: All tests pass. The depolarizing model gives *higher* fidelity than ESP (due to the +λ/d correction), so all `< 1.0` and `> 0.0` assertions still hold. The `zero_error_gives_perfect_operational_fidelity` test: with `gate_error=0.0`, `λ=0`, so `F_q = (1-0)*F_q + 0 = F_q` — stays at 1.0.

**Step 5: Commit**

```bash
git add src/noise/mod.rs
git commit -m "feat: replace ESP gate model with depolarizing channel (paper Algorithm 1)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Fix coherence model to per-layer multiplicative decay

**Files:**
- Modify: `src/noise/mod.rs:361-378` (`update_busy_and_coherence`)
- Modify: `src/noise/mod.rs:147-158` (coherence grid initialization and carry-forward)
- Modify: `src/noise/mod.rs:209-219` (call site)

**Step 1: Rewrite update_busy_and_coherence**

Replace the function (lines 361-378) with per-layer multiplicative decay:

```rust
#[inline]
fn update_busy_and_coherence(
    layer_time: f64,
    params: &ArchitectureParams,
    layer_busy_time: &[f64],
    layer_coh_grid: &mut [f64],
) {
    for q in 0..layer_coh_grid.len() {
        // Idle time is how long the qubit was idle during THIS layer only
        let layer_idle = (layer_time - layer_busy_time[q]).max(0.0);
        if layer_idle > 0.0 {
            let q_t1 = params.t1_per_qubit.as_ref().map_or(params.t1, |v| v[q]);
            let q_t2 = params.t2_per_qubit.as_ref().map_or(params.t2, |v| v[q]);
            layer_coh_grid[q] *= decoherence_fidelity(layer_idle, q_t1, q_t2);
        }
    }
}
```

**Step 2: Update the call site in estimate_fidelity**

The call to `update_busy_and_coherence` (around line 212-219) currently passes `total_circuit_time` and `&mut qubit_busy_time`. Update to the new signature:

```rust
        // 4. Update coherence based on per-layer idle time
        let layer_coh_grid =
            &mut coherence_fidelity_grid[layer * num_qubits..(layer + 1) * num_qubits];
        update_busy_and_coherence(
            layer_time,
            params,
            &layer_busy_time,
            layer_coh_grid,
        );
```

**Step 3: Remove unused variables**

Remove the now-unused variables from `estimate_fidelity`:
- Remove: `let mut qubit_busy_time = vec![0.0_f64; num_qubits];` (line 141)
- The `total_circuit_time` variable is still needed for the `FidelityReport` output, keep it.

**Step 4: Run tests**

Run: `cargo test`
Expected: All tests pass. Per-layer multiplicative coherence gives *higher* fidelity than cumulative absolute (less aggressive decay), so range assertions hold. Zero-error test: with infinite T1/T2, `decoherence_fidelity` returns 1.0, so `1.0 * 1.0 = 1.0` each layer.

**Step 5: Commit**

```bash
git add src/noise/mod.rs
git commit -m "fix: switch coherence model to per-layer multiplicative decay (paper Eq. 41)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Rebuild and re-run benchmark

**Files:**
- No code changes — validation only

**Step 1: Rebuild the Python extension**

Run:
```bash
cd /home/agonhid/dev/qusim && source .venv/bin/activate && maturin develop --release
```
Expected: Build succeeds.

**Step 2: Run the benchmark**

Run:
```bash
cd /home/agonhid/dev/qusim && source .venv/bin/activate && python examples/benchmark_against_paper.py
```
Expected: Script runs, produces `examples/benchmark_vs_paper.png`. qusim should now track much closer to the ESP and depolarizing series (all computed with same uniform error rates).

**Step 3: Report results**

Compare the new qusim values against ESP and depolarizing for a few circuits. The gap should have shrunk significantly. If qusim now tracks close to the depolarizing model (green dots), both fixes are working correctly.
