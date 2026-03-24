# Depolarizing Model & Coherence Fix

**Date**: 2026-03-24
**Status**: Approved

## Motivation

Benchmark comparison showed qusim's fidelity estimates are ~20% below ESP/depolarizing models run with the same uniform error rates on the same circuits. Two structural issues:

1. **Gate error model**: qusim uses ESP (`F *= 1-ε`) instead of the paper's depolarizing channel with η correction (Algorithm 1, arXiv:2503.06693v2)
2. **Coherence model**: qusim recomputes decoherence from cumulative idle time each layer (absolute), while the paper multiplies by per-layer decay (incremental). The concave T2 term `0.5*exp + 0.5` produces different results under these approaches.

## Design

### Fix 1: Depolarizing channel gate error model

Replace `gate_fidelity(error_rate) -> 1-ε` with depolarizing channel equations in `process_computational_gates`:

- Add `depolarization_lambda(gate_error, d) -> d * gate_error / (d - 1)` (paper Eq. 39)
- 1Q gates: `F_q = (1-λ)·F_q + λ/2`
- 2Q gates: η correction per paper Eq. 25, then `F_q = sqrt(1-λ)·F_q + η`
- Overall algorithmic fidelity becomes `∏ F_q` from grid (paper Eq. 40) instead of scalar product accumulation

### Fix 2: Per-layer multiplicative coherence

Replace absolute idle-time coherence with per-layer multiplicative decay in `update_busy_and_coherence`:

- `layer_idle = (layer_time - layer_busy_time[q]).max(0.0)`
- `layer_coh_grid[q] *= decoherence_fidelity(layer_idle, t1, t2)`
- Remove `total_circuit_time` and `qubit_busy_time` from coherence path (still needed for timing stats)

### Scope

All changes in `src/noise/mod.rs`. No Python API changes. Existing tests updated to match new model behavior. Benchmark re-run to validate.
