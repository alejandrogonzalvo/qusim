# Hungarian Qubit Assignment (HQA)

Rust implementation of the HQA algorithm for optimized mapping of quantum circuits on multi-core architectures.

> **Based on:** Pau Escofet *et al.*, "Hungarian Qubit Assignment for Optimized Mapping of Quantum Circuits on Multi-Core Architectures," [arXiv:2309.12182](https://arxiv.org/pdf/2309.12182)

## Algorithm Overview

HQA solves the **qubit-to-core assignment** problem for modular quantum processors. Given a quantum circuit decomposed into temporal layers (time slices), HQA iteratively:

1. **Lookahead** — builds a spatio-temporal interaction matrix that combines hard constraints from the current layer with exponentially decaying future interactions.
2. **Gate conflict detection** — identifies qubit pairs that interact but are assigned to different cores.
3. **Core balancing** — resolves odd free-space parity between cores by swapping movable qubits.
4. **Hungarian matching** — uses the Kuhn-Munkres (Hungarian) algorithm to optimally assign conflicting qubit pairs to cores, minimizing inter-core communication cost.

## Module Structure

| File | Purpose |
|---|---|
| `mod.rs` | Module root, re-exports public API |
| `interaction_tensor.rs` | `InteractionTensor` — wrapper around an `ndarray` 3D view `(layers × qubits × qubits)` |
| `mapping.rs` | Core algorithm: `hqa_mapping`, `lookahead`, `validate_partition`, `array3_from_sparse` |

## Optimizations vs. Python Reference

The Rust port introduces three key optimizations beyond the language-level speedup:

### 1. ndarray for Memory Locality
The Python version uses `scipy.sparse` matrices in a list. The Rust port stores the full interaction tensor as a contiguous `ndarray::Array3<f64>`, giving the CPU cache-friendly row-major access patterns. `InteractionTensor` provides zero-cost views via `ArrayView2` / `ArrayView3`.

### 2. Truncated Lookahead Horizon
The original algorithm sums over **all** remaining layers. With exponential decay (`σ = 1`), contributions beyond ~20 layers are negligible (`2^{-20} ≈ 10^{-6}`). The Rust version caps the lookahead at `LOOKAHEAD_HORIZON = 20` layers, turning an O(L) sweep into O(1) per time step.

### 3. Sparse Edge Traversal
Even with a truncated horizon, the dense `for i in 0..N { for j in 0..N { ... } }` loop over the interaction matrix is wasteful — quantum circuits are sparse (≤ N/2 two-qubit gates per layer out of N² possible entries). The Rust version pre-extracts an `ActiveGates` sparse edge list once, then the lookahead iterates **only over actual gates**, reducing work from O(N²) to O(|E|) per layer.

**Combined impact** on a QFT-25 benchmark (5 cores, all-to-all topology):

| Version | Time |
|---|---|
| Python (reference) | ~50 ms |
| Rust (Vec, dense, full lookahead) | ~430 µs |
| Rust (ndarray, horizon=20) | ~148 µs |
| Rust (ndarray, sparse lookahead) | ~100 µs |

## Usage

```rust
use qusim::hqa::{hqa_mapping, InteractionTensor, array3_from_sparse};

let gs_array = array3_from_sparse(&gs_sparse, num_layers, num_qubits);
let tensor = InteractionTensor::new(gs_array.view());
let result = hqa_mapping(tensor, ps, num_cores, &capacities, dist.view());
```
