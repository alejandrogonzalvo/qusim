# Hungarian Qubit Assignment (HQA)

Rust implementation of the HQA algorithm for optimized mapping of quantum circuits on multi-core architectures.

> **Based on:** Pau Escofet *et al.*, "Hungarian Qubit Assignment for Optimized Mapping of Quantum Circuits on Multi-Core Architectures," [arXiv:2309.12182](https://arxiv.org/pdf/2309.12182)

## Algorithm Overview

HQA solves the qubit-to-core assignment problem for modular quantum processors. Given a quantum circuit decomposed into L temporal layers (time slices), HQA iteratively:

1. **Lookahead**: builds a spatio-temporal interaction matrix combining hard constraints from the current layer with exponentially decaying future interactions.
2. **Conflict detection**: identifies qubit pairs that interact but sit on different cores.
3. **Core balancing**: resolves odd free-space parity between cores by swapping movable qubits.
4. **Hungarian matching**: uses Kuhn-Munkres to optimally assign conflicting qubit pairs to cores, minimizing inter-core communication cost.

## Complexity

Let:
- N = number of qubits
- L = number of circuit layers
- K = number of cores
- E = edges (two-qubit gates) per layer, at most N/2
- P = conflicting pairs per layer (gates spanning two cores)
- H = lookahead horizon (constant, 20)

**Per layer:**

| Step | Cost |
|---|---|
| Lookahead | O(H * E), effectively O(E) since H is constant |
| Active qubit set + movable check | O(E + N) |
| Conflict detection (sorted edge pairs) | O(E * log E) |
| Core likelihood (per conflicting pair) | O(P * N) |
| Hungarian matching | O(max(P, K)^3) |
| Partition validation | O(E) |

**Total: O(L * (P*N + max(P, K)^3))**

For typical QFT circuits, P is small relative to N (most gates are already co-located), so the dominant term in practice is the cubic Hungarian matching when K is large, or the P*N likelihood computation when N is large.

## Module Structure

| File | Purpose |
|---|---|
| `mod.rs` | Module root, re-exports public API |
| `interaction_tensor.rs` | `InteractionTensor`: owns sparse edge lists per layer |
| `mapping.rs` | Core algorithm: `hqa_mapping`, `lookahead`, `validate_partition` |

## Optimizations vs. Python Reference

### 1. Fully Sparse Representation
Python uses `scipy.sparse` COO matrices. The Rust version stores the interaction tensor as `Vec<Vec<(usize, usize, f64)>>` edge lists per layer. All operations (lookahead, conflict detection, movable qubit check, partition validation) iterate only over actual gates, giving O(E) instead of O(N^2) per layer.

### 2. Truncated Lookahead Horizon
The original algorithm sums over all remaining layers. With exponential decay (sigma=1), contributions beyond ~20 layers are negligible (2^-20 ~ 10^-6). The Rust version caps at `LOOKAHEAD_HORIZON = 20`, turning an O(L) sweep into O(1) per time step.

### 3. Language-level Gains
No interpreter overhead, no GC pauses, contiguous memory layout for partition matrices (`ndarray::Array2`), and zero-cost abstractions for iteration.

## Usage

```rust
use qusim::hqa::{hqa_mapping, InteractionTensor};

// gs_sparse: Vec<[f64; 4]> where each entry is [layer, q1, q2, weight]
let tensor = InteractionTensor::from_sparse(&gs_sparse, num_layers, num_qubits);
let result = hqa_mapping(tensor, ps, num_cores, &capacities, dist.view());
```
