# qusim: Quantum Architecture HQA Mapping & Fidelity Estimation

`qusim` is a high-performance simulator written in Rust with native Python bindings. It maps Qiskit quantum circuits onto multi-core modular quantum architectures using a specialized Hungarian Qubit Assignment (HQA) mapping algorithm, and then performs fine-grained hardware noise and fidelity estimation.

## Installation

Ensure you have a working Python environment (Python 3.10+) and the Rust compiler installed.
The python extension is built identically to any standard Rust extension using [Maturin](https://www.maturin.rs/).

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install build tools and python dependencies
pip install maturin qiskit numpy matplotlib

# Build the qusim python package containing the native bindings inline
maturin develop --release
```

## Usage Guide

To use `qusim`, you provide a fully transpiled `qiskit.QuantumCircuit` composed of your native hardware gates (since our cost model relies on standard 1-qubit and 2-qubit interactions). 

```python
import numpy as np
import qiskit
from qiskit.circuit.library import QFT
from qiskit import transpile
import qusim
from qusim.hqa.placement import InitialPlacement

# 1. Create and transpile your circuit
circ = QFT(30)
transp_circ = transpile(
    circ, 
    basis_gates=['x', 'cx', 'cp', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'measure'], 
    optimization_level=0,
    seed_transpiler=42
)

# 2. Define the Target Hardware Architecture
num_cores = 5
qubits_per_core = 6

# 3. Execute the HQA Mapping and Noise Estimator natively
result: qusim.QusimResult = qusim.map_circuit(
    circuit=transp_circ,
    num_cores=num_cores,
    qubits_per_core=qubits_per_core,
    initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
    # Optional hardware tuning parameters
    single_gate_error=1e-4,
    two_gate_error=1e-3,
    t1=100_000.0,
    seed=42  # Seed for deterministic initial partition mapping
)

# 4. Analyze Results
print(f"Overall Final Fidelity: {result.overall_fidelity:.4f}")
print(f"Total EPR Pairs Consumed: {result.total_epr_pairs}")

# Grab 2D arrays plotting trace fidelity curves per qubit over the whole timeslice 
op_grid = result.operational_fidelity_grid
# op_grid.shape == (num_layers, num_virtual_qubits)
```

## Simulator Output Metrics
The `map_circuit` function returns a `qusim.QusimResult` dataclass with the following fields:

- `placements: np.ndarray`: Shape `(num_layers + 1, num_qubits)`. Holds the physical core indices denoting where each logical qubit resides at every layer step of the circuit execution.
- `total_teleportations: int`: Number of inter-core teleportations needed to execute the routing.
- `overall_fidelity: float`: A single multiplier of total global error accrued.
- `operational_fidelity_grid: np.ndarray` and `coherence_fidelity_grid: np.ndarray`: Shape `(num_layers, num_qubits)`. Floating point grids detailing the exact layer drops for fidelity per logical execution strand. You can easily plot these components via `result.get_qubit_fidelity_over_time(qubit_index)`.

## Custom Hardware Topology (Core Distances)
By default, mapping assumes an *all-to-all* topology where all cores are exactly 1 hop away from one another. You can supply a custom adjacency/distance matrix indicating teleportation penalties between far cores.

```python
# Create a linear bus topology for 3 cores natively in numpy
linear_distance = np.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0]
])

result = qusim.map_circuit(
    circuit=transp_circ,
    num_cores=3,
    qubits_per_core=10,
    distance_matrix=linear_distance
)
```

## Fidelity Estimation Architecture

The `qusim` noise module models quantum hardware execution by decoupling algorithm errors from physical compilation overhead. **For a detailed breakdown of the mathematical equations governing Algorithmic, Routing, and Coherence (T1/T2) decay modeling, please refer to the [Noise Module Documentation](src/noise/README.md).**

### Pipeline Structure

The estimation occurs via a localized two-pass pipeline over the `InteractionTensor`:

1. **HQA First Pass**: Evaluates the circuit greedily to generate a static placement table (`ps`) and inter-core teleportation events.
2. **Sabre Orchestration Pass (Optional)**: If `core_topologies` are provided, the `MultiCoreOrchestrator` slices the global DAG into localized fragments based on `ps` and routes them using `SabreSwap`. The orchestrator linearly scans the newly routed schedules to extract a precise, sparse timeline of exactly when and where physical SWAPs were injected.
3. **Fast-path Re-estimation**: A dedicated Rust endpoint (`estimate_hardware_fidelity`) aggregates the `placements` and the SWAP timeline events to compute the final exponentially decaying `FidelityReport`.

### Complexity

Let:
- N = number of virtual qubits
- L = number of circuit layers
- P = number of physical SWAPs/Teleportations per layer

**Per layer:**

| Step | Cost |
|---|---|
| Algorithmic Gate Evaluation | O(N) |
| Sabre DAG Slicing & Routing | O(Python transpiler overhead) |
| Routing Overhead Aggregation | O(P) |
| Coherence Trace / Busy Time updates | O(N) |

**Total Rust Estimation Pass: O(L * N)**

The inner fidelity tensor estimator is fully parallelizable linear algebra executing in `< 1ms` for thousands of qubits, allowing lightning-fast hyperparameter sweeps over hardware noise values without re-running the costly `O(L * K^3)` HQA or SABRE mappers.

## Examples

All runnable scripts live in `examples/`. They share the helpers in `plot_fidelities_utils.py` (`SimulationConfig`, `simulate`, `build_core_mapping`, `build_all_to_all_core_mapping`).

| Script | Description |
|---|---|
| `plot_qft_fidelities.py` | Single QFT run — plots overall, algorithmic, routing, and coherence fidelity per qubit over time. |
| `plot_ghz_fidelities.py` | Same breakdown for a GHZ circuit. |
| `plot_qft_comparison.py` | Side-by-side comparison of **Random vs Spectral Clustering** placement on a QFT-30 circuit (grid+ring topology). |
| `plot_qft_topology_comparison.py` | Four-way comparison of **Grid+Ring vs Multi-core All-to-All vs Single Core All-to-All vs Single Core 2D Grid** connectivity, all using Spectral Clustering placement on a QFT-30 circuit. |
| `dse_qft_cores.py` | Design-space exploration sweeping number of cores for a fixed qubit budget. |
