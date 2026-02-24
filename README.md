# qusim: Quantum Architecture HQA Mapping & Fidelity Estimation

`qusim` is a high-performance simulator written in Rust with native Python bindings. It maps Qiskit quantum circuits onto multi-core modular quantum architectures using a specialized Heuristic Quantum Architecture (HQA) mapping algorithm, and then performs fine-grained hardware noise and fidelity estimation.

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
