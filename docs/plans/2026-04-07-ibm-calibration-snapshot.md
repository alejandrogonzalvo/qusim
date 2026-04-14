# IBM Q Calibration Snapshot & Live Benchmark Plan

**Goal:** Capture live per-qubit/per-gate calibration data from IBM Quantum hardware, save it for offline use with qusim's per-qubit error model, and optionally run benchmark circuits to validate fidelity predictions against real hardware.

**Prerequisite:** IBM Quantum account (free Open Plan). Sign up at `quantum.cloud.ibm.com`.

---

## Phase 0 — Setup

### 0.1 Install SDK

```bash
pip install qiskit-ibm-runtime
```

### 0.2 Save credentials (one-time)

Get your API token from `quantum.cloud.ibm.com` → Account settings → API token.

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="<YOUR_IQP_TOKEN>",
    overwrite=True,
    set_as_default=True,
)
```

### 0.3 Verify access

```python
service = QiskitRuntimeService()
backends = service.backends(simulator=False, operational=True)
print([b.name for b in backends])
# Expected: ['ibm_kingston'] or similar Heron r2 backend
```

---

## Phase 1 — Capture Calibration Snapshot (FREE, no runtime consumed)

Calling `backend.properties()` and `backend.target` queries metadata from IBM servers. It does **not** submit a job and does **not** consume the 10-minute runtime budget.

### 1.1 Script: `scripts/snapshot_calibration.py`

```python
"""Snapshot live IBM Q calibration data for offline qusim use."""
import json
import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_kingston")  # Heron r2, free tier

# --- Extract all calibration data ---
props = backend.properties()
num_qubits = backend.num_qubits

calibration = {
    "backend_name": backend.name,
    "num_qubits": num_qubits,
    "snapshot_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "qubits": {},
    "gates_1q": {},
    "gates_2q": {},
}

# Per-qubit: T1, T2, readout error, frequency
for q in range(num_qubits):
    calibration["qubits"][q] = {
        "t1_us": props.t1(q) * 1e6,       # seconds → microseconds
        "t2_us": props.t2(q) * 1e6,
        "readout_error": props.readout_error(q),
        "frequency_ghz": props.frequency(q) / 1e9,
    }

# Per-gate errors and durations from Target (BackendV2 interface)
target = backend.target
for gate_name in target.operation_names:
    for qargs, inst_props in target[gate_name].items():
        if inst_props is None:
            continue
        entry = {
            "error": inst_props.error,
            "duration_ns": inst_props.duration * 1e9 if inst_props.duration else None,
        }
        if len(qargs) == 1:
            calibration["gates_1q"].setdefault(gate_name, {})[str(qargs[0])] = entry
        elif len(qargs) == 2:
            key = f"{qargs[0]},{qargs[1]}"
            calibration["gates_2q"].setdefault(gate_name, {})[key] = entry

# Save
filename = f"data/calibration_{backend.name}_{datetime.date.today()}.json"
with open(filename, "w") as f:
    json.dump(calibration, f, indent=2, default=str)
print(f"Saved calibration snapshot: {filename} ({num_qubits} qubits)")
```

### 1.2 Also save the raw BackendProperties (for future-proofing)

```python
raw_props = props.to_dict()
with open(f"data/calibration_raw_{backend.name}_{datetime.date.today()}.json", "w") as f:
    json.dump(raw_props, f, indent=2, default=str)
```

> **Note:** `BackendProperties` is deprecated since Qiskit 1.2 (removal in 2.0). The `Target`-based extraction in the script above is forward-compatible.

---

## Phase 2 — Load Snapshot into qusim

### 2.1 Converter: snapshot → qusim params

```python
"""Load a calibration snapshot and convert to qusim parameters."""
import json
import numpy as np

def load_calibration(path: str, basis_1q: str = "sx", basis_2q: str = "cz"):
    """Convert a calibration snapshot JSON into qusim-compatible parameters.
    
    Args:
        path: Path to calibration JSON from Phase 1.
        basis_1q: Name of the native 1Q gate to use for error rates.
        basis_2q: Name of the native 2Q gate to use for error rates.
    
    Returns:
        dict with keys matching map_circuit() keyword arguments.
    """
    with open(path) as f:
        cal = json.load(f)

    num_qubits = cal["num_qubits"]

    # Per-qubit T1/T2 (convert μs → ns for qusim)
    t1_per_qubit = np.array([cal["qubits"][str(q)]["t1_us"] * 1e3 for q in range(num_qubits)])
    t2_per_qubit = np.array([cal["qubits"][str(q)]["t2_us"] * 1e3 for q in range(num_qubits)])

    # Per-qubit 1Q gate errors
    single_gate_error_per_qubit = np.zeros(num_qubits)
    gate_1q_data = cal["gates_1q"].get(basis_1q, {})
    for q in range(num_qubits):
        entry = gate_1q_data.get(str(q))
        if entry and entry["error"] is not None:
            single_gate_error_per_qubit[q] = entry["error"]

    # Per-pair 2Q gate errors
    two_gate_error_per_pair = {}
    gate_2q_data = cal["gates_2q"].get(basis_2q, {})
    for key, entry in gate_2q_data.items():
        if entry["error"] is not None:
            q1, q2 = map(int, key.split(","))
            two_gate_error_per_pair[(q1, q2)] = entry["error"]

    # Gate durations (use median as representative)
    durations_1q = [e["duration_ns"] for e in gate_1q_data.values() if e["duration_ns"]]
    durations_2q = [e["duration_ns"] for e in gate_2q_data.values() if e["duration_ns"]]

    return {
        "t1_per_qubit": t1_per_qubit,
        "t2_per_qubit": t2_per_qubit,
        "single_gate_error_per_qubit": single_gate_error_per_qubit,
        "two_gate_error_per_pair": two_gate_error_per_pair,
        # Use medians as scalar fallbacks for gates not in per-qubit/per-pair maps
        "single_gate_error": float(np.median(single_gate_error_per_qubit)),
        "two_gate_error": float(np.median(list(two_gate_error_per_pair.values()))) if two_gate_error_per_pair else 1e-3,
        "single_gate_time": float(np.median(durations_1q)) if durations_1q else 35.0,
        "two_gate_time": float(np.median(durations_2q)) if durations_2q else 660.0,
        "t1": float(np.median(t1_per_qubit)),
        "t2": float(np.median(t2_per_qubit)),
    }
```

### 2.2 Usage

```python
from qusim import map_circuit

hw_params = load_calibration("data/calibration_ibm_kingston_2026-04-07.json")
result = map_circuit(circuit=transp, full_coupling_map=cmap, core_mapping=core_map, **hw_params)
```

---

## Phase 3 — Live Benchmark (OPTIONAL, consumes runtime)

### 3.1 Budget estimation

| Item | QPU time |
|------|----------|
| Free tier budget | 10 min / 28 days |
| Mirror circuit, 10 qubits, 1000 shots | ~0.5-2 seconds |
| Mirror circuit, 50 qubits, 1000 shots | ~2-5 seconds |
| **Estimated runs per budget** | **100-300 circuits** |

### 3.2 Script: `scripts/benchmark_live.py`

Run the paper's mirror-circuit methodology on real hardware and compare against qusim predictions.

```python
"""Run mirror-circuit benchmarks on live IBM Q and compare with qusim."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qusim import map_circuit

service = QiskitRuntimeService()
backend = service.backend("ibm_kingston")

# Use a small set of circuits to conserve runtime
CIRCUITS = [
    ("qft_5", QFT(5)),
    ("qft_10", QFT(10)),
    ("ghz_10", make_ghz(10)),
]
SHOTS = 1000

results = []
for name, circ in CIRCUITS:
    # 1. Build mirror circuit (circ^-1 & circ → should return |0...0>)
    circ_inv = circ.inverse()
    circ_inv.barrier()
    mirror = circ_inv.compose(circ)
    mirror.measure_all()

    # 2. Transpile to backend's native gates
    transp = transpile(mirror, backend=backend, optimization_level=3)

    # 3. Run on hardware
    sampler = SamplerV2(mode=backend)
    job = sampler.run([transp], shots=SHOTS)
    counts = job.result()[0].data.meas.get_counts()
    
    # Success rate = fraction of |0...0> outcomes
    zero_state = "0" * circ.num_qubits
    hw_fidelity = counts.get(zero_state, 0) / SHOTS

    # 4. Run qusim prediction with live calibration
    hw_params = load_calibration(f"data/calibration_{backend.name}_latest.json")
    qusim_result = map_circuit(circuit=transp, ...)  # setup coupling map etc.

    results.append({
        "circuit": name,
        "hw_fidelity": hw_fidelity,
        "qusim_fidelity": qusim_result.overall_fidelity,
        "num_qubits": circ.num_qubits,
        "num_gates": len(transp.data),
    })

# Save results
import json
with open("data/benchmark_live_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 3.3 What to save

After a live run, persist these artifacts in `data/`:

```
data/
├── calibration_ibm_kingston_2026-04-07.json       # Phase 1 snapshot
├── calibration_raw_ibm_kingston_2026-04-07.json    # Raw BackendProperties
├── benchmark_live_results.json                     # Phase 3 results
└── transpiled_circuits/                            # QASM of transpiled circuits
    ├── qft_5_ibm_kingston.qasm
    └── qft_10_ibm_kingston.qasm
```

Saving the transpiled QASM lets you re-run qusim predictions offline with different calibration snapshots without re-transpiling.

---

## Phase 4 — Integrate into Benchmark Suite

Update `examples/benchmark_against_paper.py` to optionally load a live calibration snapshot instead of hardcoded uniform error rates:

```python
if args.calibration:
    hw_params = load_calibration(args.calibration)
else:
    # Fall back to current uniform rates
    hw_params = {"single_gate_error": 2.3e-4, ...}
```

This makes the benchmark reproducible: anyone with the saved snapshot can re-run the comparison without IBM Q access.

---

## Execution Order

| Step | Consumes runtime? | Dependency |
|------|-------------------|------------|
| Phase 0: Install + auth | No | IBM Q account |
| Phase 1: Snapshot calibration | **No** | Phase 0 |
| Phase 2: Build loader + test offline | **No** | Phase 1 |
| Phase 3: Live benchmark | **Yes** (~1-2 min) | Phase 1 + Phase 2 |
| Phase 4: Integrate into benchmark suite | No | Phase 2 |

**Recommended:** Do Phases 0-2 first (all free). Only run Phase 3 once you've verified qusim predictions look reasonable with the calibration snapshot. This way you spend zero runtime on debugging.
