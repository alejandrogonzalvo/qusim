# qusim

Multi-core quantum architecture simulator with a built-in Design Space
Exploration (DSE) toolkit. A Rust core (PyO3-bound) does the heavy
lifting; a GUI-agnostic Python library (`qusim`, `qusim.dse`,
`qusim.analysis`) drives both interactive exploration and headless
scripts; an optional Dash app (`qusim-dse`) puts a UI on top.

```
┌──────── Dash GUI (optional)  ────────┐
│  qusim-dse                            │   pip install qusim[gui]
└─────────────────┬────────────────────┘
                  │   imports
┌─────────────────▼────────────────────┐
│  Python library (qusim, qusim.dse,    │   pip install qusim
│  qusim.analysis)                      │
└─────────────────┬────────────────────┘
                  │   PyO3
┌─────────────────▼────────────────────┐
│  Rust core (HQA, noise, routing,      │
│  TeleSABRE FFI)                       │
└──────────────────────────────────────┘
```

For the layered diagram, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

```bash
# Set up a venv and the maturin build chain
python3 -m venv .venv && source .venv/bin/activate
pip install maturin

# Build the Rust extension and install qusim in editable mode
maturin develop --release

# Optional: pull in the Dash GUI
pip install -e ".[gui]"
```

`pip install qusim` is headless — Dash isn't a runtime dependency. Add
`[gui]` (or `[dev]` for the test/docs tooling) only when you need them.

## What's in the box

| Package | What it gives you | Read more |
|---|---|---|
| `qusim` | `map_circuit`, `telesabre_map_circuit`, `estimate_fidelity_from_cache[_batch]`, `QusimResult` | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| `qusim.dse` | `DSEEngine`, `SweepResult`, `SWEEPABLE_METRICS`, `NOISE_DEFAULTS`, parameter registry | [`python/qusim/dse/README.md`](python/qusim/dse/README.md) |
| `qusim.analysis` | `FomConfig`, `compute_for_sweep`, `pareto_front`, `pareto_front_mask` | [`python/qusim/analysis/README.md`](python/qusim/analysis/README.md) |
| `qusim.hqa` | Initial-placement policies (random, spectral clustering) | — |
| GUI app | Interactive DSE explorer (Dash) | [`gui/README.md`](gui/README.md) |
| Rust core | HQA, noise model, routing, TeleSABRE | [`src/hqa/README.md`](src/hqa/README.md), [`src/noise/README.md`](src/noise/README.md) |

## Single-circuit example

```python
import qiskit
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

import qusim
from qusim.hqa.placement import InitialPlacement

# 1. Transpile to the basis the noise model expects.
circ = QFT(30)
transp = transpile(
    circ,
    basis_gates=["x", "cx", "cp", "rz", "h", "s", "sdg", "t", "tdg", "measure"],
    optimization_level=0, seed_transpiler=42,
)

# 2. Describe the device: which physical qubit lives on which core, and how
#    the cores are wired. (The GUI / qusim.dse builders compute this for you.)
num_cores, qubits_per_core = 5, 6
core_mapping = {q: q // qubits_per_core for q in range(num_cores * qubits_per_core)}
edges = [(c, (c + 1) % num_cores) for c in range(num_cores)]   # ring of cores
intra = [(c * qubits_per_core + i, c * qubits_per_core + j)
         for c in range(num_cores)
         for i in range(qubits_per_core) for j in range(i + 1, qubits_per_core)]
coupling = CouplingMap.from_edge_list(intra + [
    (a * qubits_per_core, b * qubits_per_core) for a, b in edges
])

# 3. Run HQA + SABRE + fidelity estimation.
result: qusim.QusimResult = qusim.map_circuit(
    circuit=transp,
    full_coupling_map=coupling,
    core_mapping=core_mapping,
    seed=42,
    initial_placement=InitialPlacement.SPECTRAL_CLUSTERING,
    single_gate_error=1e-4, two_gate_error=1e-3, t1=100_000.0,
)

print(f"Overall fidelity: {result.overall_fidelity:.4f}")
print(f"EPR pairs:        {result.total_epr_pairs}")
print(f"SABRE swaps:      {result.total_swaps}")

# Per-qubit / per-layer fidelity grids — shape (num_layers, num_qubits)
algo_grid    = result.algorithmic_fidelity_grid
routing_grid = result.routing_fidelity_grid
coh_grid     = result.coherence_fidelity_grid
```

## DSE example (parameter sweep)

```python
from qusim.dse import DSEEngine, NOISE_DEFAULTS

engine = DSEEngine()
cached = engine.run_cold(
    circuit_type="qft", num_qubits=32, num_cores=4,
    topology_type="ring", intracore_topology="all_to_all",
    placement_policy="spectral", seed=0,
    communication_qubits=2, buffer_qubits=1,
)
xs, results = engine.sweep_1d(
    cached=cached, metric_key="two_gate_error",
    low=-5.0, high=-1.0, fixed_noise=dict(NOISE_DEFAULTS),
)
print({"xs": xs[:3], "fidelities": [r["overall_fidelity"] for r in results[:3]]})
```

See [`examples/`](examples/) for three full end-to-end scripts:

| Script | Demonstrates |
|---|---|
| [`dse_1d_noise_sweep.py`](examples/dse_1d_noise_sweep.py) | Cold compile once, hot-sweep 2Q gate error |
| [`dse_2d_pareto.py`](examples/dse_2d_pareto.py) | (cores × comm-qubits) sweep with Pareto frontier |
| [`dse_fom_heatmap.py`](examples/dse_fom_heatmap.py) | User-defined Figure-of-Merit on a 2-D noise grid |

## DSE GUI

```bash
pip install -e ".[gui]"
qusim-dse              # → http://127.0.0.1:8050
```

On startup it auto-runs a 3-D sweep (T1 × T2 × 2Q gate time) on a
4-core ring. From there you can swap circuit / topology / routing
algorithm, drop any of the 20 swept metrics on up to 6 axes, and pick
between line / heatmap / 3-D / parallel-coords / Pareto / FoM views —
all rendered in your browser. See [`gui/README.md`](gui/README.md) for
the full layout, [`docs/DSE_VIEWS.md`](docs/DSE_VIEWS.md) for the view
catalog, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the
GUI sits on top of the library.

## Cold path vs. hot path

DSE separates two cost regimes:

- **Cold path** — circuit transpilation + HQA + SABRE routing, ~1–10 s
  per configuration. Required when *structural* parameters change
  (circuit type, qubit count, topology, comm/buffer-qubit count,
  placement policy, routing algorithm).
- **Hot path** — fidelity re-estimation from cached structural data,
  <1 ms per noise config. Used for sweeps over T1/T2, gate errors,
  EPR rates, etc.

The cold path runs once per unique (cold-cfg) tuple; hot-path
evaluations are batched through a single Rust call. For multi-cell
sweeps, an empirical RAM model schedules cold compilations across a
forkserver pool so concurrent workers never exceed `MemAvailable`.

## Noise model

Three disjoint error mechanisms tracked per layer per qubit:

1. **Algorithmic** — depolarising error per native gate.
2. **Routing** — SABRE-injected SWAPs (3 × CNOT each) plus inter-core
   teleportation cost (1 EPR + Bell measurement + buffer SWAP, derived
   from constituent gate / EPR / measurement error rates by default).
3. **Coherence** — exponential T1/T2 decay over per-qubit busy/idle time.

Full equations + Rust implementation pointers in
[`src/noise/README.md`](src/noise/README.md).

## Tests

```bash
pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

The repo also has `cargo test` for the Rust crate and `cargo bench`
for the HQA / noise hot loops.

## Project layout

```
csrc/             # Vendored TeleSABRE C library
docs/             # Long-form docs (this README links into them)
examples/         # End-to-end Python scripts (library only)
gui/              # Dash app (qusim-dse). Re-exports library names for back-compat.
python/qusim/     # Public Python package
  __init__.py        # map_circuit, telesabre_map_circuit, ...
  dse/               # DSEEngine, sweeps, parameter registry, backends
  analysis/          # FoM evaluator + Pareto frontier
  hqa/               # Initial-placement policies
src/              # Rust crate (PyO3 bindings, HQA, noise, routing, telesabre)
tests/            # Python test suite
```

## License & citation

If you use `qusim` for academic work, the HQA algorithm is based on
Pau Escofet *et al.*, "Hungarian Qubit Assignment for Optimized Mapping
of Quantum Circuits on Multi-Core Architectures"
([arXiv:2309.12182](https://arxiv.org/pdf/2309.12182)).
