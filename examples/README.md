# `examples/`

Runnable scripts showing the `qusim` library API end-to-end. Every
script here is *library-only* — none of them import from `gui/`, so
they double as a smoke test that the library is decoupled from the
Dash app.

Run any script with the project venv active:

```bash
.venv/bin/python examples/dse_1d_noise_sweep.py
```

## DSE workflows (the new public API)

| Script | What it shows | Cold compiles |
|---|---|---|
| [`dse_1d_noise_sweep.py`](dse_1d_noise_sweep.py) | Compile once, sweep 2-Q gate error from 1e-5 → 1e-1; plot overall / routing / coherence fidelity | 1 |
| [`dse_2d_pareto.py`](dse_2d_pareto.py)           | Architectural sweep over `(num_cores × communication_qubits)`; compute Pareto frontier in (EPR pairs, fidelity); annotate Pareto-optimal configurations | up to 32 |
| [`dse_fom_heatmap.py`](dse_fom_heatmap.py)       | Compile once, sweep `T1 × two_gate_error`; evaluate a user-defined Figure-of-Merit (`log(1/infidelity) / max(EPR_pairs, 1)`) and render a heatmap | 1 |

These three are the canonical reference for "how do I drive `qusim`
from a script?" — they cover hot-path sweeps, cold-path sweeps,
custom FoM, and Pareto analysis.

## Validation / paper benchmarks

Older scripts that pre-date the `qusim.dse` package, kept for
reproducibility of published results.

| Script | What it shows |
|---|---|
| `benchmark_against_paper.py` | Validates `qusim` fidelity against the HQA paper's reference numbers |
| `benchmark_validation.py`     | Cross-validation against an independent IBM-Q reference |
| `benchmark_real_hw.py`        | Compares against IBM Q calibration data |
| `benchmark_live_hw.py`        | Live IBM Q backend run |
| `benchmark_telesabre_vs_hqa.py` | HQA+SABRE vs. TeleSABRE on the same circuits |
| `dse_qft_cores.py`            | Pre-`qusim.dse` core-count sweep on QFT (manual loop) |

## Per-qubit fidelity plots

| Script | What it shows |
|---|---|
| `plot_qft_fidelities.py` | Per-qubit overall / algorithmic / routing / coherence curves for QFT |
| `plot_ghz_fidelities.py` | Same, on GHZ |
| `plot_qft_comparison.py` | Random vs. spectral-clustering placement, side-by-side |
| `plot_qft_topology_comparison.py` | Grid+ring vs. all-to-all (multi-core) vs. single-core all-to-all vs. single-core grid |
| `plot_fidelities_utils.py` | Shared helpers (`SimulationConfig`, `simulate`, core-mapping builders) used by the four `plot_*` scripts above |

## QASM fixtures

`qaoa_maxcut_ring8.qasm` is a small QAOA-MaxCut circuit (8 qubits, ring
topology) used by `qaoa_maxcut.py` to demonstrate routing on a custom
QASM input.

## See also

- [`../python/qusim/dse/README.md`](../python/qusim/dse/README.md) — DSE library reference.
- [`../python/qusim/analysis/README.md`](../python/qusim/analysis/README.md) — Figure-of-Merit + Pareto.
- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — layered diagram.
