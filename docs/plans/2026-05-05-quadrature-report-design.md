# Quadrature Report — design

**Audience.** Alejandro's supervisor, who will mine this document for
the Quadrature project deliverable.

**Output.** A single markdown file at `docs/QUADRATURE_REPORT.md`,
≤10 pages, optimised for paste-able paragraphs / tables / mermaid
figures rather than narrative polish.

**Naming.** The package is referred to as `quadris` throughout. A
note up front says the name is provisional and the codebase currently
exposes it as `qusim`.

## Approved scope (after debate)

- **Audience-aware cuts.** No context/motivation (supervisor provides
  it), no recent-refactor section (internal history), no roadmap /
  open work, no contributor-only details (worktrees, deployment
  branches, packaging shims, GUI shims).
- **Emphasis: measured numbers > complexity proofs.** §3 (algorithms
  & complexity) is shorter and more table-driven; §5 (performance &
  benchmarks) is the headline section.
- **Code excerpts.** Short pseudocode for HQA, FFI signatures, the
  noise equations as LaTeX. No large verbatim source blocks.
- **Diagrams.** Mermaid + ASCII fall-backs (paste-able into TikZ).
- **DSE I/O is a first-class focus** — under approved feedback the
  supervisor will use it heavily.

## Approved table of contents (10-page target, ~5 000 words)

```
0.  Executive summary                                        ½ p
1.  Functional capabilities                                  1½ p
    1.1 Single-circuit mapping
    1.2 Hot-path fidelity re-estimation
    1.3 DSE: inputs, outputs, sweep model        ★ supervisor focus
    1.4 Custom Figures of Merit
    1.5 Pareto frontier analysis
    1.6 Routing-backend selection (HQA+SABRE / TeleSABRE)
    1.7 Interactive GUI (view catalogue summary)
2.  System architecture                                      1½ p
    2.1 Three-layer model (Rust core / Python lib / Dash GUI)
    2.2 Module map (directory → responsibility)
    2.3 Two-stage execution model (cold / hot)
    2.4 Public API surface (one-line per entry)
3.  Algorithms & computational complexity                    2½ p ★ slimmed
    3.1 HQA + lookahead + Hungarian matching
    3.2 Initial-placement policies
    3.3 SABRE intra-core routing
    3.4 TeleSABRE unified routing (vendored — integration model)
    3.5 Noise model (algorithmic / routing / coherence)
    3.6 Teleportation cost decomposition
    3.7 Sweep-axis count derivation (split-budget model)
    3.8 Pareto front + FoM evaluation
4.  Implementation overview                                  ¾ p   (basics)
    Rust crate, PyO3 surface, Python packaging, backend strategy,
    sweep orchestration. Cross-link to per-package READMEs for depth.
5.  Performance & benchmarks                                 2 p   ★ headline
    Test bench: this host (CPU, RAM, OS).
    5.1 Cold-path latency vs. num_qubits
    5.2 Cold-path peak RSS vs. num_qubits
    5.3 Hot-path single vs. batched
    5.4 Parallel cold-pool scaling (1 / 2 / 4 / 8 workers)
    5.5 Mapping quality vs. HQA paper reference
    5.6 TeleSABRE vs. HQA+SABRE
    5.7 Sweep-grid memory: structured vs. dict
6.  GUI overview                                             ½ p
7.  Validation & testing                                     ½ p
Appendix A  File index                                       ½ p
Appendix B  Public API signatures                            ½ p
Appendix C  Glossary                                         margins
```

## Benchmarks I will run

Run on the dev host (described in §5 caption); 5 reps each, median
reported with min/max bounds where wall-clock variance > 10 %.

| # | Benchmark | Source |
|---|---|---|
| 1 | Cold-path latency at num_qubits ∈ {4, 16, 32, 64, 128, 256} | new bench script |
| 2 | Cold-path peak RSS at the same points | extends `tests/test_cores_qubits_oom.py` |
| 3 | Hot-path single `run_hot` latency | new bench script |
| 4 | Hot-path batched `run_hot_batch` (1, 10, 100, 1000) | new bench script |
| 5 | Parallel cold-pool scaling (1, 2, 4, 8 workers) | extends `tests/bench_parallel_sweep.py` |
| 6 | Mapping quality vs. HQA paper | uses `examples/benchmark_against_paper.py` |
| 7 | TeleSABRE vs. HQA+SABRE on QFT/GHZ | uses `tests/bench_telesabre_vs_hqa.py` |
| 8 | Sweep-grid cell memory: structured (~64 B) vs. dict (~280 B) | computed + verified on 1 000-cell grid |

Bench scripts live under `tests/` (existing convention). One new
runner `tests/bench_quadrature_report.py` produces a deterministic
text dump that the report quotes from.

## QPC refactor (relevant; will be mentioned)

The codebase has a partially-landed refactor that anchors
``num_logical_qubits`` (algorithm size) and pins exactly one of
``num_cores`` / ``qubits_per_core``; the other is *derived* by
``qusim.dse.config._resolve_architecture`` so the architecture
absorbs the comm/buffer overhead without changing the logical qubit
count across sweep cells.

This is exactly what the supervisor wants reflected under "DSE
inputs / outputs", so §1.3 will describe the pinned-axis model with
a short table:

| Input axis | Constancy | Notes |
|---|---|---|
| `num_logical_qubits` (L) | constant during a sweep | algorithm size |
| pin axis ∈ {`num_cores`, `qubits_per_core`} | one is pinned | other is derived |
| comm/buffer config | derived from K, B, topology | absorbed by the unpinned axis |
| noise dict | per-cell / per-axis | hot path |

The report will note this is "the model the engine resolves to";
implementation-level state ("partial refactor in flight") will not
appear (per "skip the refactor" cut).

## Single-output deliverable

`docs/QUADRATURE_REPORT.md` — committed once the benchmarks are
captured. The plan stays under `docs/plans/` per repo convention.

## Out of scope

- Worktrees, deployment branches, `gui/` shims, packaging extras
  (covered only as far as "the GUI is optional").
- The 2026-05 refactor narrative (lift / leaf split / backend
  strategy as a story).
- Open roadmap items.
