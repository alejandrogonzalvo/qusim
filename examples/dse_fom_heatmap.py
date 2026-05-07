"""
DSE example #3 — custom Figure-of-Merit heatmap over a noise grid.

Compiles the circuit once, then sweeps a 2-D noise grid (T1 × 2Q error)
and evaluates a *user-defined* Figure of Merit at every point. The FoM
expression is parsed and validated against the same safe AST whitelist
the GUI uses — arithmetic + ``log/exp/sqrt/min/max/abs/clip/pow`` only.

Run::

    python examples/dse_fom_heatmap.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quadris.analysis import FomConfig, compute_for_sweep
from quadris.dse import DSEEngine, NOISE_DEFAULTS


def main() -> None:
    engine = DSEEngine()

    cached = engine.run_cold(
        circuit_type="qft",
        num_qubits=24,
        num_cores=4,
        topology_type="ring",
        intracore_topology="all_to_all",
        placement_policy="spectral",
        seed=0,
        communication_qubits=2,
        buffer_qubits=1,
    )

    # 2-D noise sweep: T1 (log) × 2Q gate error (log).
    # Endpoints are log10 exponents because both axes have log_scale=True.
    xs, ys, grid = engine.sweep_2d(
        cached=cached,
        metric_key1="t1",                 low1=4.0, high1=6.5,
        metric_key2="two_gate_error",     low2=-5.0, high2=-2.0,
        fixed_noise=dict(NOISE_DEFAULTS),
    )

    # Build the sweep_data dict the FoM evaluator expects (same shape as
    # SweepResult.to_sweep_data() for a 2-D sweep).
    sweep_data = {
        "metric_keys": ["t1", "two_gate_error"],
        "xs": list(xs),
        "ys": list(ys),
        "grid": grid,
    }

    # Custom FoM: log(1/infidelity) per EPR pair — a "fidelity yield" proxy.
    # Intermediates make the formula readable; the engine vectorises it.
    fom = FomConfig(
        name="Fidelity yield",
        intermediates=(
            ("infidelity", "max(1 - overall_fidelity, 1e-12)"),
        ),
        numerator="log(1 / infidelity)",
        denominator="max(total_epr_pairs, 1)",
    )

    result = compute_for_sweep(sweep_data, fom)
    if result.error:
        raise SystemExit(f"FoM evaluation failed: {result.error}")

    # FoM values come back flattened over the sweep grid; reshape for the heatmap.
    z = result.values.reshape(len(xs), len(ys)).T   # (ys, xs) for imshow

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        cmap="RdYlBu",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T1 (ns)")
    ax.set_ylabel("Two-qubit gate error")
    ax.set_title(f"QFT-24 on 4-core ring — {fom.name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{fom.numerator}  /  {fom.denominator}")
    fig.tight_layout()

    out = "examples/dse_fom_heatmap.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
