"""
DSE example #2 — 2-D architectural sweep with Pareto frontier.

Sweeps the design space ``(num_cores × communication_qubits)`` for a fixed
QFT circuit, then plots the Pareto frontier in (EPR pairs vs. overall
fidelity) — the textbook DSE output: which (cores, K) configurations are
not dominated by any other?

Each cell of the grid runs a fresh cold compilation, so this script
takes longer than the 1-D example. Reduce the bounds (or the engine's
``MAX_COLD_COMPILATIONS``) for a quick smoke test.

Run::

    python examples/dse_2d_pareto.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qusim.analysis import pareto_front
from qusim.dse import DSEEngine, NOISE_DEFAULTS, SweepProgress


NUM_QUBITS = 32


def main() -> None:
    engine = DSEEngine()

    cold_config = {
        "circuit_type": "qft",
        "num_qubits": NUM_QUBITS,
        "num_cores": 1,                    # overridden by sweep_axes
        "topology_type": "ring",
        "intracore_topology": "all_to_all",
        "placement_policy": "spectral",
        "routing_algorithm": "hqa_sabre",
        "seed": 0,
        "communication_qubits": 1,         # overridden by sweep_axes
        "buffer_qubits": 1,
    }

    def progress(p: SweepProgress) -> None:
        print(f"  [{p.completed}/{p.total}] cold {p.cold_completed}/{p.cold_total}")

    # Cold-path axes use (low, high) integer endpoints; the engine clamps to
    # the per-axis budget derived from MAX_COLD_COMPILATIONS.
    sweep = engine.sweep_nd(
        cached=None,
        sweep_axes=[
            ("num_cores",            1, 8),
            ("communication_qubits", 1, 4),
        ],
        fixed_noise=dict(NOISE_DEFAULTS),
        cold_config=cold_config,
        progress_callback=progress,
        parallel=True,
        max_workers=2,
    )
    print(f"Sweep complete: shape={sweep.shape}, total={sweep.total_points} points")

    # pareto_front works directly on the SweepResult.
    front = pareto_front(
        sweep,
        objective_x="total_epr_pairs",   # minimise
        objective_y="overall_fidelity",  # maximise
    )

    x = front["x"]
    y = front["y"]
    mask = front["mask"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x[~mask], y[~mask], s=40, alpha=0.45, color="#888888",
               label="dominated")
    ax.scatter(x[mask], y[mask], s=80, color="#d73027",
               edgecolors="black", linewidths=0.5, label="Pareto-optimal")

    # Annotate Pareto points with their (cores, K) coordinates.
    for i in np.where(mask)[0]:
        cores = int(front["axes"]["num_cores"][i])
        comm = int(front["axes"]["communication_qubits"][i])
        ax.annotate(
            f"  C={cores}, K={comm}",
            (x[i], y[i]),
            fontsize=8,
            color="#2B2B2B",
        )

    ax.set_xlabel("Total EPR pairs (lower is better)")
    ax.set_ylabel("Overall fidelity (higher is better)")
    ax.set_title(
        f"QFT-{NUM_QUBITS} — Pareto frontier over (cores × comm-qubits)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()

    out = "examples/dse_2d_pareto.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out} — {int(mask.sum())} Pareto-optimal configurations")


if __name__ == "__main__":
    main()
