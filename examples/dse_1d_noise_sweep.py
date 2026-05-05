"""
DSE example #1 — 1-D noise sweep on a fixed mapping.

The cheapest possible DSE: compile a circuit *once*, then re-evaluate
fidelity for many noise configurations. Demonstrates the cold-vs-hot
path split — the cold compilation runs in seconds, the hot sweep in
milliseconds per point.

Run::

    python examples/dse_1d_noise_sweep.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from qusim.dse import DSEEngine, NOISE_DEFAULTS


def main() -> None:
    engine = DSEEngine()

    # Cold path — once. Builds the circuit, the topology, and the HQA+SABRE
    # mapping; the resulting CachedMapping is reused by every hot evaluation.
    cached = engine.run_cold(
        circuit_type="qft",
        num_qubits=32,
        num_cores=4,
        topology_type="ring",
        intracore_topology="all_to_all",
        placement_policy="spectral",
        seed=0,
        communication_qubits=2,
        buffer_qubits=1,
    )
    print(
        f"Cold compilation: {cached.cold_time_s:.2f}s "
        f"({cached.total_teleportations} teleports, "
        f"{cached.total_swaps} swaps, "
        f"{cached.total_epr_pairs} EPR pairs)"
    )

    # Hot path — sweep two-qubit gate error from 1e-5 to 1e-1.
    # The metric is log_scale (see qusim.dse.axes), so endpoints are exponents.
    # Default count is SWEEP_POINTS_1D = 60.
    xs, results = engine.sweep_1d(
        cached=cached,
        metric_key="two_gate_error",
        low=-5.0,
        high=-1.0,
        fixed_noise=dict(NOISE_DEFAULTS),
    )

    fids = np.asarray([r["overall_fidelity"] for r in results])
    routing = np.asarray([r["routing_fidelity"] for r in results])
    coherence = np.asarray([r["coherence_fidelity"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(xs, fids, marker="o", linewidth=1.6, color="#2B2B2B",
                label="Overall")
    ax.semilogx(xs, routing, linewidth=1.2, color="#d73027",
                linestyle="--", label="Routing")
    ax.semilogx(xs, coherence, linewidth=1.2, color="#4575b4",
                linestyle="--", label="Coherence")
    ax.set_xlabel("Two-qubit gate error rate")
    ax.set_ylabel("Fidelity")
    ax.set_title("QFT-32 on 4-core ring — sensitivity to 2Q gate error")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out = "examples/dse_1d_noise_sweep.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
