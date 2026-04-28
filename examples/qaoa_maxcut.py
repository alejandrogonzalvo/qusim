"""
Generate a logical OpenQASM 2.0 file for QAOA / MaxCut on a ring graph.

The output is written next to this script as ``qaoa_maxcut_ring8.qasm``
and is intended to be uploaded via the Quadris DSE GUI's "Custom circuit"
control. The circuit is *logical* — no transpilation, no coupling map,
no SWAP insertion — so Quadris can do its own placement and routing.
"""

from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit, qasm2


def qaoa_maxcut_ring(num_nodes: int, p: int, gammas: list[float], betas: list[float]) -> QuantumCircuit:
    """QAOA-MaxCut on a ring graph with ``p`` alternating layers."""
    if len(gammas) != p or len(betas) != p:
        raise ValueError("len(gammas) and len(betas) must equal p")

    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    qc = QuantumCircuit(num_nodes, name=f"qaoa_maxcut_ring{num_nodes}_p{p}")

    # Initial state |+>^n
    for q in range(num_nodes):
        qc.h(q)

    for layer in range(p):
        # Cost layer: e^{-i γ Σ_{(i,j)∈E} (I - Z_i Z_j)/2}
        # Up to a global phase, each edge term decomposes into CX-RZ-CX.
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2.0 * gammas[layer], j)
            qc.cx(i, j)

        # Mixer layer: e^{-i β Σ_i X_i} = ⊗ RX(2β)
        for q in range(num_nodes):
            qc.rx(2.0 * betas[layer], q)

    return qc


def main() -> None:
    num_nodes = 8
    p = 1
    # Near-optimal QAOA-1 angles for ring MaxCut.
    gammas = [0.39269908]   # π/8
    betas = [0.39269908]    # π/8

    qc = qaoa_maxcut_ring(num_nodes, p, gammas, betas)
    qasm_str = qasm2.dumps(qc)

    out = Path(__file__).with_name(f"qaoa_maxcut_ring{num_nodes}.qasm")
    out.write_text(qasm_str)
    print(f"Wrote {out} — {qc.num_qubits} qubits, depth {qc.depth()}")


if __name__ == "__main__":
    main()
