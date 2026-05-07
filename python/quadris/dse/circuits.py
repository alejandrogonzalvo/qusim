"""
Qiskit circuit constructors used by the cold path.

QFT / GHZ / random-circuit builders, plus the project's standard
transpile pass (basis gates pinned for deterministic gate counts).
"""

from __future__ import annotations

import qiskit
from qiskit import transpile
from qiskit.circuit.library import QFT

def _build_circuit(
    circuit_type: str,
    num_qubits: int,
    seed: int,
    qasm_str: str | None = None,
) -> qiskit.QuantumCircuit:
    if circuit_type == "custom":
        if not qasm_str:
            raise ValueError("circuit_type='custom' requires a non-empty qasm_str")
        from qiskit import qasm2
        return qasm2.loads(qasm_str)
    if circuit_type == "qft":
        circ = QFT(num_qubits)
    elif circuit_type == "ghz":
        circ = qiskit.QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(0, i)
    elif circuit_type == "random":
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(num_qubits, depth=max(3, num_qubits // 2), seed=seed)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    return circ


def _transpile_circuit(circ: qiskit.QuantumCircuit, seed: int) -> qiskit.QuantumCircuit:
    return transpile(
        circ,
        basis_gates=["x", "cx", "cp", "rz", "h", "s", "sdg", "t", "tdg", "measure"],
        optimization_level=0,
        seed_transpiler=seed,
    )
