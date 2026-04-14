import pytest
import qiskit
import numpy as np
from qusim import map_circuit
from qusim.hqa.placement import InitialPlacement
from qiskit.transpiler import CouplingMap

class TestGateTypeBehavior:
    def test_rz_has_zero_error_and_duration_by_default(self):
        circ = qiskit.QuantumCircuit(4)
        circ.rz(1.0, 0)
        circ.rz(1.0, 1)

        core_mapping = {0:0, 1:0, 2:0, 3:0}
        full_coupling_map = CouplingMap.from_line(4)

        # Baseline: High single gate error.
        result = map_circuit(
            circ, full_coupling_map, core_mapping,
            single_gate_error=0.1,
            single_gate_time=1000.0,
            initial_placement=InitialPlacement.RANDOM
        )

        assert result.algorithmic_fidelity == 1.0, "rz should natively incur no error"
        assert result.total_circuit_time_ns == 0.0, "rz should natively incur no duration"

    def test_gate_type_overrides_scalar(self):
        circ = qiskit.QuantumCircuit(4)
        circ.sx(0) # 1Q 
        circ.h(1)  # 1Q

        core_mapping = {0:0, 1:0, 2:0, 3:0}
        full_coupling_map = CouplingMap.from_line(4)

        result = map_circuit(
            circ, full_coupling_map, core_mapping,
            single_gate_error=0.0,
            gate_error_per_type={"sx": 0.5, "h": 0.0},
            initial_placement=InitialPlacement.RANDOM
        )

        # fidelity 1 - 0.5 = 0.5 for sx, h=0.0 error
        assert result.algorithmic_fidelity == 0.5, f"Expected 0.5 alg fidelity, got {result.algorithmic_fidelity}"
