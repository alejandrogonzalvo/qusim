"""
Tests for per-qubit single-gate and per-pair two-gate error support.

Regression: uniform per-qubit arrays must produce identical fidelity to scalar params.
New behavior: degraded qubits/pairs must produce lower fidelity.
"""
import sys
import os
import numpy as np
import pytest

import qiskit
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from qusim import map_circuit, InitialPlacement


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_qft_setup():
    """5-qubit QFT on a single all-to-all core (no routing overhead)."""
    nq = 5
    circ = QFT(nq)
    transp = qiskit.transpile(
        circ,
        basis_gates=['x', 'cx', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'measure'],
        optimization_level=0,
        seed_transpiler=42,
    )

    edges = []
    for i in range(nq):
        for j in range(i + 1, nq):
            edges.append((i, j))
            edges.append((j, i))
    coupling_map = CouplingMap(edges)
    core_mapping = {i: 0 for i in range(nq)}

    return transp, coupling_map, core_mapping, nq


SCALAR_1Q_ERROR = 2.3e-4
SCALAR_2Q_ERROR = 8.2e-3
COMMON_KWARGS = dict(
    seed=42,
    initial_placement=InitialPlacement.RANDOM,
    single_gate_error=SCALAR_1Q_ERROR,
    two_gate_error=SCALAR_2Q_ERROR,
    teleportation_error_per_hop=0.0,
    single_gate_time=35.0,
    two_gate_time=660.0,
    teleportation_time_per_hop=1000.0,
    t1=120_000.0,
    t2=116_000.0,
)


# ---------------------------------------------------------------------------
# Regression: uniform arrays == scalar
# ---------------------------------------------------------------------------

class TestUniformRegression:
    """Uniform per-qubit/per-pair arrays must produce identical results to scalar params."""

    def test_uniform_per_qubit_1q_matches_scalar(self, small_qft_setup):
        transp, cmap, core_map, nq = small_qft_setup

        scalar_result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            **COMMON_KWARGS,
        )

        uniform_1q = np.full(nq, SCALAR_1Q_ERROR)
        per_qubit_result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            single_gate_error_per_qubit=uniform_1q,
            **COMMON_KWARGS,
        )

        assert scalar_result.algorithmic_fidelity == pytest.approx(
            per_qubit_result.algorithmic_fidelity, abs=1e-12
        ), "uniform per-qubit 1Q errors must match scalar"

        assert scalar_result.overall_fidelity == pytest.approx(
            per_qubit_result.overall_fidelity, abs=1e-12
        ), "overall fidelity must match"

    def test_uniform_per_pair_2q_matches_scalar(self, small_qft_setup):
        transp, cmap, core_map, nq = small_qft_setup

        scalar_result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            **COMMON_KWARGS,
        )

        # Build a per-pair dict with the scalar value for every possible pair
        pair_errors = {}
        for i in range(nq):
            for j in range(nq):
                if i != j:
                    pair_errors[(i, j)] = SCALAR_2Q_ERROR

        per_pair_result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            two_gate_error_per_pair=pair_errors,
            **COMMON_KWARGS,
        )

        assert scalar_result.algorithmic_fidelity == pytest.approx(
            per_pair_result.algorithmic_fidelity, abs=1e-12
        ), "uniform per-pair 2Q errors must match scalar"

        assert scalar_result.overall_fidelity == pytest.approx(
            per_pair_result.overall_fidelity, abs=1e-12
        ), "overall fidelity must match"

    def test_none_per_qubit_same_as_omitted(self, small_qft_setup):
        """Passing None explicitly must be identical to not passing at all."""
        transp, cmap, core_map, nq = small_qft_setup

        result_omitted = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            **COMMON_KWARGS,
        )

        result_none = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            single_gate_error_per_qubit=None,
            two_gate_error_per_pair=None,
            **COMMON_KWARGS,
        )

        assert result_omitted.overall_fidelity == pytest.approx(
            result_none.overall_fidelity, abs=1e-12
        )


# ---------------------------------------------------------------------------
# New behavior: per-qubit/per-pair errors
# ---------------------------------------------------------------------------

class TestPerQubitBehavior:
    """Degraded per-qubit/per-pair errors must lower fidelity."""

    def test_degraded_qubit_lowers_fidelity(self, small_qft_setup):
        transp, cmap, core_map, nq = small_qft_setup

        baseline = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            **COMMON_KWARGS,
        )

        # Make qubit 0 have 100x worse 1Q error
        bad_1q = np.full(nq, SCALAR_1Q_ERROR)
        bad_1q[0] = SCALAR_1Q_ERROR * 100

        degraded = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            single_gate_error_per_qubit=bad_1q,
            **COMMON_KWARGS,
        )

        assert degraded.algorithmic_fidelity < baseline.algorithmic_fidelity, \
            f"degraded qubit should worsen fidelity: {degraded.algorithmic_fidelity} vs {baseline.algorithmic_fidelity}"

        # Coherence should be unaffected
        assert degraded.coherence_fidelity == pytest.approx(
            baseline.coherence_fidelity, abs=1e-12
        ), "gate errors should not affect coherence"

    def test_degraded_pair_lowers_fidelity(self, small_qft_setup):
        transp, cmap, core_map, nq = small_qft_setup

        baseline = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            **COMMON_KWARGS,
        )

        # Make all pairs 10x worse
        pair_errors = {}
        for i in range(nq):
            for j in range(nq):
                if i != j:
                    pair_errors[(i, j)] = min(SCALAR_2Q_ERROR * 10, 1.0)

        degraded = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            two_gate_error_per_pair=pair_errors,
            **COMMON_KWARGS,
        )

        assert degraded.algorithmic_fidelity < baseline.algorithmic_fidelity, \
            f"worse 2Q errors should worsen fidelity: {degraded.algorithmic_fidelity} vs {baseline.algorithmic_fidelity}"

    def test_zero_errors_give_perfect_fidelity(self, small_qft_setup):
        transp, cmap, core_map, nq = small_qft_setup

        result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            seed=42,
            initial_placement=InitialPlacement.RANDOM,
            single_gate_error=0.0,
            two_gate_error=0.0,
            teleportation_error_per_hop=0.0,
            single_gate_time=35.0,
            two_gate_time=660.0,
            teleportation_time_per_hop=1000.0,
            t1=float('inf'),
            t2=float('inf'),
            single_gate_error_per_qubit=np.zeros(nq),
            two_gate_error_per_pair={},
        )

        assert result.algorithmic_fidelity == pytest.approx(1.0, abs=1e-12)

    def test_per_qubit_grid_reflects_degraded_qubit(self, small_qft_setup):
        """The per-qubit fidelity grid should show qubit 0 worse than others."""
        transp, cmap, core_map, nq = small_qft_setup

        bad_1q = np.full(nq, SCALAR_1Q_ERROR)
        bad_1q[0] = SCALAR_1Q_ERROR * 100

        result = map_circuit(
            circuit=transp, full_coupling_map=cmap, core_mapping=core_map,
            single_gate_error_per_qubit=bad_1q,
            **COMMON_KWARGS,
        )

        # Final layer algorithmic fidelity for qubit 0 should be worst
        final_algo = result.algorithmic_fidelity_grid[-1, :]
        assert final_algo[0] < np.min(final_algo[1:]), \
            f"qubit 0 should have worst algo fidelity: {final_algo}"


# ---------------------------------------------------------------------------
# Virtual gate filtering (rz should not contribute error or time)
# ---------------------------------------------------------------------------

class TestVirtualGateFiltering:
    """Virtual gates (rz, id) must be excluded from the sparse tensor."""

    def test_rz_gates_excluded_from_sparse_tensor(self):
        """rz gates should not appear in the sparse interaction tensor."""
        from qusim import _qiskit_circ_to_sparse_list

        # Build a circuit with both rz (virtual) and sx (physical) gates
        circ = qiskit.QuantumCircuit(2)
        circ.rz(0.5, 0)
        circ.rz(1.0, 1)
        circ.sx(0)
        circ.cx(0, 1)

        sparse = _qiskit_circ_to_sparse_list(circ)

        # Should have 1 sx (1Q) + 1 cx (2Q) = 2 entries, NOT 4 (which includes 2 rz)
        assert len(sparse) == 2, \
            f"rz gates should be filtered out: expected 2 entries, got {len(sparse)}"

    def test_rz_only_circuit_produces_empty_tensor(self):
        """A circuit with only rz gates should produce an empty sparse tensor."""
        from qusim import _qiskit_circ_to_sparse_list

        circ = qiskit.QuantumCircuit(3)
        circ.rz(0.5, 0)
        circ.rz(1.0, 1)
        circ.rz(0.3, 2)

        sparse = _qiskit_circ_to_sparse_list(circ)
        assert len(sparse) == 0, \
            f"rz-only circuit should produce empty tensor, got {len(sparse)} entries"

    def test_rz_does_not_affect_fidelity(self):
        """Adding rz gates should NOT change fidelity (they are virtual/free)."""
        nq = 3

        # Circuit without rz
        circ_no_rz = qiskit.QuantumCircuit(nq)
        circ_no_rz.sx(0)
        circ_no_rz.cx(0, 1)
        circ_no_rz.sx(2)

        # Same circuit with rz gates added
        circ_with_rz = qiskit.QuantumCircuit(nq)
        circ_with_rz.rz(0.5, 0)
        circ_with_rz.sx(0)
        circ_with_rz.rz(1.0, 1)
        circ_with_rz.cx(0, 1)
        circ_with_rz.rz(0.3, 2)
        circ_with_rz.sx(2)
        circ_with_rz.rz(0.7, 0)

        edges = [(i, j) for i in range(nq) for j in range(i+1, nq)]
        edges += [(j, i) for i, j in edges]
        cmap = CouplingMap(edges)
        core_map = {i: 0 for i in range(nq)}

        kwargs = dict(
            full_coupling_map=cmap, core_mapping=core_map,
            seed=42, initial_placement=InitialPlacement.RANDOM,
            single_gate_error=1e-3, two_gate_error=1e-2,
            teleportation_error_per_hop=0.0,
            single_gate_time=36.0, two_gate_time=68.0,
            teleportation_time_per_hop=1000.0,
            t1=100_000.0, t2=50_000.0,
        )

        result_no_rz = map_circuit(circuit=circ_no_rz, **kwargs)
        result_with_rz = map_circuit(circuit=circ_with_rz, **kwargs)

        assert result_no_rz.algorithmic_fidelity == pytest.approx(
            result_with_rz.algorithmic_fidelity, abs=1e-12
        ), "rz gates must not affect algorithmic fidelity"

        assert result_no_rz.total_circuit_time_ns == pytest.approx(
            result_with_rz.total_circuit_time_ns, abs=1e-6
        ), "rz gates must not affect circuit time"

        assert result_no_rz.overall_fidelity == pytest.approx(
            result_with_rz.overall_fidelity, abs=1e-12
        ), "rz gates must not affect overall fidelity"
