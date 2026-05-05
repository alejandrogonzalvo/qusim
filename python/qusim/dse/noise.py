"""
Noise-parameter merging + dse_pau-equivalent teleportation cost
derivation. ``_merge_noise`` is the entry point used by every
cold/hot path call.
"""

from __future__ import annotations

import numpy as np

from .axes import NOISE_DEFAULTS

# --- dse_pau-equivalent teleportation cost decomposition ---------------------
#
# dse_pau models a single teleportation hop as the following gate sequence
# (see ``dse_pau/utils.py:646-679`` `get_operational_fidelity_depol`):
#
#   1. Generate EPR pair          → fidelity sqrt(1 − EPR_error)
#   2. Source-side preprocessing  → 1 CNOT (Bell measurement)
#                                   + 1 single-qubit gate (Hadamard)
#                                   + 1 measurement
#   3. Classical send             → no fidelity cost (latency only)
#   4. Destination postprocessing → 1 single-qubit gate (X/Z correction)
#                                   + 3 CNOTs (SWAP into the buffer qubit)
#
# Total per hop: 4 two-qubit gates + 2 single-qubit gates + 1 measurement
# + 1 EPR generation, plus the classical-comm latency.
#
# We bundle these into a single ``teleportation_error_per_hop`` /
# ``teleportation_time_per_hop`` pair so the existing Rust noise model can
# stay product-of-fidelities without per-qubit η-tracking.  This is an
# approximation — it agrees with dse_pau in the small-error limit and to
# first order otherwise, but skips the asymmetric-depolarisation η coupling
# between the qubit and its EPR/buffer partner that dse_pau models.

_TELE_PROTOCOL_TWO_GATE_COUNT = 4   # 1 (preprocess CNOT) + 3 (buffer SWAP)
_TELE_PROTOCOL_SINGLE_GATE_COUNT = 2  # 1 H (preprocess) + 1 X/Z (correction)
_TELE_PROTOCOL_MEAS_COUNT = 1       # Bell measurement on the source side


def _derived_tele_error(p: dict) -> float:
    """Bundled per-hop teleportation fidelity loss derived from constituents.

    .. note::
        With the η-coupled depolarising-channel model now in the Rust
        noise core, the *bundled* number is no longer used to compute
        per-hop routing fidelity directly — Rust applies the protocol
        gate-by-gate.  This bundle is kept as a back-compat / reporting
        knob (it shows up in the GUI's "Teleport Error/Hop" readout)
        and as a fallback for legacy callers that override
        ``teleportation_error_per_hop`` directly.

    Per-hop bundle (small-error linearisation of the η model):
        F_hop ≈ √(1 − EPR_err)
                · (1 − (2/3)·ε_2q)^4    ← per-qubit marginal of CNOT
                · (1 − ε_1q)^2
                · (1 − ε_meas)
    The (2/3)·ε factor on each CNOT is the per-qubit marginal of a
    d=4 depolarising channel under the η formula (see
    ``src/noise/mod.rs::eta_cnot_update``).
    """
    epr_err = max(0.0, float(p.get("epr_error_per_hop", 0.0)))
    sq_err = max(0.0, float(p.get("single_gate_error", 0.0)))
    tq_err = max(0.0, float(p.get("two_gate_error", 0.0)))
    meas_err = max(0.0, float(p.get("measurement_error", 0.0)))
    f = (
        max(0.0, 1.0 - epr_err) ** 0.5
        * (1.0 - (2.0 / 3.0) * tq_err) ** _TELE_PROTOCOL_TWO_GATE_COUNT
        * (1.0 - sq_err) ** _TELE_PROTOCOL_SINGLE_GATE_COUNT
        * (1.0 - meas_err) ** _TELE_PROTOCOL_MEAS_COUNT
    )
    return max(0.0, min(1.0, 1.0 - f))


def _derived_tele_time(p: dict) -> float:
    """Per-hop teleportation latency derived from the protocol gate budget."""
    return (
        float(p.get("epr_time_per_hop", 0.0))
        + _TELE_PROTOCOL_TWO_GATE_COUNT * float(p.get("two_gate_time", 0.0))
        + _TELE_PROTOCOL_SINGLE_GATE_COUNT * float(p.get("single_gate_time", 0.0))
        + _TELE_PROTOCOL_MEAS_COUNT * float(p.get("measurement_time", 0.0))
    )


def _merge_noise(overrides: dict) -> dict:
    """Return NOISE_DEFAULTS with any overrides applied, deriving the bundled
    teleportation cost from constituent EPR / gate / measurement parameters.

    The derivation runs unless the caller *explicitly* set
    ``teleportation_error_per_hop`` or ``teleportation_time_per_hop`` —
    that escape hatch lets test fixtures and benchmarks pin a known
    bundled value without going through the protocol-cost model.
    """
    merged = {**NOISE_DEFAULTS, **overrides}
    if "teleportation_error_per_hop" not in overrides:
        merged["teleportation_error_per_hop"] = _derived_tele_error(merged)
    if "teleportation_time_per_hop" not in overrides:
        merged["teleportation_time_per_hop"] = _derived_tele_time(merged)
    return merged


# Scalar result fields kept in the N-D sweep grid. The per-qubit ``*_grid``
# ndarrays produced by the Rust hot path are read only for single-point
# inspection (benchmarks, per-qubit heatmap), never from the sweep grid, so
# carrying them into every cell is pure dead weight — at 64 qubits each cell

def _make_gate_arrays(gate_names: list, noise: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build gate_error_arr and gate_time_arr from gate_names and scalar noise params."""
    VIRTUAL = {"rz", "id", "delay", "barrier", "measure"}
    err_map = {**{g: 0.0 for g in VIRTUAL}}
    time_map = {**{g: 0.0 for g in VIRTUAL}}

    error_arr = np.array(
        [err_map.get(name, np.nan) for name in gate_names], dtype=np.float64
    )
    time_arr = np.array(
        [time_map.get(name, np.nan) for name in gate_names], dtype=np.float64
    )
    return error_arr, time_arr
