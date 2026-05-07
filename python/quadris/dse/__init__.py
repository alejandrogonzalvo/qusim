"""
Design Space Exploration (DSE) toolkit.

Build and run multi-dimensional sweeps over circuit, topology, and noise
parameters; cache the expensive structural pass; cheaply re-evaluate noise
configs from cached mappings.

Logical-first parameterization
------------------------------

The user pins exactly one of ``num_cores`` or ``qubits_per_core``; the
unpinned axis is deduced so the chip fits ``num_logical_qubits`` even
as the user sweeps comm/buffer overhead. ``num_qubits`` (= ``num_cores
· qubits_per_core``) and ``idle_reserved_qubits`` are derived outputs.

Typical usage::

    from quadris.dse import DSEEngine

    engine = DSEEngine()
    sweep = engine.sweep_nd(
        cached=None,
        sweep_axes=[
            ("num_cores",      1, 8),     # cold: pinned cores
            ("two_gate_error", -4, -2),   # hot: log10 endpoints
        ],
        fixed_noise={},
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 32,
            "qubits_per_core": 16,
            "num_cores": 4,                # overridden by the sweep
            "pin_axis": "cores",
            "topology_type": "ring",
            "intracore_topology": "all_to_all",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 0,
        },
    )
    print(sweep.shape, sweep.metric_keys)
"""

from .axes import (
    CAT_METRIC_BY_KEY,
    CATEGORICAL_METRICS,
    CatMetricDef,
    DEFAULT_SWEEP_AXES,
    FIDELITY_METRICS,
    MAX_SWEEP_AXES,
    METRIC_BY_KEY,
    MetricDef,
    NOISE_DEFAULTS,
    OUTPUT_METRIC_LABEL,
    OUTPUT_METRICS,
    PARETO_METRIC_ORIENTATION,
    SWEEPABLE_METRICS,
)
from .config import (
    DEFAULT_PIN_AXIS,
    PIN_CORES,
    PIN_QPC,
    _resolve_architecture,
)
from .engine import (
    CachedMapping,
    DSEEngine,
    SweepProgress,
    SweepResult,
    deduce_num_cores,
    deduce_qubits_per_core,
    g_max,
    idle_reserved_qubits,
    inter_core_neighbors,
    max_data_slots,
    total_reserved_slots,
)
from .flatten import flatten_sweep_to_table

__all__ = [
    # Engine
    "DSEEngine",
    "SweepResult",
    "SweepProgress",
    "CachedMapping",
    # Axes registry
    "MetricDef",
    "CatMetricDef",
    "SWEEPABLE_METRICS",
    "METRIC_BY_KEY",
    "CATEGORICAL_METRICS",
    "CAT_METRIC_BY_KEY",
    "OUTPUT_METRICS",
    "OUTPUT_METRIC_LABEL",
    "PARETO_METRIC_ORIENTATION",
    "FIDELITY_METRICS",
    "NOISE_DEFAULTS",
    "DEFAULT_SWEEP_AXES",
    "MAX_SWEEP_AXES",
    # Pin-axis vocabulary
    "DEFAULT_PIN_AXIS",
    "PIN_CORES",
    "PIN_QPC",
    # Topology helpers — deduction + uniform-G_max accounting
    "deduce_num_cores",
    "deduce_qubits_per_core",
    "g_max",
    "idle_reserved_qubits",
    "inter_core_neighbors",
    "max_data_slots",
    "total_reserved_slots",
    # Tabular access
    "flatten_sweep_to_table",
]
