"""
Design Space Exploration (DSE) toolkit.

Build and run multi-dimensional sweeps over circuit, topology, and noise
parameters; cache the expensive structural pass; cheaply re-evaluate noise
configs from cached mappings.

Typical usage::

    from qusim.dse import DSEEngine

    engine = DSEEngine()
    sweep = engine.sweep_nd(
        cached=None,
        sweep_axes=[
            ("num_cores",            1, 8),       # cold: num_qubits/num_cores etc.
            ("two_gate_error",      -4, -2),      # hot: log10 endpoints for log_scale
        ],
        fixed_noise={},
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 32,
            "num_cores": 4,                       # overridden by sweep_axes
            "topology_type": "ring",
            "intracore_topology": "all_to_all",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 0,
        },
    )

    # SweepResult exposes axes, grid, shape, total_points.
    print(sweep.shape, sweep.metric_keys)

    # Hand off to qusim.analysis for FoM / Pareto-front analysis.
    from qusim.analysis import FomConfig, compute_for_sweep
    fom = FomConfig(numerator="overall_fidelity",
                    denominator="max(total_epr_pairs, 1)")
    result = compute_for_sweep(sweep.to_sweep_data(), fom)
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
from .engine import (
    CachedMapping,
    DSEEngine,
    SweepProgress,
    SweepResult,
    clamp_b_for_topology,
    clamp_k_for_topology,
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
    # Topology helpers
    "inter_core_neighbors",
    "clamp_k_for_topology",
    "clamp_b_for_topology",
    "max_data_slots",
    "total_reserved_slots",
    # Tabular access
    "flatten_sweep_to_table",
]
