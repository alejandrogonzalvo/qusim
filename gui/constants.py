"""
Parameter registry for the DSE GUI.

Each MetricDef describes a sweepable hardware parameter (left sidebar).
Each FixedParam describes a configuration knob (right panel).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricDef:
    key: str
    label: str
    # Slider positions (exponents for log-scale, raw values for linear)
    slider_min: float
    slider_max: float
    slider_default_low: float
    slider_default_high: float
    num_steps: int
    log_scale: bool
    unit: str
    is_cold_path: bool
    description: str


# All parameters that can be placed on the sweep axes (left sidebar)
SWEEPABLE_METRICS: List[MetricDef] = [
    MetricDef(
        key="single_gate_error",
        label="1Q Gate Error",
        slider_min=-6, slider_max=-1,
        slider_default_low=-5, slider_default_high=-3,
        num_steps=50, log_scale=True, unit="",
        is_cold_path=False,
        description="Per-gate error probability for single-qubit operations",
    ),
    MetricDef(
        key="two_gate_error",
        label="2Q Gate Error",
        slider_min=-5, slider_max=-1,
        slider_default_low=-4, slider_default_high=-2,
        num_steps=50, log_scale=True, unit="",
        is_cold_path=False,
        description="Per-gate error probability for two-qubit operations",
    ),
    MetricDef(
        key="teleportation_error_per_hop",
        label="Teleport Error/Hop",
        slider_min=-5, slider_max=-1,
        slider_default_low=-3, slider_default_high=-1,
        num_steps=50, log_scale=True, unit="",
        is_cold_path=False,
        description="Fidelity loss per inter-core teleportation hop",
    ),
    MetricDef(
        key="t1",
        label="T1 (Relaxation)",
        slider_min=3, slider_max=7,
        slider_default_low=4, slider_default_high=6,
        num_steps=50, log_scale=True, unit="ns",
        is_cold_path=False,
        description="T1 energy relaxation time constant in nanoseconds",
    ),
    MetricDef(
        key="t2",
        label="T2 (Dephasing)",
        slider_min=3, slider_max=7,
        slider_default_low=4, slider_default_high=6,
        num_steps=50, log_scale=True, unit="ns",
        is_cold_path=False,
        description="T2 dephasing time constant in nanoseconds",
    ),
    MetricDef(
        key="single_gate_time",
        label="1Q Gate Time",
        slider_min=0, slider_max=3,
        slider_default_low=1, slider_default_high=2,
        num_steps=40, log_scale=True, unit="ns",
        is_cold_path=False,
        description="Single-qubit gate execution time in nanoseconds",
    ),
    MetricDef(
        key="two_gate_time",
        label="2Q Gate Time",
        slider_min=1, slider_max=4,
        slider_default_low=2, slider_default_high=3,
        num_steps=40, log_scale=True, unit="ns",
        is_cold_path=False,
        description="Two-qubit gate execution time in nanoseconds",
    ),
    MetricDef(
        key="teleportation_time_per_hop",
        label="Teleport Time/Hop",
        slider_min=2, slider_max=5,
        slider_default_low=3, slider_default_high=4,
        num_steps=40, log_scale=True, unit="ns",
        is_cold_path=False,
        description="Inter-core teleportation time per hop in nanoseconds",
    ),
    MetricDef(
        key="readout_mitigation_factor",
        label="Readout Mitigation",
        slider_min=0.0, slider_max=1.0,
        slider_default_low=0.0, slider_default_high=0.95,
        num_steps=20, log_scale=False, unit="",
        is_cold_path=False,
        description="TREX readout error mitigation factor (0=none, 1=full)",
    ),
    MetricDef(
        key="classical_link_width",
        label="Classical Link Width",
        slider_min=1, slider_max=15,
        slider_default_low=1, slider_default_high=15,
        num_steps=15, log_scale=False, unit="wires",
        is_cold_path=False,
        description="Number of parallel wires in the classical inter-core link",
    ),
    MetricDef(
        key="classical_clock_freq_hz",
        label="Clock Frequency",
        slider_min=7, slider_max=9,
        slider_default_low=7, slider_default_high=8.3,
        num_steps=30, log_scale=True, unit="Hz",
        is_cold_path=False,
        description="Classical network clock frequency (10 MHz – 1 GHz)",
    ),
    MetricDef(
        key="classical_routing_cycles",
        label="Routing Cycles",
        slider_min=1, slider_max=10,
        slider_default_low=1, slider_default_high=5,
        num_steps=10, log_scale=False, unit="cycles",
        is_cold_path=False,
        description="Clock cycles for routing/arbitration overhead per hop",
    ),
    MetricDef(
        key="num_qubits",
        label="Qubits",
        slider_min=4, slider_max=256,
        slider_default_low=4, slider_default_high=20,
        num_steps=15, log_scale=False, unit="",
        is_cold_path=True,
        description="Number of logical qubits in the circuit",
    ),
    MetricDef(
        key="num_cores",
        label="Cores",
        slider_min=1, slider_max=8,
        slider_default_low=1, slider_default_high=8,
        num_steps=8, log_scale=False, unit="",
        is_cold_path=True,
        description="Number of processor cores in the multi-core architecture",
    ),
]

METRIC_BY_KEY = {m.key: m for m in SWEEPABLE_METRICS}

# Maximum number of axes that can be swept simultaneously.
# Derived from the registry so adding/removing entries in SWEEPABLE_METRICS
# automatically updates every consumer (slider rows, callbacks, etc.).
MAX_SWEEP_AXES = len(SWEEPABLE_METRICS)

DEFAULT_SWEEP_AXES = ["t1", "t2", "two_gate_time"]

# Default scalar noise values shown in the right panel
NOISE_DEFAULTS = {
    "single_gate_error": 1e-4,
    "two_gate_error": 1e-3,
    "teleportation_error_per_hop": 1e-2,
    "t1": 100_000.0,
    "t2": 50_000.0,
    "single_gate_time": 20.0,
    "two_gate_time": 100.0,
    "teleportation_time_per_hop": 1_000.0,
    "readout_mitigation_factor": 0.0,
    "classical_link_width": 0,
    "classical_clock_freq_hz": 200e6,
    "classical_routing_cycles": 2,
}

# Circuit types available in the dropdown
CIRCUIT_TYPES = [
    {"label": "Quantum Fourier Transform (QFT)", "value": "qft"},
    {"label": "GHZ State", "value": "ghz"},
    {"label": "Random Circuit", "value": "random"},
]

# Inter-core topology options
TOPOLOGY_TYPES = [
    {"label": "Ring", "value": "ring"},
    {"label": "All-to-All", "value": "all_to_all"},
    {"label": "Linear Chain", "value": "linear"},
]

# Intra-core (on-chip) qubit connectivity
INTRACORE_TOPOLOGY_TYPES = [
    {"label": "All-to-All", "value": "all_to_all"},
    {"label": "Linear Chain", "value": "linear"},
    {"label": "Ring", "value": "ring"},
    {"label": "Grid", "value": "grid"},
]

# HQA initial placement policies
PLACEMENT_OPTIONS = [
    {"label": "Random", "value": "random"},
    {"label": "Spectral Clustering", "value": "spectral"},
]

# Output metrics that can be shown on the Y-axis
OUTPUT_METRICS = [
    {"label": "Overall Fidelity", "value": "overall_fidelity"},
    {"label": "Algorithmic Fidelity", "value": "algorithmic_fidelity"},
    {"label": "Routing Fidelity", "value": "routing_fidelity"},
    {"label": "Coherence Fidelity", "value": "coherence_fidelity"},
    {"label": "Circuit Time (ns)", "value": "total_circuit_time_ns"},
    {"label": "Total EPR Pairs", "value": "total_epr_pairs"},
    {"label": "Intra-core Swaps", "value": "total_swaps"},
    {"label": "Inter-core Swaps", "value": "total_teleportations"},
    {"label": "Network Distance", "value": "total_network_distance"},
]

# Number of sweep points per dimension (legacy, kept for backward compat)
SWEEP_POINTS_1D = 60
SWEEP_POINTS_2D = 30
SWEEP_POINTS_3D = 12

# Cold-path sweeps (num_qubits, num_cores) are much slower per point
SWEEP_POINTS_COLD_1D = 15
SWEEP_POINTS_COLD_2D = 8
SWEEP_POINTS_COLD_3D = 5

# N-D grid point budgets — split by cost model:
#   Cold compilations: expensive (~1-10s each), only needed per unique
#     (num_qubits, num_cores) combination.
#   Hot evaluations:   near-free when batched via Rust, safe to run many.
MAX_COLD_COMPILATIONS = 64       # cap on unique cold-path configs
MAX_TOTAL_POINTS_HOT = 50_000   # cap on total grid points (hot eval is cheap)

# Legacy alias kept for backward compat in _points_per_axis
MAX_TOTAL_POINTS_COLD = MAX_COLD_COMPILATIONS

# Minimum points per axis regardless of dimensionality
MIN_POINTS_PER_AXIS = 3

# View tab definitions per sweep dimensionality
VIEW_TABS: dict[int, list[dict]] = {
    1: [
        {"value": "line", "label": "Line"},
    ],
    2: [
        {"value": "heatmap", "label": "Heatmap"},
        {"value": "contour", "label": "Contour"},
    ],
    3: [
        {"value": "scatter3d", "label": "Scatter"},
        {"value": "isosurface", "label": "Isosurface"},
        {"value": "frozen_heatmap", "label": "Frozen Heat"},
        {"value": "frozen_contour", "label": "Frozen Ctr"},
    ],
}

VIEW_TAB_DEFAULTS: dict[int, str] = {
    1: "line",
    2: "contour",
    3: "isosurface",
}

# For N >= 4, the default view is the first analysis tab
VIEW_TAB_DEFAULT_ND = "parallel"

ANALYSIS_TABS: list[dict] = [
    {"value": "parallel", "label": "Parallel"},
    {"value": "slices", "label": "Slices"},
    {"value": "importance", "label": "Importance"},
    {"value": "pareto", "label": "Pareto"},
    {"value": "correlation", "label": "Corr."},
]
