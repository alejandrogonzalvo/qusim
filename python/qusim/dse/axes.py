"""
Sweepable parameter registry — the pure data part of the DSE configuration.

This module is GUI-agnostic: every dataclass and constant here can be consumed
from a Python script, a notebook, or the Dash GUI without pulling any UI deps.

GUI-only presentation knobs (slider min/max, view tabs, view modes, ...) live
in ``gui/constants.py`` and decorate the entries here.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CatMetricDef:
    key: str
    label: str
    options: list  # list of {"label": ..., "value": ...} dicts
    is_cold_path: bool
    description: str
    cold_config_key: str = ""  # key name in cold_config; defaults to self.key

    def __post_init__(self):
        if not self.cold_config_key:
            self.cold_config_key = self.key


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


# All parameters that can be placed on the sweep axes (left sidebar).
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
        key="epr_error_per_hop",
        label="EPR Error/Hop",
        slider_min=-5, slider_max=-1,
        slider_default_low=-3, slider_default_high=-1,
        num_steps=50, log_scale=True, unit="",
        is_cold_path=False,
        description=(
            "Per-hop EPR generation error.  Dominant term in the "
            "teleportation cost (matches dse_pau ``EPR_error``)."
        ),
    ),
    MetricDef(
        key="measurement_error",
        label="Meas. Error",
        slider_min=-6, slider_max=-1,
        slider_default_low=-5, slider_default_high=-3,
        num_steps=50, log_scale=True, unit="",
        is_cold_path=False,
        description=(
            "Mid-circuit measurement error charged once per "
            "teleportation hop (Bell measurement on the source side)."
        ),
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
        key="epr_time_per_hop",
        label="EPR Time/Hop",
        slider_min=1, slider_max=4,
        slider_default_low=2, slider_default_high=3,
        num_steps=40, log_scale=True, unit="ns",
        is_cold_path=False,
        description="Per-hop EPR generation latency in nanoseconds.",
    ),
    MetricDef(
        key="measurement_time",
        label="Meas. Time",
        slider_min=1, slider_max=3,
        slider_default_low=1, slider_default_high=2,
        num_steps=40, log_scale=True, unit="ns",
        is_cold_path=False,
        description="Mid-circuit measurement latency in nanoseconds.",
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
        key="num_logical_qubits",
        label="Logical Qubits",
        slider_min=2, slider_max=256,
        slider_default_low=4, slider_default_high=64,
        num_steps=15, log_scale=False, unit="",
        is_cold_path=True,
        description=(
            "Number of logical qubits in the algorithm circuit. "
            "Held constant during a sweep — the architecture (cores or "
            "qubits-per-core, whichever is *not* pinned) grows to absorb "
            "comm/buffer overhead. Auto-set when a custom QASM file is "
            "uploaded."
        ),
    ),
    MetricDef(
        key="num_cores",
        label="Cores",
        slider_min=1, slider_max=64,
        slider_default_low=1, slider_default_high=8,
        num_steps=8, log_scale=False, unit="",
        is_cold_path=True,
        description=(
            "Number of processor cores. Sweepable only when *Cores* is "
            "the pinned architectural axis; otherwise it is a derived "
            "output computed from logical qubits + qpc + comm/buffer."
        ),
    ),
    MetricDef(
        key="qubits_per_core",
        label="Qubits per Core",
        slider_min=4, slider_max=128,
        slider_default_low=8, slider_default_high=64,
        num_steps=15, log_scale=False, unit="",
        is_cold_path=True,
        description=(
            "Physical slots per core (uniform across the chip). "
            "Sweepable only when *Qubits per core* is the pinned "
            "architectural axis; otherwise it is a derived output."
        ),
    ),
    MetricDef(
        key="communication_qubits",
        label="Comm Qubits",
        slider_min=1, slider_max=16,
        slider_default_low=1, slider_default_high=2,
        num_steps=4, log_scale=False, unit="",
        is_cold_path=True,
        description=(
            "Comm qubits per group (per inter-core link). Each core's "
            "G inter-core neighbours each reserve K + B slots (K comm + "
            "B buffer per group)."
        ),
    ),
    MetricDef(
        key="buffer_qubits",
        label="Buffer Qubits",
        slider_min=1, slider_max=16,
        slider_default_low=1, slider_default_high=1,
        num_steps=4, log_scale=False, unit="",
        is_cold_path=True,
        description=(
            "Buffer qubits per group. Bounded above by communication "
            "qubits (B ≤ K) and by data-slot feasibility."
        ),
    ),
]

METRIC_BY_KEY = {m.key: m for m in SWEEPABLE_METRICS}
MAX_SWEEP_AXES = len(SWEEPABLE_METRICS)
DEFAULT_SWEEP_AXES = ["t1", "t2", "two_gate_time"]

# Default scalar noise values. Programmatic sweeps merge these with overrides
# (see ``qusim.dse.engine._merge_noise``).
NOISE_DEFAULTS = {
    "single_gate_error": 1e-4,
    "two_gate_error": 1e-3,
    "epr_error_per_hop": 0.0,
    "measurement_error": 1e-3,
    "teleportation_error_per_hop": 1e-2,
    "t1": 100_000.0,
    "t2": 50_000.0,
    "single_gate_time": 20.0,
    "two_gate_time": 100.0,
    "epr_time_per_hop": 130.0,
    "measurement_time": 40.0,
    "teleportation_time_per_hop": 1_000.0,
    "readout_mitigation_factor": 0.0,
    "classical_link_width": 0,
    "classical_clock_freq_hz": 200e6,
    "classical_routing_cycles": 2,
    "communication_qubits": 1,
    "buffer_qubits": 1,
    "num_logical_qubits": 16,
    "qubits_per_core": 16,
    "num_cores": 1,
    "pin_axis": "cores",
}


CIRCUIT_TYPES = [
    {"label": "Quantum Fourier Transform (QFT)", "value": "qft"},
    {"label": "GHZ State", "value": "ghz"},
    {"label": "Random Circuit", "value": "random"},
]

TOPOLOGY_TYPES = [
    {"label": "Ring", "value": "ring"},
    {"label": "All-to-All", "value": "all_to_all"},
    {"label": "Linear Chain", "value": "linear"},
]

INTRACORE_TOPOLOGY_TYPES = [
    {"label": "All-to-All", "value": "all_to_all"},
    {"label": "Linear Chain", "value": "linear"},
    {"label": "Ring", "value": "ring"},
    {"label": "Grid", "value": "grid"},
]

ROUTING_ALGORITHM_OPTIONS = [
    {"label": "HQA + Sabre", "value": "hqa_sabre"},
    {"label": "TeleSABRE", "value": "telesabre"},
]

PLACEMENT_OPTIONS = [
    {"label": "Random", "value": "random"},
    {"label": "Spectral Clustering", "value": "spectral"},
]


CATEGORICAL_METRICS: List[CatMetricDef] = [
    CatMetricDef(
        key="circuit_type",
        label="Circuit Type",
        options=CIRCUIT_TYPES,
        is_cold_path=True,
        description="Quantum circuit benchmark (QFT, GHZ, Random)",
    ),
    CatMetricDef(
        key="topology_type",
        label="Inter-core Topology",
        options=TOPOLOGY_TYPES,
        is_cold_path=True,
        description="Inter-core connectivity (ring / all-to-all / linear)",
    ),
    CatMetricDef(
        key="intracore_topology",
        label="Intra-core Topology",
        options=INTRACORE_TOPOLOGY_TYPES,
        is_cold_path=True,
        description="On-chip qubit connectivity within each core",
    ),
    CatMetricDef(
        key="routing_algorithm",
        label="Routing Algorithm",
        options=ROUTING_ALGORITHM_OPTIONS,
        is_cold_path=True,
        description="Routing algorithm (HQA+Sabre / TeleSABRE)",
    ),
    CatMetricDef(
        key="placement",
        label="Placement",
        options=PLACEMENT_OPTIONS,
        is_cold_path=True,
        description="Initial qubit placement policy (random / spectral)",
        cold_config_key="placement_policy",
    ),
]

CAT_METRIC_BY_KEY: dict = {m.key: m for m in CATEGORICAL_METRICS}


# Output metrics that can be shown on the Y-axis of any plot.
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
    {"label": "Physical Qubits (derived)", "value": "num_qubits"},
    {"label": "Cores (derived)", "value": "derived_num_cores"},
    {"label": "Qubits/Core (derived)", "value": "derived_qubits_per_core"},
    {"label": "Idle Reserved Qubits", "value": "idle_reserved_qubits"},
]

# Optimization direction per output metric. Consumed by Pareto computation.
PARETO_METRIC_ORIENTATION: dict[str, str] = {
    "overall_fidelity": "max",
    "algorithmic_fidelity": "max",
    "routing_fidelity": "max",
    "coherence_fidelity": "max",
    "total_circuit_time_ns": "min",
    "total_epr_pairs": "min",
    "total_swaps": "min",
    "total_teleportations": "min",
    "total_network_distance": "min",
    "num_qubits": "min",
    "derived_num_cores": "min",
    "derived_qubits_per_core": "min",
    "idle_reserved_qubits": "min",
}

# Metrics bounded in [0, 1] — used to decide when to clamp axes / draw
# fidelity threshold lines.
FIDELITY_METRICS: set[str] = {
    "overall_fidelity",
    "algorithmic_fidelity",
    "routing_fidelity",
    "coherence_fidelity",
}

OUTPUT_METRIC_LABEL: dict[str, str] = {m["value"]: m["label"] for m in OUTPUT_METRICS}


# Sweep budgets / safety knobs — referenced from the engine and the GUI.
SWEEP_POINTS_1D = 60
SWEEP_POINTS_2D = 30
SWEEP_POINTS_3D = 12

SWEEP_POINTS_COLD_1D = 15
SWEEP_POINTS_COLD_2D = 8
SWEEP_POINTS_COLD_3D = 5

MAX_COLD_COMPILATIONS = 64
MAX_TOTAL_POINTS_HOT = 5_000
MAX_WORKERS_DEFAULT = 1

# Legacy alias
MAX_TOTAL_POINTS_COLD = MAX_COLD_COMPILATIONS

MIN_POINTS_PER_AXIS = 3
