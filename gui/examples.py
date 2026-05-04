"""Pre-computed example sessions for the DSE GUI.

Each entry in :data:`EXAMPLES` describes one canned session: the sweep axes,
fixed circuit / noise configuration, and the view that should be active when
the session opens. The offline ``scripts/generate_example_sessions.py`` script
runs the sweep for each entry and writes a gzipped session file (the same
format the Save/Load flow uses) under ``gui/assets/examples/``. The Topbar
"Examples" dropdown then loads the canned bytes through the regular load path.

Slider ranges in :class:`ExampleSpec.sweep_axes` use the GUI's slider-position
convention: log10 exponents for log-scale metrics, raw values for linear ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent / "assets" / "examples"


@dataclass
class ExampleSpec:
    id: str
    label: str
    description: str = ""

    # Numeric sweep axes: list of (metric_key, slider_low, slider_high).
    # ``slider_low``/``slider_high`` are slider positions: log10 exponents
    # for log-scale metrics, raw values for linear ones.
    sweep_axes: list[tuple[str, float, float]] = field(default_factory=list)

    # Fixed circuit / topology config (cold path).
    cold_config: dict = field(default_factory=dict)

    # Fixed noise overrides on top of NOISE_DEFAULTS (raw values, not slider
    # positions). Anything omitted falls back to NOISE_DEFAULTS.
    fixed_noise: dict = field(default_factory=dict)

    # View opened when the session loads.
    view_type: str = "isosurface"
    frozen_axis: int = 2
    frozen_slider_value: float | None = None
    output_metric: str = "overall_fidelity"
    view_mode: str = "absolute"

    # Optional grid-size hints for the generator. None lets the engine pick
    # its default budgets (SWEEP_POINTS_*).
    max_cold: int | None = None
    max_hot: int | None = None


# Five canned sessions, one per dimensionality the GUI exposes plus an extra
# 2-D view. Names lead with the dimensionality + the headline graph property
# so the user can pick the visualisation they want without reading
# descriptions.
EXAMPLES: list[ExampleSpec] = [
    ExampleSpec(
        id="1d_cores_sweet_spot",
        label="1D — Cores: sweet spot",
        description=(
            "Adding cores reduces intra-core swap depth on a constrained "
            "linear chip but pays inter-core EPR cost. Fidelity peaks at an "
            "intermediate core count instead of growing monotonically."
        ),
        sweep_axes=[("num_cores", 1, 4)],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 24,
            "num_logical_qubits": 8,
            "num_cores": 2,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            "single_gate_error": 1e-4,
            "two_gate_error": 5e-4,
            "epr_error_per_hop": 5e-3,
            "measurement_error": 1e-3,
            "t1": 5e5,
            "t2": 2e5,
            "two_gate_time": 80.0,
            "epr_time_per_hop": 150.0,
        },
        view_type="line",
        output_metric="overall_fidelity",
        max_cold=4,
    ),
    ExampleSpec(
        id="2d_two_gate_error_vs_t1",
        label="2D — 2Q error × T1 (speed-accuracy)",
        description=(
            "Heatmap of overall fidelity over the dominant noise pair on a "
            "single-core 6-qubit GHZ. Two regimes meet on a diagonal iso-"
            "fidelity ridge: the lower-left (gate-error-limited) and upper-"
            "right (T1-limited) corners frame the trade-off."
        ),
        sweep_axes=[
            ("two_gate_error", -5, -2),
            ("t1", 3.5, 6.5),
        ],
        cold_config={
            "circuit_type": "ghz",
            "num_qubits": 8,
            "num_logical_qubits": 6,
            "num_cores": 1,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "all_to_all",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            "single_gate_error": 1e-4,
            "measurement_error": 5e-4,
            "t2": 5e4,
            "two_gate_time": 80.0,
            "single_gate_time": 20.0,
        },
        view_type="heatmap",
        output_metric="overall_fidelity",
    ),
    ExampleSpec(
        id="2d_two_gate_error_vs_cores",
        label="2D — 2Q error × Cores (cores affordability)",
        description=(
            "Heatmap pairs the dominant noise channel with the architecture's "
            "cold-path core count. Low gate error tolerates more cores (extra "
            "EPR overhead doesn't dominate); high gate error pushes the iso-"
            "fidelity ridge toward fewer, larger cores."
        ),
        sweep_axes=[
            ("two_gate_error", -5, -2),
            ("num_cores", 1, 4),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 24,
            "num_logical_qubits": 12,
            "num_cores": 2,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            "single_gate_error": 1e-4,
            "epr_error_per_hop": 5e-3,
            "measurement_error": 1e-3,
            "t1": 5e5,
            "t2": 2e5,
            "epr_time_per_hop": 150.0,
            "two_gate_time": 80.0,
        },
        view_type="heatmap",
        output_metric="overall_fidelity",
        max_cold=4,
    ),
    ExampleSpec(
        id="3d_two_gate_error_t1_cores",
        label="3D — 2Q error × T1 × Cores (isosurface)",
        description=(
            "Fidelity volume over the dominant noise pair plus the core "
            "count. Frozen-heatmap views slice along the cores axis to show "
            "how the (error, T1) iso-fidelity ridge shifts with topology."
        ),
        sweep_axes=[
            ("two_gate_error", -5, -2),
            ("t1", 4, 6.5),
            ("num_cores", 1, 4),
        ],
        cold_config={
            "circuit_type": "ghz",
            "num_qubits": 24,
            "num_logical_qubits": 8,
            "num_cores": 2,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            "single_gate_error": 1e-4,
            "epr_error_per_hop": 5e-3,
            "measurement_error": 5e-4,
            "t2": 1e5,
            "two_gate_time": 80.0,
            "epr_time_per_hop": 150.0,
        },
        view_type="isosurface",
        frozen_axis=2,
        output_metric="overall_fidelity",
        max_cold=4,
    ),
    ExampleSpec(
        id="4d_noise_parallel",
        label="4D — 1Q × 2Q × T1 × Meas (parallel)",
        description=(
            "Four-axis hot-path scan over the dominant noise channels on a "
            "small two-core GHZ. Parallel coordinates expose how each noise "
            "source trades against the others against overall fidelity."
        ),
        sweep_axes=[
            ("single_gate_error", -5, -2),
            ("two_gate_error", -5, -2),
            ("t1", 3.5, 6.5),
            ("measurement_error", -5, -2),
        ],
        cold_config={
            "circuit_type": "ghz",
            "num_qubits": 16,
            "num_logical_qubits": 8,
            "num_cores": 2,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "all_to_all",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            "epr_error_per_hop": 5e-3,
            "t2": 5e4,
            "two_gate_time": 80.0,
        },
        view_type="parallel",
        output_metric="overall_fidelity",
    ),
]


def example_options() -> list[dict]:
    """Build the dropdown ``options`` list for the Topbar selector."""
    return [{"label": e.label, "value": e.id} for e in EXAMPLES]


def example_by_id(eid: str) -> ExampleSpec | None:
    for e in EXAMPLES:
        if e.id == eid:
            return e
    return None


def example_path(eid: str) -> Path:
    """Filesystem path to the gzipped session for *eid*."""
    return EXAMPLES_DIR / f"{eid}.qusim.json.gz"


def load_example_bytes(eid: str) -> bytes:
    """Read the canned session bytes for *eid*. Raises FileNotFoundError if
    the example hasn't been generated yet."""
    return example_path(eid).read_bytes()
