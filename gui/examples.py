"""Pre-computed example sessions for the DSE GUI (logical-first schema).

Each entry in :data:`EXAMPLES` describes one canned session: the sweep axes,
fixed circuit / noise configuration, and the view that should be active when
the session opens. The offline ``scripts/generate_example_sessions.py`` script
runs the sweep for each entry and writes a gzipped session file (the same
format the Save/Load flow uses) under ``gui/assets/examples/``. The Topbar
"Examples" dropdown then loads the canned bytes through the regular load path.

Slider ranges in :class:`ExampleSpec.sweep_axes` use the GUI's slider-position
convention: log10 exponents for log-scale metrics, raw values for linear ones.

Logical-first model
-------------------

``num_qubits`` (physical) and ``num_cores`` are no longer free parameters —
the user pins one of {num_cores, qubits_per_core} via ``pin_axis`` and the
unpinned axis is deduced per cell so the device fits ``num_logical_qubits``.
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

    sweep_axes: list[tuple] = field(default_factory=list)
    cold_config: dict = field(default_factory=dict)
    fixed_noise: dict = field(default_factory=dict)

    view_type: str = "isosurface"
    frozen_axis: int = 2
    frozen_slider_value: float | None = None
    output_metric: str = "overall_fidelity"
    view_mode: str = "absolute"

    max_cold: int | None = None
    max_hot: int | None = None
    max_workers: int | None = None
    fom: dict | None = None
    pareto_x: str | None = None
    pareto_y: str | None = None


_LOW_NOISE: dict = {
    "single_gate_error": 1e-6,
    "two_gate_error": 1e-5,
    "epr_error_per_hop": 5e-3,
    "measurement_error": 1e-4,
    "t1": 1e8,
    "t2": 5e7,
    "single_gate_time": 20.0,
    "two_gate_time": 50.0,
    "epr_time_per_hop": 150.0,
}


# Five canned sessions covering 1-D through Pareto. The headline 1-D
# example showcases the pain point the logical-first model fixes:
# sweeping K with logical qubits held constant, watching the device
# grow rather than the circuit shrink.
EXAMPLES: list[ExampleSpec] = [
    ExampleSpec(
        id="1d_comm_qft64_pinned_cores",
        label="1D — QFT-64 sweep over comm qubits (cores pinned)",
        description=(
            "Headline logical-first showcase. QFT-64 stays QFT-64 in every "
            "cell; we pin cores=4 and watch qubits-per-core grow as comm "
            "qubits per group climb from 1 to 5. Each cell uses a "
            "*different* device — same circuit, more silicon, same router "
            "— so the fidelity curve is purely about per-hop teleportation "
            "cost vs. inter-core link multiplicity."
        ),
        sweep_axes=[("communication_qubits", 1, 5)],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,  # derived per cell since pin=cores
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="line",
        output_metric="overall_fidelity",
        max_cold=8,
        max_workers=1,
    ),
    ExampleSpec(
        id="2d_comm_buffer_infeasible",
        label="2D — Comm × Buffer (white = infeasible)",
        description=(
            "Showcases the infeasible-cell rendering. Sweeps comm "
            "qubits per group K from 1..6 and buffer qubits per group "
            "B from 1..6 with QFT-32 on a 4-core ring, cores pinned. "
            "Every cell with B > K violates the per-group rule "
            "(buffer count cannot exceed comm count) and skips "
            "compilation entirely — the engine writes a NaN result row "
            "and the heatmap renders that cell as a white square. The "
            "result is a clean upper-triangular block of skipped cells, "
            "with the feasible diagonal-and-below filling in coloured."
        ),
        sweep_axes=[
            ("communication_qubits", 1, 6),
            ("buffer_qubits", 1, 6),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 32,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "all_to_all",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="heatmap",
        output_metric="overall_fidelity",
        max_cold=64,
        max_workers=1,
    ),
    ExampleSpec(
        id="2d_cores_logical",
        label="2D — Cores × Logical qubits (architecture heatmap)",
        description=(
            "Joint sweep over partition count and circuit size on QFT "
            "with linear intra-core chips. Cores is pinned as the "
            "architectural input, qpc derives per cell to fit the "
            "logical circuit. Heatmap reveals where splitting starts to "
            "pay off and how the cores=2 trough widens as the algorithm "
            "grows."
        ),
        sweep_axes=[
            ("num_cores", 1, 6),
            ("num_logical_qubits", 16, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="heatmap",
        output_metric="overall_fidelity",
        max_cold=36,
        max_workers=1,
    ),
    ExampleSpec(
        id="2d_facet_routing_cores_logical",
        label="2D faceted — HQA+Sabre vs TeleSABRE × Cores × Logical",
        description=(
            "Side-by-side cores × logical-qubits heatmaps for the two "
            "routing algorithms. TeleSABRE's joint inter/intra-core "
            "scheduler vs HQA + Sabre's two-stage pipeline produce "
            "visibly different regions of best fidelity, especially at "
            "higher core counts."
        ),
        sweep_axes=[
            ("routing_algorithm", ["hqa_sabre", "telesabre"]),
            ("num_cores", 2, 6),
            ("num_logical_qubits", 32, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="heatmap",
        output_metric="overall_fidelity",
        max_cold=20,
        max_workers=1,
    ),
    ExampleSpec(
        id="3d_comm_cores_logical",
        label="3D — Comm × Cores × Logical (DSE cube)",
        description=(
            "Three architectural axes form the full DSE cube: how many "
            "comm slots per group, how many cores, and at what logical "
            "scale. Frozen-heatmap slices along the logical axis show "
            "where extra comm slots pay off — at small circuits cores=1 "
            "wins regardless of K, but as logical qubits grow the higher "
            "K cells unlock fidelity that single-core can't match."
        ),
        sweep_axes=[
            ("communication_qubits", 1, 3),
            ("num_cores", 2, 6),
            ("num_logical_qubits", 32, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="frozen_heatmap",
        frozen_axis=2,
        output_metric="overall_fidelity",
        max_cold=48,
        max_workers=1,
    ),
    ExampleSpec(
        id="4d_circuit_cores_logical_2qerr",
        label="4D — Circuit × Cores × Logical × 2Q error (parallel)",
        description=(
            "Four-axis scan that mixes architecture (cores), circuit "
            "size (logical qubits), algorithm (QFT vs GHZ vs random), "
            "and the dominant noise channel (2Q gate error). Parallel "
            "coordinates expose which circuits soak up extra cores "
            "cleanly (GHZ) and which collapse early as 2Q error rises "
            "(QFT)."
        ),
        sweep_axes=[
            ("circuit_type", ["qft", "ghz", "random"]),
            ("num_cores", 2, 6),
            ("num_logical_qubits", 32, 128),
            ("two_gate_error", -5, -3.3),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={**_LOW_NOISE},
        view_type="parallel",
        output_metric="overall_fidelity",
        max_cold=20,
        max_workers=1,
    ),
    ExampleSpec(
        id="3d_pareto_qft_arch",
        label="Pareto — QFT arch DSE (F vs EPR cost)",
        description=(
            "Architectural DSE on QFT with explicit fidelity-vs-cost "
            "tension. Sweeps cores (1–6), comm slots (1–3), and logical "
            "qubits (48–96) and opens on the Pareto tab — total EPR pairs "
            "(cost, lower=better) on X, overall fidelity (quality, "
            "higher=better) on Y. cores=1 sits at EPR=0 (no inter-core "
            "traffic) but its intra-core swap depth caps fidelity; "
            "higher cores trade EPR cost for shallower intra-core "
            "routing. The bundled FoM, F / (EPR + α·time), tells the "
            "same story through a scalar lens."
        ),
        sweep_axes=[
            ("num_cores", 1, 6),
            ("communication_qubits", 1, 3),
            ("num_logical_qubits", 48, 96),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_logical_qubits": 64,
            "num_cores": 4,
            "qubits_per_core": 16,
            "pin_axis": "cores",
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise=_LOW_NOISE,
        view_type="pareto",
        output_metric="overall_fidelity",
        fom={
            "name": "Fidelity / (EPR + α·time)",
            "numerator": "overall_fidelity",
            "denominator": "cost",
            "intermediates": [
                ["cost", "max(total_epr_pairs + 1e-9 * total_circuit_time_ns, 1)"],
            ],
        },
        pareto_x="total_epr_pairs",
        pareto_y="overall_fidelity",
        max_cold=64,
        max_workers=1,
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
