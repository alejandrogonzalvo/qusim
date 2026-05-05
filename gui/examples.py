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

    # Sweep axes mix two row shapes:
    #   numeric:     ``(metric_key, slider_low, slider_high)``
    #   categorical: ``(metric_key, [value, value, …])``
    # ``slider_low``/``slider_high`` are slider positions: log10 exponents
    # for log-scale metrics, raw values for linear ones.
    sweep_axes: list[tuple] = field(default_factory=list)

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
    # Worker pool size for the parallel cold-path. None lets the scheduler
    # decide based on available memory; pin to 1 for examples that
    # individually need a lot of RAM (128-qubit QFT compiles).
    max_workers: int | None = None
    # Optional FoM override (matches ``FomConfig.to_dict()``). None falls
    # back to ``DEFAULT_FOM`` in the generator.
    fom: dict | None = None
    # Optional Pareto-view axis overrides written into the session
    # ``view`` block. ``None`` keeps the GUI defaults
    # (``total_epr_pairs`` × ``overall_fidelity``).
    pareto_x: str | None = None
    pareto_y: str | None = None


# Quiet-noise baseline. The architectural examples crank gate / coherence
# noise way down so that the *architectural* differences (cores, comm
# slots, routing strategy) drive the fidelity variation rather than noise
# floor swamping everything at large qubit counts. Examples that *do*
# sweep noise (the 4-D one) override these per-axis.
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


# Five canned sessions covering 1-D through 4-D. Each one is built around
# *architectural* sweep axes (qubits, cores, comm slots, routing strategy,
# circuit type) — the dimensions where the design space actually has
# tension. The 4-D example folds in a single noise channel so the loaded
# session shows what real DSE feels like.
#
# Names lead with the dimensionality + the headline graph property so the
# user can pick a visualisation without reading the descriptions.
EXAMPLES: list[ExampleSpec] = [
    ExampleSpec(
        id="1d_cores_sweet_spot_qft64",
        label="1D — Cores sweet spot (QFT-64, line)",
        description=(
            "Sweep over partition count for a 64-qubit QFT on a 64-physical "
            "ring of linear-chain cores. cores=1 is the trivial winner "
            "(fully on-chip routing); cores=2 collapses because the 2-core "
            "ring is degenerate (one inter-core link carries all traffic); "
            "cores=4–6 recover much of the lost fidelity by spreading the "
            "EPR pressure. The d²F/dx² view (mode picker on the right) "
            "highlights the sharp inflection between cores=1 and cores=2."
        ),
        sweep_axes=[("num_cores", 1, 6)],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
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
        max_cold=6,
        max_workers=1,
    ),
    ExampleSpec(
        id="2d_cores_qubits",
        label="2D — Cores × Qubits (architecture heatmap)",
        description=(
            "Joint sweep over partition count and algorithm size on QFT "
            "with linear intra-core chips. The heatmap reveals where "
            "splitting starts to pay off and how the cores=2 trough widens "
            "as the algorithm grows. cores=1 wins outright at small "
            "scales; many small chips beat a single large chip past ~64 "
            "qubits."
        ),
        sweep_axes=[
            ("num_cores", 1, 6),
            ("qubits", 16, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
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
        id="2d_facet_routing_cores_qubits",
        label="2D faceted — HQA+Sabre vs TeleSABRE × Cores × Qubits",
        description=(
            "Side-by-side cores × qubits heatmaps for the two routing "
            "algorithms. TeleSABRE's joint inter/intra-core scheduler vs "
            "HQA + Sabre's two-stage pipeline produce visibly different "
            "regions of best fidelity, especially at higher core counts."
        ),
        sweep_axes=[
            ("routing_algorithm", ["hqa_sabre", "telesabre"]),
            ("num_cores", 2, 6),
            ("qubits", 32, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
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
        id="3d_comm_cores_qubits",
        label="3D — Comm × Cores × Qubits (DSE cube)",
        description=(
            "Three architectural axes form the full DSE cube: how many "
            "comm slots per group, how many cores, and at what scale. "
            "Frozen-heatmap slices along the qubits axis show where extra "
            "comm slots actually pay off; in the rest of the volume they "
            "only steal data slots from the algorithm."
        ),
        sweep_axes=[
            ("communication_qubits", 1, 3),
            ("num_cores", 2, 6),
            ("qubits", 32, 128),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
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
        id="4d_circuit_cores_qubits_2qerr",
        label="4D — Circuit × Cores × Qubits × 2Q error (parallel)",
        description=(
            "Four-axis scan that mixes architecture (cores, qubits), "
            "algorithm (QFT vs GHZ vs random), and the dominant noise "
            "channel (2Q gate error). Parallel coordinates expose which "
            "circuits soak up extra cores cleanly (GHZ) and which collapse "
            "early as 2Q error rises (QFT)."
        ),
        sweep_axes=[
            ("circuit_type", ["qft", "ghz", "random"]),
            ("num_cores", 2, 6),
            ("qubits", 32, 128),
            ("two_gate_error", -5, -3.3),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
            "communication_qubits": 1,
            "buffer_qubits": 1,
            "topology_type": "ring",
            "intracore_topology": "linear",
            "placement_policy": "spectral",
            "routing_algorithm": "hqa_sabre",
            "seed": 42,
        },
        fixed_noise={
            **_LOW_NOISE,
            # 2Q error is on the sweep axis — drop the baseline override
            # so the engine reads it from each cell.
        },
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
            "tension.  Sweeps cores (1–6), comm slots (1–3), and qubit "
            "count (48–96) and opens on the Pareto tab — total EPR pairs "
            "(cost, lower=better) on X, overall fidelity (quality, "
            "higher=better) on Y.\n\n"
            "Two non-dominated regimes show up on the frontier:\n"
            "  • cores=1 (EPR=0): no inter-core traffic, fidelity is "
            "limited by intra-core swap depth on the linear chip.\n"
            "  • cores=6 (≈61 EPR pairs at 48 qubits): pays a small "
            "inter-core cost in exchange for a much shallower intra-core "
            "routing problem, raising fidelity from 0.90 to 0.99.\n\n"
            "Everything in between (mid-cores, larger qubit counts) is "
            "dominated — strictly worse on at least one axis.  The bundled "
            "FoM, F / (EPR + α·time), tells the same story through a "
            "scalar lens; the Merit tab's iso-FoM lines mirror the "
            "frontier."
        ),
        sweep_axes=[
            ("num_cores", 1, 6),
            ("communication_qubits", 1, 3),
            ("qubits", 48, 96),
        ],
        cold_config={
            "circuit_type": "qft",
            "num_qubits": 64,
            "num_logical_qubits": 64,
            "num_cores": 4,
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
        # Same shape as ``PRESETS["fidelity_over_cost"]`` from gui/fom.py.
        # Kept inline here so editing examples.py doesn't require touching
        # fom.py — the generator round-trips it through ``FomConfig``.
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
