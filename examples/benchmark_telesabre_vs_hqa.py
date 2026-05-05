"""
Benchmark: TeleSABRE vs HQA+Sabre
===================================
Compares SWAP count and inter-core communication count for both algorithms
on QFT-25, GHZ-25 and AE-25 circuits mapped to the A_grid_2_2_3_3 device
(4 cores × 9 qubits = 36 qubits total).

Run from the repo root:
    python examples/benchmark_telesabre_vs_hqa.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from qiskit import qasm2
from qiskit.qasm2 import CustomInstruction
from qiskit.circuit.library import SXGate
from qiskit.transpiler import CouplingMap

# sx is an IBM native gate; Qiskit's qelib1.inc doesn't include it,
# so we register it manually whenever loading these QASM files.
_CUSTOM_INSTRUCTIONS = [
    CustomInstruction("sx", 0, 1, SXGate, builtin=True),
]

# Make sure the qusim package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from qusim import map_circuit, telesabre_map_circuit
from qusim.hqa.placement import InitialPlacement

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "telesabre"
DEVICE_JSON = FIXTURE_DIR / "devices" / "A_grid_2_2_3_3.json"
CONFIG_JSON = FIXTURE_DIR / "configs" / "default.json"

CIRCUITS = {
    "QFT-25":  FIXTURE_DIR / "circuits" / "qasm_25" / "qft_nativegates_ibm_qiskit_opt3_25.qasm",
    "GHZ-25":  FIXTURE_DIR / "circuits" / "qasm_25" / "ghz_nativegates_ibm_qiskit_opt3_25.qasm",
    "AE-25":   FIXTURE_DIR / "circuits" / "qasm_25" / "ae_nativegates_ibm_qiskit_opt3_25.qasm",
}

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def load_device_topology():
    """Parse A_grid_2_2_3_3.json and build CouplingMap + core_mapping."""
    with open(DEVICE_JSON) as f:
        dev = json.load(f)["device"]

    num_qubits = dev["num_qubits"]       # 36
    num_cores  = dev["num_cores"]        # 4
    core_cap   = num_qubits // num_cores  # 9

    core_mapping = {p: p // core_cap for p in range(num_qubits)}

    edges = []
    for e in dev["intra_core_edges"]:
        edges += [(e[0], e[1]), (e[1], e[0])]
    for e in dev["inter_core_edges"]:
        edges += [(e[0], e[1]), (e[1], e[0])]

    coupling_map = CouplingMap(couplinglist=edges)
    return coupling_map, core_mapping

# ---------------------------------------------------------------------------
# HQA+Sabre runner
# ---------------------------------------------------------------------------

def run_hqa_sabre(circuit_path: Path, coupling_map: CouplingMap, core_mapping: dict) -> dict:
    """
    Run HQA+Sabre via qusim.map_circuit and return {swaps, teleportations, fidelity}.
    """
    circuit = qasm2.load(str(circuit_path), custom_instructions=_CUSTOM_INSTRUCTIONS)
    result = map_circuit(
        circuit=circuit,
        full_coupling_map=coupling_map,
        core_mapping=core_mapping,
        seed=42,
        initial_placement=InitialPlacement.RANDOM,
    )
    return {
        "swaps":          result.total_swaps,
        "teleportations": result.total_teleportations,
        "fidelity":       result.overall_fidelity,
    }


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark():
    print("Loading device topology …")
    coupling_map, core_mapping = load_device_topology()

    ts_results  = {}
    hqa_results = {}

    for name, path in CIRCUITS.items():
        print(f"\n[{name}] Running TeleSABRE …", flush=True)
        ts_result = telesabre_map_circuit(
            circuit_path=path,
            device_json_path=DEVICE_JSON,
            config_json_path=CONFIG_JSON,
        )
        ts_results[name] = {
            "swaps":    ts_result.total_swaps,
            "teleportations": ts_result.total_teleportations,
            "fidelity": ts_result.overall_fidelity,
        }
        print(f"  TeleSABRE → swaps={ts_results[name]['swaps']}, "
              f"teleportations={ts_results[name]['teleportations']}, "
              f"fidelity={ts_results[name]['fidelity']:.4f}")

        print(f"[{name}] Running HQA+Sabre …", flush=True)
        hqa_results[name] = run_hqa_sabre(path, coupling_map, core_mapping)
        print(f"  HQA+Sabre → swaps={hqa_results[name]['swaps']}, "
              f"teleportations={hqa_results[name]['teleportations']}, "
              f"fidelity={hqa_results[name]['fidelity']:.4f}")

    return ts_results, hqa_results

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_benchmark(ts_results: dict, hqa_results: dict, out_file: str = "examples/benchmark_telesabre_vs_hqa.png"):
    circuit_names = list(CIRCUITS.keys())
    n = len(circuit_names)
    x = np.arange(n)
    w = 0.35  # bar width

    ts_swaps   = [ts_results[c]["swaps"]                              for c in circuit_names]
    ts_comms   = [ts_results[c]["teleportations"]                              for c in circuit_names]
    ts_fid     = [ts_results[c]["fidelity"]                           for c in circuit_names]
    hqa_swaps  = [hqa_results[c]["swaps"]                             for c in circuit_names]
    hqa_comms  = [hqa_results[c]["teleportations"]                    for c in circuit_names]
    hqa_fid    = [hqa_results[c]["fidelity"]                          for c in circuit_names]

    BLUE   = "#4C72B0"
    ORANGE = "#DD8452"

    fig, (ax_swap, ax_comm, ax_fid) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "TeleSABRE vs HQA+Sabre — A_grid_2_2_3_3 (4 cores × 9 qubits)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── SWAPs ────────────────────────────────────────────────────────────────
    bars_ts  = ax_swap.bar(x - w/2, ts_swaps,  w, label="TeleSABRE", color=BLUE,   zorder=3)
    bars_hqa = ax_swap.bar(x + w/2, hqa_swaps, w, label="HQA+Sabre", color=ORANGE, zorder=3)

    ax_swap.set_title("SWAP gates (intra-core routing)", fontsize=11)
    ax_swap.set_ylabel("Count")
    ax_swap.set_xticks(x)
    ax_swap.set_xticklabels(circuit_names)
    ax_swap.legend()
    ax_swap.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_swap.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _label_bars(ax_swap, bars_ts)
    _label_bars(ax_swap, bars_hqa)

    # ── Cross-core communications ─────────────────────────────────────────────
    bars_ts2  = ax_comm.bar(x - w/2, ts_comms,  w, label="TeleSABRE (teleportations)",     color=BLUE,   zorder=3)
    bars_hqa2 = ax_comm.bar(x + w/2, hqa_comms, w, label="HQA+Sabre (teleportations)",    color=ORANGE, zorder=3)

    ax_comm.set_title("Cross-core communications", fontsize=11)
    ax_comm.set_ylabel("Count")
    ax_comm.set_xticks(x)
    ax_comm.set_xticklabels(circuit_names)
    ax_comm.legend(fontsize=8)
    ax_comm.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax_comm.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _label_bars(ax_comm, bars_ts2)
    _label_bars(ax_comm, bars_hqa2)

    # ── Overall fidelity (log scale) ─────────────────────────────────────────
    # Clamp to a minimum floor so log scale doesn't break on true-zero values
    _floor = 1e-12
    ts_fid_plot  = [max(v, _floor) for v in ts_fid]
    hqa_fid_plot = [max(v, _floor) for v in hqa_fid]

    bars_ts3  = ax_fid.bar(x - w/2, ts_fid_plot,  w, label="TeleSABRE",             color=BLUE,   zorder=3)
    bars_hqa3 = ax_fid.bar(x + w/2, hqa_fid_plot, w, label="HQA+Sabre",             color=ORANGE, zorder=3)

    ax_fid.set_title("Overall fidelity (log scale)", fontsize=11)
    ax_fid.set_ylabel("Fidelity")
    ax_fid.set_xticks(x)
    ax_fid.set_xticklabels(circuit_names)
    ax_fid.legend(fontsize=8)
    ax_fid.set_yscale("log")
    ax_fid.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    _label_bars_float(ax_fid, bars_ts3, fmt="{:.2e}")
    _label_bars_float(ax_fid, bars_hqa3, fmt="{:.2e}")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_file}")
    plt.show()


def _label_bars(ax, bars):
    """Add an integer value label on top of each bar."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                str(int(h)),
                ha="center", va="bottom", fontsize=8,
            )


def _label_bars_float(ax, bars, fmt="{:.2e}"):
    """Add a float value label on top of each bar (works on log-scale axes)."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.5,  # 1.5× on log scale gives a small upward nudge
                fmt.format(h),
                ha="center", va="bottom", fontsize=7,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_summary_table(ts_results: dict, hqa_results: dict) -> None:
    """Print a comparison table the report can quote verbatim.

    Includes raw counts plus the SWAP and teleport ratios that drive
    the fidelity verdict (SWAP cost = 3 × CNOT under the noise model;
    each teleport hop carries the bundled EPR + buffer-SWAP cost).
    """
    rows = []
    for name in ts_results:
        ts = ts_results[name]
        hq = hqa_results[name]
        rows.append({
            "circuit": name,
            "ts_swaps": ts["swaps"], "hq_swaps": hq["swaps"],
            "ts_tele":  ts["teleportations"], "hq_tele":  hq["teleportations"],
            "ts_fid":   ts["fidelity"], "hq_fid":   hq["fidelity"],
        })

    print()
    print("=" * 78)
    print("Summary — TeleSABRE vs HQA+Sabre on A_grid_2_2_3_3 (4 cores × 9 qubits)")
    print("=" * 78)
    print(f"{'circuit':>10} {'algo':>10} "
          f"{'swaps':>7} {'teleports':>10} {'fidelity':>10}")
    for r in rows:
        print(f"{r['circuit']:>10} {'TeleSABRE':>10} "
              f"{r['ts_swaps']:>7} {r['ts_tele']:>10} {r['ts_fid']:>10.4f}")
        print(f"{r['circuit']:>10} {'HQA+Sabre':>10} "
              f"{r['hq_swaps']:>7} {r['hq_tele']:>10} {r['hq_fid']:>10.4f}")

    # Aggregate: TeleSABRE / HQA ratio per metric, geometric mean across
    # circuits. Helpful one-line takeaway for the report.
    import math
    swap_ratios = [r["ts_swaps"] / max(1, r["hq_swaps"]) for r in rows]
    tele_ratios = [r["ts_tele"]  / max(1, r["hq_tele"])  for r in rows]
    fid_ratios  = [
        r["ts_fid"] / r["hq_fid"] if r["hq_fid"] > 0 else float("inf")
        for r in rows
    ]
    def _gmean(xs):
        finite = [x for x in xs if math.isfinite(x) and x > 0]
        if not finite: return float("nan")
        return math.exp(sum(math.log(x) for x in finite) / len(finite))
    print()
    print(f"  geo-mean SWAP ratio      (TeleSABRE / HQA): {_gmean(swap_ratios):.2f}")
    print(f"  geo-mean teleport ratio  (TeleSABRE / HQA): {_gmean(tele_ratios):.2f}")
    print(f"  geo-mean fidelity ratio  (TeleSABRE / HQA): {_gmean(fid_ratios):.2f}")


if __name__ == "__main__":
    ts_results, hqa_results = run_benchmark()
    print_summary_table(ts_results, hqa_results)
    plot_benchmark(ts_results, hqa_results)
