"""
DSE computation engine.

Two-stage execution model:
  Cold path – full circuit transpilation + HQA mapping + SABRE routing.
              Required when circuit structure or topology changes (~1-10s).
  Hot path  – fidelity estimation only, reusing cached placements/swaps.
              Required when only noise parameters change (<1ms per point).
"""

import sys
import os

# Cap per-process thread pools BEFORE importing numpy/qiskit/qusim so their
# Rayon/OpenMP/BLAS pools initialize at 1 thread.  Otherwise each cold-path
# worker spawns ~34 threads and 8 parallel workers oversubscribe the CPU
# (272 threads on 16 cores) until the scheduler stalls and Dash's heartbeat
# times out — which the UI shows as a crash.  Workers use process-level
# parallelism already, so there is nothing to lose by disabling library
# threads here.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import time
import multiprocessing
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

_MP_CONTEXT = multiprocessing.get_context("forkserver")

# Minimum free RAM (MB) to keep available for OS + UI + browser
_RESERVED_RAM_MB = 1024

# Empirical peak RSS (MB) for one cold compilation vs. num_qubits, measured on
# QFT (densest common circuit) with the forkserver worker used in production.
# See tests/test_cores_qubits_oom.py for the measurement.
_EMPIRICAL_COLD_MB: list[tuple[int, float]] = [
    (4,   150.0),
    (40,  160.0),
    (76,  260.0),
    (112, 450.0),
    (148, 820.0),
    (184, 1600.0),
    (220, 2100.0),
    (256, 3800.0),
]


# Peak memory per hot-path grid cell. The sweep grid is a structured numpy
# array of ``_RESULT_DTYPE``: 7 × float64 = 56 B per cell, flat in memory
# (no per-cell Python overhead). 128 B keeps a ~2x safety margin for the
# transient noise-dict / chunk buffers the sweep holds alongside the grid.
_BYTES_PER_HOT_POINT = 128

# Reserved headroom for OS + browser + UI when computing the hot-path
# ceiling. Larger than ``_RESERVED_RAM_MB`` (cold path) because the main
# process — not a short-lived worker — is the one that has to hold the
# full result grid for the lifetime of the plot.
_RESERVED_RAM_MB_HOT = 3072


def _max_hot_points_for_memory() -> int:
    """Maximum hot-path grid size that fits in currently available RAM.

    Sweep results live in a numpy object grid of Python dicts until the
    plot is rendered. Exceeding ``MemAvailable`` gets the process
    OOM-killed by the kernel, which surfaces in the UI as a "crash".
    Returning a conservative ceiling lets :meth:`DSEEngine.sweep_nd` clamp
    a user-requested ``max_hot`` below the danger zone — the sweep then
    either fits, or fails loudly with the usual "hot budget too tight"
    guard. Either way, the process stays alive.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
                    break
            else:
                return 1_000_000
    except OSError:
        return 1_000_000
    budget_bytes = max(0, avail_mb - _RESERVED_RAM_MB_HOT) * 1024 * 1024
    return max(10_000, budget_bytes // _BYTES_PER_HOT_POINT)


def _estimate_cold_mb(num_qubits: int) -> float:
    """Peak RSS estimate (MB) for one cold compilation at ``num_qubits``.

    Piecewise-linear interpolation over :data:`_EMPIRICAL_COLD_MB`, with
    linear extrapolation beyond the measured range.
    """
    pts = _EMPIRICAL_COLD_MB
    if num_qubits <= pts[0][0]:
        return pts[0][1]
    if num_qubits >= pts[-1][0]:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (num_qubits - x2)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if x1 <= num_qubits <= x2:
            return y1 + (y2 - y1) * (num_qubits - x1) / (x2 - x1)
    return pts[-1][1]

import numpy as np
import qiskit
from qiskit import transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import qusim
from qusim.hqa.placement import InitialPlacement

from .constants import (
    CAT_METRIC_BY_KEY,
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    SWEEP_POINTS_1D,
    SWEEP_POINTS_2D,
    SWEEP_POINTS_3D,
    SWEEP_POINTS_COLD_1D,
    SWEEP_POINTS_COLD_2D,
    SWEEP_POINTS_COLD_3D,
    MAX_TOTAL_POINTS_HOT,
    MAX_TOTAL_POINTS_COLD,
    MAX_COLD_COMPILATIONS,
    MIN_POINTS_PER_AXIS,
)


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _build_circuit(
    circuit_type: str,
    num_qubits: int,
    seed: int,
    qasm_str: str | None = None,
) -> qiskit.QuantumCircuit:
    if circuit_type == "custom":
        if not qasm_str:
            raise ValueError("circuit_type='custom' requires a non-empty qasm_str")
        from qiskit import qasm2
        return qasm2.loads(qasm_str)
    if circuit_type == "qft":
        circ = QFT(num_qubits)
    elif circuit_type == "ghz":
        circ = qiskit.QuantumCircuit(num_qubits)
        circ.h(0)
        for i in range(1, num_qubits):
            circ.cx(0, i)
    elif circuit_type == "random":
        from qiskit.circuit.random import random_circuit
        circ = random_circuit(num_qubits, depth=max(3, num_qubits // 2), seed=seed)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    return circ


def _transpile_circuit(circ: qiskit.QuantumCircuit, seed: int) -> qiskit.QuantumCircuit:
    return transpile(
        circ,
        basis_gates=["x", "cx", "cp", "rz", "h", "s", "sdg", "t", "tdg", "measure"],
        optimization_level=0,
        seed_transpiler=seed,
    )


# ---------------------------------------------------------------------------
# Topology builders  (all-to-all intra-core, configurable inter-core)
# ---------------------------------------------------------------------------

def inter_core_neighbors(num_cores: int, inter_topology: str) -> list[list[int]]:
    """Per-core list of unique neighbouring core indices."""
    import math
    if num_cores < 2:
        return [[] for _ in range(num_cores)]
    nbrs: list[list[int]] = [[] for _ in range(num_cores)]
    inter = (inter_topology or "ring").lower()
    if inter == "ring":
        for c in range(num_cores):
            nbrs[c].append((c + 1) % num_cores)
            if num_cores > 2:
                nbrs[c].append((c - 1) % num_cores)
    elif inter == "linear":
        for c in range(num_cores):
            if c + 1 < num_cores:
                nbrs[c].append(c + 1)
            if c - 1 >= 0:
                nbrs[c].append(c - 1)
    elif inter == "all_to_all":
        for c in range(num_cores):
            for c2 in range(num_cores):
                if c2 != c:
                    nbrs[c].append(c2)
    elif inter == "grid":
        side = math.ceil(math.sqrt(num_cores))
        for c in range(num_cores):
            row, col = divmod(c, side)
            if col + 1 < side and c + 1 < num_cores:
                nbrs[c].append(c + 1)
            if col > 0 and c - 1 >= 0:
                nbrs[c].append(c - 1)
            if c + side < num_cores:
                nbrs[c].append(c + side)
            if c - side >= 0:
                nbrs[c].append(c - side)
    else:
        for c in range(num_cores):
            nbrs[c].append((c + 1) % num_cores)
            if num_cores > 2:
                nbrs[c].append((c - 1) % num_cores)
    return [sorted(set(n)) for n in nbrs]


def core_groups_for(num_cores: int, inter_topology: str) -> list[list[int]]:
    """Per-core *ordered* list of partner cores — one entry per inter-core link.

    A core's ``g``-th group of comm qubits is dedicated to the ``g``-th
    partner returned here.  Ordering is the sorted neighbour list (already
    deterministic in :func:`inter_core_neighbors`).
    """
    return inter_core_neighbors(num_cores, inter_topology)


def num_comm_groups(num_cores: int, inter_topology: str) -> list[int]:
    """Number of comm-qubit groups per core (= number of inter-core neighbours)."""
    return [len(g) for g in core_groups_for(num_cores, inter_topology)]


def core_slot_layout(core_size: int, num_groups: int, k_per_group: int) -> dict:
    """Legacy slot layout — kept for back-compat callers.

    The layout places data first, then ``G`` consecutive groups of
    ``K`` comm + ``1`` buffer slots in slot-order.  New code should use
    :func:`assign_core_slots` instead, which respects the
    "comm on edge, buffer adjacent to comm" placement rules and works
    for grid intra-core topologies.
    """
    G = max(0, int(num_groups))
    K = max(0, int(k_per_group))
    reserved = G * (K + 1)
    data_count = core_size - reserved
    feasible = data_count >= 1 and G >= 0 and K >= 0

    def comm_slot(g: int, k: int) -> int:
        return data_count + g * (K + 1) + k

    def buffer_slot(g: int) -> int:
        return data_count + g * (K + 1) + K

    return {
        "data_count": data_count,
        "reserved": reserved,
        "feasible": feasible,
        "comm_slot": comm_slot,
        "buffer_slot": buffer_slot,
    }


def _grid_side(core_size: int) -> int:
    import math
    side = math.isqrt(core_size)
    if side * side < core_size:
        side += 1
    return side


def _grid_neighbours(slot: int, side: int, core_size: int) -> list[int]:
    """4-neighbours of ``slot`` in a side×side grid (row-major), filtered to
    valid slots (0 ≤ neighbour < core_size)."""
    row, col = divmod(slot, side)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < side and 0 <= nc < side:
            n = nr * side + nc
            if 0 <= n < core_size:
                out.append(n)
    return out


def assign_core_slots(
    core_size: int,
    intracore_topology: str,
    num_groups: int,
    k_per_group: int,
    b_per_group: int = 1,
) -> dict:
    """Decide which local slot each role (data / comm / buffer) occupies.

    Returns a dict::

      {
        "data": [slot_idx, ...],
        "groups": [
          {"comm": [slot_idx, ...], "buffer": [slot_idx, ...]},
          ...
        ],
        "data_count": int,
        "feasible": bool,
      }

    Each group reserves ``K`` comm + ``B`` buffer slots (``B ≤ K`` by
    rule).  For grid intra-core, comm slots are pinned to a side edge
    (left / right / top / bottom in priority order) and buffer slots
    are picked from cells **adjacent** (4-neighbourhood) to the comm
    column, preferring interior cells for visual cleanliness.

    For non-grid intra-core (linear / ring / all-to-all) we use a
    slot-order fallback (data first, then per-group [K comm + B buffer]).
    """
    G = max(0, int(num_groups))
    K = max(0, int(k_per_group))
    B = max(1, int(b_per_group or 1)) if K > 0 else 0
    reserved = G * (K + B)
    data_count = core_size - reserved
    intra = (intracore_topology or "all_to_all").lower()
    # ``all_to_all`` shares the row-major grid layout (the local-positions
    # helper falls through to the grid case), so apply the same
    # edge-aware placement rules here.
    use_grid_layout = intra in ("grid", "all_to_all")

    if G == 0 or K == 0 or not use_grid_layout:
        # Slot-order fallback — also covers K=0 / single-core paths and
        # genuinely non-grid layouts (linear, ring) where "edge" isn't a
        # meaningful concept.
        groups = []
        offset = max(0, data_count)
        for _ in range(G):
            comm = list(range(offset, offset + K))
            offset += K
            buffer = list(range(offset, offset + B))
            offset += B
            groups.append({"comm": comm, "buffer": buffer})
        data_slots = list(range(max(0, data_count)))
        return {
            "data": data_slots,
            "groups": groups,
            "data_count": max(0, data_count),
            "feasible": data_count >= 1,
        }

    # ---- grid intra-core: pick edge slots per group + adjacent buffer ----
    side = _grid_side(core_size)
    rows_per_side = side
    cols_per_side = side
    left_edge = [r * side for r in range(rows_per_side)
                 if r * side < core_size]
    right_edge = [r * side + (cols_per_side - 1) for r in range(rows_per_side)
                  if r * side + (cols_per_side - 1) < core_size]
    top_edge = [c for c in range(cols_per_side) if c < core_size]
    bottom_edge = [(rows_per_side - 1) * side + c for c in range(cols_per_side)
                   if (rows_per_side - 1) * side + c < core_size]
    side_options = [left_edge, right_edge, top_edge, bottom_edge]
    edge_set = set().union(left_edge, right_edge, top_edge, bottom_edge)

    used_comm: set[int] = set()
    used_buffer: set[int] = set()
    groups_out: list[dict] = []

    def _centre_dist(s: int) -> float:
        r, c = divmod(s, side)
        return abs(r - (side - 1) / 2) + abs(c - (side - 1) / 2)

    for g in range(G):
        # Pick this group's preferred side; cycle through if more than 4 groups.
        side_slots = side_options[g % len(side_options)]
        avail = [s for s in side_slots
                 if s not in used_comm and s not in used_buffer]
        if K > len(avail):
            extras = [
                s for s in range(core_size)
                if s not in used_comm and s not in used_buffer
                and s not in avail
            ]
            avail = avail + extras
        start = max(0, (len(avail) - K) // 2)
        comm_slots = list(avail[start:start + K])
        for s in comm_slots:
            used_comm.add(s)

        # Buffer slots: pick B distinct cells adjacent to *any* comm in
        # this group, preferring interior cells.  Once interior options
        # are exhausted, allow any free adjacent cell, then any free
        # cell anywhere (last-resort fallback).
        buf_slots: list[int] = []
        candidates: list[int] = []
        seen = set()
        for cs in comm_slots:
            for n in _grid_neighbours(cs, side, core_size):
                if n in used_comm or n in used_buffer or n in seen:
                    continue
                seen.add(n)
                candidates.append(n)

        def _buf_key(s: int) -> tuple[int, int]:
            on_edge = 1 if s in edge_set else 0
            return (on_edge, int(_centre_dist(s) * 100))

        candidates.sort(key=_buf_key)
        for s in candidates:
            if len(buf_slots) >= B:
                break
            buf_slots.append(s)
            used_buffer.add(s)

        if len(buf_slots) < B:
            # Fall back to any free cell so we always emit B buffers.
            for s in range(core_size):
                if len(buf_slots) >= B:
                    break
                if s in used_comm or s in used_buffer:
                    continue
                buf_slots.append(s)
                used_buffer.add(s)

        groups_out.append({
            "comm": comm_slots,
            "buffer": buf_slots,
        })

    data_slots = [s for s in range(core_size)
                  if s not in used_comm and s not in used_buffer]
    return {
        "data": data_slots,
        "groups": groups_out,
        "data_count": len(data_slots),
        "feasible": len(data_slots) >= 1,
    }


def inter_core_edges(
    num_cores: int,
    communication_qubits: int,
    inter_topology: str,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Comm-qubit-to-comm-qubit edges across the multi-core fabric.

    Each comm qubit hosts **exactly one** inter-core link.  A core with
    ``G`` neighbours therefore exposes ``G`` *groups* of comm qubits — one
    per partner core — and the slider value ``K`` is the number of comm
    qubits in each group (so the per-core comm count is ``G·K``).  For
    every unordered pair of neighbouring cores ``(c, c')`` we pair comm
    qubit ``i`` of ``c``'s group-toward-``c'`` with comm qubit ``i`` of
    ``c'``'s group-toward-``c``, giving ``K`` parallel cross-links between
    them (e.g. ring with ``K=2`` becomes two concentric rings).

    Returns
    -------
    list of ``((core_a, group_a, k), (core_b, group_b, k))`` edges, where
    ``group_a`` is core ``a``'s group index dedicated to core ``b`` (and
    vice versa) and ``k`` is the same comm-qubit index on both ends.
    """
    if num_cores < 2 or communication_qubits < 1:
        return []
    K = int(communication_qubits)
    nbrs = inter_core_neighbors(num_cores, inter_topology)
    # Cache c2 → group index lookups
    group_of: list[dict[int, int]] = [
        {n: g for g, n in enumerate(nbrs[c])} for c in range(num_cores)
    ]
    edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for c in range(num_cores):
        for c2 in nbrs[c]:
            if c2 <= c:
                continue
            g_a = group_of[c][c2]
            g_b = group_of[c2][c]
            for k in range(K):
                edges.append(((c, g_a, k), (c2, g_b, k)))
    return edges


def _max_K_for_layout(core_size: int, num_groups: int, b_per_group: int = 1) -> int:
    """Largest ``K`` (comm per group) that leaves at least one data qubit
    given ``B`` buffer qubits per group.

    Each group reserves ``K + B`` slots, so we need
    ``core_size − G·(K+B) ≥ 1`` ⇒ ``K ≤ ⌊(core_size − 1)/G⌋ − B``.
    Returns 0 when no positive ``K`` is feasible.
    """
    if num_groups <= 0:
        return 0
    B = max(1, int(b_per_group or 1))
    return max(0, (core_size - 1) // num_groups - B)


def _max_B_for_layout(core_size: int, num_groups: int, k_per_group: int) -> int:
    """Largest feasible ``B`` (buffers per group) given ``K``.

    Bounded by both architectural feasibility (``K + B ≤ ⌊(qpc−1)/G⌋``)
    and the per-group rule (``B ≤ K``) — buffer count never exceeds
    the comm count of the same group.
    """
    if num_groups <= 0 or k_per_group <= 0:
        return 0
    K = max(1, int(k_per_group))
    arch_cap = (core_size - 1) // num_groups - K
    return max(0, min(K, arch_cap))


def _build_topology(
    num_qubits: int,
    num_cores: int,
    topology_type: str,
    intracore_topology: str = "all_to_all",
    communication_qubits: int = 1,
    buffer_qubits: int = 1,
) -> tuple[CouplingMap, dict[int, int]]:
    """Build full_coupling_map and core_mapping for the given topology.

    Each core lays out its qubits as
    ``[D data slots, group_0, group_1, …, group_{G-1}]`` where ``G`` is
    the number of inter-core neighbours and each group is ``K`` comm
    slots followed by ``1`` buffer slot.  Inter-core links pair comm
    qubit ``i`` of one core's group-toward-partner with comm qubit ``i``
    of the partner's group-toward-this — so each comm qubit hosts
    exactly one inter-core link (see :func:`inter_core_edges`).  Buffer
    slots are reserved (no inter-core edges); they participate in the
    intra-core topology only.

    Qubits are distributed across cores so that physical count == logical
    count exactly.  With 7 qubits and 3 cores the sizes are [3, 2, 2].
    """
    import math
    if num_cores < 1:
        num_cores = 1
    num_cores = min(num_cores, num_qubits)

    base = num_qubits // num_cores
    remainder = num_qubits % num_cores
    core_sizes = [base + (1 if c < remainder else 0) for c in range(num_cores)]

    cm = CouplingMap()
    for i in range(num_qubits):
        cm.add_physical_qubit(i)

    core_mapping: dict[int, int] = {}
    core_offsets: list[int] = []
    offset = 0
    for c, size in enumerate(core_sizes):
        core_offsets.append(offset)
        for q in range(size):
            core_mapping[offset + q] = c

        _add_intracore_edges(cm, offset, size, intracore_topology)
        offset += size

    if num_cores > 1:
        groups_per_core = num_comm_groups(num_cores, topology_type)
        B_req = max(1, int(buffer_qubits or 1))
        K_caps = [
            _max_K_for_layout(core_sizes[c], groups_per_core[c],
                              b_per_group=B_req)
            for c in range(num_cores)
            if groups_per_core[c] > 0
        ]
        K_max = min(K_caps) if K_caps else 0
        K = min(int(communication_qubits or 1), max(0, K_max))
        if K >= 1:
            B_caps = [
                _max_B_for_layout(core_sizes[c], groups_per_core[c], K)
                for c in range(num_cores)
                if groups_per_core[c] > 0
            ]
            B_max = min(B_caps) if B_caps else 0
            B = min(B_req, max(1, B_max))
            per_core_layout = [
                assign_core_slots(core_sizes[c], intracore_topology,
                                  groups_per_core[c], K, b_per_group=B)
                for c in range(num_cores)
            ]
            for (a_core, a_g, a_k), (b_core, b_g, b_k) in inter_core_edges(
                num_cores, K, topology_type,
            ):
                p_a = core_offsets[a_core] + per_core_layout[a_core]["groups"][a_g]["comm"][a_k]
                p_b = core_offsets[b_core] + per_core_layout[b_core]["groups"][b_g]["comm"][b_k]
                cm.add_edge(p_a, p_b)
                cm.add_edge(p_b, p_a)

    return cm, core_mapping


def _add_intracore_edges(
    cm: CouplingMap, offset: int, size: int, topology: str,
) -> None:
    import math
    if size < 2:
        return
    if topology == "all_to_all":
        for q in range(size):
            for r in range(q + 1, size):
                cm.add_edge(offset + q, offset + r)
                cm.add_edge(offset + r, offset + q)
    elif topology == "linear":
        for q in range(size - 1):
            cm.add_edge(offset + q, offset + q + 1)
            cm.add_edge(offset + q + 1, offset + q)
    elif topology == "ring":
        for q in range(size):
            nxt = (q + 1) % size
            cm.add_edge(offset + q, offset + nxt)
            cm.add_edge(offset + nxt, offset + q)
    elif topology == "grid":
        side = math.isqrt(size)
        if side * side < size:
            side += 1
        for q in range(size):
            row, col = divmod(q, side)
            if col + 1 < side and q + 1 < size:
                cm.add_edge(offset + q, offset + q + 1)
                cm.add_edge(offset + q + 1, offset + q)
            if q + side < size:
                cm.add_edge(offset + q, offset + q + side)
                cm.add_edge(offset + q + side, offset + q)


# ---------------------------------------------------------------------------
# Cached mapping data
# ---------------------------------------------------------------------------

@dataclass
class CachedMapping:
    gs_sparse: np.ndarray
    placements: np.ndarray
    distance_matrix: np.ndarray
    sparse_swaps: np.ndarray
    gate_error_arr: np.ndarray
    gate_time_arr: np.ndarray
    gate_names: list
    total_epr_pairs: int
    total_swaps: int
    total_teleportations: int
    total_network_distance: int
    # Key used to detect when re-mapping is required
    config_key: tuple
    cold_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Noise param helpers
# ---------------------------------------------------------------------------

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
# would otherwise cost ~96 KB instead of a few hundred bytes.
_RESULT_SCALAR_KEYS: tuple[str, ...] = (
    "overall_fidelity",
    "algorithmic_fidelity",
    "routing_fidelity",
    "coherence_fidelity",
    "readout_fidelity",
    "total_circuit_time_ns",
    "total_epr_pairs",
    "total_swaps",
    "total_teleportations",
    "total_network_distance",
)

# Structured dtype for the sweep grid. One float64 field per scalar output
# replaces the Python-dict cell, cutting per-cell cost from ~280 B (dict
# overhead dominates) to 7 × 8 B = 56 B + an 8 B numpy slot ≈ 64 B.
_RESULT_DTYPE = np.dtype([(k, np.float64) for k in _RESULT_SCALAR_KEYS])


def _strip_for_grid(result: dict) -> dict:
    """Return a new dict containing only the scalar fields stored per cell."""
    return {k: result[k] for k in _RESULT_SCALAR_KEYS if k in result}


# Per-qubit grids exposed to the topology view (and any other consumer that
# wants per-physical-qubit overlays).  Kept separate from the scalar grid so
# the structured-dtype memory layout is unchanged for sweep cells that don't
# need this data.
_PER_QUBIT_GRID_KEYS: tuple[str, ...] = (
    "algorithmic_fidelity_grid",
    "routing_fidelity_grid",
    "coherence_fidelity_grid",
)


def _extract_per_qubit(result: dict, cached: "CachedMapping | None", *,
                       cold_cfg: dict | None = None) -> dict:
    """Bundle per-qubit ndarrays + the cold-config bits the topology view
    needs to redraw the device structure for this cell.

    Output keys match what ``_compute_per_qubit_for_cell`` produces in
    ``gui/app.py`` so the topology view can consume either with the same
    accessor logic.
    """
    out: dict = {}
    for k in _PER_QUBIT_GRID_KEYS:
        v = result.get(k)
        if v is not None:
            out[k] = v
    if cached is not None:
        out["placements"] = getattr(cached, "placements", None)
    if cold_cfg is not None:
        # Only the bits that drive the topology view's graph layout — keeps
        # this dict small and serialisable.
        if "num_qubits" in cold_cfg:
            out["num_physical"] = int(cold_cfg["num_qubits"])
        for k in (
            "num_cores", "communication_qubits",
            "num_logical_qubits", "topology_type", "intracore_topology",
        ):
            if k in cold_cfg:
                out[k] = cold_cfg[k]
    return out


def _expand_qubits_alias(cfg: dict) -> None:
    """Expand the virtual ``qubits`` cold-path key in-place.

    ``qubits`` is a sweep alias for "physical qubits == logical qubits".
    The engine itself only knows ``num_qubits`` and ``num_logical_qubits``,
    so every place that consumes a swept cold cfg needs to translate
    ``qubits`` -> both keys before running the cold path.
    """
    if "qubits" not in cfg:
        return
    n = int(cfg.pop("qubits"))
    cfg["num_qubits"] = n
    cfg["num_logical_qubits"] = n


def total_reserved_slots(num_qubits: int, num_cores: int,
                         topology_type: str, k_per_group: int,
                         b_per_group: int = 1) -> int:
    """Total non-data slots reserved across the chip = Σ_c G(c)·(K+B).

    A core with ``G`` inter-core neighbours reserves ``G·(K+B)`` slots
    (``K`` comm + ``B`` buffer per group).  Summed over all cores this is
    the chip-wide capacity stolen from the data pool.
    """
    nc = max(1, int(num_cores or 1))
    K = max(0, int(k_per_group or 0))
    B = max(1, int(b_per_group or 1))
    if nc < 2 or K < 1:
        # Single core has no inter-core links; with K=0 nothing is reserved.
        return 0
    groups = num_comm_groups(nc, topology_type or "ring")
    return sum(groups) * (K + B)


def max_data_slots(num_qubits: int, num_cores: int,
                   topology_type: str, k_per_group: int,
                   b_per_group: int = 1) -> int:
    """Number of data (non-comm, non-buffer) slots available chip-wide."""
    nq = max(1, int(num_qubits or 1))
    return max(0, nq - total_reserved_slots(
        nq, num_cores, topology_type, k_per_group, b_per_group,
    ))


def clamp_k_for_topology(num_qubits: int, num_cores: int,
                         topology_type: str, requested_k: int,
                         b_per_group: int = 1) -> int:
    """Clamp slider value ``K`` so every core keeps ≥ 1 data slot.

    With ``B`` buffers per group, the cap shrinks: each group reserves
    ``K+B`` slots, so ``K ≤ ⌊(qpc−1)/G⌋ − B``.
    """
    nq = max(1, int(num_qubits or 1))
    nc = max(1, int(num_cores or 1))
    nc = min(nc, nq)
    base = nq // nc
    remainder = nq % nc
    core_sizes = [base + (1 if c < remainder else 0) for c in range(nc)]
    if nc < 2:
        # No inter-core links, comm slots aren't physically meaningful.
        return 1
    groups = num_comm_groups(nc, topology_type or "ring")
    B = max(1, int(b_per_group or 1))
    K_caps = [
        _max_K_for_layout(core_sizes[c], groups[c], b_per_group=B)
        for c in range(nc) if groups[c] > 0
    ]
    K_max = min(K_caps) if K_caps else 0
    return max(1, min(int(requested_k or 1), max(1, K_max)))


def clamp_b_for_topology(num_qubits: int, num_cores: int,
                         topology_type: str, k_per_group: int,
                         requested_b: int) -> int:
    """Clamp slider value ``B`` so every core keeps ≥ 1 data slot AND
    ``B ≤ K`` (per-group rule: buffers never exceed comm count)."""
    nq = max(1, int(num_qubits or 1))
    nc = max(1, int(num_cores or 1))
    nc = min(nc, nq)
    base = nq // nc
    remainder = nq % nc
    core_sizes = [base + (1 if c < remainder else 0) for c in range(nc)]
    if nc < 2:
        return 1
    K = max(1, int(k_per_group or 1))
    groups = num_comm_groups(nc, topology_type or "ring")
    B_caps = [
        _max_B_for_layout(core_sizes[c], groups[c], K)
        for c in range(nc) if groups[c] > 0
    ]
    B_max = min(B_caps) if B_caps else 0
    return max(1, min(int(requested_b or 1), max(1, B_max)))


def _clamp_cfg_comm_and_logical(cfg: dict) -> None:
    """In-place: clamp ``communication_qubits``, ``buffer_qubits``, and
    ``num_logical_qubits`` to the architectural caps for the current
    (num_qubits, num_cores, topology_type) combination.

    Each of a core's ``G`` inter-core neighbours reserves ``K+B`` slots
    (K comm + B buffer per group), so logical qubits can use only
    ``num_qubits − Σ_c G(c)·(K+B)`` slots.  The per-group rule
    ``B ≤ K`` is also enforced here.
    """
    nq = int(cfg.get("num_qubits", 1) or 1)
    nc = int(cfg.get("num_cores", 1) or 1)
    topo = cfg.get("topology_type") or "ring"
    B = int(cfg.get("buffer_qubits", 1) or 1)
    if "communication_qubits" in cfg:
        cfg["communication_qubits"] = clamp_k_for_topology(
            nq, nc, topo, int(cfg["communication_qubits"] or 1),
            b_per_group=B,
        )
    K = int(cfg.get("communication_qubits", 1) or 1)
    if "buffer_qubits" in cfg:
        cfg["buffer_qubits"] = clamp_b_for_topology(
            nq, nc, topo, K, int(cfg["buffer_qubits"] or 1),
        )
        B = cfg["buffer_qubits"]
    if "num_logical_qubits" in cfg:
        cap = max(2, min(nq, max_data_slots(nq, nc, topo, K, B) or nq))
        cfg["num_logical_qubits"] = max(
            2, min(int(cfg["num_logical_qubits"]), cap)
        )


def _resolve_cell_cold_cfg(
    cold_config: dict, swept: dict[str, float],
) -> dict:
    """Apply a cell's swept overrides + standard clamps to a cold_config.

    Mirrors the clamp logic in ``DSEEngine._eval_point`` and
    ``_eval_cold_batch`` so the captured per-cell cold cfg matches what
    the engine actually compiled with.
    """
    cfg = dict(cold_config)
    for k, v in swept.items():
        if k in DSEEngine.COLD_PATH_KEYS:
            cfg[k] = int(v) if k in DSEEngine.INTEGER_KEYS else v
    _expand_qubits_alias(cfg)
    cfg["num_cores"] = min(int(cfg.get("num_cores", 1)), int(cfg.get("num_qubits", 1)))
    _clamp_cfg_comm_and_logical(cfg)
    return cfg


def _result_to_row(result: dict) -> tuple:
    """Pack a stripped result dict into the tuple order of ``_RESULT_DTYPE``."""
    return tuple(result.get(k, 0.0) for k in _RESULT_SCALAR_KEYS)


def _row_to_dict(row) -> dict:
    """Unpack a structured-array cell (numpy.void) into a plain dict.

    Leaves plain dicts untouched so legacy 1–3 D sweep paths keep working.
    """
    if isinstance(row, np.void):
        return {k: float(row[k]) for k in row.dtype.names}
    return row


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


# ---------------------------------------------------------------------------
# Sweep progress tracking
# ---------------------------------------------------------------------------

@dataclass
class SweepProgress:
    """Snapshot of sweep progress, emitted on every iteration."""
    completed: int
    total: int
    current_params: dict[str, float | str] = field(default_factory=dict)
    cold_completed: int = 0
    cold_total: int = 0

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.completed / self.total * 100, 2)


# ---------------------------------------------------------------------------
# N-D sweep result
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """N-dimensional sweep result with a compact structured-array grid.

    Attributes:
        metric_keys: Names of the swept parameters.
        axes: One array of sample values per swept parameter.
        grid: numpy structured array of shape (len(axes[0]), ..., len(axes[N-1]))
              with dtype ``_RESULT_DTYPE``. Each cell is a ``numpy.void`` row
              holding the scalar result fields. ``to_sweep_data`` converts
              cells back to plain dicts for the browser payload.
        per_qubit_data: optional ``{idx_tuple: {alg_grid, rt_grid, coh_grid,
              placements, num_physical, num_logical, num_cores,
              communication_qubits, topology_type, intracore_topology}}``.
              Populated when the sweep was run with
              ``keep_per_qubit_grids=True``.  Heavy — kept server-side only.
    """
    metric_keys: list[str]
    axes: list[np.ndarray]
    grid: np.ndarray  # dtype=_RESULT_DTYPE, shape matches axis lengths
    per_qubit_data: dict | None = None

    @property
    def ndim(self) -> int:
        return len(self.metric_keys)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(ax) for ax in self.axes)

    @property
    def total_points(self) -> int:
        return int(np.prod([len(ax) for ax in self.axes]))

    def to_sweep_data(self) -> dict:
        """Convert to the dict format consumed by plotting functions.

        For 1-3D: produces the legacy xs/ys/zs + nested-list grid.
        For 4D+:  produces axes list + flat grid list with shape metadata.
        """
        sd: dict = {"metric_keys": self.metric_keys}
        n = self.ndim

        if n >= 1:
            sd["xs"] = self.axes[0].tolist()
        if n >= 2:
            sd["ys"] = self.axes[1].tolist()
        if n >= 3:
            sd["zs"] = self.axes[2].tolist()

        # Convert structured rows back to plain dicts at the browser
        # boundary; plotting/interpolation consume the dict form.
        cell = _row_to_dict

        if n == 1:
            sd["grid"] = [cell(self.grid[i]) for i in range(len(self.axes[0]))]
        elif n == 2:
            sd["grid"] = [
                [cell(self.grid[i, j]) for j in range(len(self.axes[1]))]
                for i in range(len(self.axes[0]))
            ]
        elif n == 3:
            sd["grid"] = [
                [[cell(self.grid[i, j, k]) for k in range(len(self.axes[2]))]
                 for j in range(len(self.axes[1]))]
                for i in range(len(self.axes[0]))
            ]
        else:
            # N >= 4: keep the structured grid in place. Flattening to a
            # list of 85M+ dicts used to allocate ~24 GB transiently and
            # OOM-killed the process right before plotting. The plotting
            # and CSV paths vectorise over the structured array directly.
            sd["axes"] = [ax.tolist() for ax in self.axes]
            sd["shape"] = self.shape
            if self.grid.dtype.names:
                sd["grid"] = self.grid
            else:
                # Legacy object-dtype grids (test fixtures) still need the
                # flat dict-list form; callers rely on ``for r in grid``.
                sd["grid"] = [cell(v) for v in self.grid.ravel()]

        if self.per_qubit_data is not None:
            sd["per_qubit_data"] = self.per_qubit_data

        return sd


# ---------------------------------------------------------------------------
# DSE Engine
# ---------------------------------------------------------------------------

class DSEEngine:
    """
    Wraps qusim's map_circuit for Design Space Exploration.

    Cold path: build circuit + topology, run HQA+SABRE, cache structural results.
    Hot path:  call estimate_hardware_fidelity directly with cached data + new noise.
    """

    def __init__(self) -> None:
        self._cache: Optional[CachedMapping] = None

    # -- Cold path -----------------------------------------------------------

    def run_cold(
        self,
        circuit_type: str,
        num_qubits: int,
        num_cores: int,
        topology_type: str,
        placement_policy: str,
        seed: int,
        noise: Optional[dict] = None,
        intracore_topology: str = "all_to_all",
        routing_algorithm: str = "hqa_sabre",
        communication_qubits: int = 1,
        buffer_qubits: int = 1,
        num_logical_qubits: Optional[int] = None,
        custom_qasm: Optional[str] = None,
    ) -> CachedMapping:
        """
        Full circuit transpilation + mapping. Returns a CachedMapping that
        can be reused for many hot-path fidelity evaluations.

        ``num_qubits`` is the physical device size (drives topology /
        coupling map).  ``num_logical_qubits`` is the algorithm size used
        to build the circuit; defaults to ``num_qubits`` when omitted, and
        is clamped to ``[2, num_qubits]``.

        ``communication_qubits`` (K) carves out per-group EPR endpoints
        on each core; ``buffer_qubits`` (B) reserves additional buffer
        slots adjacent to each comm group (B ≤ K by rule).  Each core
        with ``G`` neighbours therefore reserves ``G·(K+B)`` total slots.
        """
        if circuit_type == "custom":
            if not custom_qasm:
                raise ValueError("circuit_type='custom' requires a custom_qasm string")
            from qiskit import qasm2
            _custom_circ = qasm2.loads(custom_qasm)
            num_logical_qubits = int(_custom_circ.num_qubits)
            if num_logical_qubits > int(num_qubits):
                raise ValueError(
                    f"Custom circuit uses {num_logical_qubits} logical qubits, "
                    f"which exceeds the device's {int(num_qubits)} physical qubits."
                )
        elif num_logical_qubits is None:
            num_logical_qubits = num_qubits
        num_logical_qubits = max(2, min(int(num_logical_qubits), int(num_qubits)))
        import hashlib as _hashlib
        qasm_fingerprint = (
            _hashlib.sha256(custom_qasm.encode("utf-8")).hexdigest()[:16]
            if custom_qasm else None
        )
        config_key = (
            circuit_type, num_qubits, num_cores, topology_type, intracore_topology,
            placement_policy, seed, routing_algorithm,
            int(communication_qubits or 1), int(buffer_qubits or 1),
            int(num_logical_qubits), qasm_fingerprint,
        )
        if self._cache is not None and self._cache.config_key == config_key:
            return self._cache

        if routing_algorithm == "telesabre":
            return self._run_cold_telesabre(
                circuit_type, num_qubits, num_cores, topology_type,
                intracore_topology, seed, noise, config_key,
                num_logical_qubits=num_logical_qubits,
                communication_qubits=communication_qubits,
                buffer_qubits=buffer_qubits,
                custom_qasm=custom_qasm,
            )

        t0 = time.time()

        circ = _build_circuit(circuit_type, num_logical_qubits, seed, qasm_str=custom_qasm)
        transp = _transpile_circuit(circ, seed)

        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type,
            intracore_topology=intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )

        actual_qubits = transp.num_qubits

        placement = (
            InitialPlacement.SPECTRAL_CLUSTERING
            if placement_policy == "spectral"
            else InitialPlacement.RANDOM
        )

        merged = _merge_noise(noise or {})

        result = qusim.map_circuit(
            circuit=transp,
            full_coupling_map=full_coupling_map,
            core_mapping=core_mapping,
            seed=seed,
            initial_placement=placement,
            single_gate_error=merged["single_gate_error"],
            two_gate_error=merged["two_gate_error"],
            teleportation_error_per_hop=merged["teleportation_error_per_hop"],
            single_gate_time=merged["single_gate_time"],
            two_gate_time=merged["two_gate_time"],
            teleportation_time_per_hop=merged["teleportation_time_per_hop"],
            epr_error_per_hop=merged["epr_error_per_hop"],
            measurement_error=merged["measurement_error"],
            t1=merged["t1"],
            t2=merged["t2"],
            dynamic_decoupling=False,
            readout_mitigation_factor=merged["readout_mitigation_factor"],
            classical_link_width=int(merged["classical_link_width"]),
            classical_clock_freq_hz=merged["classical_clock_freq_hz"],
            classical_routing_cycles=int(merged["classical_routing_cycles"]),
        )

        # Reconstruct internal gs_sparse + gate arrays as map_circuit does
        from qusim import _qiskit_circ_to_sparse_list

        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, merged)

        # Rebuild distance matrix (mirrors map_circuit logic)
        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(full_coupling_map, core_mapping, num_cores_actual)

        # SABRE-injected swaps are emitted by ``map_circuit`` and now exposed
        # on ``QusimResult.sparse_swaps``.  These represent intra-core SWAP
        # gates and MUST be passed to the hot-path noise estimator —
        # ``extract_inter_core_communications`` only sees inter-core changes
        # in the placements diff and therefore never charges intra-core
        # SWAPs on its own.  Without this, HQA's swap cost is silently
        # dropped while TeleSABRE's is correctly counted, making the two
        # algorithms structurally incomparable on the noise sweep.
        sparse_swaps = (
            result.sparse_swaps
            if getattr(result, "sparse_swaps", None) is not None
            else np.zeros((0, 3), dtype=np.int32)
        )

        cold_time = time.time() - t0

        self._cache = CachedMapping(
            gs_sparse=gs_sparse,
            placements=result.placements,
            distance_matrix=dist_mat,
            sparse_swaps=sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            gate_names=gate_names,
            total_epr_pairs=result.total_epr_pairs,
            total_swaps=result.total_swaps,
            total_teleportations=result.total_teleportations,
            total_network_distance=result.total_network_distance,
            config_key=config_key,
            cold_time_s=cold_time,
        )
        return self._cache

    # -- TeleSABRE cold path -------------------------------------------------

    def _run_cold_telesabre(
        self,
        circuit_type: str,
        num_qubits: int,
        num_cores: int,
        topology_type: str,
        intracore_topology: str,
        seed: int,
        noise: Optional[dict],
        config_key: tuple,
        num_logical_qubits: Optional[int] = None,
        communication_qubits: int = 1,
        buffer_qubits: int = 1,
        custom_qasm: Optional[str] = None,
    ) -> CachedMapping:
        """Cold path using TeleSABRE routing. Writes QASM + device JSON to
        temp files, calls telesabre_map_and_estimate, and caches the resulting
        placements/sparse_swaps for hot-path re-evaluation."""
        import json
        import tempfile
        import os
        from qiskit import qasm2
        from qusim.rust_core import telesabre_map_and_estimate

        t0 = time.time()
        merged = _merge_noise(noise or {})

        if circuit_type == "custom":
            if not custom_qasm:
                raise ValueError("circuit_type='custom' requires a custom_qasm string")
            from qiskit import qasm2
            _custom_circ = qasm2.loads(custom_qasm)
            num_logical_qubits = int(_custom_circ.num_qubits)
        elif num_logical_qubits is None:
            num_logical_qubits = num_qubits
        num_logical_qubits = max(2, min(int(num_logical_qubits), int(num_qubits)))

        circ = _build_circuit(circuit_type, num_logical_qubits, seed, qasm_str=custom_qasm)
        transp = _transpile_circuit(circ, seed)

        # Write QASM to temp file
        qasm_str = qasm2.dumps(transp)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".qasm", delete=False
        ) as f:
            f.write(qasm_str)
            qasm_path = f.name

        # Build and write TeleSABRE device JSON
        device_json = _build_telesabre_device_json(
            num_qubits, num_cores, topology_type, intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(device_json, f)
            device_path = f.name

        # Use the bundled default config
        config_path = os.path.join(
            os.path.dirname(__file__), "..",
            "tests", "fixtures", "telesabre", "configs", "default.json",
        )

        try:
            raw = telesabre_map_and_estimate(
                circuit_path=qasm_path,
                device_path=device_path,
                config_path=os.path.abspath(config_path),
                single_gate_error=merged["single_gate_error"],
                two_gate_error=merged["two_gate_error"],
                teleportation_error_per_hop=merged["teleportation_error_per_hop"],
                single_gate_time=merged["single_gate_time"],
                two_gate_time=merged["two_gate_time"],
                teleportation_time_per_hop=merged["teleportation_time_per_hop"],
                t1=merged["t1"],
                t2=merged["t2"],
                dynamic_decoupling=bool(merged.get("dynamic_decoupling", False)),
                readout_mitigation_factor=merged["readout_mitigation_factor"],
                classical_link_width=int(merged["classical_link_width"]),
                classical_clock_freq_hz=merged["classical_clock_freq_hz"],
                classical_routing_cycles=int(merged["classical_routing_cycles"]),
            )
        finally:
            os.unlink(qasm_path)
            os.unlink(device_path)

        # Use the Qiskit circuit's gs_sparse so the hot path has the correct
        # num_layers (circuit DAG depth) and gate structure for algorithmic /
        # coherence fidelity.  TeleSABRE's placements/swaps are remapped from
        # their internal gate-step space to the DAG-layer space so both grids
        # have a consistent row count.
        from qusim import _qiskit_circ_to_sparse_list
        gs_sparse, gate_names = _qiskit_circ_to_sparse_list(transp)
        gate_error_arr, gate_time_arr = _make_gate_arrays(gate_names, merged)

        dag_layers = (
            int(gs_sparse[:, 0].max()) + 1 if len(gs_sparse) > 0 else 1
        )

        placements = _remap_to_dag_layers(raw["placements"], dag_layers)
        sparse_swaps = _remap_swaps_to_dag_layers(raw["sparse_swaps"], dag_layers)

        # Real inter-core distances so routing fidelity is non-trivial
        full_coupling_map, core_mapping = _build_topology(
            num_qubits, num_cores, topology_type, intracore_topology=intracore_topology,
            communication_qubits=communication_qubits,
            buffer_qubits=buffer_qubits,
        )
        num_cores_actual = max(core_mapping.values()) + 1 if core_mapping else 1
        dist_mat = _compute_distance_matrix(full_coupling_map, core_mapping, num_cores_actual)

        cold_time = time.time() - t0
        self._cache = CachedMapping(
            gs_sparse=gs_sparse,
            placements=placements,
            distance_matrix=dist_mat,
            sparse_swaps=sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            gate_names=gate_names,
            total_epr_pairs=raw["total_epr_pairs"],
            total_swaps=raw["total_swaps"],
            total_teleportations=raw["total_teleportations"],
            total_network_distance=0,
            config_key=config_key,
            cold_time_s=cold_time,
        )
        return self._cache

    # -- Hot path ------------------------------------------------------------

    def run_hot(self, cached: CachedMapping, noise: dict) -> dict:
        """
        Fast fidelity estimation reusing cached placements and distance matrix.
        Returns a dict with overall_fidelity, algorithmic_fidelity, etc.
        """
        merged = _merge_noise(noise)
        gate_error_arr, gate_time_arr = _make_gate_arrays(cached.gate_names, merged)

        result = qusim.estimate_fidelity_from_cache(
            gs_sparse=cached.gs_sparse,
            placements=cached.placements,
            distance_matrix=cached.distance_matrix,
            sparse_swaps=cached.sparse_swaps,
            gate_error_arr=gate_error_arr,
            gate_time_arr=gate_time_arr,
            single_gate_error=merged["single_gate_error"],
            two_gate_error=merged["two_gate_error"],
            teleportation_error_per_hop=merged["teleportation_error_per_hop"],
            single_gate_time=merged["single_gate_time"],
            two_gate_time=merged["two_gate_time"],
            teleportation_time_per_hop=merged["teleportation_time_per_hop"],
            epr_error_per_hop=merged["epr_error_per_hop"],
            measurement_error=merged["measurement_error"],
            t1=merged["t1"],
            t2=merged["t2"],
            dynamic_decoupling=merged.get("dynamic_decoupling", False),
            readout_mitigation_factor=merged["readout_mitigation_factor"],
            classical_link_width=int(merged["classical_link_width"]),
            classical_clock_freq_hz=merged["classical_clock_freq_hz"],
            classical_routing_cycles=int(merged["classical_routing_cycles"]),
        )
        result["total_epr_pairs"] = cached.total_epr_pairs
        result["total_swaps"] = cached.total_swaps
        result["total_teleportations"] = cached.total_teleportations
        result["total_network_distance"] = cached.total_network_distance
        return result

    # Max noise configs per Rust batch call to bound peak memory
    _HOT_BATCH_CHUNK = 5_000

    def run_hot_batch(
        self,
        cached: CachedMapping,
        noise_dicts: list[dict],
        keep_grids: bool = False,
    ) -> list[dict]:
        """
        Batch fidelity estimation: chunked Rust calls for many noise configs.

        Structural data (tensor, placements, routing) is parsed once per chunk.
        Chunks keep peak memory bounded instead of allocating all results at once.

        ``keep_grids=True`` returns the per-qubit ``*_fidelity_grid`` ndarrays
        per result. The Rust batch entry point is scalar-only by design (for
        throughput), so this branch falls back to a per-cell ``run_hot`` loop —
        slower per cell, but the only path that actually produces the
        (num_layers, num_qubits) grids the topology overlay needs.
        """
        if keep_grids:
            return [self.run_hot(cached, n) for n in noise_dicts]

        gate_error_arr, gate_time_arr = _make_gate_arrays(
            cached.gate_names, _merge_noise(noise_dicts[0]),
        )
        all_results: list[dict] = []

        for start in range(0, len(noise_dicts), self._HOT_BATCH_CHUNK):
            chunk = noise_dicts[start : start + self._HOT_BATCH_CHUNK]
            merged_chunk = [_merge_noise(n) for n in chunk]

            chunk_results = qusim.estimate_fidelity_from_cache_batch(
                gs_sparse=cached.gs_sparse,
                placements=cached.placements,
                distance_matrix=cached.distance_matrix,
                sparse_swaps=cached.sparse_swaps,
                gate_error_arr=gate_error_arr,
                gate_time_arr=gate_time_arr,
                noise_dicts=merged_chunk,
            )
            for r in chunk_results:
                r["total_epr_pairs"] = cached.total_epr_pairs
                r["total_swaps"] = cached.total_swaps
                r["total_teleportations"] = cached.total_teleportations
                r["total_network_distance"] = cached.total_network_distance
            # Strip per-qubit ``*_grid`` ndarrays before they cross the
            # process-pool pickle boundary or enter the sweep grid. The
            # batch Rust call already returns scalar-only dicts; this strip
            # is a defensive no-op there but matters when callers feed in
            # an alternative result source.
            all_results.extend(_strip_for_grid(r) for r in chunk_results)

        return all_results

    # -- Sweep methods -------------------------------------------------------

    COLD_PATH_KEYS = frozenset({
        "qubits",  # virtual alias: expands to num_qubits == num_logical_qubits
        "num_qubits", "num_cores", "communication_qubits", "buffer_qubits",
        "num_logical_qubits",
        "circuit_type", "topology_type", "intracore_topology", "routing_algorithm",
    })
    INTEGER_KEYS = frozenset({
        "qubits", "num_qubits", "num_cores", "communication_qubits",
        "buffer_qubits", "num_logical_qubits",
        "classical_link_width", "classical_routing_cycles",
    })

    def _memory_capped_max_hot(self, max_hot: int | None) -> tuple[int, int]:
        """Clamp a requested ``max_hot`` to what current RAM can hold.

        Returns ``(effective, requested)`` so callers can report when the
        cap kicked in. ``requested`` mirrors the user's configured value
        (or the registry default when ``None``); ``effective`` is the
        value the sweep should actually respect.
        """
        requested = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT
        return min(requested, _max_hot_points_for_memory()), requested

    def _metric_values(self, metric_key: str, low, high, n: int) -> np.ndarray:
        if metric_key in CAT_METRIC_BY_KEY:
            # ``low`` is the pre-selected list of option values; high/n unused.
            return np.array(low, dtype=object)
        m = METRIC_BY_KEY[metric_key]
        if m.log_scale:
            return np.logspace(low, high, n)
        vals = np.linspace(low, high, n)
        if metric_key in self.INTEGER_KEYS:
            vals = np.unique(np.round(vals).astype(int))
        return vals

    def _has_cold(self, *keys: str) -> bool:
        return any(k in self.COLD_PATH_KEYS for k in keys)

    def _sweep_points(self, ndim: int, has_cold: bool) -> int:
        if has_cold:
            return [SWEEP_POINTS_COLD_1D, SWEEP_POINTS_COLD_2D, SWEEP_POINTS_COLD_3D][ndim - 1]
        return [SWEEP_POINTS_1D, SWEEP_POINTS_2D, SWEEP_POINTS_3D][ndim - 1]

    def _points_per_axis(self, ndim: int, has_cold: bool) -> int:
        """Compute uniform points per axis for an N-D sweep within the total budget."""
        budget = MAX_TOTAL_POINTS_COLD if has_cold else MAX_TOTAL_POINTS_HOT
        # For 1-3D, use legacy values for backward compatibility
        if ndim <= 3:
            return self._sweep_points(ndim, has_cold)
        # For N >= 4, compute from budget
        n = max(MIN_POINTS_PER_AXIS, int(budget ** (1.0 / ndim)))
        # Clamp so total doesn't exceed budget
        while n ** ndim > budget and n > MIN_POINTS_PER_AXIS:
            n -= 1
        return n

    def _compute_axis_counts(
        self, sweep_axes: list, has_cold: bool,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> list[int]:
        """Compute per-axis point counts using a split budget model.

        ``sweep_axes`` entries are either:
          - ``(key, low, high)`` for numeric axes, or
          - ``(key, values_list)`` for categorical axes (2-tuple).

        Categorical axes have a fixed count (len of selected values) and are
        not subject to budget reduction.  Numeric cold-path axes (num_qubits,
        num_cores) are capped by ``max_cold``.  Hot-path axes share the
        remaining hot budget.
        """
        ndim = len(sweep_axes)
        cold_budget = max_cold if max_cold is not None else MAX_COLD_COMPILATIONS
        hot_budget = max_hot if max_hot is not None else MAX_TOTAL_POINTS_HOT

        # Separate indices by kind
        cat_indices = []     # categorical: fixed count, always cold-path
        cold_indices = []    # numeric cold-path (num_qubits, num_cores)
        hot_indices = []     # numeric hot-path (noise params)
        for i, ax in enumerate(sweep_axes):
            key = ax[0]
            if key in CAT_METRIC_BY_KEY:
                cat_indices.append(i)
            elif key in self.COLD_PATH_KEYS:
                cold_indices.append(i)
            else:
                hot_indices.append(i)

        counts: list[int] = [0] * ndim

        # Categorical axes: fixed at the number of selected options
        for i in cat_indices:
            counts[i] = len(sweep_axes[i][1])

        cat_product = 1
        for i in cat_indices:
            cat_product *= counts[i]

        # Numeric cold axes: distribute remaining cold budget
        # (divide by cat_product so cold_budget applies to numeric cold combos)
        effective_cold_budget = max(1, cold_budget // max(1, cat_product))
        if cold_indices:
            n_cold = len(cold_indices)
            per_cold = max(MIN_POINTS_PER_AXIS, int(effective_cold_budget ** (1.0 / n_cold)))
            for i in cold_indices:
                key, low, high = sweep_axes[i]
                natural = int(high) - int(low) + 1
                counts[i] = max(MIN_POINTS_PER_AXIS, min(natural, per_cold))
            # Verify product is within budget, reduce if needed
            cold_product = 1
            for i in cold_indices:
                cold_product *= counts[i]
            while cold_product > effective_cold_budget:
                largest = max(cold_indices, key=lambda i: counts[i])
                counts[largest] = max(MIN_POINTS_PER_AXIS, counts[largest] - 1)
                cold_product = 1
                for i in cold_indices:
                    cold_product *= counts[i]
                if all(counts[i] == MIN_POINTS_PER_AXIS for i in cold_indices):
                    break

        # Total cold multiplier (cat × numeric-cold) for hot budget division
        cold_product = cat_product
        for i in cold_indices:
            cold_product *= counts[i]

        if hot_indices:
            n_hot = len(hot_indices)
            min_total = (MIN_POINTS_PER_AXIS ** n_hot) * cold_product
            if min_total > hot_budget:
                raise RuntimeError(
                    f"Hot budget too tight for this sweep: "
                    f"{n_hot} hot axes at {MIN_POINTS_PER_AXIS} points each × "
                    f"{cold_product} cold combinations = {min_total:,} points, "
                    f"but max_hot={hot_budget:,}. "
                    f"Raise Max hot evaluations to at least {min_total:,}, "
                    f"reduce dimensions, or lower Max cold compilations."
                )
            effective_hot_budget = max(1, hot_budget // max(1, cold_product))
            per_hot = max(MIN_POINTS_PER_AXIS, int(effective_hot_budget ** (1.0 / n_hot)))
            while per_hot ** n_hot * cold_product > hot_budget and per_hot > MIN_POINTS_PER_AXIS:
                per_hot -= 1
            for i in hot_indices:
                counts[i] = per_hot
        elif not cold_indices and not cat_indices:
            # All axes are neither cold nor hot — shouldn't happen, but fallback
            n = max(MIN_POINTS_PER_AXIS, int(hot_budget ** (1.0 / ndim)))
            counts = [n] * ndim

        return counts

    def _eval_point(
        self,
        cold_config: dict,
        noise: dict,
        swept: dict[str, float],
    ) -> dict:
        """Evaluate a single design point, re-running cold path if needed.

        Returns the scalar-only result dict destined for the sweep grid; the
        per-qubit ``*_grid`` ndarrays are dropped here (see
        ``_strip_for_grid``).
        """
        import math as _math
        cfg = dict(cold_config)
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k in self.COLD_PATH_KEYS:
                cfg[k] = int(v) if k in self.INTEGER_KEYS else v
            else:
                hot_noise[k] = v
        _expand_qubits_alias(cfg)
        cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
        # Clamp communication_qubits to the architectural cap and
        # num_logical_qubits to the remaining data-slot count for the
        # current topology / cores / num_qubits.
        _clamp_cfg_comm_and_logical(cfg)
        cached = self.run_cold(**cfg, noise=hot_noise)
        # Return the full dict (per-qubit grids included) so callers that
        # need them can capture before stripping.  ``_result_to_row`` reads
        # only the scalar keys it knows about, so feeding it a full dict is
        # equivalent to the previously-stripped one.
        return self.run_hot(cached, hot_noise)

    @staticmethod
    def _mem_budget_mb() -> int:
        """Memory budget (MB) for concurrent cold compilations.

        Reserves the larger of ``_RESERVED_RAM_MB`` (1 GB floor) or 30% of
        total RAM so the budget scales with the host. Keeping 30% free
        prevents the swap-thrashing deadlock that can freeze the whole OS
        when multiple 3+ GB workers run alongside the browser, desktop,
        and a growing page cache.
        """
        try:
            avail_mb = 0
            total_mb = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemTotal:"):
                        total_mb = int(line.split()[1]) // 1024
                    if avail_mb and total_mb:
                        break
            if avail_mb and total_mb:
                reserved = max(_RESERVED_RAM_MB, total_mb * 30 // 100)
                return max(1, avail_mb - reserved)
        except OSError:
            pass
        # Fallback: a conservative ~1.6 GB when /proc is unavailable
        return 2 * 800

    def _parallel_cold_sweep(
        self,
        cold_config: dict,
        noise: dict,
        indexed_points: Iterable[tuple[tuple, dict]],
        total: int,
        progress_callback: Callable[[SweepProgress], None] | None,
        max_workers: int | None,
        keep_grids: bool = False,
    ) -> dict[tuple, dict]:
        """Run cold-path sweep points with qubit-aware parallel scheduling.

        Each unique cold config (one (num_qubits, num_cores) combo) becomes one
        group. The scheduler submits groups to a process pool so that the sum
        of their estimated peak RSS stays under the memory budget — small
        groups parallelize aggressively, large-qubit groups serialize. A single
        oversized group is always allowed to run alone to avoid deadlock.
        """
        groups: dict[tuple, list[tuple[tuple, dict]]] = {}
        for idx_key, swept in indexed_points:
            cold_vals = tuple(sorted(
                (k, int(v) if k in self.INTEGER_KEYS else v)
                for k, v in swept.items() if k in self.COLD_PATH_KEYS
            ))
            groups.setdefault(cold_vals, []).append((idx_key, swept))

        fallback_nq = int(cold_config.get("num_qubits", 16))

        def _group_nq(cv: tuple) -> int:
            # Also recognise the ``qubits`` alias (logical == physical sweep)
            # which expands to ``num_qubits`` inside the worker.  Without
            # this, memory estimation falls back to the default and large
            # qubit-count cells get scheduled in parallel and OOM.
            for k, v in cv:
                if k == "num_qubits" or k == "qubits":
                    return int(v)
            return fallback_nq

        group_cost: dict[tuple, float] = {
            cv: _estimate_cold_mb(_group_nq(cv)) for cv in groups
        }

        cpu_cap = max(1, (os.cpu_count() or 2) // 2)
        slot_cap = max_workers if max_workers else cpu_cap
        mem_budget = self._mem_budget_mb()

        # Per-worker address-space cap (RLIMIT_AS). With arbitrary
        # user-supplied circuits, the cost estimate can be severely wrong
        # (e.g. dense random circuits on grid topology allocate 10x the
        # QFT-calibrated estimate).  Capping each worker to its fair share
        # of the budget ensures a runaway allocation raises MemoryError
        # inside the worker — surfacing as a "Sweep failed" banner — rather
        # than OOM-killing the whole system.
        rss_cap_bytes = (mem_budget * 1024 * 1024) // max(1, slot_cap)

        # Guard: if even the cheapest group's estimate exceeds the budget by a
        # wide margin, the run will OOM the worker.  Fail early with a
        # message the user can act on instead of a BrokenProcessPool crash.
        if group_cost:
            min_cost = min(group_cost.values())
            if min_cost > mem_budget * 1.5:
                largest_nq = max(_group_nq(cv) for cv in groups)
                raise RuntimeError(
                    f"Not enough RAM for this sweep: smallest cold-compile "
                    f"needs ~{min_cost:.0f} MB, budget is {mem_budget} MB "
                    f"(largest qubits = {largest_nq}). Reduce max qubits, "
                    f"close other apps, or lower Max cold compilations."
                )

        # Schedule largest-first so cheap groups can tuck into the remaining budget.
        pending = sorted(groups, key=lambda cv: -group_cost[cv])
        active: dict = {}  # future -> (cold_vals, cost_mb)
        results: dict[tuple, dict] = {}
        completed = 0
        cold_total = len(groups)
        cold_completed = 0

        # Recycle workers after a handful of cold compilations to flush
        # qiskit/BLAS caches that otherwise grow across reused processes
        # and drift real RSS above the per-job estimate.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=slot_cap, mp_context=_MP_CONTEXT,
            max_tasks_per_child=4,
        ) as pool:

            def _submit_fitting() -> None:
                used = sum(c for _, c in active.values())
                kept: list[tuple] = []
                for cv in pending:
                    cost = group_cost[cv]
                    if len(active) >= slot_cap:
                        kept.append(cv)
                        continue
                    # Always allow one job when the pool is idle, even if its
                    # estimate exceeds the budget (otherwise we'd deadlock on
                    # a single oversized group).
                    mem_ok = not active or used + cost <= mem_budget
                    if mem_ok:
                        swept_list = [s for _, s in groups[cv]]
                        fut = pool.submit(
                            _eval_cold_batch,
                            cold_config, noise, swept_list, rss_cap_bytes,
                            keep_grids,
                        )
                        active[fut] = (cv, cost)
                        used += cost
                    else:
                        kept.append(cv)
                pending[:] = kept

            _submit_fitting()
            while active:
                done, _still = concurrent.futures.wait(
                    list(active), return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    cv, _cost = active.pop(fut)
                    batch_results = fut.result()
                    cold_completed += 1
                    for (idx_key, swept), result in zip(groups[cv], batch_results):
                        results[idx_key] = result
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(SweepProgress(
                                completed=completed,
                                total=total,
                                current_params={k: float(v) for k, v in swept.items()},
                                cold_completed=cold_completed,
                                cold_total=cold_total,
                            ))
                _submit_fitting()

        return results

    def sweep_1d(
        self,
        cached: CachedMapping,
        metric_key: str,
        low: float,
        high: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, list]:
        has_cold = self._has_cold(metric_key)
        counts = self._compute_axis_counts(
            [(metric_key, low, high)], has_cold, max_cold, max_hot,
        )
        n = counts[0]
        xs = self._metric_values(metric_key, low, high, n)
        total = len(xs)

        if parallel and has_cold and cold_config is not None:
            indexed = [((i,), {metric_key: v}) for i, v in enumerate(xs)]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            return xs, [rmap[(i,)] for i in range(total)]

        results = []
        for i, v in enumerate(xs):
            if has_cold:
                results.append(self._eval_point(cold_config, fixed_noise, {metric_key: v}))
            else:
                noise = {**fixed_noise, metric_key: v}
                results.append(_strip_for_grid(self.run_hot(cached, noise)))
            if progress_callback is not None:
                progress_callback(SweepProgress(
                    completed=i + 1,
                    total=total,
                    current_params={metric_key: float(v)},
                ))
        return xs, results

    def sweep_2d(
        self,
        cached: CachedMapping,
        metric_key1: str, low1: float, high1: float,
        metric_key2: str, low2: float, high2: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2)
        counts = self._compute_axis_counts(
            [(metric_key1, low1, high1), (metric_key2, low2, high2)],
            has_cold, max_cold, max_hot,
        )
        xs = self._metric_values(metric_key1, low1, high1, counts[0])
        ys = self._metric_values(metric_key2, low2, high2, counts[1])
        total = len(xs) * len(ys)

        if parallel and has_cold and cold_config is not None:
            indexed = [
                ((i, j), {metric_key1: v1, metric_key2: v2})
                for i, v1 in enumerate(xs)
                for j, v2 in enumerate(ys)
            ]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            grid = [[rmap[(i, j)] for j in range(len(ys))] for i in range(len(xs))]
            return xs, ys, grid

        grid = []
        count = 0
        for v1 in xs:
            row = []
            for v2 in ys:
                swept = {metric_key1: v1, metric_key2: v2}
                if has_cold:
                    row.append(self._eval_point(cold_config, fixed_noise, swept))
                else:
                    noise = {**fixed_noise, **swept}
                    row.append(_strip_for_grid(self.run_hot(cached, noise)))
                count += 1
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        completed=count,
                        total=total,
                        current_params={
                            metric_key1: float(v1),
                            metric_key2: float(v2),
                        },
                    ))
            grid.append(row)
        return xs, ys, grid

    def sweep_3d(
        self,
        cached: CachedMapping,
        metric_key1: str, low1: float, high1: float,
        metric_key2: str, low2: float, high2: float,
        metric_key3: str, low3: float, high3: float,
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        has_cold = self._has_cold(metric_key1, metric_key2, metric_key3)
        counts = self._compute_axis_counts(
            [
                (metric_key1, low1, high1),
                (metric_key2, low2, high2),
                (metric_key3, low3, high3),
            ],
            has_cold, max_cold, max_hot,
        )
        xs = self._metric_values(metric_key1, low1, high1, counts[0])
        ys = self._metric_values(metric_key2, low2, high2, counts[1])
        zs = self._metric_values(metric_key3, low3, high3, counts[2])
        total = len(xs) * len(ys) * len(zs)

        if parallel and has_cold and cold_config is not None:
            indexed = [
                ((i, j, k), {metric_key1: v1, metric_key2: v2, metric_key3: v3})
                for i, v1 in enumerate(xs)
                for j, v2 in enumerate(ys)
                for k, v3 in enumerate(zs)
            ]
            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, indexed, total, progress_callback, max_workers,
            )
            grid = [
                [[rmap[(i, j, k)] for k in range(len(zs))] for j in range(len(ys))]
                for i in range(len(xs))
            ]
            return xs, ys, zs, grid

        grid = []
        count = 0
        for v1 in xs:
            plane = []
            for v2 in ys:
                row_list = []
                for v3 in zs:
                    swept = {metric_key1: v1, metric_key2: v2, metric_key3: v3}
                    if has_cold:
                        row_list.append(self._eval_point(cold_config, fixed_noise, swept))
                    else:
                        noise = {**fixed_noise, **swept}
                        row_list.append(_strip_for_grid(self.run_hot(cached, noise)))
                    count += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=count,
                            total=total,
                            current_params={
                                metric_key1: float(v1),
                                metric_key2: float(v2),
                                metric_key3: float(v3),
                            },
                        ))
                plane.append(row_list)
            grid.append(plane)
        return xs, ys, zs, grid

    # -- N-D sweep (unified) ---------------------------------------------------

    def sweep_nd(
        self,
        cached: CachedMapping | None,
        sweep_axes: list[tuple[str, float, float]],
        fixed_noise: dict,
        cold_config: dict | None = None,
        progress_callback: Callable[[SweepProgress], None] | None = None,
        parallel: bool = False,
        max_workers: int | None = None,
        max_cold: int | None = None,
        max_hot: int | None = None,
        keep_per_qubit_grids: bool = False,
    ) -> SweepResult:
        """N-dimensional sweep over arbitrary axes.

        Parameters
        ----------
        sweep_axes : list of (metric_key, low, high) tuples
        max_cold : override for cold compilation budget
        max_hot : override for total hot-path point budget
        keep_per_qubit_grids : populate ``SweepResult.per_qubit_data`` with
            per-cell per-qubit fidelity grids + placements + effective cold
            config so the topology view can colour nodes / replot the device
            for any cell without re-running the engine.  Adds memory cost
            roughly proportional to ``num_cells × num_layers × num_qubits``.
        """
        from itertools import islice

        ndim = len(sweep_axes)
        keys = [ax[0] for ax in sweep_axes]
        has_cold = self._has_cold(*keys)

        # Clamp user-requested max_hot to what RAM can hold before computing
        # per-axis counts. Without this, an oversized budget silently
        # produces a grid the kernel OOM-kills mid-sweep.
        effective_max_hot, requested_max_hot = self._memory_capped_max_hot(max_hot)
        try:
            axis_counts = self._compute_axis_counts(
                sweep_axes, has_cold, max_cold, effective_max_hot,
            )
        except RuntimeError as e:
            if effective_max_hot < requested_max_hot:
                raise RuntimeError(
                    f"{e} Memory cap: hot budget reduced from requested "
                    f"{requested_max_hot:,} to {effective_max_hot:,} to fit in "
                    f"available RAM."
                ) from e
            raise

        # Build per-axis value arrays.
        # Categorical axes: ax = (key, values_list) — 2-tuple.
        # Numeric axes:     ax = (key, low, high)   — 3-tuple.
        axes = []
        for i, ax in enumerate(sweep_axes):
            k = ax[0]
            if k in CAT_METRIC_BY_KEY:
                # ax[1] is the list of selected option values
                axes.append(self._metric_values(k, ax[1], None, axis_counts[i]))
            else:
                axes.append(self._metric_values(k, ax[1], ax[2], axis_counts[i]))
        shape = tuple(len(ax) for ax in axes)
        total = int(np.prod(shape))

        grid = np.empty(shape, dtype=_RESULT_DTYPE)

        def _make_swept(idx):
            """Build swept dict for one grid index, preserving value types."""
            return {keys[d]: axes[d][idx[d]] for d in range(ndim)}

        def _progress_params(swept):
            """Coerce swept values to float where possible for progress display."""
            out = {}
            for k, v in swept.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = 0.0
            return out

        # Collect per-cell per-qubit grids during the sweep when requested,
        # so the topology view can scrub cells without paying the cold
        # compile a second time.  Memory ≈ num_cells × num_layers ×
        # num_qubits × 4 grids × 8 bytes (~13 MB for 32×100×128×4).
        per_qubit_cells: dict[tuple, dict] = {}

        def _capture(idx: tuple, full_res: dict, cell_swept: dict) -> None:
            if not keep_per_qubit_grids:
                return
            cell_cfg = (
                _resolve_cell_cold_cfg(cold_config, cell_swept)
                if cold_config is not None else None
            )
            per_qubit_cells[tuple(idx)] = _extract_per_qubit(
                full_res, cached=None, cold_cfg=cell_cfg,
            )

        if parallel and has_cold and cold_config is not None:
            def _indexed_iter():
                for idx in np.ndindex(shape):
                    yield idx, _make_swept(idx)

            rmap = self._parallel_cold_sweep(
                cold_config, fixed_noise, _indexed_iter(), total,
                progress_callback, max_workers,
                keep_grids=keep_per_qubit_grids,
            )
            for idx in np.ndindex(shape):
                full_res = rmap[idx]
                grid[idx] = _result_to_row(full_res)
                _capture(idx, full_res, _make_swept(idx))
        elif has_cold and cold_config is not None:
            count = 0
            for idx in np.ndindex(shape):
                swept = _make_swept(idx)
                full_res = self._eval_point(cold_config, fixed_noise, swept)
                grid[idx] = _result_to_row(full_res)
                _capture(idx, full_res, swept)
                count += 1
                if progress_callback is not None:
                    progress_callback(SweepProgress(
                        completed=count,
                        total=total,
                        current_params=_progress_params(swept),
                    ))
        else:
            # Pure hot-path: stream indices in chunks.
            index_iter = iter(np.ndindex(shape))
            completed = 0
            while True:
                chunk_indices = list(islice(index_iter, self._HOT_BATCH_CHUNK))
                if not chunk_indices:
                    break
                noise_dicts = []
                for idx in chunk_indices:
                    noise = dict(fixed_noise)
                    for d in range(ndim):
                        noise[keys[d]] = axes[d][idx[d]]
                    noise_dicts.append(noise)

                chunk_results = self.run_hot_batch(
                    cached, noise_dicts, keep_grids=keep_per_qubit_grids,
                )
                for i, idx in enumerate(chunk_indices):
                    full_res = chunk_results[i]
                    grid[idx] = _result_to_row(full_res)
                    _capture(idx, full_res, _make_swept(idx))
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(SweepProgress(
                            completed=completed,
                            total=total,
                            current_params=_progress_params(_make_swept(idx)),
                        ))
                del noise_dicts, chunk_results

        # Snapshot the engine inputs so the topology view's overlay can
        # rebuild any cell on demand via run_cold/run_hot without
        # requiring callers to keep the original config around.
        per_qubit_meta: dict | None = None
        if keep_per_qubit_grids:
            per_qubit_meta = {
                "cold_config": dict(cold_config) if cold_config else None,
                "fixed_noise": dict(fixed_noise) if fixed_noise else None,
                "axis_keys": list(keys),
                "axis_values": [ax.tolist() for ax in axes],
                "shape": list(shape),
                # Per-cell per-qubit grids captured during the sweep so the
                # topology view does an O(1) lookup instead of paying the
                # cold compile a second time per slider move.
                "cells": per_qubit_cells,
            }

        return SweepResult(
            metric_keys=keys, axes=axes, grid=grid,
            per_qubit_data=per_qubit_meta,
        )


# ---------------------------------------------------------------------------
# Process-pool worker (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _eval_cold_batch(
    cold_config: dict,
    noise: dict,
    swept_list: list[dict],
    rss_cap_bytes: int | None = None,
    keep_grids: bool = False,
) -> list[dict]:
    """Evaluate a batch of design points sharing the same cold-path config.

    Compiles the circuit once (cold path), then evaluates all noise
    variations in a single batched Rust call (no per-point Python↔Rust overhead).

    ``rss_cap_bytes`` caps the worker's address space via RLIMIT_AS so a
    runaway allocation from a pathologically large circuit raises
    MemoryError inside this process instead of thrashing the whole system.
    """
    if rss_cap_bytes and rss_cap_bytes > 0:
        try:
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS, (int(rss_cap_bytes), int(rss_cap_bytes)),
            )
        except (ImportError, ValueError, OSError):
            # RLIMIT_AS unavailable (non-Linux) or kernel refused — fall
            # back to unbounded execution; the scheduler's mem_ok check
            # still provides some protection.
            pass

    engine = DSEEngine()

    # Apply cold-path overrides from the first swept dict (all share the same cold keys)
    cfg = dict(cold_config)
    for k, v in swept_list[0].items():
        if k in DSEEngine.COLD_PATH_KEYS:
            cfg[k] = int(v)
    _expand_qubits_alias(cfg)
    cfg["num_cores"] = min(cfg["num_cores"], cfg["num_qubits"])
    _clamp_cfg_comm_and_logical(cfg)

    # Build the noise dicts for each swept point
    noise_dicts = []
    for swept in swept_list:
        hot_noise = dict(noise)
        for k, v in swept.items():
            if k not in DSEEngine.COLD_PATH_KEYS:
                hot_noise[k] = v
        noise_dicts.append(hot_noise)

    # Single cold compilation
    cached = engine.run_cold(**cfg, noise=noise_dicts[0])

    # Single batched Rust call for all hot-path variations
    return engine.run_hot_batch(cached, noise_dicts, keep_grids=keep_grids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_distance_matrix(
    full_coupling_map: CouplingMap,
    core_mapping: dict[int, int],
    num_cores: int,
) -> np.ndarray:
    """Replicate the Floyd-Warshall distance computation from map_circuit."""
    core_adj = np.zeros((num_cores, num_cores), dtype=np.int32)
    for edge in full_coupling_map.get_edges():
        c1 = core_mapping.get(edge[0], 0)
        c2 = core_mapping.get(edge[1], 0)
        if c1 != c2:
            core_adj[c1, c2] = 1
            core_adj[c2, c1] = 1

    dist_mat = np.full((num_cores, num_cores), fill_value=9999, dtype=np.int32)
    for c in range(num_cores):
        dist_mat[c, c] = 0
    for c1 in range(num_cores):
        for c2 in range(num_cores):
            if core_adj[c1, c2]:
                dist_mat[c1, c2] = 1

    for k in range(num_cores):
        for i in range(num_cores):
            for j in range(num_cores):
                if dist_mat[i, k] + dist_mat[k, j] < dist_mat[i, j]:
                    dist_mat[i, j] = dist_mat[i, k] + dist_mat[k, j]

    return dist_mat


def _remap_to_dag_layers(placements: np.ndarray, dag_layers: int) -> np.ndarray:
    """Remap TeleSABRE placements from ts-gate-step space to DAG-layer space.

    TeleSABRE's placements have shape (ts_layers+1, num_qubits) where
    ts_layers = num_teledata + num_telegate + num_swaps.  The hot path needs
    (dag_layers+1, num_qubits) to match the Qiskit gs_sparse layer count.
    """
    ts_layers = placements.shape[0] - 1
    num_qubits = placements.shape[1]
    out = np.zeros((dag_layers + 1, num_qubits), dtype=np.int32)
    for l in range(dag_layers + 1):
        ts_l = round(l * ts_layers / dag_layers) if dag_layers > 0 else 0
        out[l] = placements[min(ts_l, ts_layers)]
    return out


def _remap_swaps_to_dag_layers(sparse_swaps: np.ndarray, dag_layers: int) -> np.ndarray:
    """Remap TeleSABRE sparse_swaps layer indices to DAG-layer space."""
    if len(sparse_swaps) == 0:
        return sparse_swaps
    ts_layers = int(sparse_swaps[:, 0].max()) + 1 if len(sparse_swaps) > 0 else 1
    out = sparse_swaps.copy()
    for i in range(len(out)):
        ts_l = int(out[i, 0])
        dag_l = round(ts_l * dag_layers / ts_layers) if ts_layers > 0 else 0
        out[i, 0] = min(dag_l, dag_layers - 1)
    return out


def _build_telesabre_device_json(
    num_qubits: int,
    num_cores: int,
    topology_type: str,
    intracore_topology: str,
    communication_qubits: int = 1,
    buffer_qubits: int = 1,
) -> dict:
    """Build a TeleSABRE-format device JSON dict from DSE topology parameters.

    Two invariants that the TeleSABRE C library requires:

    1. Uniform core sizes: ``device_from_json`` computes core ownership as
       ``phys_to_core[i] = i / core_capacity`` where ``core_capacity =
       total_qubits / num_cores`` (integer division). Non-uniform sizes
       cause the last qubit to map to an out-of-bounds core index,
       corrupting the heap → double-free segfault.  Fix: always use
       ``ceil(num_qubits / num_cores) + SLACK_PER_CORE`` qubits per core.

    2. Free teleportation slots: the round-robin initial layout fills every
       physical qubit when ``total_qubits == num_qubits``.  With no free
       qubits, all nearest-free-qubit heaps are empty → Dijkstra node
       weights equal TS_INF → integer overflow → heap corruption → crash.
       Fix: ``SLACK_PER_CORE = 3`` guarantees at least 3 free slots per
       core after initial layout for mediator/target teleportation.
    """
    import math
    num_cores = max(1, min(num_cores, num_qubits))
    # TeleSABRE's routing requires free physical qubits in each core as
    # teleportation mediators and landing slots.  Without slack, round-robin
    # initial layout fills every slot and routing deadlocks with a heap
    # corruption crash (Dijkstra gets TS_INF node weights, integer arithmetic
    # overflows, and glibc detects a double-free).  Add at least 3 spare
    # qubits per core so each core always has free slots after initial layout.
    SLACK_PER_CORE = 3
    qubits_per_core = math.ceil(num_qubits / num_cores) + SLACK_PER_CORE
    total_qubits = qubits_per_core * num_cores
    offsets = [c * qubits_per_core for c in range(num_cores)]

    # Intra-core edges
    intra_edges = []
    for c in range(num_cores):
        off = offsets[c]
        size = qubits_per_core
        if size < 2:
            continue
        if intracore_topology == "all_to_all":
            for q in range(size):
                for r in range(q + 1, size):
                    intra_edges.append([off + q, off + r])
        elif intracore_topology == "ring":
            for q in range(size):
                intra_edges.append([off + q, off + (q + 1) % size])
        elif intracore_topology == "grid":
            side = math.isqrt(size)
            if side * side < size:
                side += 1
            for q in range(size):
                row, col = divmod(q, side)
                if col + 1 < side and q + 1 < size:
                    intra_edges.append([off + q, off + q + 1])
                if q + side < size:
                    intra_edges.append([off + q, off + q + side])
        else:  # linear (default)
            for q in range(size - 1):
                intra_edges.append([off + q, off + q + 1])

    # Inter-core edges: each comm qubit hosts exactly one inter-core link.
    # A core's qubits are laid out as ``[D data, group_0, group_1, ...]``
    # with each group = ``K`` comm + ``1`` buffer slots; inter_core_edges
    # pairs comm qubit ``k`` of one core's group-toward-partner with comm
    # qubit ``k`` of the partner's group-toward-this.
    inter_edges = []
    if num_cores > 1:
        groups_per_core = num_comm_groups(num_cores, topology_type)
        B_req = max(1, int(buffer_qubits or 1))
        K_caps = [
            _max_K_for_layout(qubits_per_core, groups_per_core[c],
                              b_per_group=B_req)
            for c in range(num_cores)
            if groups_per_core[c] > 0
        ]
        K_max = min(K_caps) if K_caps else 0
        K = min(int(communication_qubits or 1), max(0, K_max))
        if K >= 1:
            B_caps = [
                _max_B_for_layout(qubits_per_core, groups_per_core[c], K)
                for c in range(num_cores)
                if groups_per_core[c] > 0
            ]
            B_max = min(B_caps) if B_caps else 0
            B = min(B_req, max(1, B_max))
            per_core_layout = [
                assign_core_slots(qubits_per_core, intracore_topology,
                                  groups_per_core[c], K, b_per_group=B)
                for c in range(num_cores)
            ]
            for (a_core, a_g, a_k), (b_core, b_g, b_k) in inter_core_edges(
                num_cores, K, topology_type,
            ):
                p1 = offsets[a_core] + per_core_layout[a_core]["groups"][a_g]["comm"][a_k]
                p2 = offsets[b_core] + per_core_layout[b_core]["groups"][b_g]["comm"][b_k]
                inter_edges.append([p1, p2])

    # Simple grid node positions
    side = math.isqrt(total_qubits)
    if side * side < total_qubits:
        side += 1
    node_positions = [
        [float(q % side), -float(q // side)] for q in range(total_qubits)
    ]

    return {
        "device": {
            "name": f"dse_{total_qubits}q_{num_cores}c_{topology_type}",
            "num_cores": num_cores,
            "num_qubits": total_qubits,
            "intra_core_edges": intra_edges,
            "inter_core_edges": inter_edges,
            "node_positions": node_positions,
        }
    }
