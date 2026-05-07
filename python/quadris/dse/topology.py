"""
Multi-core device topology: inter-core graph, intra-core slot
layout, comm/buffer placement, and the deduction helpers that
turn ``(num_logical_qubits, qubits_per_core, K, B)`` into a
concrete ``num_cores`` (or vice versa).

Logical-first model
-------------------

Every core reserves the **same** ``G_max · (K + B)`` slots,
regardless of its actual neighbour count.  Cores at the corners /
edges of non-uniform inter-topologies (grid, linear) carry idle
comm slots — the trade-off for "all cores look identical" and
uniform per-core data capacity.

  data_per_core = qpc − G_max · (K + B)

The user pins exactly one of ``num_cores`` or ``qubits_per_core``;
the other is deduced so ``nc · data_per_core ≥ num_logical_qubits``.

Pure-python + numpy; no qiskit dep aside from CouplingMap which the
Rust core consumes.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit.transpiler import CouplingMap


# ---------------------------------------------------------------------------
# Inter-core neighbour graphs (purely topological — no slot accounting)
# ---------------------------------------------------------------------------

def inter_core_neighbors(num_cores: int, inter_topology: str) -> list[list[int]]:
    """Per-core list of unique neighbouring core indices."""
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
    """Per-core *ordered* list of partner cores — one entry per inter-core link."""
    return inter_core_neighbors(num_cores, inter_topology)


def num_comm_groups(num_cores: int, inter_topology: str) -> list[int]:
    """Number of comm-qubit groups per core (= number of inter-core neighbours)."""
    return [len(g) for g in core_groups_for(num_cores, inter_topology)]


def g_max(num_cores: int, inter_topology: str) -> int:
    """Worst-case neighbour count across the chip — drives uniform reservation.

    Every core in the device reserves ``G_max · (K + B)`` slots regardless
    of its actual ``G(c)``.  Corner/edge cores in grid or linear
    topologies therefore carry idle comm slots, but every core has the
    same data capacity ``qpc − G_max · (K+B)``.

    nc=1 returns 0 (no inter-core links exist).
    """
    if num_cores < 2:
        return 0
    return max(len(n) for n in inter_core_neighbors(num_cores, inter_topology))


# ---------------------------------------------------------------------------
# Local slot layout (data + comm + buffer roles within one core)
# ---------------------------------------------------------------------------

def _grid_side(core_size: int) -> int:
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

    Reserves ``num_groups · (K + B)`` slots — i.e., one group per *active*
    inter-core neighbour.  Idle comm groups (the ``G_max − G(c)`` extras
    that come from uniform reservation) are NOT placed here; the caller
    is responsible for accounting for them in the total core size.

    For grid intra-core, comm slots are pinned to a side edge (left /
    right / top / bottom in priority order) and buffer slots are picked
    from cells adjacent (4-neighbourhood) to the comm column.  For
    non-grid intra-core (linear / ring / all-to-all) we use a slot-order
    fallback (data first, then per-group [K comm + B buffer]).
    """
    G = max(0, int(num_groups))
    K = max(0, int(k_per_group))
    B = max(1, int(b_per_group or 1)) if K > 0 else 0
    reserved = G * (K + B)
    data_count = core_size - reserved
    intra = (intracore_topology or "all_to_all").lower()
    use_grid_layout = intra in ("grid", "all_to_all")

    if G == 0 or K == 0 or not use_grid_layout:
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


# ---------------------------------------------------------------------------
# Inter-core edges (comm-qubit pairs across the fabric)
# ---------------------------------------------------------------------------

def inter_core_edges(
    num_cores: int,
    communication_qubits: int,
    inter_topology: str,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Comm-qubit-to-comm-qubit edges across the multi-core fabric.

    Returns ``((core_a, group_a, k), (core_b, group_b, k))`` edges, where
    ``group_a`` is core ``a``'s active group index dedicated to core ``b``
    (and vice versa) and ``k`` is the same comm-qubit index on both ends.
    Idle (reserved-but-unused) groups have no edges and never appear.
    """
    if num_cores < 2 or communication_qubits < 1:
        return []
    K = int(communication_qubits)
    nbrs = inter_core_neighbors(num_cores, inter_topology)
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


# ---------------------------------------------------------------------------
# Logical-first deduction
# ---------------------------------------------------------------------------

def _data_per_core(qpc: int, num_cores: int, k: int, b: int,
                   inter_topology: str) -> int:
    """Uniform per-core data capacity under G_max reservation.

    Returns 0 (or negative) when the reservation eats every slot — caller
    must check for feasibility.
    """
    return int(qpc) - g_max(num_cores, inter_topology) * (int(k) + int(b))


def deduce_num_cores(
    num_logical_qubits: int,
    qubits_per_core: int,
    communication_qubits: int,
    buffer_qubits: int,
    inter_topology: str,
    *,
    nc_max: int = 256,
) -> int | None:
    """Smallest ``nc`` such that ``nc · data_per_core ≥ num_logical_qubits``.

    Returns ``None`` when no feasible ``nc`` exists in ``[1, nc_max]``.
    For all-to-all the capacity is non-monotone in ``nc`` (downward
    parabola); we still scan linearly because the search space is small
    and the cost of a few extra iterations is negligible.
    """
    L = max(1, int(num_logical_qubits))
    qpc = max(1, int(qubits_per_core))
    K = max(0, int(communication_qubits))
    B = max(0, int(buffer_qubits))
    for nc in range(1, max(2, int(nc_max)) + 1):
        dpc = _data_per_core(qpc, nc, K, B, inter_topology)
        if dpc <= 0:
            continue
        if nc * dpc >= L:
            return nc
    return None


def deduce_qubits_per_core(
    num_logical_qubits: int,
    num_cores: int,
    communication_qubits: int,
    buffer_qubits: int,
    inter_topology: str,
) -> int:
    """Smallest ``qpc`` such that ``nc · (qpc − G_max·(K+B)) ≥ logical``.

    Always feasible: ``qpc`` grows freely to absorb whatever overhead the
    pinned ``nc`` + comm/buffer demand.
    """
    L = max(1, int(num_logical_qubits))
    nc = max(1, int(num_cores))
    K = max(0, int(communication_qubits))
    B = max(0, int(buffer_qubits))
    overhead = g_max(nc, inter_topology) * (K + B)
    return math.ceil(L / nc) + overhead


def idle_reserved_qubits(
    num_cores: int,
    communication_qubits: int,
    buffer_qubits: int,
    inter_topology: str,
) -> int:
    """Sum of unused comm slots across the chip.

    A core with ``G(c) < G_max`` reserves ``G_max·(K+B)`` slots but only
    uses ``G(c)·(K+B)`` of them — the difference is idle hardware.
    """
    if num_cores < 2:
        return 0
    g_active = num_comm_groups(num_cores, inter_topology)
    Gm = g_max(num_cores, inter_topology)
    K = max(0, int(communication_qubits))
    B = max(0, int(buffer_qubits))
    return sum((Gm - g) * (K + B) for g in g_active)


# ---------------------------------------------------------------------------
# Coupling map construction (uniform G_max reservation per core)
# ---------------------------------------------------------------------------

def _build_topology(
    num_qubits: int,
    num_cores: int,
    topology_type: str,
    intracore_topology: str = "all_to_all",
    communication_qubits: int = 1,
    buffer_qubits: int = 1,
) -> tuple[CouplingMap, dict[int, int]]:
    """Build full_coupling_map and core_mapping.

    Every core has the same physical size ``qpc = num_qubits // num_cores``
    and reserves ``G_max · (K + B)`` slots regardless of its actual
    neighbour count.  Idle reserved slots have no inter-core edges (they
    participate in the intra-core topology only — like buffer slots).

    ``num_qubits`` MUST equal ``num_cores · qpc`` exactly; the deduction
    layer in :mod:`quadris.dse.config` is responsible for making that
    arithmetic land on a whole number.  When it doesn't (e.g. legacy
    callers pass mismatched values), the remainder cores get an extra
    qubit to keep the total honest, but the data/comm slot accounting
    assumes uniform sizes — heterogeneous device sizes are not supported
    in the logical-first model.
    """
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
        K = max(0, int(communication_qubits or 0))
        B = max(0, int(buffer_qubits or 0))
        if K >= 1:
            # Each core reserves G_max active groups' worth of slots, but
            # only its actual G(c) groups host inter-core edges.  We lay
            # out the *active* groups via assign_core_slots; idle groups
            # would consume additional slots in core_size accounting but
            # carry no inter-core edges here.
            g_active = num_comm_groups(num_cores, topology_type)
            per_core_layout = [
                assign_core_slots(core_sizes[c], intracore_topology,
                                  g_active[c], K, b_per_group=B)
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
# Aggregate accounting
# ---------------------------------------------------------------------------

def total_reserved_slots(num_qubits: int, num_cores: int,
                         topology_type: str, k_per_group: int,
                         b_per_group: int = 1) -> int:
    """Total non-data slots reserved across the chip = nc · G_max · (K+B).

    Uniform reservation: every core reserves ``G_max · (K+B)`` slots,
    even cores with fewer actual neighbours.  Idle reserved slots count
    here too — see :func:`idle_reserved_qubits` for that breakdown.
    """
    nc = max(1, int(num_cores or 1))
    K = max(0, int(k_per_group or 0))
    B = max(0, int(b_per_group or 0))
    if nc < 2 or K < 1:
        return 0
    return nc * g_max(nc, topology_type or "ring") * (K + B)


def max_data_slots(num_qubits: int, num_cores: int,
                   topology_type: str, k_per_group: int,
                   b_per_group: int = 1) -> int:
    """Number of data (non-comm, non-buffer) slots available chip-wide."""
    nq = max(1, int(num_qubits or 1))
    return max(0, nq - total_reserved_slots(
        nq, num_cores, topology_type, k_per_group, b_per_group,
    ))


def _compute_distance_matrix(
    full_coupling_map: CouplingMap,
    core_mapping: dict[int, int],
    num_cores: int,
) -> np.ndarray:
    """All-pairs shortest-hop matrix between cores (Floyd-Warshall).

    Mirrors the construction in ``quadris.map_circuit`` so that hot-path
    fidelity estimation can run without re-deriving it.
    """
    core_adj = np.zeros((num_cores, num_cores), dtype=np.int32)
    for edge in full_coupling_map.get_edges():
        c1, c2 = core_mapping[edge[0]], core_mapping[edge[1]]
        if c1 != c2:
            core_adj[c1, c2] = 1
            core_adj[c2, c1] = 1
    dist_mat = np.full((num_cores, num_cores), 9999, dtype=np.int32)
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


# ---------------------------------------------------------------------------
# Legacy slot-layout entry point — kept for the GUI's topology-overlay
# renderer which constructs core slots without going through the full
# coupling-map builder.
# ---------------------------------------------------------------------------

def core_slot_layout(core_size: int, num_groups: int, k_per_group: int) -> dict:
    """Legacy slot layout (data + ``G·(K+1)`` comm/buffer columns).

    Kept for back-compat callers that need the simple slot-order layout
    without grid-edge placement.  New code should use
    :func:`assign_core_slots` instead.
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


# ---------------------------------------------------------------------------
# Legacy placement caps — only used inside _build_telesabre_device_json
# for sanity (the GUI no longer calls these directly).
# ---------------------------------------------------------------------------

def _max_K_for_layout(core_size: int, num_groups: int, b_per_group: int = 1) -> int:
    """Largest ``K`` (comm per group) that leaves at least one data qubit."""
    if num_groups <= 0:
        return 0
    B = max(0, int(b_per_group or 0))
    return max(0, (core_size - 1) // num_groups - B)


def _max_B_for_layout(core_size: int, num_groups: int, k_per_group: int) -> int:
    """Largest feasible ``B`` (buffers per group) given ``K``."""
    if num_groups <= 0 or k_per_group <= 0:
        return 0
    K = max(1, int(k_per_group))
    arch_cap = (core_size - 1) // num_groups - K
    return max(0, min(K, arch_cap))
