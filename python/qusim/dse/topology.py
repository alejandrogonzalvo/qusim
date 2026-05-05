"""
Multi-core device topology: inter-core graph, intra-core slot
layout, comm/buffer placement, and the topology-aware clamps that
keep a configuration physically realisable.

Pure-python + numpy; no qiskit dep aside from CouplingMap which the
Rust core consumes.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit.transpiler import CouplingMap

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

def _compute_distance_matrix(
    full_coupling_map: CouplingMap,
    core_mapping: dict[int, int],
    num_cores: int,
) -> np.ndarray:
    """All-pairs shortest-hop matrix between cores (Floyd-Warshall).

    Mirrors the construction in ``qusim.map_circuit`` so that hot-path
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
