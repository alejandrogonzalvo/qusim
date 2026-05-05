"""
Cold-config normalisation: alias expansion + clamping the user-
supplied dict to whatever the current (num_qubits, num_cores,
topology) combination can physically support.
"""

from __future__ import annotations


# Mirror of DSEEngine class attributes — kept here so this module
# is independent of the engine class. Engine re-exports them for
# back-compat with code that reads DSEEngine.COLD_PATH_KEYS.
COLD_PATH_KEYS: frozenset[str] = frozenset({
    "num_qubits", "num_cores", "topology_type", "intracore_topology",
    "placement_policy", "communication_qubits", "buffer_qubits",
    "num_logical_qubits", "circuit_type", "routing_algorithm",
    "qubits", "seed", "custom_qasm",
})
INTEGER_KEYS: frozenset[str] = frozenset({
    "num_qubits", "num_cores", "communication_qubits",
    "buffer_qubits", "num_logical_qubits", "qubits",
    "classical_link_width", "classical_routing_cycles",
})

from .topology import (
    clamp_b_for_topology,
    clamp_k_for_topology,
    max_data_slots,
)

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
        if k in COLD_PATH_KEYS:
            cfg[k] = int(v) if k in INTEGER_KEYS else v
    _expand_qubits_alias(cfg)
    cfg["num_cores"] = min(int(cfg.get("num_cores", 1)), int(cfg.get("num_qubits", 1)))
    _clamp_cfg_comm_and_logical(cfg)
    return cfg
