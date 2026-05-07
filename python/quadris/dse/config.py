"""
Cold-config normalisation under the **logical-first** parameterization.

The user pins exactly one of ``num_cores`` or ``qubits_per_core`` (via
``pin_axis``); the unpinned axis is deduced from
``(num_logical_qubits, K, B, inter_topology)`` so the circuit always
fits.  ``num_qubits`` (physical device size) is a derived field
written here for downstream consumption — it is never user-set.

When deduction fails (no ``nc`` satisfies the constraints, or the
pinned ``qpc`` is too small to host even one group of ``K+B`` slots),
:func:`_resolve_architecture` returns ``feasible=False`` and the caller
is responsible for rendering the cell as NaN.
"""

from __future__ import annotations

from .topology import (
    deduce_num_cores,
    deduce_qubits_per_core,
    g_max,
    idle_reserved_qubits as _idle_reserved_qubits,
)


# Cold-path keys: changing any of these forces a full re-compile.
COLD_PATH_KEYS: frozenset[str] = frozenset({
    "num_logical_qubits", "qubits_per_core", "num_cores",
    "communication_qubits", "buffer_qubits",
    "topology_type", "intracore_topology",
    "placement_policy", "circuit_type", "routing_algorithm",
    "pin_axis", "seed", "custom_qasm",
})

# Integer-typed keys.  Sweep grids quantise these values via np.round.
INTEGER_KEYS: frozenset[str] = frozenset({
    "num_logical_qubits", "qubits_per_core", "num_cores",
    "communication_qubits", "buffer_qubits",
    "classical_link_width", "classical_routing_cycles",
})


# Pin axis vocabulary
PIN_CORES = "cores"
PIN_QPC = "qubits_per_core"
DEFAULT_PIN_AXIS = PIN_CORES


def _resolve_architecture(cfg: dict) -> dict:
    """Logical-first deduction — fills in the unpinned architectural axis.

    Reads (and respects):
      * ``num_logical_qubits``
      * ``communication_qubits`` (K), ``buffer_qubits`` (B) — clamped so
        ``B ≤ K`` (per-group rule)
      * ``topology_type`` (inter-core)
      * ``pin_axis`` ∈ {"cores", "qubits_per_core"} — defaults to
        :data:`DEFAULT_PIN_AXIS`

    Writes (in-place on ``cfg``):
      * the *derived* axis
      * ``num_qubits`` (= ``num_cores · qubits_per_core``)
      * ``idle_reserved_qubits``

    Returns a small dict::

        {
            "feasible": bool,
            "reason": str | None,   # human-readable when not feasible
        }

    Mutates ``cfg`` even when infeasible (so callers can still log /
    inspect the partial state); however ``num_qubits`` is set to 0 so
    downstream cold-compile attempts fail fast instead of running on
    a half-resolved config.
    """
    L = max(2, int(cfg.get("num_logical_qubits", 2) or 2))
    cfg["num_logical_qubits"] = L

    K = max(0, int(cfg.get("communication_qubits", 1) or 0))
    B = max(0, int(cfg.get("buffer_qubits", 1) or 0))
    if B > K:
        cfg["feasible"] = False
        cfg["num_qubits"] = 0
        cfg["idle_reserved_qubits"] = 0
        return {
            "feasible": False,
            "reason": f"buffer_qubits ({B}) > communication_qubits ({K}); per-group rule violated.",
        }
    cfg["communication_qubits"] = K
    cfg["buffer_qubits"] = B

    inter_topo = cfg.get("topology_type") or "ring"
    pin = cfg.get("pin_axis", DEFAULT_PIN_AXIS)
    if pin not in (PIN_CORES, PIN_QPC):
        pin = DEFAULT_PIN_AXIS
    cfg["pin_axis"] = pin

    if pin == PIN_CORES:
        nc = max(1, int(cfg.get("num_cores", 1) or 1))
        qpc = deduce_qubits_per_core(L, nc, K, B, inter_topo)
        cfg["num_cores"] = nc
        cfg["qubits_per_core"] = int(qpc)
    else:
        qpc = max(1, int(cfg.get("qubits_per_core", 1) or 1))
        nc = deduce_num_cores(L, qpc, K, B, inter_topo)
        if nc is None:
            cfg["feasible"] = False
            cfg["num_qubits"] = 0
            cfg["idle_reserved_qubits"] = 0
            return {
                "feasible": False,
                "reason": (
                    f"no num_cores satisfies logical={L} with qpc={qpc}, "
                    f"K={K}, B={B} on inter-topology '{inter_topo}'."
                ),
            }
        cfg["num_cores"] = int(nc)
        cfg["qubits_per_core"] = int(qpc)

    cfg["num_qubits"] = int(cfg["num_cores"]) * int(cfg["qubits_per_core"])
    cfg["idle_reserved_qubits"] = _idle_reserved_qubits(
        cfg["num_cores"], K, B, inter_topo,
    )
    cfg["feasible"] = True
    return {"feasible": True, "reason": None}


def _resolve_cell_cold_cfg(
    cold_config: dict, swept: dict,
) -> dict:
    """Apply a cell's swept overrides + resolve the architecture.

    Mirrors the resolution in :func:`DSEEngine._eval_point` and
    :func:`_eval_cold_batch` so the captured per-cell cold cfg matches
    what the engine actually compiled with.
    """
    cfg = dict(cold_config)
    for k, v in swept.items():
        if k in COLD_PATH_KEYS:
            cfg[k] = int(v) if k in INTEGER_KEYS else v
    _resolve_architecture(cfg)
    return cfg
