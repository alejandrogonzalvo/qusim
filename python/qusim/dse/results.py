"""
Result containers — the dataclasses and structured-array dtype
carried through every DSE call.

Kept separate from the engine so callers can construct or inspect
these without importing the full sweep machinery (and the qiskit
deps that come with it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    # Derived architectural metrics — written by the resolver-backed
    # ``DSEEngine._eval_point`` and read by ``run_hot`` so each cell
    # carries its (cores, qpc, num_qubits, idle) metadata into the
    # result dict alongside the fidelity numbers.
    num_qubits: int = 0
    derived_num_cores: int = 0
    derived_qubits_per_core: int = 0
    idle_reserved_qubits: int = 0

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
    # Derived architectural metrics — written per cell so the heat-map /
    # FoM / Pareto plots can show them as outputs alongside the
    # fidelity/cost columns.  NaN flags an infeasible cell.
    "num_qubits",
    "derived_num_cores",
    "derived_qubits_per_core",
    "idle_reserved_qubits",
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
            "num_cores", "qubits_per_core", "communication_qubits",
            "buffer_qubits", "num_logical_qubits", "topology_type",
            "intracore_topology", "pin_axis", "idle_reserved_qubits",
        ):
            if k in cold_cfg:
                out[k] = cold_cfg[k]
    return out

def _result_to_row(result: dict) -> tuple:
    """Pack a stripped result dict into the tuple order of ``_RESULT_DTYPE``."""
    return tuple(result.get(k, 0.0) for k in _RESULT_SCALAR_KEYS)


def _nan_result_row(reason: str | None = None) -> dict:
    """Result dict for an infeasible sweep cell.

    All numeric fields are NaN so heat-maps render the cell as
    transparent / white and Pareto plots skip it.  ``reason`` is kept
    in the dict so the GUI tooltip / banner can surface why the cell
    was skipped.
    """
    import numpy as _np
    out: dict = {k: _np.nan for k in _RESULT_SCALAR_KEYS}
    out["infeasible"] = True
    if reason:
        out["infeasible_reason"] = reason
    return out


def _row_to_dict(row) -> dict:
    """Unpack a structured-array cell (numpy.void) into a plain dict.

    Leaves plain dicts untouched so legacy 1–3 D sweep paths keep working.
    """
    if isinstance(row, np.void):
        return {k: float(row[k]) for k in row.dtype.names}
    return row

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
