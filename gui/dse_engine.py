"""
Backwards-compatibility shim — the DSE engine now lives in :mod:`qusim.dse`.

Existing imports continue to work::

    from gui.dse_engine import DSEEngine, SweepResult

…but new code should prefer the library path::

    from qusim.dse import DSEEngine, SweepResult
"""

from qusim.dse.engine import *  # noqa: F401, F403
from qusim.dse.engine import (  # noqa: F401  — re-export private names tests rely on
    _RESERVED_RAM_MB,
    _build_circuit,
    _build_topology,
    _clamp_cfg_comm_and_logical,
    _compute_distance_matrix,
    _derived_tele_error,
    _derived_tele_time,
    _estimate_cold_mb,
    _eval_cold_batch,
    _expand_qubits_alias,
    _extract_per_qubit,
    _grid_neighbours,
    _grid_side,
    _make_gate_arrays,
    _max_B_for_layout,
    _max_K_for_layout,
    _max_hot_points_for_memory,
    _merge_noise,
    _resolve_cell_cold_cfg,
    _result_to_row,
    _row_to_dict,
    _strip_for_grid,
    _transpile_circuit,
)
