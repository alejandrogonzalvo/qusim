"""
Backwards-compatibility shim — the DSE engine now lives in :mod:`quadris.dse`.

Existing imports continue to work::

    from gui.dse_engine import DSEEngine, SweepResult

…but new code should prefer the library path::

    from quadris.dse import DSEEngine, SweepResult

The logical-first parameterization removed several helpers (the
``_clamp_cfg_*`` family, ``clamp_k_for_topology``, the ``qubits``
alias) so this shim no longer re-exports them.
"""

from quadris.dse.engine import *  # noqa: F401, F403
from quadris.dse.config import (  # noqa: F401
    DEFAULT_PIN_AXIS,
    PIN_CORES,
    PIN_QPC,
    _resolve_architecture,
    _resolve_cell_cold_cfg,
)
from quadris.dse.engine import (  # noqa: F401  — re-export private names tests rely on
    _RESERVED_RAM_MB,
    _build_circuit,
    _build_topology,
    _compute_distance_matrix,
    _derived_tele_error,
    _derived_tele_time,
    _estimate_cold_mb,
    _eval_cold_batch,
    _extract_per_qubit,
    _grid_neighbours,
    _grid_side,
    _make_gate_arrays,
    _max_B_for_layout,
    _max_K_for_layout,
    _max_hot_points_for_memory,
    _merge_noise,
    _result_to_row,
    _row_to_dict,
    _strip_for_grid,
    _transpile_circuit,
    deduce_num_cores,
    deduce_qubits_per_core,
    g_max,
    idle_reserved_qubits,
)
