#!/usr/bin/env python3
"""Generate canned example sessions for the Topbar "Examples" dropdown.

Run once after editing ``gui/examples.py``::

    poetry run python scripts/generate_example_sessions.py

The script runs each :class:`gui.examples.ExampleSpec` through the live DSE
engine and writes a gzipped session file to ``gui/assets/examples/`` using the
same schema as the GUI's Save button. Loading the file from the dropdown is
indistinguishable from re-opening a hand-saved session.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui.constants import (  # noqa: E402
    METRIC_BY_KEY,
    NOISE_DEFAULTS,
    SWEEPABLE_METRICS,
)
from gui.dse_engine import DSEEngine  # noqa: E402
from gui.examples import EXAMPLES, ExampleSpec, EXAMPLES_DIR  # noqa: E402
from gui.fom import DEFAULT_FOM  # noqa: E402
from gui.session import (  # noqa: E402
    build_controls_dict,
    build_view_dict,
    collect_session,
    dump,
)


def _value_to_slider(val: float, log_scale: bool) -> float:
    """Mirror ``app._value_to_slider`` so the canned controls dict matches
    what the load callback expects."""
    if log_scale and val is not None and val > 0:
        return math.log10(val)
    return val


def _grid_to_jsonable(grid, ndim: int):
    """Convert a sweep grid to JSON-serializable form.

    1–3 D sweeps already produce nested lists of plain dicts. The N≥4 path
    in :meth:`SweepResult.to_sweep_data` instead returns a structured numpy
    array — that's a fast in-process representation but ``json.dumps`` won't
    accept it. Flatten to a list of plain-dict cells matching what the
    ``_flatten_nd`` plotting path consumes.
    """
    if isinstance(grid, np.ndarray) and grid.dtype.names:
        names = list(grid.dtype.names)
        flat = grid.reshape(-1)
        return [{n: float(row[n]) for n in names} for row in flat]
    return grid


def _build_noise_values(spec: ExampleSpec) -> dict[str, float]:
    """Raw-value noise dict mirroring the right-panel slider state."""
    out: dict[str, float] = {}
    sweep_keys = {ax[0] for ax in spec.sweep_axes}
    for m in SWEEPABLE_METRICS:
        if m.is_cold_path:
            continue
        # Skip metrics that are on the sweep axes — the GUI hides those
        # from the right-panel by leaving the slider at no_update, which
        # serialises as no entry in noise_values.
        if m.key in sweep_keys:
            continue
        if m.key in spec.fixed_noise:
            out[m.key] = float(spec.fixed_noise[m.key])
        elif m.key in NOISE_DEFAULTS:
            out[m.key] = float(NOISE_DEFAULTS[m.key])
    return out


def _build_engine_noise(spec: ExampleSpec) -> dict[str, float]:
    """Raw-value noise dict actually fed to ``DSEEngine.sweep_nd``.

    Same shape as ``_build_noise_values`` plus ``dynamic_decoupling`` and
    keeping the swept-axis defaults so the engine has a complete dict.
    """
    noise: dict = {}
    for m in SWEEPABLE_METRICS:
        if m.is_cold_path:
            continue
        if m.key in spec.fixed_noise:
            noise[m.key] = float(spec.fixed_noise[m.key])
        elif m.key in NOISE_DEFAULTS:
            noise[m.key] = float(NOISE_DEFAULTS[m.key])
    noise["dynamic_decoupling"] = False
    return noise


def _build_controls(spec: ExampleSpec) -> dict:
    cc = spec.cold_config
    n_axes = len(spec.sweep_axes)

    dropdown_vals = [ax[0] for ax in spec.sweep_axes]
    slider_vals = [[ax[1], ax[2]] for ax in spec.sweep_axes]
    checklist_vals = [None] * n_axes

    return build_controls_dict(
        num_metrics=n_axes,
        dropdown_vals=dropdown_vals,
        slider_vals=slider_vals,
        checklist_vals=checklist_vals,
        cfg_circuit_type=cc.get("circuit_type", "qft"),
        cfg_num_qubits=int(cc.get("num_qubits", 16)),
        cfg_num_cores=int(cc.get("num_cores", 4)),
        cfg_communication_qubits=int(cc.get("communication_qubits", 1)),
        cfg_num_logical_qubits=int(cc.get("num_logical_qubits", cc.get("num_qubits", 16))),
        cfg_topology=cc.get("topology_type", "ring"),
        cfg_intracore_topology=cc.get("intracore_topology", "all_to_all"),
        cfg_placement=cc.get("placement_policy", "spectral"),
        cfg_routing_algorithm=cc.get("routing_algorithm", "hqa_sabre"),
        cfg_seed=int(cc.get("seed", 42)),
        cfg_dynamic_decoupling=[],
        cfg_max_cold=spec.max_cold,
        cfg_max_hot=spec.max_hot,
        cfg_max_workers=None,
        cfg_output_metric=spec.output_metric,
        cfg_view_mode=spec.view_mode,
        cfg_threshold_enable=[],
        num_thresholds=3,
        threshold_values=[None, None, None, None, None],
        threshold_colors=[None, None, None, None, None],
        noise_values=_build_noise_values(spec),
        hot_reload=["on"],
        fom_config=DEFAULT_FOM.to_dict(),
    )


def _engine_cold_config(spec: ExampleSpec) -> dict:
    """Build the cold_config dict ``DSEEngine.sweep_nd`` expects."""
    cc = dict(spec.cold_config)
    cc.setdefault("circuit_type", "qft")
    cc.setdefault("num_qubits", 16)
    cc.setdefault("num_logical_qubits", cc["num_qubits"])
    cc.setdefault("num_cores", 4)
    cc.setdefault("communication_qubits", 1)
    cc.setdefault("buffer_qubits", 1)
    cc.setdefault("topology_type", "ring")
    cc.setdefault("intracore_topology", "all_to_all")
    cc.setdefault("placement_policy", "spectral")
    cc.setdefault("routing_algorithm", "hqa_sabre")
    cc.setdefault("seed", 42)
    cc.setdefault("custom_qasm", None)
    return cc


def _run_sweep(spec: ExampleSpec, engine: DSEEngine) -> dict:
    cold_cfg = _engine_cold_config(spec)
    noise = _build_engine_noise(spec)

    sweep_axes = [(ax[0], float(ax[1]), float(ax[2])) for ax in spec.sweep_axes]
    keys = [ax[0] for ax in sweep_axes]
    has_cold = engine._has_cold(*keys)

    cached = (
        None if has_cold
        else engine.run_cold(**cold_cfg, noise=noise)
    )
    # Mirror the Run-button path: cold-path axes go through the parallel
    # worker pool because the serial ``_eval_point`` route currently passes
    # ``epr_error_per_hop`` / ``measurement_error`` to the Rust binding,
    # which doesn't accept them.
    result = engine.sweep_nd(
        cached=cached,
        sweep_axes=sweep_axes,
        fixed_noise=noise,
        cold_config=cold_cfg,
        progress_callback=None,
        parallel=has_cold,
        keep_per_qubit_grids=False,
        max_workers=1,
        max_cold=spec.max_cold,
        max_hot=spec.max_hot,
    )
    sweep_data = result.to_sweep_data()
    if "grid" in sweep_data:
        sweep_data["grid"] = _grid_to_jsonable(sweep_data["grid"], result.ndim)
    sweep_data.pop("per_qubit_data", None)
    return sweep_data


def _interesting(sweep_data: dict, output: str = "overall_fidelity") -> tuple[float, float]:
    """Compute (min, max) of the output field across the sweep — sanity
    check that the chosen ranges actually traverse interesting regions."""
    grid = sweep_data["grid"]
    if isinstance(grid, list):
        flat = []

        def walk(g):
            if isinstance(g, list):
                for v in g:
                    walk(v)
            elif isinstance(g, dict):
                v = g.get(output)
                if v is not None:
                    flat.append(float(v))

        walk(grid)
        if not flat:
            return (0.0, 0.0)
        return (min(flat), max(flat))
    if isinstance(grid, np.ndarray) and grid.dtype.names:
        col = grid[output].ravel()
        return (float(col.min()), float(col.max()))
    return (0.0, 0.0)


def _generate_one(spec: ExampleSpec, engine: DSEEngine, out_dir: Path) -> None:
    print(f"[{spec.id}] sweeping ({len(spec.sweep_axes)}-D, {spec.label})")
    t0 = time.time()
    sweep_data = _run_sweep(spec, engine)
    sweep_dt = time.time() - t0

    fid_lo, fid_hi = _interesting(sweep_data, spec.output_metric)
    print(
        f"[{spec.id}] {spec.output_metric}: "
        f"min={fid_lo:.4f}  max={fid_hi:.4f}   ({sweep_dt:.1f}s)"
    )

    controls = _build_controls(spec)
    view = build_view_dict(spec.view_type, spec.frozen_axis, spec.frozen_slider_value)
    session = collect_session(controls, view, sweep_data, name=spec.label)
    raw = dump(session)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.id}.qusim.json.gz"
    out_path.write_bytes(raw)
    size_kb = len(raw) / 1024
    print(f"[{spec.id}] wrote {out_path}  ({size_kb:.1f} KB)")


def main() -> int:
    out_dir = EXAMPLES_DIR
    print(f"Output dir: {out_dir}")
    engine = DSEEngine()

    t0 = time.time()
    for spec in EXAMPLES:
        try:
            _generate_one(spec, engine, out_dir)
        except Exception as exc:
            print(f"[{spec.id}] FAILED: {exc!r}", file=sys.stderr)
            raise
    print(f"All examples generated in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
