"""
Figure-of-Merit (FoM) evaluator for sweep results.

A FoM is a user-defined scalar = ``numerator_expr / denominator_expr`` where
both sides are arithmetic expressions over *primitives*:

- currently-swept input axes (e.g. ``t1``, ``two_gate_error``, ``num_qubits``)
- simulation outputs (``overall_fidelity``, ``total_epr_pairs``, ...)

Users may also define named intermediate variables to keep formulas readable.

The evaluator parses expressions with :mod:`ast` and rejects every construct
outside a small whitelist — no attribute access, subscripting, lambdas,
comprehensions, imports, or arbitrary function calls. Evaluation is fully
vectorised on numpy columns, so the cost per grid point is a few fused BLAS
ops.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class FomError(ValueError):
    """Raised when a FoM expression is invalid or unsafe."""


# Functions exposed to user expressions. All are vectorised numpy variants.
SAFE_FUNCS: dict[str, Any] = {
    "log": np.log,
    "ln": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "pow": np.power,
    "clip": np.clip,
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Constant,
    ast.Name, ast.Load,
    ast.Call,
)


def _validate(tree: ast.AST, allowed_names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise FomError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FomError("only direct function calls are allowed")
            if node.func.id not in SAFE_FUNCS:
                raise FomError(f"unknown function: {node.func.id!r}")
            if node.keywords:
                raise FomError("keyword arguments are not supported")
        if isinstance(node, ast.Name):
            name = node.id
            if name in SAFE_FUNCS or name in allowed_names:
                continue
            raise FomError(f"unknown variable: {name!r}")
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise FomError(
                    f"only numeric constants are allowed, got {type(node.value).__name__}"
                )


def _compile_expr(expr: str, allowed_names: set[str]) -> Any:
    expr = (expr or "").strip()
    if not expr:
        raise FomError("expression is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FomError(f"syntax error: {exc.msg}") from exc
    _validate(tree, allowed_names)
    return compile(tree, "<fom>", "eval")


def _eval_compiled(code: Any, env: dict[str, Any]) -> Any:
    # ``SAFE_FUNCS`` live in globals so user-supplied columns (locals) can't
    # shadow them. ``__builtins__`` is scrubbed so no implicit names leak in.
    globs: dict[str, Any] = {"__builtins__": {}, **SAFE_FUNCS}
    return eval(code, globs, env)  # noqa: S307 — AST already validated


@dataclass(frozen=True)
class FomConfig:
    """User-authored Figure of Merit formula.

    ``intermediates`` is an ordered list of ``(name, expr)`` pairs. Each
    intermediate may reference primitives and previously-defined intermediates.
    """

    name: str = "Figure of Merit"
    numerator: str = "overall_fidelity"
    denominator: str = "1"
    intermediates: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_dict(cls, d: dict | None) -> "FomConfig":
        if not d:
            return cls()
        intermediates_raw = d.get("intermediates") or ()
        intermediates: list[tuple[str, str]] = []
        for row in intermediates_raw:
            if isinstance(row, (list, tuple)) and len(row) == 2:
                intermediates.append((str(row[0]), str(row[1])))
        return cls(
            name=str(d.get("name") or "Figure of Merit"),
            numerator=str(d.get("numerator") or "overall_fidelity"),
            denominator=str(d.get("denominator") or "1"),
            intermediates=tuple(intermediates),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "intermediates": [list(t) for t in self.intermediates],
        }


def evaluate(config: FomConfig, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised FoM evaluation over a column-wise primitive table.

    ``columns[k]`` must be a 1-D ndarray of the same length for every ``k``.
    Returns a float64 array of the same length; division-by-zero becomes NaN.
    """
    if not columns:
        raise FomError("no primitives available")
    size = next(iter(columns.values())).shape[0]
    allowed: set[str] = set(columns.keys())
    env: dict[str, Any] = dict(columns)

    for raw_name, raw_expr in config.intermediates:
        name = (raw_name or "").strip()
        if not name:
            continue
        if not name.isidentifier():
            raise FomError(f"invalid intermediate name: {name!r}")
        if name in allowed:
            raise FomError(f"intermediate {name!r} shadows a primitive")
        code = _compile_expr(raw_expr, allowed)
        value = _eval_compiled(code, env)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(size, float(arr))
        elif arr.shape != (size,):
            raise FomError(f"intermediate {name!r} produced wrong shape {arr.shape}")
        env[name] = arr
        allowed.add(name)

    num_code = _compile_expr(config.numerator or "1", allowed)
    den_code = _compile_expr(config.denominator or "1", allowed)
    num = np.asarray(_eval_compiled(num_code, env), dtype=float)
    den = np.asarray(_eval_compiled(den_code, env), dtype=float)
    if num.ndim == 0:
        num = np.full(size, float(num))
    if den.ndim == 0:
        den = np.full(size, float(den))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "fidelity": {
        "name": "Fidelity",
        "numerator": "overall_fidelity",
        "denominator": "1",
        "intermediates": [],
    },
    "fidelity_per_epr": {
        "name": "Fidelity / EPR",
        "numerator": "overall_fidelity",
        "denominator": "max(total_epr_pairs, 1)",
        "intermediates": [],
    },
    "fidelity_per_time": {
        "name": "Fidelity / Time",
        "numerator": "overall_fidelity",
        "denominator": "max(total_circuit_time_ns, 1)",
        "intermediates": [],
    },
    "fidelity_over_cost": {
        "name": "Fidelity / (EPR + α·time)",
        "numerator": "overall_fidelity",
        "denominator": "cost",
        "intermediates": [
            ["cost", "max(total_epr_pairs + 1e-9 * total_circuit_time_ns, 1)"],
        ],
    },
    "neg_log_infidelity_per_epr": {
        "name": "−log(1−F) / EPR",
        "numerator": "log(1 / max(1 - overall_fidelity, 1e-12))",
        "denominator": "max(total_epr_pairs, 1)",
        "intermediates": [],
    },
}


PRESET_OPTIONS = [
    {"label": "Fidelity (sanity check)", "value": "fidelity"},
    {"label": "Fidelity / EPR pairs", "value": "fidelity_per_epr"},
    {"label": "Fidelity / Circuit time", "value": "fidelity_per_time"},
    {"label": "Fidelity / (EPR + α·time)", "value": "fidelity_over_cost"},
    {"label": "−log(1−F) / EPR", "value": "neg_log_infidelity_per_epr"},
]


DEFAULT_FOM = FomConfig.from_dict(PRESETS["fidelity_per_epr"])


# ---------------------------------------------------------------------------
# Parse + compute helpers used by the plot/view layer
# ---------------------------------------------------------------------------


@dataclass
class FomResult:
    """Outcome of computing a FoM over a sweep.

    ``values`` is None on failure; ``error`` is None on success. ``primitives``
    lists the names available to the user (useful for the UI helper text).
    """

    values: np.ndarray | None
    primitives: list[str] = field(default_factory=list)
    error: str | None = None


def primitives_for_sweep(sweep_data: dict) -> tuple[list[str], list[str]]:
    """Return (input_primitives, output_primitives) for a given sweep.

    Input primitives are whatever is currently on the sweep axes;
    output primitives are every QuadrisResult field discovered in a sample row.
    Kept out of ``plotting.py`` to avoid the heavy import on the UI path.
    """
    # Local import avoids a circular dependency at module load time.
    from quadris.dse.flatten import flatten_sweep_to_table as _flatten_sweep_to_table

    metric_keys, outputs, _ = _flatten_sweep_to_table(sweep_data)
    return list(metric_keys), list(outputs)


def compute_for_sweep(sweep_data: dict, config: FomConfig) -> FomResult:
    """Evaluate the FoM at every sweep point and return a ``FomResult``.

    Never raises — user-facing errors are surfaced via ``FomResult.error``.
    """
    from quadris.dse.flatten import flatten_sweep_to_table as _flatten_sweep_to_table

    try:
        metric_keys, outputs, rows = _flatten_sweep_to_table(sweep_data)
    except Exception as exc:  # pragma: no cover - defensive
        return FomResult(values=None, error=f"could not flatten sweep: {exc}")

    primitives = list(metric_keys) + list(outputs)
    if rows.size == 0:
        return FomResult(values=None, primitives=primitives, error="sweep is empty")

    columns: dict[str, np.ndarray] = {}
    for j, k in enumerate(metric_keys):
        columns[k] = rows[:, j]
    for j, k in enumerate(outputs):
        columns[k] = rows[:, len(metric_keys) + j]

    try:
        values = evaluate(config, columns)
    except FomError as exc:
        return FomResult(values=None, primitives=primitives, error=str(exc))
    except Exception as exc:  # arithmetic issues, etc.
        return FomResult(values=None, primitives=primitives, error=f"evaluation error: {exc}")

    return FomResult(values=values, primitives=primitives, error=None)


# ---------------------------------------------------------------------------
# Per-point breakdown: exposes numerator, denominator, intermediates, primitives
# ---------------------------------------------------------------------------


@dataclass
class FomBreakdown:
    """Per-point values of every quantity feeding into the FoM.

    Used by the Merit view's scatter plot to show a full breakdown on hover
    without having to re-parse the expressions for each render.
    """

    numerator: np.ndarray | None = None
    denominator: np.ndarray | None = None
    fom: np.ndarray | None = None
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    primitives: dict[str, np.ndarray] = field(default_factory=dict)
    sweep_axes: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    error: str | None = None


def _referenced_names(expr: str) -> set[str]:
    """Return the set of free names referenced in ``expr`` (best-effort)."""
    try:
        tree = ast.parse(expr or "", mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def compute_breakdown(sweep_data: dict, config: FomConfig) -> FomBreakdown:
    """Evaluate numerator, denominator, intermediates, and FoM at every sweep
    point, along with every primitive value needed to show a breakdown.

    Never raises — user-facing errors are surfaced via ``FomBreakdown.error``.
    """
    from quadris.dse.flatten import flatten_sweep_to_table as _flatten_sweep_to_table

    try:
        metric_keys, outputs, rows = _flatten_sweep_to_table(sweep_data)
    except Exception as exc:  # pragma: no cover - defensive
        return FomBreakdown(error=f"could not flatten sweep: {exc}")

    if rows.size == 0:
        return FomBreakdown(
            sweep_axes=list(metric_keys), output_keys=list(outputs),
            error="sweep is empty",
        )

    columns: dict[str, np.ndarray] = {}
    for j, k in enumerate(metric_keys):
        columns[k] = rows[:, j]
    for j, k in enumerate(outputs):
        columns[k] = rows[:, len(metric_keys) + j]

    allowed: set[str] = set(columns.keys())
    env: dict[str, Any] = dict(columns)
    size = rows.shape[0]
    intermediates: dict[str, np.ndarray] = {}

    try:
        for raw_name, raw_expr in config.intermediates:
            name = (raw_name or "").strip()
            if not name:
                continue
            if not name.isidentifier():
                raise FomError(f"invalid intermediate name: {name!r}")
            if name in allowed:
                raise FomError(f"intermediate {name!r} shadows a primitive")
            code = _compile_expr(raw_expr, allowed)
            arr = np.asarray(_eval_compiled(code, env), dtype=float)
            if arr.ndim == 0:
                arr = np.full(size, float(arr))
            elif arr.shape != (size,):
                raise FomError(f"intermediate {name!r} produced wrong shape {arr.shape}")
            env[name] = arr
            allowed.add(name)
            intermediates[name] = arr

        num_code = _compile_expr(config.numerator or "1", allowed)
        den_code = _compile_expr(config.denominator or "1", allowed)
        num = np.asarray(_eval_compiled(num_code, env), dtype=float)
        den = np.asarray(_eval_compiled(den_code, env), dtype=float)
    except FomError as exc:
        return FomBreakdown(
            intermediates=intermediates,
            primitives=columns,
            sweep_axes=list(metric_keys),
            output_keys=list(outputs),
            error=str(exc),
        )
    except Exception as exc:
        return FomBreakdown(
            intermediates=intermediates,
            primitives=columns,
            sweep_axes=list(metric_keys),
            output_keys=list(outputs),
            error=f"evaluation error: {exc}",
        )

    if num.ndim == 0:
        num = np.full(size, float(num))
    if den.ndim == 0:
        den = np.full(size, float(den))

    with np.errstate(divide="ignore", invalid="ignore"):
        fom = num / den
    fom = np.where(np.isfinite(fom), fom, np.nan)

    return FomBreakdown(
        numerator=num,
        denominator=den,
        fom=fom,
        intermediates=intermediates,
        primitives=columns,
        sweep_axes=list(metric_keys),
        output_keys=list(outputs),
        error=None,
    )


def _trim_label(name: str) -> str:
    """Shorten an internal output/metric key for display in hover text."""
    # Map Rust-side snake_case keys to concise human labels without
    # importing the heavy METRIC_BY_KEY registry here.
    return name.replace("_", " ")
