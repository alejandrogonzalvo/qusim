"""
Routing-algorithm backends for the DSE cold path.

Each backend takes a cold-config dict (circuit, topology, placement,
seed, ...) plus a merged-noise dict and returns a fully-populated
:class:`quadris.dse.results.CachedMapping` that the hot path consumes.

Currently registered:

* ``hqa_sabre`` — HQA initial mapping + SABRE swap insertion (the
  default; calls :func:`quadris.map_circuit`).
* ``telesabre`` — TeleSABRE C library via :mod:`quadris.rust_core`
  (writes QASM + device JSON to temp files, then re-imports the
  routed placements into DAG-layer space).
"""

from .base import Backend
from .hqa_sabre import HqaSabreBackend
from .telesabre import TeleSabreBackend


_BACKENDS: dict[str, Backend] = {
    "hqa_sabre": HqaSabreBackend(),
    "telesabre": TeleSabreBackend(),
}


def get_backend(name: str) -> Backend:
    """Look up a routing backend by name (e.g. ``"hqa_sabre"``)."""
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown routing backend {name!r}. "
            f"Known: {sorted(_BACKENDS)}"
        ) from e


__all__ = ["Backend", "HqaSabreBackend", "TeleSabreBackend", "get_backend"]
