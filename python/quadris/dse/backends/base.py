"""
Backend protocol shared by HQA+SABRE and TeleSABRE.

A backend is the only thing that differs between routing algorithms in
the cold path — circuit building, topology construction, noise merging,
and result wrapping all live in shared leaf modules. Adding a new
backend is one new file in this directory plus an entry in
``_BACKENDS``.
"""

from __future__ import annotations

from typing import Protocol

from ..results import CachedMapping


class Backend(Protocol):
    """Routing-algorithm backend.

    Implementations must be picklable (forkserver workers spawn a new
    interpreter and re-instantiate the backend), so avoid bound methods
    of large objects or unpickleable state.
    """

    name: str

    def compile(
        self,
        cold_cfg: dict,
        noise: dict,
        config_key: tuple,
    ) -> CachedMapping:
        """Run the cold path: build circuit + topology, route, return the
        cached mapping. ``cold_cfg`` is already alias-expanded and
        clamped; ``noise`` is already merged via ``_merge_noise``.

        ``config_key`` is the engine's cache key for this configuration;
        the returned ``CachedMapping`` must carry it so :class:`DSEEngine`
        can detect cache hits on subsequent ``run_cold`` calls.
        """
        ...
