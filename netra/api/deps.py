"""Dependency-injection wiring for the operator API.

A single process-wide :class:`~netra.api.providers.SituationProvider` is created
lazily from configuration (env var ``NETRA_API_PROVIDER`` -> ``demo`` | ``live``)
and handed to every route via FastAPI's ``Depends``. Swapping Demo for Live is a
one-line/one-env-var change; no route code changes.

The integrator can also call :func:`set_provider` at startup to inject a
pre-constructed/already-wired provider (e.g. a ``LiveProvider`` with engine
handles attached).
"""

from __future__ import annotations

from threading import Lock

from netra.api.providers import SituationProvider, make_provider

_provider: SituationProvider | None = None
_lock = Lock()


def set_provider(provider: SituationProvider) -> None:
    """Install a specific provider instance (call before serving requests)."""
    global _provider
    with _lock:
        _provider = provider


def get_provider() -> SituationProvider:
    """FastAPI dependency: return the process-wide provider (build on first use)."""
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                _provider = make_provider()
    return _provider


def reset_provider() -> None:
    """Drop the cached provider (used by tests to rebuild with a fresh seed)."""
    global _provider
    with _lock:
        _provider = None


__all__ = ["get_provider", "set_provider", "reset_provider"]
