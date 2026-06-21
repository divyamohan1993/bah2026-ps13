"""netra.api — FastAPI surface for the operator console (Workstream 6).

Exposes the analytics, copilot and incident read model over HTTP/JSON + SSE for
the operator UI (``ui/``) and Grafana. Serialises the existing
``netra.contracts`` types; defines no new contracts. Returns identical shapes
whether the copilot used the LLM or the template fallback.

Runs **standalone** on the bundled :class:`~netra.api.providers.DemoProvider`
(seeded, contract-conformant data — no analytics/copilot/sim engine required),
and is wired to the real engines by swapping in
:class:`~netra.api.providers.LiveProvider` (env ``NETRA_API_PROVIDER=live``).

Launch::

    uvicorn netra.api.app:app --host 127.0.0.1 --port 8000

The public objects are imported lazily-friendly: ``app`` and ``create_app`` come
from :mod:`netra.api.app`, the providers from :mod:`netra.api.providers`.
"""

from __future__ import annotations

__all__ = ["create_app", "app", "SituationProvider", "DemoProvider", "LiveProvider"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Lazy so ``import netra.api`` doesn't require fastapi unless the app/providers
    # are actually used (keeps the package import-light, like the contracts).
    if name in {"create_app", "app"}:
        from netra.api.app import app, create_app

        return {"create_app": create_app, "app": app}[name]
    if name in {"SituationProvider", "DemoProvider", "LiveProvider"}:
        from netra.api import providers

        return getattr(providers, name)
    raise AttributeError(f"module 'netra.api' has no attribute {name!r}")
