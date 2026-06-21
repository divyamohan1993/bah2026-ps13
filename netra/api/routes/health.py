"""Health / readiness route."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from netra.api.deps import get_provider
from netra.api.providers import DemoProvider, SituationProvider

router = APIRouter(tags=["health"])


@router.get("/health")
def health(provider: SituationProvider = Depends(get_provider)) -> dict:
    """Liveness probe + which provider is active (demo vs live)."""
    return {
        "status": "ok",
        "service": "netra-api",
        "provider": "demo" if isinstance(provider, DemoProvider) else "live",
        "time": datetime.now(UTC).isoformat(),
    }


__all__ = ["router"]
