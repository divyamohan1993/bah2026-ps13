"""Copilot route: POST a ``CopilotRequest`` -> get a ``CopilotResponse``.

The request/response models ARE the contracts, so FastAPI validates the body
against ``CopilotRequest`` and serialises the answer as ``CopilotResponse`` — the
same schema whether the answer came from the LLM (LiveProvider) or the
deterministic template fallback (DemoProvider). The UI's chat box POSTs here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from netra.api.deps import get_provider
from netra.api.providers import SituationProvider
from netra.contracts import CopilotRequest, CopilotResponse

router = APIRouter(tags=["copilot"])


@router.post("/copilot/query", response_model=CopilotResponse)
def copilot_query(
    request: CopilotRequest,
    provider: SituationProvider = Depends(get_provider),
) -> CopilotResponse:
    """Answer an operator query or auto-trigger as a grounded ``CopilotResponse``."""
    return provider.copilot(request)


class _ChatBody(CopilotRequest):
    """Convenience body so the UI can POST just a question.

    Inherits every ``CopilotRequest`` field but makes ``request_id`` /
    ``created_at`` optional with sensible defaults, so the minimal chat payload is
    ``{"operator_query": "..."}``. Still a strict contract subtype.
    """

    # Override the two required fields with defaults so a bare
    # ``{"operator_query": "..."}`` body validates; the route fills real values.
    request_id: str = ""  # type: ignore[assignment]
    created_at: datetime | None = None  # type: ignore[assignment]


@router.post("/copilot/chat", response_model=CopilotResponse)
def copilot_chat(
    body: _ChatBody,
    provider: SituationProvider = Depends(get_provider),
) -> CopilotResponse:
    """Lightweight chat entrypoint for the UI (``{operator_query}`` is enough).

    Normalises the minimal payload into a full :class:`CopilotRequest` (filling a
    ``request_id`` and ``created_at`` if the caller omitted them) and delegates to
    the same provider path as :func:`copilot_query`.
    """
    import uuid

    req = CopilotRequest(
        request_id=body.request_id or f"chat-{uuid.uuid4().hex[:12]}",
        created_at=body.created_at or datetime.now(timezone.utc),
        operator_query=body.operator_query,
        auto_trigger=body.auto_trigger,
        incident_ref=body.incident_ref,
        entity_refs=body.entity_refs,
        fused_risk_refs=body.fused_risk_refs,
        max_context_chunks=body.max_context_chunks,
    )
    return provider.copilot(req)


__all__ = ["router"]
