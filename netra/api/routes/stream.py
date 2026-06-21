"""Live risk stream via Server-Sent Events (SSE).

Implemented with a plain Starlette ``StreamingResponse`` emitting
``text/event-stream`` — **no extra dependency** (works with just fastapi/
starlette), which keeps the air-gap footprint minimal. Each event is a
``risk_tick`` frame from the provider (per-entity risk + the headline ETA
counting down) so the UI can recolour the graph and tick the timeline live.

A ``limit`` query param bounds the number of frames (default unbounded for the
UI; the tests pass a small limit so the response terminates deterministically).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import StreamingResponse

from netra.api.deps import get_provider
from netra.api.providers import SituationProvider

router = APIRouter(tags=["stream"])


def _sse(event: str, data: dict) -> str:
    """Format one SSE frame (``event:`` + ``data:`` lines, blank-line terminated)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/stream/risk")
async def stream_risk(
    request: Request,
    interval: float = Query(
        default=2.0, ge=0.05, le=30.0, description="Seconds between frames."
    ),
    limit: int | None = Query(
        default=None, ge=1, le=10000, description="Max frames before closing (None=open)."
    ),
    provider: SituationProvider = Depends(get_provider),
) -> StreamingResponse:
    """Stream live ``risk_tick`` events as SSE until the client disconnects."""

    async def gen() -> AsyncIterator[str]:
        # An initial comment frame lets EventSource open immediately.
        yield ": netra risk stream open\n\n"
        count = 0
        while True:
            if await request.is_disconnected():
                break
            frame = provider.risk_tick()
            yield _sse("risk", frame)
            count += 1
            if limit is not None and count >= limit:
                yield _sse("end", {"type": "end", "frames": count})
                break
            await asyncio.sleep(interval)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering if one is ever present
    }
    return StreamingResponse(
        gen(), media_type="text/event-stream", headers=headers
    )


__all__ = ["router"]
