"""API route modules for the NETRA operator console.

Each module exposes an ``APIRouter`` named ``router`` mounted under ``/api`` by
``netra.api.app``:

  * ``health``    — GET /api/health
  * ``analytics`` — GET /api/situation, /api/incidents, /api/risk/timeline,
                    /api/topology
  * ``copilot``   — POST /api/copilot/query
  * ``stream``    — GET /api/stream/risk (Server-Sent Events live updates)
"""

from __future__ import annotations

__all__: list[str] = []
