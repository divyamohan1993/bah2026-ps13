"""NETRA operator API — FastAPI app (Workstream 6).

Surfaces the analytics / copilot / incident read model over HTTP/JSON + SSE for
the offline operator console (``ui/``) and serves that console's static assets.
Everything is **local-only**: CORS is restricted to localhost origins and the
server is intended to bind ``127.0.0.1`` (the air-gap posture — no external
origin, no CDN).

Launch (standalone, DemoProvider, no other module needed)::

    uvicorn netra.api.app:app --host 127.0.0.1 --port 8000
    # then open http://127.0.0.1:8000/  (serves ui/index.html)

Switch to the wired engines by setting ``NETRA_API_PROVIDER=live`` (the
integrator fills in ``netra.api.providers.LiveProvider``).

Routes (all under ``/api``):
  GET  /api/health
  GET  /api/situation            combined Q1/Q2/Q3 snapshot
  GET  /api/incidents            prioritised Incident[] (P1 first)
  GET  /api/incidents/{id}       one Incident
  GET  /api/risk/timeline        risk-over-time (+ conformal band + breach marker)
  GET  /api/topology             nodes/edges + per-node risk (Cytoscape elements)
  POST /api/copilot/query        CopilotRequest -> CopilotResponse
  POST /api/copilot/chat         {operator_query} -> CopilotResponse (UI helper)
  GET  /api/stream/risk          Server-Sent Events live risk updates
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from netra.api.routes import analytics as analytics_routes
from netra.api.routes import copilot as copilot_routes
from netra.api.routes import health as health_routes
from netra.api.routes import stream as stream_routes

# Repo-relative path to the vendored, offline UI (``<repo>/ui``).
_UI_DIR = Path(__file__).resolve().parents[2] / "ui"

# Localhost-only CORS: the console is same-origin, but allowing 127.0.0.1 /
# localhost on common dev ports keeps things working if the UI is served
# separately during development. No wildcard, no external origins.
_LOCAL_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (factory for tests/integrator)."""
    app = FastAPI(
        title="NETRA Operator API",
        version="1.0.0",
        description=(
            "Air-gapped predictive NOC copilot — operator console backend. "
            "Serves the 3-answer snapshot (Q1/Q2/Q3), the topology + blast radius, "
            "the live risk timeline, and the copilot chat. Fully offline."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_LOCAL_ORIGINS,
        allow_origin_regex=r"^http://(127\.0\.0\.1|localhost)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # --- API routers (all under /api) ---
    app.include_router(health_routes.router, prefix="/api")
    app.include_router(analytics_routes.router, prefix="/api")
    app.include_router(copilot_routes.router, prefix="/api")
    app.include_router(stream_routes.router, prefix="/api")

    # --- Serve the offline UI (if present) ---
    _mount_ui(app)

    return app


def _mount_ui(app: FastAPI) -> None:
    """Serve ``ui/`` at ``/`` (index) and ``/ui`` (static assets), if it exists."""
    if not _UI_DIR.is_dir():
        @app.get("/", include_in_schema=False)
        def _no_ui() -> JSONResponse:
            return JSONResponse(
                {
                    "service": "netra-api",
                    "note": "UI directory not found; API is up. See /docs.",
                    "ui_expected_at": str(_UI_DIR),
                }
            )

        return

    index = _UI_DIR / "index.html"

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(index)

    # Static mount for app.js / style.css / vendor/* (HTML disabled so unknown
    # paths 404 rather than silently returning index.html).
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=False), name="ui")

    # Also expose the most-used assets at the root so index.html can reference
    # them with relative paths (./app.js, ./style.css, ./vendor/...).
    @app.get("/app.js", include_in_schema=False)
    def _appjs() -> FileResponse:
        return FileResponse(_UI_DIR / "app.js", media_type="application/javascript")

    @app.get("/style.css", include_in_schema=False)
    def _stylecss() -> FileResponse:
        return FileResponse(_UI_DIR / "style.css", media_type="text/css")

    @app.get("/vendor/{path:path}", include_in_schema=False)
    def _vendor(path: str) -> FileResponse:
        # Resolve safely within the vendor dir (no path traversal).
        base = (_UI_DIR / "vendor").resolve()
        target = (base / path).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="vendor asset not found")
        media = (
            "application/javascript"
            if target.suffix == ".js"
            else "text/css"
            if target.suffix == ".css"
            else None
        )
        return FileResponse(target, media_type=media)


# Module-level app for ``uvicorn netra.api.app:app``.
app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual launch convenience
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("NETRA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("NETRA_API_PORT", "8000")),
    )
