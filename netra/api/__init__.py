"""netra.api — FastAPI surface (Workstream 6).

Exposes the analytics, copilot and incident state over HTTP/JSON + WebSocket for
the operator UI (``ui/``) and Grafana. Serialises the existing
``netra.contracts`` types; defines no new contracts. Returns identical shapes
whether the copilot used the LLM or the template fallback.

Builder: ``app.py`` (FastAPI app + routes: /incidents, /risk,
/forecast/{entity}, POST /copilot -> CopilotResponse, /ws stream),
``routes_copilot.py``, ``routes_analytics.py``. Core tier deps only
(fastapi/uvicorn/httpx). Bind to 127.0.0.1.
"""
