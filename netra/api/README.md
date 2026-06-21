# `netra/api/` — Operator API (Workstream 6)

FastAPI backend for the NETRA operator console. It surfaces the analytics /
copilot / incident **read model** over HTTP/JSON + Server-Sent-Events and serves
the offline single-page UI in [`../../ui/`](../../ui). Every response is a
serialised [`netra.contracts`](../contracts) type — **the wire schema *is* the
contract** — and the API runs **standalone** (no analytics/copilot/sim engine
required) on a bundled, seeded `DemoProvider`.

## Run it (standalone, offline)

```bash
# install the (light) extra deps once
pip install -r netra/api/requirements-api.txt        # fastapi, uvicorn, httpx

# launch — binds localhost only (air-gap posture)
uvicorn netra.api.app:app --host 127.0.0.1 --port 8000

# open the console
#   http://127.0.0.1:8000/        ← serves ui/index.html
#   http://127.0.0.1:8000/docs    ← OpenAPI (schemas are the contracts)
```

No GPU, no internet, no other NETRA module needed: the `DemoProvider` fabricates
realistic, deterministic, contract-conformant state (3 correlated incidents
across scenarios A/C/B, a rising risk timeline, the ~20-node reference topology
with blast-radius shading, and template-fallback copilot answers).

## Endpoints (all under `/api`)

| Method & path | Returns | Answers |
|---|---|---|
| `GET /api/health` | `{status, provider, time}` | — |
| `GET /api/situation` | `{headline_incident: Incident, copilot: CopilotResponse, answers:{q1,q2,q3}, fleet}` | Q1+Q2+Q3 snapshot |
| `GET /api/incidents` | `Incident[]` (P1 first) | the triage queue |
| `GET /api/incidents/{id}` | `Incident` (404 if unknown) | one incident |
| `GET /api/risk/timeline?entity_id=` | `{threshold, breach_index, points:[{timestamp,risk,lower,upper}]}` | Q1 lead-time (risk rises *before* impact) |
| `GET /api/topology` | `{root_cause_devices, blast_radius_devices, elements:{nodes,edges}}` (Cytoscape) | Q1 where/how-bad |
| `POST /api/copilot/query` | body `CopilotRequest` → `CopilotResponse` | Q1/Q2/Q3 grounded |
| `POST /api/copilot/chat` | body `{operator_query}` → `CopilotResponse` | UI chat helper |
| `GET /api/stream/risk?interval=&limit=` | `text/event-stream` of `risk_tick` frames | live updates |

The UI is served at `/` (index), `/app.js`, `/style.css`, and `/vendor/*`.

### SSE stream format

`GET /api/stream/risk` emits `event: risk` frames whose `data:` is a JSON
`risk_tick` (`{tick, timestamp, headline_entity, headline_risk,
headline_eta_minutes, predicted_issue, entities:[{entity_id, device, risk}]}`).
Implemented with a plain Starlette `StreamingResponse` — **no `sse-starlette` /
websocket dependency** (smaller air-gap footprint). Pass `limit=N` to bound the
number of frames (used by the tests); omit it for an open stream the UI keeps.

## Architecture: the provider DI seam

The routes never import another builder's engine. They depend on
`providers.SituationProvider`, resolved per-request via `deps.get_provider()`:

```
routes/*  ──Depends(get_provider)──▶  SituationProvider
                                          ├─ DemoProvider  (default; self-contained, seeded)
                                          └─ LiveProvider  (wiring stub for real engines)
```

Select with the `NETRA_API_PROVIDER` env var (`demo` | `live`, default `demo`),
or inject a pre-built instance at startup with `deps.set_provider(...)`.

```python
# files
app.py            # FastAPI app, CORS (localhost-only), serves ui/, mounts routers
deps.py           # get_provider / set_provider / reset_provider (DI)
providers.py      # SituationProvider ABC + DemoProvider + LiveProvider + make_provider
routes/health.py        # GET /api/health
routes/analytics.py     # situation / incidents / risk timeline / topology
routes/copilot.py       # POST copilot query + chat
routes/stream.py        # SSE live risk stream
requirements-api.txt    # fastapi, uvicorn, httpx (extra deps beyond core tier)
```

> **Note on the topology import.** The DemoProvider reuses the shared reference
> topology in `netra/datagen/topology.py` (import-light: only contract enums +
> stdlib) so node ids / blast radius stay consistent with the rest of NETRA. It
> loads that *submodule directly via importlib*, bypassing `netra/datagen/__init__.py`
> (which eagerly imports the numpy-backed generator), so the API keeps its
> "runs on light deps" promise. No analytics engine is imported.

## Wiring `LiveProvider` to the real engines (integrator)

`providers.LiveProvider` is a stub: every method raises `NotImplementedError`
with a note on which engine call fills it. To go live, implement each method and
either set `NETRA_API_PROVIDER=live` or call `deps.set_provider(LiveProvider(engines=...))`
at startup. Expected wiring:

| Method | Wire to |
|---|---|
| `incidents()` | `netra.analytics.risk.prioritize` output — the ranked `Incident[]` (already correlated, with `BlastRadius` + `ContributingSignal[]` + `Playbook`). |
| `situation()` | the headline `Incident` + a `CopilotResponse` from `netra.copilot.orchestrate.answer(...)`. |
| `risk_timeline(entity_id)` | a PromQL/MetricsQL query over the VictoriaMetrics risk series, **or** a ring-buffer of recent `FusedRisk.risk_score` per entity. Keep the `{points:[{timestamp,risk,lower,upper}], threshold, breach_index}` shape. |
| `topology()` | project `netra.analytics.correlation.graph` (the networkx digital twin) to Cytoscape `{nodes,edges}`, copying per-node `risk`, `is_root_cause`, `in_blast_radius` from the live incidents. |
| `copilot(request)` | `netra.copilot.orchestrate.answer(request)` → `CopilotResponse` (the LLM path *or* its template fallback — same schema, so the API/UI are unchanged). |
| `risk_tick()` | the latest frame off the NATS `alerts.>` / `FusedRisk` stream (or poll the engine), shaped like the demo `risk_tick`. |

Because both the LLM and the template fallback return `CopilotResponse`, and the
DemoProvider already emits the *fallback* shape (`used_fallback=True`,
`model_id="template-fallback"`, confidence sourced from the analytics objects —
never invented), the UI renders identically in every degradation mode.

`POST /api/copilot/query` also accepts a `llama-server` (OpenAI-compatible) wiring
inside `LiveProvider.copilot` — `httpx` is already a dependency for that call.

## Tests

`../../tests/test_api.py` (TestClient, light deps) hits every endpoint, validates
payloads against the contracts, asserts the demo provider is deterministic, the
live provider is a stub, and the UI is air-gapped:

```bash
pip install fastapi httpx pydantic pytest
pytest -q tests/test_api.py
```

## Constraints honoured

- Request/response models **are** `netra.contracts` types; no new contracts.
- No other builder's engine module is imported (provider/DI seam only).
- Extra deps live here in `requirements-api.txt` (not in the top-level manifest).
- Everything is CPU-only / offline; the server binds `127.0.0.1`; CORS is
  localhost-only (no wildcard, no external origin).
