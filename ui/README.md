# `ui/` — Operator console frontend (Workstream 6)

A lightweight, **fully-offline** single-page operator console. Plain vanilla JS —
no npm build, no framework, no CDN. Served by the FastAPI backend in
[`../netra/api/`](../netra/api) and reachable at `http://127.0.0.1:8000/`.

## What's here

| File | Role |
|---|---|
| `index.html` | The page: topology panel, risk-timeline panel, 3-answer card, incident queue, copilot chat. References only local assets. |
| `app.js` | All logic (no build step): fetches the API, renders the Cytoscape graph, hand-rolls the `<canvas>` risk chart, fills the 3-answer card + queue, drives the chat, and consumes the SSE live stream. |
| `style.css` | Dark NOC theme. **System fonts only** — no web-font download. |
| `vendor/cytoscape.min.js` | Vendored [Cytoscape.js](https://js.cytoscape.org/) 3.30.2 (MIT). The **only** third-party runtime asset. See [`vendor/README.md`](vendor/README.md) for version/SHA. |

The **risk timeline chart is hand-rolled** in `app.js` on an HTML5 canvas (no
charting library) to keep the vendored footprint to a single file.

## What it shows (maps to the 3 answers)

- **Topology & blast radius** (Cytoscape): nodes sized/coloured by risk, the
  **root-cause node** ringed red and the **blast-radius subgraph** shaded amber;
  click a node to focus its incident. Live SSE updates recolour nodes in place.
- **Risk timeline** (canvas): the fused risk curve with a conformal band, the
  dashed **action threshold**, and a **breach marker** — the visual proof that
  *risk rises before impact* (predictive lead time). **(Q1 — when)**
- **3-answer incident card:** **Q1** predicted issue + time-to-impact +
  calibrated-confidence bar + affected scope; **Q2** root-cause hypothesis +
  SHAP-weighted contributing signals; **Q3** ordered playbook actions with
  approval / urgency / runbook-citation badges.
- **Incident queue:** the prioritised triage list (P1 first) with per-incident
  risk and ETA; selecting one syncs the card, the graph focus and the timeline.
- **Copilot chat:** POSTs the operator's question to `/api/copilot/chat` and
  renders the grounded `CopilotResponse` (Q1/Q2/Q3 + citations + model/grounding
  footer). Works identically whether the backend used the LLM or the template
  fallback.

## Contracts rendered

`Incident`, `FusedRisk`, `TimeToImpact`, `ContributingSignal`, `BlastRadius`,
`Playbook`, `CopilotResponse` — all delivered as JSON by the API. The card
renders identically whether the copilot used the LLM or the deterministic
template fallback (same `CopilotResponse` schema).

## Air-gap (hard requirement)

Every asset is local. There are **no CDNs, no Google Fonts, no remote imports**.
The test `tests/test_api.py::test_ui_authored_assets_have_no_external_refs`
(and siblings) greps the authored html/js/css and fails on any external
`http(s)://` reference. Re-vendoring instructions and the integrity hash are in
[`vendor/README.md`](vendor/README.md).

## Run

```bash
pip install -r ../netra/api/requirements-api.txt
uvicorn netra.api.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS6.
