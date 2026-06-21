# `ui/` — Operator console frontend (Workstream 6)

A lightweight, fully-offline single-page operator console served from localhost.
The FastAPI backend it talks to lives in [`../netra/api/`](../netra/api).

**What goes here:**
- `index.html` + `app.js` + `styles.css` — the console: a **Cytoscape.js**
  topology graph (nodes colored/sized by risk; root-cause node highlighted and
  blast-radius subgraph shaded), a **risk timeline** (risk trending up *before*
  impact — the visual proof of lead time), the **3-answer incident card**
  (Q1 what/when · Q2 why · Q3 what-to-do), and a **chat box** that streams from
  the local model via the API.
- `vendor/` — ALL JS/CSS/fonts vendored locally (Cytoscape.js, charting lib).
  **No CDNs, no Google Fonts** — this is an air-gap requirement.

**Contracts:** renders `Incident`, `FusedRisk`, `TimeToImpact`,
`ContributingSignal`, `CopilotResponse` (served as JSON by the API). See
[`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS6.

**Note:** the card renders identically whether the copilot used the LLM or the
template fallback (same `CopilotResponse` schema).
