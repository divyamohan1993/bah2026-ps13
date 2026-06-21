# `corpus/` — Internal NOC artifacts for RAG (Workstream 5)

The **internal-only** knowledge base the copilot grounds on. RAG retrieves from
here and nowhere else — no external sources, ever. Every chunk is keyed with an
id so the copilot can cite it (`CopilotResponse.citations`).

**What goes here:**
- `runbooks/*.md` — operator runbooks (structure-aware: header sections, intact
  tables/code/CLI), one per fault class.
- `playbooks/*.json` — CACAO-style machine-readable playbooks (ordered steps,
  each with command template, safety class, verification, rollback) — map 1:1 to
  `netra.contracts.Playbook` / `RecommendedAction`.
- `incidents/*.json` — past-incident records (symptom / RCA / resolution /
  timeline) for incident-similarity retrieval.
- `topology/*.json` — topology metadata (sites, devices, roles, VRFs, links) —
  also loaded into the NetworkX digital twin for GraphRAG-lite.

**Contracts:** consumed by [`../netra/copilot/`](../netra/copilot); playbooks
serialise to `Playbook`. See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS5.

**Note:** keep entries small and realistic (≥1 per validation scenario) so the
offline grounding metrics (RAGAS/HHEM) and citation enforcement are demonstrable.
