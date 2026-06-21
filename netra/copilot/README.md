# `netra/copilot/` — Offline LLM + RAG + Grounding Copilot (Workstream 5)

The NETRA copilot turns the analytics engine's predictions and the internal NOC
corpus into a **grounded, schema-valid** answer to the three operator questions —
**Q1** (what fails next & when), **Q2** (why / which signals), **Q3** (what to
do) — emitted as a [`netra.contracts.CopilotResponse`](../contracts/copilot.py).

This subtree drives **Copilot Effectiveness (35%)** — correct, operator-relevant,
**grounded with no hallucination** — and supports **Offline Compliance (20%)**
(verifiable zero egress). Its defining property is **graceful degradation**: the
copilot runs, demos and passes its tests with **no heavy model present** (a
deterministic template fallback fills the same schema), and **lights up fully**
when a local model / embeddings / vector DB exist.

```
netra/copilot/
├── llm/                      # structured-output LLM backends (one interface)
│   ├── base.py               #   LLMClient ABC + CopilotPrompt / CopilotGrounding
│   ├── llama_cpp_client.py   #   local llama-server (GBNF, localhost-only, no egress)
│   ├── template_client.py    #   deterministic model-free fallback (same schema)
│   ├── grammar.py            #   GBNF + JSON schema derived from CopilotResponse
│   └── grammar.gbnf          #   the bundled, generated grammar artifact
├── rag/                      # offline hybrid RAG over internal artifacts
│   ├── embed.py              #   bge-m3 (optional) | TF-IDF/hashing fallback
│   ├── store.py              #   Qdrant (opt) | FAISS (opt) | numpy cosine fallback
│   ├── retrieve.py           #   dense + BM25 hybrid, Reciprocal Rank Fusion
│   ├── rerank.py             #   bge-reranker cross-encoder (opt) | identity fallback
│   ├── ingest.py             #   chunk + index corpus/ (md runbooks, JSON records)
│   └── graphrag.py           #   GraphRAG-lite: deterministic topology + blast-radius
├── grounding/                # post-generation anti-hallucination gate
│   ├── faithfulness.py       #   HHEM-2.1 (opt) | lexical-overlap NLI fallback
│   └── citations.py          #   closed-set citation check + abstain logic
├── orchestrator.py           # Copilot.answer(CopilotRequest, *, analytics_context)
├── requirements-copilot.txt  # OPTIONAL-heavy deps (all import-guarded)
└── README.md                 # this file
```

The grounding corpus lives in [`corpus/`](../../corpus) (this workstream also
owns it): markdown runbooks, CACAO-style playbooks, past-incident records, and
topology notes — one set per validation scenario.

---

## Quickstart (CPU-only, no model, fully offline)

```python
from datetime import datetime, timezone
from netra.copilot import Copilot, AnalyticsContext
from netra.contracts import CopilotRequest, FusedRisk, TimeToImpact, IssueType  # + others

cop = Copilot(prefer_models=False)          # template fallback + TF-IDF retriever
req = CopilotRequest(
    request_id="req-1",
    created_at=datetime.now(timezone.utc),
    operator_query="Why is the Mumbai hub uplink at risk and what do I do?",
    max_context_chunks=5,
)
resp = cop.answer(req, analytics_context=AnalyticsContext(fused_risk=..., ...))
# resp is a schema-valid CopilotResponse: predicted_issue, confidence_score,
# time_to_impact_minutes, root_cause_hypothesis, contributing_signals,
# affected_scope, recommended_actions, citations, insufficient_context.
```

`Copilot(prefer_models=True)` (and/or `NETRA_LLAMA_URL=http://127.0.0.1:8080`)
opts into the heavy path: a local `llama-server`, bge-m3 embeddings, the
cross-encoder reranker and the HHEM faithfulness gate — **only if** they are
installed and reachable; otherwise each silently uses its light fallback.

---

## How it answers Q1 / Q2 / Q3 (and stays grounded)

| Question | Source | Response fields |
|---|---|---|
| **Q1 — what & when** | Analytics: `FusedRisk.predicted_issue`, `TimeToImpact.eta_seconds` → minutes; `BlastRadius` → scope (deterministic, **reported not guessed**) | `predicted_issue`, `time_to_impact_minutes`, `affected_scope` |
| **Q2 — why / signals** | Analytics: `ContributingSignal[]` (SHAP) + the retrieved runbook/incident prose | `root_cause_hypothesis`, `contributing_signals` |
| **Q3 — action** | The matched `Playbook` (from the corpus), ordered approval-gated steps with rollback | `recommended_actions` (with `runbook_ref`) |

**Grounding is enforced, not hoped for:**

1. **Structured output guaranteed.** The LLM is constrained by a **GBNF grammar
   generated from the `CopilotResponse` contract** (`grammar.py` →
   `grammar.gbnf`) so it cannot emit malformed JSON or an out-of-vocabulary
   `predicted_issue`. The template fallback produces the same schema by
   construction.
2. **Confidence is sourced from analytics, never invented.** `confidence_score`
   = `FusedRisk.calibrated_confidence`; `time_to_impact_minutes` = the survival/
   trajectory ETA. The LLM client *overwrites* any model-emitted values with
   these; the model only *explains* them.
3. **Closed-set citations.** Every id in `citations` (and every action
   `runbook_ref`) must be a member of the retrieved-context universe
   (`grounding/citations.py`); ids not present are dropped.
4. **Abstain when thin.** With no retrieved knowledge-base chunks the copilot
   sets `insufficient_context=True`, forces a low confidence, and recommends
   "gather more data / escalate" — still schema-valid.
5. **Faithfulness gate.** A post-generation HHEM-2.1 / lexical-NLI score
   (`grounding/faithfulness.py`) is written to `grounding_score`.

---

## Integrator notes — wiring the API to the copilot

The API layer (Workstream 6) calls the copilot like this:

```python
from netra.copilot import Copilot, AnalyticsContext

copilot = Copilot()                          # build once at startup (holds the
                                             # retriever, topology graph, backend)

# Per request (POST /copilot with a CopilotRequest):
analytics_context = AnalyticsContext(
    incident=incident,                       # the richest single input, OR pass
    # fused_risk=..., time_to_impact=...,    #   looser pieces directly; explicit
    # contributing_signals=[...],            #   fields win over incident-derived
    # blast_radius=..., playbook=...,        #   ones.
    # root_cause_entity=...,
)
response = copilot.answer(request, analytics_context=analytics_context)
# return response  (CopilotResponse) — identical shape whether the LLM or the
# template fallback produced it.
```

**`AnalyticsContext` shape** (all fields optional; all are
`netra.contracts` types — the copilot **consumes contract outputs, never imports
other builders**):

| Field | Type | Provides |
|---|---|---|
| `incident` | `Incident` | The full correlated incident (preferred; everything below is derived from it if not given explicitly). |
| `fused_risk` | `FusedRisk` | Authoritative `predicted_issue` + `calibrated_confidence`. |
| `time_to_impact` | `TimeToImpact` | Q1 "when" (`eta_seconds` → `time_to_impact_minutes`). |
| `contributing_signals` | `list[ContributingSignal]` | Q2 SHAP signals. |
| `blast_radius` | `BlastRadius` | Q1 deterministic affected scope. |
| `playbook` | `Playbook` | Q3 remediation steps. |
| `root_cause_entity` | `EntityRef` | GraphRAG-lite root-cause device description. |

Field precedence: an explicit value wins; otherwise it is derived from
`incident` (e.g. `blast_radius` falls back to `incident.blast_radius`). Pass an
empty `AnalyticsContext` (or none) and the copilot abstains gracefully.

The API returns the `CopilotResponse` **unchanged** whether the copilot used the
LLM or the fallback (`used_fallback` / `model_id` tell which) — the UI and tests
are identical either way.

---

## Air-gap / no-egress

- The llama client (`llama_cpp_client.py`) **refuses any non-loopback base URL**
  at construction (`127.0.0.1` / `localhost` / `::1` only) — it can never be
  pointed at a remote API.
- No other component makes network calls. `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` are set defensively before any HF backend could load,
  so even the optional heavy models load from local disk only.
- The corpus is **internal-only**; RAG retrieves from `corpus/` and nowhere else.

---

## Dependencies & testing

- **Light tier (required):** `pydantic`, `numpy`, `scikit-learn` (all in
  [`requirements-core.txt`](../../requirements-core.txt)); `networkx` (core)
  powers GraphRAG-lite. The whole pipeline — template fallback, TF-IDF + numpy
  cosine + pure-Python BM25 hybrid retrieval, lexical faithfulness — runs on this
  tier alone.
- **Optional-heavy (upgrade only):** see
  [`requirements-copilot.txt`](requirements-copilot.txt) — `llama-cpp-python`,
  `sentence-transformers`/`FlagEmbedding` (bge-m3, bge-reranker-v2-m3),
  `qdrant-client`, `faiss-cpu`, `bm25s`/`rank-bm25`, `transformers`/`torch`
  (HHEM). All import-guarded; none required to run or test.
- **Tests:** [`tests/test_copilot.py`](../../tests/test_copilot.py) — 34 CPU-only
  tests (schema validity, citations present, abstain on empty corpus, Q1/Q2/Q3
  population, the four scenarios, the no-egress guard, grammar generation, the
  grounding gate). Run: `pytest -q tests/test_copilot.py`.
