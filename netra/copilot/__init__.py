"""netra.copilot — offline LLM + RAG + grounding orchestration (Workstream 5).

Produces the required :class:`netra.contracts.CopilotResponse` either from the
GBNF-constrained local LLM (Qwen2.5-7B on llama.cpp ``llama-server``) OR — when
no model is present — from a deterministic template fallback that fills the same
schema (graceful degradation). Grounded by hybrid RAG (bge-m3 + Qdrant +
bge-reranker-v2-m3 + contextual prefixes, all import-guarded) over internal
artifacts only, plus GraphRAG-lite over the topology, with an offline HHEM/NLI
faithfulness gate, mandatory citations and an abstain flag.

Subtree (this workstream owns ``netra/copilot/**`` and ``corpus/**``):
  * ``llm/``       — ``LLMClient`` ABC, llama.cpp client (GBNF), template
                     fallback, ``grammar.py`` (GBNF from the contract).
  * ``rag/``       — embed / store / hybrid retrieve+RRF / rerank / ingest /
                     graphrag-lite (each heavy dep guarded with a light fallback).
  * ``grounding/`` — faithfulness (HHEM/NLI) + citation closed-set + abstain.
  * ``orchestrator.py`` — ``Copilot.answer(CopilotRequest, *, analytics_context)``.

The public entry point is :class:`Copilot`; the analytics inputs arrive as the
contract-typed :class:`AnalyticsContext` (the copilot never imports other
builders, only their contract outputs).

Top-level imports are kept light (only the orchestrator, which pulls pydantic +
numpy + scikit-learn). Heavy backends are loaded lazily inside ``rag``/``llm``
only when ``prefer_models`` is requested and the local weights/server exist.
"""

from __future__ import annotations

from .orchestrator import AnalyticsContext, Copilot

__all__ = ["Copilot", "AnalyticsContext"]
