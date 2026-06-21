"""netra.copilot — offline LLM + RAG + grounding orchestration (Workstream 5).

Produces the required ``netra.contracts.CopilotResponse`` either from the
GBNF-constrained local LLM (Qwen2.5-7B on llama.cpp ``llama-server``) OR — when
no model is present — from a deterministic template fallback that fills the same
schema (graceful degradation). Grounded by hybrid RAG (bge-m3 + Qdrant +
bge-reranker-v2-m3 + contextual prefixes) over internal artifacts only, plus
GraphRAG-lite over the topology, with an offline HHEM/NLI faithfulness gate,
mandatory citations and an abstain flag.

Builder: ``llm.py`` (llama-server client + template fallback — both return
CopilotResponse), ``grammar.py`` (GBNF from the CopilotResponse schema),
``rag.py`` (hybrid retrieval + in-process fallback), ``grounding.py`` (HHEM/NLI
+ citation check + abstain), ``orchestrate.py`` (prompt assembly). All heavy
deps import-guarded; the template fallback is a hard requirement.
"""
