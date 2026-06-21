"""netra.copilot.llm — structured-output LLM backends + auto-selection.

Two backends, one interface (:class:`LLMClient`), both returning the same
:class:`~netra.contracts.CopilotResponse`:

  * :class:`LlamaCppClient` — grammar-constrained local ``llama-server``
    (Qwen2.5-7B, localhost-only, verifiable no-egress). The "lit-up" path.
  * :class:`TemplateClient` — deterministic, model-free fallback that fills the
    same schema from the structured analytics/RAG inputs. The hard requirement
    that keeps the copilot fully working offline with zero model.

:func:`select_llm_client` performs auto-selection: use the llama.cpp client iff
it is configured and reachable on loopback, else the template fallback. This is
the single decision that realises graceful degradation for the whole copilot.
"""

from __future__ import annotations

import os

from .base import CopilotGrounding, CopilotPrompt, LLMClient
from .grammar import copilot_gbnf, copilot_json_schema
from .llama_cpp_client import LlamaCppClient
from .template_client import TemplateClient


def select_llm_client(
    *,
    prefer_llm: bool | None = None,
    base_url: str | None = None,
    probe: bool = True,
) -> LLMClient:
    """Return the best available backend (llama.cpp if reachable, else template).

    Parameters
    ----------
    prefer_llm:
        Tri-state. ``True`` forces an attempt to use llama.cpp (still verifies
        reachability when ``probe`` is set); ``False`` forces the template
        fallback (useful for tests/CI and the CPU-only default); ``None``
        (default) auto-detects from the ``NETRA_LLAMA_URL`` env var — if it is
        set we try the LLM, otherwise we go straight to the template.
    base_url:
        Override the llama-server URL (must be loopback). Defaults to
        ``$NETRA_LLAMA_URL`` / ``http://127.0.0.1:8080``.
    probe:
        When True, actually health-probe the server before selecting it; when
        False, select the LLM client without a network probe (the orchestrator
        will still fall back at call time if it is down).

    Notes
    -----
    Never raises on an unreachable/misconfigured server — it degrades to the
    deterministic template client so the pipeline always runs.
    """
    if prefer_llm is False:
        return TemplateClient()

    want_llm = prefer_llm is True or (
        prefer_llm is None and bool(os.environ.get("NETRA_LLAMA_URL"))
    )
    if not want_llm:
        return TemplateClient()

    try:
        client = LlamaCppClient(base_url=base_url)
    except Exception:
        # e.g. a non-loopback URL was configured — refuse egress, degrade safely.
        return TemplateClient()

    if not probe or client.available():
        return client
    return TemplateClient()


__all__ = [
    "LLMClient",
    "CopilotPrompt",
    "CopilotGrounding",
    "LlamaCppClient",
    "TemplateClient",
    "select_llm_client",
    "copilot_gbnf",
    "copilot_json_schema",
]
