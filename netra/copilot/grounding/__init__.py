"""netra.copilot.grounding — post-generation anti-hallucination gate.

Two deterministic + one optional-model control that together make ungrounded
output hard to emit and easy to catch (the heart of the "grounded, no
hallucination" score):

  * :class:`FaithfulnessScorer` — answer-vs-context consistency, HHEM-2.1 if
    present else a lexical-overlap NLI heuristic (writes ``grounding_score``).
  * :func:`enforce_citations` — closed-set citation check: drop any id not in the
    retrieved context, abstain if nothing valid remains.
  * :func:`should_abstain` — decide insufficiency from context size / top score.

All run for both the LLM and the template backend.
"""

from __future__ import annotations

from .citations import (
    CitationCheck,
    enforce_citations,
    should_abstain,
    validate_citations,
)
from .faithfulness import FaithfulnessResult, FaithfulnessScorer

__all__ = [
    "FaithfulnessScorer",
    "FaithfulnessResult",
    "enforce_citations",
    "validate_citations",
    "should_abstain",
    "CitationCheck",
]
