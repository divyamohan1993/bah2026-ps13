"""``LLMClient`` — the abstract interface every copilot backend satisfies.

Both the grammar-constrained local-LLM client (:mod:`.llama_cpp_client`) and the
deterministic, model-free template client (:mod:`.template_client`) implement
this same interface and return the **same** :class:`~netra.contracts.CopilotResponse`
schema. That is the contract that makes graceful degradation transparent to the
orchestrator, the API and the UI: whether or not a 7B model is loaded, the call
site is identical and the output is schema-valid (architecture §5.2).

The interface is intentionally tiny and structured-output oriented:

  * :meth:`LLMClient.available` — cheap, side-effect-free reachability/health
    probe used for auto-selection (no network egress beyond localhost).
  * :meth:`LLMClient.complete_copilot` — produce a validated ``CopilotResponse``
    from a fully-assembled :class:`CopilotPrompt` (system + user text already
    composed by the orchestrator) plus the structured ``grounding`` inputs the
    template fallback needs to compose an answer without any model.

The ``grounding`` payload (:class:`CopilotGrounding`) carries the analytics +
RAG context in a backend-agnostic shape so the template client can deterministically
fill every Q1/Q2/Q3 field, while the LLM client mostly ignores it (it reads the
already-rendered prompt) but still uses it for the closed-set citation universe.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from netra.contracts import (
    AffectedScope,
    CopilotAction,
    CopilotSignal,
    IssueType,
)


@dataclass
class CopilotPrompt:
    """A fully-rendered prompt ready to send to a chat-completions endpoint.

    The orchestrator composes the grounded ``system`` and ``user`` strings
    (analytics + SHAP + retrieved chunks + graph facts + the operator question)
    so the LLM client stays dumb about *how* grounding was assembled and only has
    to enforce the structured-output contract.
    """

    system: str
    user: str
    #: Soft cap on generated tokens (the schema is compact; keep it small).
    max_tokens: int = 768
    #: Low temperature for deterministic, conservative answers (research 05 §5.1).
    temperature: float = 0.2


@dataclass
class CopilotGrounding:
    """Backend-agnostic structured inputs used to compose a CopilotResponse.

    This is the *deterministic* substrate the template client turns into a valid
    answer with **no model at all**, and the value source the LLM client uses for
    the fields that must come from the analytics engine (not be invented):
    ``confidence_score``, ``time_to_impact_minutes`` and the citation universe.

    Every field is optional/defaulted so a caller can produce a (correctly
    abstaining) response even with an empty corpus and no analytics.
    """

    request_id: str
    predicted_issue: IssueType = IssueType.NONE
    #: Calibrated confidence sourced from FusedRisk/TimeToImpact (never the LLM).
    confidence_score: float = 0.0
    time_to_impact_minutes: float | None = None
    root_cause_hypothesis: str = ""
    contributing_signals: list[CopilotSignal] = field(default_factory=list)
    affected_scope: AffectedScope = field(default_factory=AffectedScope)
    recommended_actions: list[CopilotAction] = field(default_factory=list)
    #: Closed set of citation ids that are actually present in the context. Any
    #: citation an answer emits MUST be a member of this set (grounding gate).
    citation_universe: list[str] = field(default_factory=list)
    #: True when retrieval/analytics evidence is too thin to answer (abstain).
    insufficient_context: bool = False
    #: Operator-facing question, if any (used to tailor the abstain message).
    operator_query: str | None = None


class LLMClient(abc.ABC):
    """Abstract base for any backend that emits a structured CopilotResponse."""

    #: Stable identifier surfaced in ``CopilotResponse.model_id``.
    model_id: str = "abstract-llm"
    #: True if this backend runs with no heavy model (sets ``used_fallback``).
    is_fallback: bool = False

    @abc.abstractmethod
    def available(self) -> bool:
        """Return True if this backend can serve a request right now.

        Must be cheap and must not perform any non-localhost network I/O. Used by
        the auto-selector to decide between the LLM client and the template
        fallback.
        """

    @abc.abstractmethod
    def complete_copilot(
        self, prompt: CopilotPrompt, grounding: CopilotGrounding
    ) -> CopilotResponse:  # noqa: F821  (forward ref to avoid import cost here)
        """Produce a validated :class:`CopilotResponse` for ``prompt``.

        Implementations MUST return a schema-valid response (the contract enforces
        ``>=1`` action and ``>=1`` citation); the template client guarantees this
        deterministically and the LLM client enforces it via grammar + Pydantic
        validation with one constrained retry, then falls back if still invalid.
        """


__all__ = ["LLMClient", "CopilotPrompt", "CopilotGrounding"]
