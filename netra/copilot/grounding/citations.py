"""Citation enforcement + abstain logic — the hard grounding guarantee.

Two deterministic grounding controls applied to a :class:`CopilotResponse`
*after* generation (research 05 §5.1-§5.2, 06 §8.3):

  * **Closed-set citation check** (:func:`enforce_citations`): every id in
    ``CopilotResponse.citations`` (and every ``runbook_ref`` on an action) must
    be a member of the supplied context universe. Ids not in the universe are
    dropped — the model cannot cite something it was not given. If nothing valid
    remains, the response is forced into the abstain state.
  * **Abstain decision** (:func:`should_abstain`): decide whether evidence is too
    thin to answer, from the size of the retrieved context and (optionally) the
    top rerank/grounding score, so the copilot says "insufficient local evidence"
    instead of hallucinating.

These run for **both** backends (LLM and template), so grounding holds regardless
of which produced the answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from netra.contracts import CopilotAction, CopilotResponse, Urgency


@dataclass
class CitationCheck:
    """Result of validating a response's citations against the context universe."""

    valid_citations: list[str]
    dropped_citations: list[str]
    all_valid: bool  # True if no citation had to be dropped


def validate_citations(
    citations: Iterable[str], universe: Iterable[str]
) -> CitationCheck:
    """Partition ``citations`` into those present in ``universe`` and those not."""
    uni = set(universe)
    valid: list[str] = []
    dropped: list[str] = []
    for c in citations:
        (valid if c in uni else dropped).append(c)
    # De-duplicate while preserving order.
    valid = list(dict.fromkeys(valid))
    return CitationCheck(
        valid_citations=valid,
        dropped_citations=dropped,
        all_valid=not dropped,
    )


def should_abstain(
    *,
    n_context_chunks: int,
    top_score: float | None = None,
    min_chunks: int = 1,
    min_score: float = 0.0,
) -> bool:
    """Return True if the copilot should abstain (insufficient context).

    Abstain when there is no retrieved context at all, or (when a top
    rerank/grounding score is provided) when it falls below ``min_score``.
    """
    if n_context_chunks < min_chunks:
        return True
    if top_score is not None and top_score < min_score:
        return True
    return False


def _abstain_action() -> CopilotAction:
    """The single safe action used when forcing an abstain."""
    return CopilotAction(
        step=(
            "Gather more data: retrieve the relevant telemetry windows and runbooks "
            "for the affected entities and re-run; escalate if evidence remains "
            "insufficient."
        ),
        runbook_ref=None,
        urgency=Urgency.SOON,
        requires_approval=False,
    )


def enforce_citations(
    response: CopilotResponse,
    *,
    universe: Iterable[str],
    grounding_score: float | None = None,
) -> CopilotResponse:
    """Return a copy of ``response`` with citations pinned to the context universe.

    * Drops any citation / ``runbook_ref`` not in ``universe``.
    * If no valid citation remains, forces the abstain state (``insufficient_context
      =True``, a single sentinel ``"no-context"`` citation, a low confidence and a
      single "gather more data" action) so the result stays schema-valid *and*
      honestly grounded.
    * Records ``grounding_score`` if supplied.

    Pure/deterministic; mutates a model copy, not the input.
    """
    uni = set(universe)
    check = validate_citations(response.citations, uni)

    data = response.model_dump()

    if grounding_score is not None:
        data["grounding_score"] = max(0.0, min(1.0, grounding_score))

    if not check.valid_citations:
        # Nothing grounded -> abstain (still valid: >=1 citation, >=1 action).
        data["citations"] = ["no-context"]
        data["insufficient_context"] = True
        data["confidence_score"] = min(data.get("confidence_score", 0.0), 0.2)
        data["recommended_actions"] = [_abstain_action().model_dump()]
        # Strip now-dangling runbook refs.
        for _sig in data.get("contributing_signals", []):
            pass
        return CopilotResponse.model_validate(data)

    data["citations"] = check.valid_citations

    # Prune action runbook_refs that aren't in the universe (keep the action).
    pruned_actions = []
    for act in data.get("recommended_actions", []):
        ref = act.get("runbook_ref")
        if ref is not None and ref not in uni:
            act = {**act, "runbook_ref": None}
        pruned_actions.append(act)
    if pruned_actions:
        data["recommended_actions"] = pruned_actions

    return CopilotResponse.model_validate(data)


__all__ = [
    "validate_citations",
    "enforce_citations",
    "should_abstain",
    "CitationCheck",
]
