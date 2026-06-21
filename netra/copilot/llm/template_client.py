"""Deterministic, model-free copilot backend (the graceful-degradation path).

:class:`TemplateClient` composes a fully schema-valid
:class:`~netra.contracts.CopilotResponse` from the structured analytics + RAG
inputs (:class:`~netra.copilot.llm.base.CopilotGrounding`) **without any model**.
It is the hard requirement that makes the copilot run, demo and pass tests with
*zero* heavy dependencies and no LLM present (architecture §5.2): when
``llama-server`` is absent the orchestrator selects this client and the API/UI/
tests see the identical response shape, just with ``used_fallback=True`` and
``model_id="template-fallback"``.

Design guarantees:
  * **Always valid.** Output satisfies the contract's constraints
    (``>=1`` recommended action, ``>=1`` citation, ``root_cause_hypothesis``
    non-empty) for *every* input, including an empty corpus (it then abstains
    with ``insufficient_context=True`` and a "gather more data" action — still
    schema-valid).
  * **Grounded, never invented.** ``confidence_score`` and
    ``time_to_impact_minutes`` come straight from the analytics grounding;
    citations are drawn only from the closed ``citation_universe``; the prose is
    templated from real signal/observation strings, so nothing is fabricated.
  * **Deterministic.** Same inputs -> byte-identical output (no RNG, no clock),
    which makes it trivially testable and reproducible.
"""

from __future__ import annotations

from netra.contracts import (
    AffectedScope,
    CopilotAction,
    CopilotResponse,
    CopilotSignal,
    IssueType,
    Urgency,
)

from .base import CopilotGrounding, CopilotPrompt, LLMClient

#: Human-readable phrasing for each closed IssueType (used in the Q1 sentence).
_ISSUE_PHRASING: dict[IssueType, str] = {
    IssueType.INTERFACE_CONGESTION: "progressive interface congestion",
    IssueType.LATENCY_DRIFT: "latency drift",
    IssueType.BGP_ROUTE_FLAP: "a BGP route-flap cascade",
    IssueType.OSPF_CONVERGENCE_STRESS: "OSPF convergence stress",
    IssueType.TUNNEL_DEGRADATION: "IPSec/MPLS tunnel degradation",
    IssueType.MPLS_UNDERLAY_FAILURE: "an MPLS underlay failure",
    IssueType.POLICY_DRIFT: "controller-induced policy drift",
    IssueType.PATH_ASYMMETRY: "forwarding path asymmetry",
    IssueType.NONE: "no specific fault",
}


def _clamp_unit(x: float) -> float:
    """Clamp a value into the [0,1] range the confidence field requires."""
    return max(0.0, min(1.0, float(x)))


def _fmt_minutes(minutes: float | None) -> str:
    """Render a time-to-impact in operator-friendly text."""
    if minutes is None:
        return "no imminent threshold crossing is predicted"
    if minutes < 1:
        return "an SLA/security impact is imminent (under a minute)"
    return f"an estimated ~{minutes:.0f} min to SLA/security impact"


class TemplateClient(LLMClient):
    """Compose a valid CopilotResponse deterministically, with no model."""

    model_id = "template-fallback"
    is_fallback = True

    def available(self) -> bool:
        """Always available — this is the no-dependency fallback."""
        return True

    # ``prompt`` is accepted for interface symmetry with the LLM client but the
    # template path reads only the structured ``grounding`` (it does not parse
    # free text), which is exactly what keeps it deterministic and model-free.
    def complete_copilot(
        self, prompt: CopilotPrompt, grounding: CopilotGrounding
    ) -> CopilotResponse:
        """Build the structured answer from ``grounding`` alone."""
        if grounding.insufficient_context or not grounding.citation_universe:
            return self._abstain(grounding)
        return self._answer(grounding)

    # -- the grounded (sufficient-context) answer -------------------------------
    def _answer(self, g: CopilotGrounding) -> CopilotResponse:
        issue_phrase = _ISSUE_PHRASING.get(g.predicted_issue, g.predicted_issue.value)
        eta_phrase = _fmt_minutes(g.time_to_impact_minutes)

        # Q2 — root-cause hypothesis, templated from real signals (grounded).
        root_cause = self._compose_root_cause(g, issue_phrase, eta_phrase)

        # Q3 — recommended actions: prefer the analytics-supplied playbook steps;
        # the contract requires >=1 so synthesise a safe diagnostic step if none.
        actions = list(g.recommended_actions) or [self._default_diag_action(g)]

        # Citations: everything actually present in the context (closed set). The
        # grounding gate downstream will prune to the truly-used subset.
        citations = list(dict.fromkeys(g.citation_universe))  # dedupe, keep order

        return CopilotResponse(
            request_id=g.request_id,
            predicted_issue=g.predicted_issue,
            confidence_score=_clamp_unit(g.confidence_score),
            time_to_impact_minutes=g.time_to_impact_minutes,
            root_cause_hypothesis=root_cause,
            contributing_signals=list(g.contributing_signals),
            affected_scope=g.affected_scope or AffectedScope(),
            recommended_actions=actions,
            citations=citations,
            insufficient_context=False,
            used_fallback=True,
            model_id=self.model_id,
        )

    def _compose_root_cause(
        self, g: CopilotGrounding, issue_phrase: str, eta_phrase: str
    ) -> str:
        """Template a grounded one-paragraph root-cause hypothesis (Q2)."""
        scope = g.affected_scope or AffectedScope()
        where = ""
        if scope.devices:
            where = f" centred on {', '.join(scope.devices[:3])}"
        elif scope.sites:
            where = f" at {', '.join(scope.sites[:3])}"

        lead = (
            f"The analytics ensemble predicts {issue_phrase}{where}, with "
            f"{eta_phrase} at a calibrated confidence of "
            f"{_clamp_unit(g.confidence_score):.2f}."
        )

        if g.contributing_signals:
            drivers = "; ".join(
                self._signal_phrase(s) for s in g.contributing_signals[:4]
            )
            why = f" The leading contributing signals are: {drivers}."
        else:
            why = " No individual contributing signals were attributed."

        # If the orchestrator supplied a richer retrieved/RCA hypothesis, append
        # it verbatim (it is itself grounded in the corpus); otherwise the
        # templated text already stands alone.
        extra = ""
        if g.root_cause_hypothesis:
            extra = f" {g.root_cause_hypothesis.strip()}"

        text = (lead + why + extra).strip()
        # Respect the contract's max_length (1200) defensively.
        return text[:1200]

    @staticmethod
    def _signal_phrase(s: CopilotSignal) -> str:
        """Render one contributing signal as 'name (observation)'."""
        if s.observation:
            return f"{s.signal} ({s.observation})"
        return s.signal

    @staticmethod
    def _default_diag_action(g: CopilotGrounding) -> CopilotAction:
        """A safe, read-only diagnostic step when no playbook is available."""
        runbook = g.citation_universe[0] if g.citation_universe else None
        return CopilotAction(
            step=(
                "Collect current interface/queue/routing/tunnel statistics for the "
                "affected entities and correlate against the predicted issue before "
                "any state-changing action."
            ),
            runbook_ref=runbook,
            urgency=Urgency.IMMEDIATE,
            requires_approval=False,
        )

    # -- the abstain (insufficient-context) answer ------------------------------
    def _abstain(self, g: CopilotGrounding) -> CopilotResponse:
        """Produce a valid, low-confidence abstaining response.

        Still schema-valid: a single "gather more data / escalate" action and a
        single sentinel citation so the contract's ``>=1`` constraints hold while
        ``insufficient_context=True`` and confidence is forced low (research 05
        §5.1 abstain rule).
        """
        q = (
            f' for the question "{g.operator_query}"'
            if g.operator_query
            else ""
        )
        root_cause = (
            "Insufficient local evidence to attribute a root cause"
            f"{q}. The retrieved corpus/analytics context did not contain enough "
            "grounded support, so the copilot is abstaining rather than guessing. "
            "Gather additional telemetry (interface/routing/tunnel/config diffs) "
            "or escalate to a human operator."
        )
        action = CopilotAction(
            step=(
                "Gather more data: pull the relevant telemetry windows and runbooks "
                "for the affected entities, then re-run the copilot; escalate if "
                "evidence remains insufficient."
            ),
            runbook_ref=None,
            urgency=Urgency.SOON,
            requires_approval=False,
        )
        return CopilotResponse(
            request_id=g.request_id,
            predicted_issue=g.predicted_issue,
            # Abstaining => confidence must be low regardless of any upstream value.
            confidence_score=min(_clamp_unit(g.confidence_score), 0.2),
            time_to_impact_minutes=g.time_to_impact_minutes,
            root_cause_hypothesis=root_cause[:1200],
            contributing_signals=[],
            affected_scope=g.affected_scope or AffectedScope(),
            recommended_actions=[action],
            citations=["no-context"],
            insufficient_context=True,
            used_fallback=True,
            model_id=self.model_id,
        )


__all__ = ["TemplateClient"]
