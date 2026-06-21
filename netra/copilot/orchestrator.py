"""``Copilot.answer`` — the grounded, structured copilot orchestrator (WS5).

Wires the analytics inputs (Q1/Q2 from WS3/WS4) and the RAG context (the corpus +
GraphRAG-lite topology) into a single grounded prompt, runs the selected backend
(grammar-constrained ``llama-server`` *or* the deterministic template fallback),
and applies the grounding gate (closed-set citations + faithfulness + abstain) to
produce a schema-valid :class:`~netra.contracts.CopilotResponse` that answers:

  * **Q1** — ``predicted_issue`` + ``time_to_impact_minutes`` + ``affected_scope``
  * **Q2** — ``root_cause_hypothesis`` + ``contributing_signals``
  * **Q3** — ``recommended_actions``

with ``confidence_score`` sourced from the analytics engine (never invented),
mandatory ``citations`` and an ``insufficient_context`` abstain flag.

**Contract-only inputs.** The analytics objects arrive via
:class:`AnalyticsContext`, which holds the canonical ``netra.contracts`` types
(:class:`Incident`, :class:`FusedRisk`, :class:`TimeToImpact`,
:class:`ContributingSignal`, :class:`BlastRadius`, :class:`Playbook`). The
copilot **consumes these contract outputs as inputs** — it never imports the
analytics/correlation builders. The API layer constructs an ``AnalyticsContext``
from whatever the analytics engine produced and calls :meth:`Copilot.answer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from netra.contracts import (
    AffectedScope,
    BlastRadius,
    ContributingSignal,
    CopilotAction,
    CopilotRequest,
    CopilotResponse,
    CopilotSignal,
    EntityRef,
    FusedRisk,
    Incident,
    IssueType,
    Playbook,
    TimeToImpact,
    Urgency,
)

from .grounding import FaithfulnessScorer, enforce_citations, should_abstain
from .llm import CopilotGrounding, CopilotPrompt, LLMClient, select_llm_client
from .rag import (
    HybridRetriever,
    TopologyGraph,
    affected_scope_from_blast_radius,
    build_retriever,
    document_ids,
    graph_facts,
)

# The system prompt (research 05 §6.2), adapted to the contract field names.
_SYSTEM_PROMPT = """\
You are the NOC Copilot for an air-gapped SD-WAN-over-MPLS network operations \
center. You turn the predictive analytics engine's output and retrieved internal \
documents into clear, correct, operator-ready guidance.

GROUNDING RULES (strict):
- Use ONLY the information in the provided CONTEXT block: analytics predictions, \
contributing signals, retrieved runbooks/playbooks/incidents/topology (each has \
an ID), and the graph facts. Do NOT use outside knowledge or invent device \
names, metrics, or causes.
- Every factual claim must be supported by the CONTEXT. Put the IDs of the \
chunks/facts you actually relied on into "citations". If you cannot cite it, do \
not say it.
- If the CONTEXT is insufficient, set "insufficient_context": true, give a low \
"confidence_score", and recommend gathering specific data or escalating - do NOT \
guess a root cause.
- Use the analytics engine's probability as "confidence_score" and its estimate \
for "time_to_impact_minutes". Explain these numbers; do not fabricate new ones.

ANSWER THE OPERATOR'S THREE QUESTIONS, grounded in the context:
  Q1 What is likely to fail next, and when? -> predicted_issue, \
time_to_impact_minutes, affected_scope.
  Q2 Why is risk elevated, which signals contributed? -> root_cause_hypothesis \
+ contributing_signals (tie each to a cited signal/observation).
  Q3 What corrective action should be taken before SLA/security impact? -> \
recommended_actions with the matching runbook_ref and an urgency.

STYLE: concise, concrete NOC-operator language. Output ONLY the JSON object that \
conforms to the provided schema."""


@dataclass
class AnalyticsContext:
    """Structured analytics inputs the copilot grounds Q1/Q2 on (contract types).

    The API constructs this from the analytics engine's outputs and passes it to
    :meth:`Copilot.answer`. Every field is an optional ``netra.contracts`` object,
    so the copilot can run from a full :class:`Incident`, from looser pieces
    (a :class:`FusedRisk` + :class:`TimeToImpact`), or from nothing (it abstains).

    Field precedence: an explicit value wins; otherwise it is derived from
    ``incident`` (e.g. ``blast_radius`` falls back to ``incident.blast_radius``).
    """

    #: The single correlated, ranked incident (richest input); preferred source.
    incident: Incident | None = None
    #: Fused, calibrated risk — the authoritative confidence + predicted issue.
    fused_risk: FusedRisk | None = None
    #: Lead-time estimate (Q1 "when"); ``eta_seconds`` -> ``time_to_impact_minutes``.
    time_to_impact: TimeToImpact | None = None
    #: Ranked SHAP "why" signals (Q2).
    contributing_signals: list[ContributingSignal] = field(default_factory=list)
    #: Deterministic affected scope (Q1 where/how-bad) from WS4.
    blast_radius: BlastRadius | None = None
    #: Matched remediation playbook (Q3).
    playbook: Playbook | None = None
    #: Root-cause node for the GraphRAG-lite device description.
    root_cause_entity: EntityRef | None = None

    # -- resolved accessors (explicit value > derived-from-incident) ------------
    def resolved_issue(self) -> IssueType:
        if self.fused_risk is not None:
            return self.fused_risk.predicted_issue
        if self.incident is not None:
            return self.incident.predicted_issue
        return IssueType.NONE

    def resolved_confidence(self) -> float:
        if self.fused_risk is not None:
            return float(self.fused_risk.calibrated_confidence)
        if self.incident is not None:
            return float(self.incident.risk.calibrated_confidence)
        return 0.0

    def resolved_tti(self) -> TimeToImpact | None:
        if self.time_to_impact is not None:
            return self.time_to_impact
        if self.fused_risk is not None and self.fused_risk.time_to_impact is not None:
            return self.fused_risk.time_to_impact
        if self.incident is not None and self.incident.risk.time_to_impact is not None:
            return self.incident.risk.time_to_impact
        return None

    def resolved_signals(self) -> list[ContributingSignal]:
        if self.contributing_signals:
            return self.contributing_signals
        if self.incident is not None:
            return list(self.incident.contributing_signals)
        return []

    def resolved_blast_radius(self) -> BlastRadius | None:
        if self.blast_radius is not None:
            return self.blast_radius
        if self.incident is not None:
            return self.incident.blast_radius
        return None

    def resolved_playbook(self) -> Playbook | None:
        if self.playbook is not None:
            return self.playbook
        if self.incident is not None:
            return self.incident.recommended_playbook
        return None

    def resolved_root_cause_entity(self) -> EntityRef | None:
        if self.root_cause_entity is not None:
            return self.root_cause_entity
        if self.incident is not None:
            return self.incident.root_cause_entity
        return None

    def resolved_root_cause_hypothesis(self) -> str:
        if self.incident is not None and self.incident.root_cause_hypothesis:
            return self.incident.root_cause_hypothesis
        return ""


def _eta_minutes(tti: TimeToImpact | None) -> float | None:
    """Convert a TimeToImpact (seconds) to the contract's minutes field."""
    if tti is None or tti.eta_seconds is None:
        return None
    return round(tti.eta_seconds / 60.0, 1)


def _to_copilot_signal(s: ContributingSignal) -> CopilotSignal:
    """Project a heavy ContributingSignal into the lean copilot signal shape."""
    return CopilotSignal(
        signal=s.signal,
        observation=s.observation or s.human_explanation,
        shap_contribution=s.shap_value,
    )


def _playbook_to_actions(pb: Playbook | None) -> list[CopilotAction]:
    """Map a contract Playbook's ordered steps to copilot actions (Q3)."""
    if pb is None:
        return []
    actions: list[CopilotAction] = []
    for step in pb.actions:
        actions.append(
            CopilotAction(
                step=step.description,
                runbook_ref=step.runbook_ref or pb.source_ref,
                urgency=step.urgency,
                requires_approval=step.requires_approval,
            )
        )
    return actions


class Copilot:
    """The offline NOC copilot: grounded, structured Q1/Q2/Q3 answers.

    Construct once (it builds/holds the RAG retriever, topology graph, the
    selected LLM backend and the faithfulness scorer), then call
    :meth:`answer` per :class:`CopilotRequest`.

    Parameters
    ----------
    retriever:
        A pre-built :class:`HybridRetriever`. If ``None``, one is built over the
        repo corpus on first construction (``prefer_models`` controls heavy deps).
    topology:
        A pre-built :class:`TopologyGraph`. If ``None``, loaded from the corpus.
    llm:
        A pre-selected :class:`LLMClient`. If ``None``, auto-selected
        (llama.cpp if reachable on loopback, else the template fallback).
    prefer_models:
        Try the heavy bge/HHEM/llama models when True; defaults to False (the
        CPU-only, fully-offline light path).
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        topology: TopologyGraph | None = None,
        llm: LLMClient | None = None,
        faithfulness: FaithfulnessScorer | None = None,
        prefer_models: bool = False,
        corpus_dir: str | None = None,
    ) -> None:
        self.retriever = (
            retriever
            if retriever is not None
            else build_retriever(corpus_dir, prefer_model=prefer_models)
        )
        self.topology = (
            topology
            if topology is not None
            else TopologyGraph.from_corpus(corpus_dir)
        )
        self.llm = llm if llm is not None else select_llm_client(
            prefer_llm=True if prefer_models else None
        )
        self.faithfulness = faithfulness or FaithfulnessScorer(
            prefer_model=prefer_models
        )

    # -- public API -------------------------------------------------------------
    def answer(
        self,
        request: CopilotRequest,
        *,
        analytics_context: AnalyticsContext | None = None,
    ) -> CopilotResponse:
        """Produce a grounded, schema-valid :class:`CopilotResponse`.

        Parameters
        ----------
        request:
            The operator query / auto-trigger (carries ``request_id`` and
            ``max_context_chunks``).
        analytics_context:
            The analytics inputs (Q1/Q2 evidence). May be ``None`` (the copilot
            then has only the corpus and will likely abstain).
        """
        ctx = analytics_context or AnalyticsContext()

        # 1) Resolve the authoritative analytics fields (never invented).
        issue = ctx.resolved_issue()
        confidence = ctx.resolved_confidence()
        tti = ctx.resolved_tti()
        tti_minutes = _eta_minutes(tti)
        signals = ctx.resolved_signals()
        blast_radius = ctx.resolved_blast_radius()
        playbook = ctx.resolved_playbook()
        root_cause_entity = ctx.resolved_root_cause_entity()

        # 2) Retrieve grounded corpus chunks for the query.
        query = self._build_query(request, issue, signals)
        top_k = request.max_context_chunks
        chunks = self.retriever.retrieve(query, top_k=top_k) if len(self.retriever) else []

        # 3) GraphRAG-lite deterministic topology + blast-radius facts.
        facts = graph_facts(
            root_cause_entity=root_cause_entity,
            blast_radius=blast_radius,
            topology=self.topology,
        )

        # 4) Affected scope = deterministic blast radius (reported, not guessed).
        affected_scope = affected_scope_from_blast_radius(blast_radius)

        # 5) Citation universe = retrieved chunk ids + their doc-level ids
        #    (so a playbook step's runbook_ref to the parent doc is grounded) +
        #    graph fact ids.
        citation_universe = (
            [c.chunk_id for c in chunks]
            + document_ids(chunks)
            + [fid for fid, _ in facts]
        )

        # 6) Decide abstain from KNOWLEDGE-BASE sufficiency. Graph facts alone are
        #    a deterministic restatement of the analytics (not retrieved knowledge),
        #    so they are citable but do NOT by themselves lift an abstain: without
        #    runbook/incident/playbook evidence the copilot cannot ground the Q2
        #    reasoning / Q3 remediation and should say "insufficient local evidence".
        insufficient = should_abstain(n_context_chunks=len(chunks))

        # 7) Assemble the structured grounding + the rendered prompt.
        grounding = CopilotGrounding(
            request_id=request.request_id,
            predicted_issue=issue,
            confidence_score=confidence,
            time_to_impact_minutes=tti_minutes,
            root_cause_hypothesis=ctx.resolved_root_cause_hypothesis(),
            contributing_signals=[_to_copilot_signal(s) for s in signals],
            affected_scope=affected_scope,
            recommended_actions=self._actions_for(playbook, citation_universe),
            citation_universe=citation_universe,
            insufficient_context=insufficient,
            operator_query=request.operator_query,
        )
        prompt = self._build_prompt(request, grounding, chunks, facts)

        # 8) Run the selected backend (LLM or deterministic template).
        response = self.llm.complete_copilot(prompt, grounding)

        # 9) Faithfulness scoring of the answer vs the cited context.
        context_text = self._context_text(chunks, facts)
        fr = self.faithfulness.score_text(
            answer=response.root_cause_hypothesis, context=context_text
        )

        # 10) Grounding gate: closed-set citations + record grounding score.
        response = enforce_citations(
            response, universe=citation_universe, grounding_score=fr.score
        )
        return response

    # -- helpers ----------------------------------------------------------------
    def _actions_for(
        self, playbook: Playbook | None, universe: list[str]
    ) -> list[CopilotAction]:
        """Build Q3 actions from the playbook, with a safe default if absent."""
        actions = _playbook_to_actions(playbook)
        if actions:
            return actions
        ref = universe[0] if universe else None
        return [
            CopilotAction(
                step=(
                    "Collect current interface/queue/routing/tunnel statistics for "
                    "the affected entities and correlate against the predicted "
                    "issue before any state-changing action."
                ),
                runbook_ref=ref,
                urgency=Urgency.IMMEDIATE,
                requires_approval=False,
            )
        ]

    @staticmethod
    def _build_query(
        request: CopilotRequest,
        issue: IssueType,
        signals: list[ContributingSignal],
    ) -> str:
        """Compose the retrieval query from the operator question + analytics."""
        parts: list[str] = []
        if request.operator_query:
            parts.append(request.operator_query)
        if issue != IssueType.NONE:
            parts.append(issue.value.replace("_", " "))
        parts.extend(s.signal for s in signals[:5])
        parts.extend(
            s.observation for s in signals[:3] if s.observation
        )
        return " ".join(parts) if parts else "network incident remediation runbook"

    def _build_prompt(
        self,
        request: CopilotRequest,
        grounding: CopilotGrounding,
        chunks,
        facts,
    ) -> CopilotPrompt:
        """Render the grounded user message (CONTEXT block + question)."""
        lines: list[str] = ["=== CONTEXT ==="]

        lines.append("\n[ANALYTICS]")
        lines.append(f"predicted_issue: {grounding.predicted_issue.value}")
        lines.append(f"confidence_score: {grounding.confidence_score:.3f}")
        lines.append(
            "time_to_impact_minutes: "
            + (
                f"{grounding.time_to_impact_minutes}"
                if grounding.time_to_impact_minutes is not None
                else "null"
            )
        )
        scope = grounding.affected_scope
        lines.append(
            "affected_scope: sites="
            f"{scope.sites} devices={scope.devices} services_or_vpns="
            f"{scope.services_or_vpns}"
        )

        if grounding.contributing_signals:
            lines.append("\n[CONTRIBUTING SIGNALS]")
            for s in grounding.contributing_signals:
                shap = (
                    f" (shap={s.shap_contribution:+.3f})"
                    if s.shap_contribution is not None
                    else ""
                )
                lines.append(f"- {s.signal}: {s.observation}{shap}")

        if facts:
            lines.append("\n[GRAPH FACTS]")
            for fid, text in facts:
                lines.append(f"[{fid}] {text}")

        if chunks:
            lines.append("\n[RETRIEVED DOCUMENTS]")
            for c in chunks:
                lines.append(f"[{c.chunk_id}] {c.text}")

        lines.append("\n=== QUESTION ===")
        if request.operator_query:
            lines.append(request.operator_query)
        else:
            lines.append(
                "Answer the three standing operator questions (Q1 what fails next "
                "and when, Q2 why / which signals, Q3 what action)."
            )

        return CopilotPrompt(system=_SYSTEM_PROMPT, user="\n".join(lines))

    @staticmethod
    def _context_text(chunks, facts) -> str:
        """Concatenate the cited context for faithfulness scoring."""
        parts = [t for _, t in facts] + [c.text for c in chunks]
        return "\n".join(parts)


__all__ = ["Copilot", "AnalyticsContext"]
