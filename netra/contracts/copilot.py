"""Copilot request/response contracts — the LLM interface (grounded, schema-valid).

:class:`CopilotResponse` is the single most important interface in NETRA for the
"Copilot Effectiveness" score (35%). It is the **required structured schema** the
offline LLM is constrained to emit (via GBNF/JSON-schema grammar in
``llama-server``) AND the schema the deterministic template fallback fills when
no model is present (graceful degradation). Because the model is grammar-
constrained to this shape, the field set / enums here MUST stay in lockstep with
the GBNF in research 05 §4.2 — treat changes as model-grammar changes.

Field-level descriptions double as model hints (Instructor-style) and as the
operator-facing documentation, so they are written to be useful to both.

It answers the three operational questions explicitly:
  Q1 (what fails next & when) -> ``predicted_issue`` + ``time_to_impact_minutes``
                                  + ``affected_scope``.
  Q2 (why / which signals)    -> ``root_cause_hypothesis`` + ``contributing_signals``.
  Q3 (what action)            -> ``recommended_actions``.
Grounding is enforced by mandatory ``citations`` and an ``insufficient_context``
abstain flag.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import NetraModel
from .enums import IssueType, Urgency


class CopilotRequest(NetraModel):
    """Input to the copilot: an operator query and/or an auto-trigger context.

    The copilot can be invoked two ways: (a) an operator types a natural-language
    question, or (b) the analytics engine auto-triggers on a fired incident. In
    both cases the orchestration layer resolves the ``*_ref`` ids into the actual
    analytics/RAG context before prompting the model; passing references (not the
    full objects) keeps the request small and lets the server fetch the freshest
    state.
    """

    request_id: str = Field(..., description="Unique id for this copilot call.")
    created_at: datetime = Field(..., description="UTC time the request was made.")
    operator_query: str | None = Field(
        default=None,
        description="Free-text operator question; None for a pure auto-trigger.",
        examples=["Why is the Mumbai hub uplink at risk and what do I do?"],
    )
    auto_trigger: bool = Field(
        default=False, description="True if raised by the engine, not a human."
    )
    incident_ref: str | None = Field(
        default=None, description="incident_id this query is about, if any."
    )
    entity_refs: list[str] = Field(
        default_factory=list,
        description="entity_ids to scope retrieval/analytics context to.",
    )
    fused_risk_refs: list[str] = Field(
        default_factory=list,
        description="Ids of FusedRisk assessments to include as context.",
    )
    max_context_chunks: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Top-k retrieved chunks to ground on (small k => better "
        "grounding + lower latency).",
    )


class CopilotSignal(NetraModel):
    """A contributing signal as rendered inside the copilot response (Q2).

    A lean projection of :class:`~netra.contracts.analytics.ContributingSignal`
    (the heavy attribution object) into the exact shape the grammar emits, so the
    LLM only ever produces these three fields and cannot invent extra structure.
    """

    signal: str = Field(..., description="Signal/feature name.")
    observation: str = Field(
        ..., description="The concrete observed value/trend the signal reflects."
    )
    shap_contribution: float | None = Field(
        default=None, description="Signed SHAP contribution, if available."
    )


class AffectedScope(NetraModel):
    """The affected-scope block of the copilot response (Q1 'where/how-bad').

    Populated deterministically from :class:`~netra.contracts.incident.BlastRadius`
    (graph reachability) so the model reports, never guesses, the scope.
    """

    sites: list[str] = Field(default_factory=list, description="Affected sites.")
    devices: list[str] = Field(default_factory=list, description="Affected devices.")
    services_or_vpns: list[str] = Field(
        default_factory=list, description="Affected services / VPNs."
    )


class CopilotAction(NetraModel):
    """A recommended action as rendered inside the copilot response (Q3)."""

    step: str = Field(..., description="What to do.")
    runbook_ref: str | None = Field(
        default=None, description="Citation to the runbook chunk id this step uses."
    )
    urgency: Urgency = Field(..., description="immediate / soon / monitor.")
    requires_approval: bool = Field(
        default=True, description="True for any state-changing action."
    )


class CopilotResponse(NetraModel):
    """The REQUIRED structured copilot answer (grammar-constrained / fallback).

    This is the contract the LLM is forced to satisfy and the template fallback
    reproduces. ``required`` fields + the closed ``IssueType`` enum + the
    mandatory ``citations`` + the ``insufficient_context`` abstain flag together
    make ungrounded or malformed output essentially impossible:

      * ``predicted_issue``        committed to a known fault class (Q1).
      * ``confidence_score``       sourced from the analytics engine, not guessed.
      * ``time_to_impact``         lead time (Q1); None when not applicable.
      * ``root_cause_hypothesis``  grounded reasoning (Q2).
      * ``contributing_signals``   the SHAP-tied 'why' (Q2).
      * ``affected_scope``         deterministic blast radius (Q1).
      * ``recommended_actions``    the playbook (Q3); >=1 required.
      * ``citations``              ids of chunks/telemetry actually used; >=1.
      * ``insufficient_context``   abstain when evidence is too thin.
    """

    request_id: str = Field(
        ..., description="Echoes CopilotRequest.request_id for correlation."
    )
    predicted_issue: IssueType = Field(
        ..., description="Predicted fault class (closed set; Q1)."
    )
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calibrated confidence (from the analytics engine; the LLM "
        "explains it, does not fabricate it).",
    )
    time_to_impact_minutes: float | None = Field(
        default=None,
        ge=0,
        description="Estimated minutes to SLA/security impact; None if N/A (Q1).",
    )
    root_cause_hypothesis: str = Field(
        ...,
        min_length=1,
        max_length=1200,
        description="Grounded root-cause explanation (Q2).",
    )
    contributing_signals: list[CopilotSignal] = Field(
        default_factory=list, description="Signals that drove the risk (Q2)."
    )
    affected_scope: AffectedScope = Field(
        default_factory=AffectedScope, description="Affected sites/devices/VPNs (Q1)."
    )
    recommended_actions: list[CopilotAction] = Field(
        ..., min_length=1, description="Ordered remediation actions (Q3)."
    )
    citations: list[str] = Field(
        ...,
        min_length=1,
        description="Ids of retrieved chunks / telemetry windows ACTUALLY used. "
        "A closed-set check rejects any id not in the supplied context.",
    )
    insufficient_context: bool = Field(
        ...,
        description="True => the model abstained because evidence was insufficient "
        "(confidence should be low and actions should say 'gather more data').",
    )
    grounding_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Offline HHEM/NLI faithfulness score of this answer vs its "
        "cited context (post-generation gate); not emitted by the LLM itself.",
    )
    used_fallback: bool = Field(
        default=False,
        description="True if produced by the deterministic template fallback "
        "(LLM absent) rather than the model — for graceful-degradation telemetry.",
    )
    model_id: str | None = Field(
        default=None,
        description="Serving model id, e.g. 'qwen2.5-7b-instruct-q4_k_m' or "
        "'template-fallback'.",
    )


__all__ = [
    "CopilotRequest",
    "CopilotSignal",
    "AffectedScope",
    "CopilotAction",
    "CopilotResponse",
]
