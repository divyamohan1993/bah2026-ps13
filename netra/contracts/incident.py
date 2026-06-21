"""Correlation / RCA / playbook contracts — the integrated-workflow outputs.

The correlation+risk workstream collapses many correlated precursor signals
into a single ranked :class:`Incident` (graph event-correlation, root-cause
hypothesis, blast-radius, calibrated severity). The copilot workstream attaches
a recommended :class:`Playbook` (ordered :class:`RecommendedAction` steps, each
gated by human approval with a rollback). These models answer Q2 (why/where) and
Q3 (what action) and feed the operator-ready incident card in the UI.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .analytics import ContributingSignal, FusedRisk
from .common import EntityRef, NetraModel
from .enums import ApprovalState, IssueType, ScenarioId, Severity, Urgency


class BlastRadius(NetraModel):
    """The deterministically-computed scope affected by a failing entity.

    Computed by graph reachability (BFS over the topology digital twin,
    intersected with NetFlow) — NOT guessed by the LLM. ``hop_distances`` doubles
    as a propagation-time/urgency proxy (research 07 A1.4).
    """

    affected_sites: list[str] = Field(
        default_factory=list, description="Downstream-reachable sites at risk."
    )
    affected_devices: list[str] = Field(
        default_factory=list, description="Downstream-reachable devices."
    )
    affected_services_or_vpns: list[str] = Field(
        default_factory=list, description="Affected services / VPNs / VRFs."
    )
    affected_slas: list[str] = Field(
        default_factory=list, description="SLAs put at risk."
    )
    affected_flow_count: int | None = Field(
        default=None, ge=0, description="# NetFlow flows traversing the failure."
    )
    hop_distances: dict[str, int] = Field(
        default_factory=dict,
        description="entity_id -> hop distance from the failure (propagation proxy).",
    )
    normalized_size: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Blast radius normalised to [0,1] for risk scoring.",
    )


class RecommendedAction(NetraModel):
    """One ordered remediation step (suggest -> approve -> execute).

    Safety posture is human-in-the-loop by default: anything that changes device
    state has ``requires_approval=True`` and starts in ``ApprovalState.PROPOSED``;
    only read-only diagnostics may be ``AUTO_OK``. Every state-changing step
    carries a ``rollback`` and a ``verification`` so the operator can undo and
    confirm.
    """

    step: int = Field(..., ge=1, description="1-based order of this step.")
    description: str = Field(..., description="Human-readable action description.")
    command_or_guidance: str = Field(
        ...,
        description="Concrete command template or operator guidance.",
        examples=["show interface eth1 | include rate", "napalm replace_config golden"],
    )
    requires_approval: bool = Field(
        default=True, description="True for any state-changing action."
    )
    urgency: Urgency = Field(
        default=Urgency.SOON, description="How soon this step should run."
    )
    rollback: str | None = Field(
        default=None, description="Rollback command/procedure if this step is undone."
    )
    verification: str | None = Field(
        default=None, description="How to verify the step achieved its goal."
    )
    runbook_ref: str | None = Field(
        default=None, description="Citation: id of the runbook chunk this came from."
    )
    approval_state: ApprovalState = Field(
        default=ApprovalState.PROPOSED, description="Current lifecycle state."
    )
    safety_class: str | None = Field(
        default=None,
        description="Risk class of the action.",
        examples=["read_only", "config_change", "service_impacting"],
    )


class Playbook(NetraModel):
    """An ordered remediation playbook matched to a predicted issue (Q3).

    Modelled after CACAO-style course-of-action: an id, the issue signature it
    remediates, and the ordered steps (each verify+rollback). Stored in the RAG
    corpus so the copilot retrieves and cites it rather than inventing steps.
    """

    playbook_id: str = Field(
        ..., description="Stable playbook id.", examples=["PB-CONGESTION-001"]
    )
    title: str = Field(..., description="Human-readable playbook title.")
    issue_type: IssueType = Field(
        ..., description="Fault class this playbook remediates."
    )
    trigger_signature: str | None = Field(
        default=None, description="Signature/condition that selects this playbook."
    )
    actions: list[RecommendedAction] = Field(
        ..., min_length=1, description="Ordered remediation steps."
    )
    source_ref: str | None = Field(
        default=None, description="Corpus source id (for citation)."
    )


class Incident(NetraModel):
    """A single correlated, ranked, operator-ready incident.

    The unit of the triage queue. One incident bundles the correlated entities
    and signals, a ranked root-cause hypothesis, the deterministic blast radius,
    a calibrated severity/risk and the time window — i.e. the structured record
    that is handed to the LLM and rendered as the 3-answer card. Designed so
    every field is auditable and traceable to evidence (defends the
    'grounded, no hallucination' score).
    """

    incident_id: str = Field(..., description="Stable incident id.")
    created_at: datetime = Field(..., description="UTC incident creation time.")
    window_start: datetime = Field(..., description="Start of the evidence window.")
    window_end: datetime = Field(..., description="End of the evidence window.")

    predicted_issue: IssueType = Field(
        ..., description="Predicted/diagnosed fault class."
    )
    severity: Severity = Field(..., description="Urgency class (P1/P2/P3/info).")
    risk: FusedRisk = Field(
        ..., description="The fused, calibrated risk driving prioritisation."
    )

    root_cause_entity: EntityRef | None = Field(
        default=None,
        description="Most likely root-cause node (max centrality x earliest "
        "onset x causal score).",
    )
    root_cause_hypothesis: str = Field(
        ..., description="One-paragraph root-cause hypothesis (grounded)."
    )
    correlated_entities: list[EntityRef] = Field(
        default_factory=list,
        description="All entities folded into this incident (symptoms + cause).",
    )
    contributing_signals: list[ContributingSignal] = Field(
        default_factory=list, description="Ranked 'why' signals (Q2)."
    )
    blast_radius: BlastRadius = Field(
        default_factory=BlastRadius, description="Affected scope (deterministic)."
    )
    recommended_playbook: Playbook | None = Field(
        default=None, description="Suggested remediation playbook (Q3)."
    )
    alarm_compression_ratio: float | None = Field(
        default=None,
        ge=1,
        description="Raw alarms / incidents (alert-fatigue reduction metric).",
    )
    scenario_label: ScenarioId | None = Field(
        default=None,
        description="Ground-truth scenario id when evaluating against injected "
        "faults (Phase 6); None in live operation.",
    )


__all__ = [
    "BlastRadius",
    "RecommendedAction",
    "Playbook",
    "Incident",
]
