"""``SituationReport`` — the end-to-end output of :class:`NetraPipeline`.

A :class:`SituationReport` is the single object that bundles everything the
pipeline produced for one batch/stream of telemetry over a topology, so the demo,
the API ``LiveProvider`` and tests can all consume one structured result instead
of re-running the chain:

  * the **ranked incidents** (``netra.contracts.Incident``) — already correlated,
    RCA'd, blast-radius'd, explained and severity-bucketed;
  * the per-entity **FusedRisk timeline** (``risk_history``) — the visual proof
    that risk rose *before* the labeled fault (lead time);
  * the **CopilotResponse(s)** for the top incident(s) (Q1/Q2/Q3, grounded);
  * a per-scenario **evaluation** (``ScenarioEval``) measuring, against the
    synthetic ground-truth ``ScenarioLabel``, whether NETRA raised risk in the
    precursor window *before* the breach and with how much lead time.

Everything here is plain dataclasses over ``netra.contracts`` types (import-light:
only the contracts + stdlib), so a report serialises cleanly and never drags a
heavy dependency into a consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from netra.contracts import (
    CopilotResponse,
    Incident,
    IssueType,
    ScenarioId,
    ScenarioLabel,
)


@dataclass
class RiskPoint:
    """One (timestamp, risk) sample on an entity's fused-risk trajectory."""

    timestamp: datetime
    risk_score: float
    calibrated_confidence: float
    predicted_issue: IssueType = IssueType.NONE
    agreement: float = 0.0


@dataclass
class ScenarioEval:
    """Evaluation of the pipeline against one ground-truth :class:`ScenarioLabel`.

    Measures the headline metric — *did NETRA raise elevated risk during the
    precursor window, before the labeled fault began, and by how much* — for the
    target entity (or any correlated entity), plus which detector families fired.

    Lead time is ``fault_window_start - first_elevated_alert_time`` for the
    earliest alert that landed in ``[precursor_window_start, fault_window_start)``
    (the positive 'early-warning' region defined by the contract). A positive
    lead time means NETRA warned that many seconds before the breach.
    """

    scenario: ScenarioId
    expected_issue: IssueType
    target_entity_id: str
    #: precursor window opened (alerts here earn lead-time credit).
    precursor_window_start: datetime
    #: when the actual fault/breach began.
    fault_window_start: datetime
    fault_window_end: datetime

    #: did the pipeline raise elevated risk before the breach? (the win condition)
    detected: bool = False
    #: earliest in-window elevated-risk alert time (None if never warned in time).
    first_alert_at: datetime | None = None
    #: lead time in seconds (fault_start - first_alert_at); None if not detected.
    lead_time_seconds: float | None = None
    #: ground-truth target lead time the generator labeled this scenario with.
    expected_lead_time_seconds: float | None = None
    #: peak fused risk observed on the target/correlated entities in-window.
    peak_risk: float = 0.0
    #: did the pipeline's predicted issue match the labeled fault class?
    predicted_issue_correct: bool = False
    #: which detector/forecaster families contributed the in-window evidence.
    methods_fired: list[str] = field(default_factory=list)
    #: the incident id the pipeline raised for this scenario, if any.
    incident_id: str | None = None
    #: time-to-impact the pipeline estimated at the first alert (seconds), if any.
    eta_seconds_at_alert: float | None = None

    @property
    def lead_time_minutes(self) -> float | None:
        if self.lead_time_seconds is None:
            return None
        return round(self.lead_time_seconds / 60.0, 2)

    @property
    def top_method(self) -> str | None:
        return self.methods_fired[0] if self.methods_fired else None


@dataclass
class SituationReport:
    """The full, structured result of a :class:`NetraPipeline` run.

    Holds the ranked incidents, the per-entity fused-risk timeline, the copilot
    answers for the top incident(s), and the per-scenario evaluation against the
    ground-truth labels. The demo prints from it; the API ``LiveProvider`` serves
    from it.
    """

    generated_at: datetime
    #: ranked, severity-bucketed incidents (highest calibrated priority first).
    incidents: list[Incident] = field(default_factory=list)
    #: per-entity fused-risk trajectory (entity_id -> ordered risk points).
    risk_history: dict[str, list[RiskPoint]] = field(default_factory=dict)
    #: copilot Q1/Q2/Q3 answers, keyed by the incident id they answer for.
    copilot_answers: dict[str, CopilotResponse] = field(default_factory=dict)
    #: per-scenario evaluation vs ground truth (empty when no labels supplied).
    scenario_evals: list[ScenarioEval] = field(default_factory=list)
    #: the ground-truth labels the run was evaluated against (if any).
    labels: list[ScenarioLabel] = field(default_factory=list)
    #: window the telemetry spanned.
    window_start: datetime | None = None
    window_end: datetime | None = None
    #: counters for the run (records processed, entities tracked, alerts raised).
    stats: dict[str, float] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    @property
    def headline_incident(self) -> Incident | None:
        """The top-ranked incident, or None if the run produced none."""
        return self.incidents[0] if self.incidents else None

    def fused_timeline(self, entity_id: str) -> list[RiskPoint]:
        """The fused-risk trajectory for one entity (empty if untracked)."""
        return self.risk_history.get(entity_id, [])

    def copilot_for(self, incident_id: str) -> CopilotResponse | None:
        return self.copilot_answers.get(incident_id)

    def eval_for(self, scenario: ScenarioId) -> ScenarioEval | None:
        for ev in self.scenario_evals:
            if ev.scenario == scenario:
                return ev
        return None


__all__ = ["RiskPoint", "ScenarioEval", "SituationReport"]
