"""Incident prioritisation — the ranked, severity-bucketed operator triage queue.

Ties the risk pieces together (research 07 A2):
  1. product-form risk per incident (:mod:`score`),
  2. Platt/isotonic calibration of the combined score (:mod:`calibrate`),
  3. BGP-style flap suppression to demote chronically-flapping entities
     (:mod:`suppress`),
  4. severity bucketing into P1/P2/P3 and a **sorted** queue.

Output: an ``ORDERED`` list of :class:`netra.contracts.Incident` (highest
calibrated risk first), each with severity set and its ``FusedRisk`` updated to
the prioritisation-time calibrated confidence. The copilot/API consume this queue
directly. CPU-only; calibrator + suppressor are optional and the queue still ranks
sensibly without either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from netra.analytics.correlation.graph import TopologyGraph
from netra.contracts import Incident, Severity

from .calibrate import RiskCalibrator
from .score import RiskFactors, RiskWeights, compute_risk_factors, geometric_mean_score
from .suppress import FlapSuppressor

# Severity bucket thresholds on the final calibrated priority score.
DEFAULT_P1_THRESHOLD = 0.66
DEFAULT_P2_THRESHOLD = 0.33


@dataclass
class PrioritizedIncident:
    """An incident plus its computed priority breakdown (for the UI/audit)."""

    incident: Incident
    priority_score: float
    calibrated_priority: float
    factors: RiskFactors
    flap_penalty: float = 0.0
    suppressed: bool = False
    demotion_factor: float = 1.0


def _asset_criticality(incident: Incident, topology: TopologyGraph | None) -> float:
    """Asset criticality of the incident's root-cause entity (default 0.5)."""
    if incident.root_cause_entity is None:
        return 0.5
    if topology is not None:
        return topology.criticality(incident.root_cause_entity.entity_id)
    return 0.5


def score_incident(
    incident: Incident,
    *,
    topology: TopologyGraph | None = None,
    weights: RiskWeights | None = None,
    calibrator: RiskCalibrator | None = None,
    suppressor: FlapSuppressor | None = None,
    now: datetime | None = None,
) -> PrioritizedIncident:
    """Compute the full priority for a single incident (factors → calibrate → demote)."""
    crit = _asset_criticality(incident, topology)
    factors = compute_risk_factors(incident, asset_criticality=crit, weights=weights)
    base = geometric_mean_score(factors)

    # calibrate the combined score → honest probability (identity if unfitted).
    if calibrator is not None:
        calibrated = float(calibrator.transform(base))  # type: ignore[arg-type]
    else:
        calibrated = base

    # flap suppression: demote chronically-flapping root-cause entities.
    penalty = 0.0
    suppressed = False
    demote = 1.0
    if suppressor is not None and incident.root_cause_entity is not None:
        ent = incident.root_cause_entity.entity_id
        penalty = suppressor.penalty_of(ent, now=now)
        suppressed = suppressor.is_suppressed(ent, now=now)
        demote = suppressor.demotion_factor(ent, now=now)

    final = round(calibrated * demote, 6)

    return PrioritizedIncident(
        incident=incident,
        priority_score=base,
        calibrated_priority=final,
        factors=factors,
        flap_penalty=penalty,
        suppressed=suppressed,
        demotion_factor=demote,
    )


def severity_for(
    score: float,
    *,
    p1: float = DEFAULT_P1_THRESHOLD,
    p2: float = DEFAULT_P2_THRESHOLD,
) -> Severity:
    """Bucket a [0,1] calibrated priority into P1 / P2 / P3 / INFO."""
    if score >= p1:
        return Severity.P1
    if score >= p2:
        return Severity.P2
    if score > 0.0:
        return Severity.P3
    return Severity.INFO


def prioritize_incidents(
    incidents: Sequence[Incident],
    *,
    topology: TopologyGraph | None = None,
    weights: RiskWeights | None = None,
    calibrator: RiskCalibrator | None = None,
    suppressor: FlapSuppressor | None = None,
    p1_threshold: float = DEFAULT_P1_THRESHOLD,
    p2_threshold: float = DEFAULT_P2_THRESHOLD,
    now: datetime | None = None,
    mutate: bool = True,
) -> list[PrioritizedIncident]:
    """Return incidents ORDERED by calibrated priority (highest first), with severity.

    When ``mutate`` is true (default) each incident's ``severity`` is set to its
    bucket and its ``risk.calibrated_confidence`` is updated to the prioritisation
    score, so the returned ``Incident`` objects are self-consistent for the API/UI.
    """
    scored = [
        score_incident(
            inc,
            topology=topology,
            weights=weights,
            calibrator=calibrator,
            suppressor=suppressor,
            now=now,
        )
        for inc in incidents
    ]

    # primary sort: calibrated priority; tie-break by imminence then blast size.
    def _key(pi: PrioritizedIncident):
        minutes = pi.factors.minutes_to_impact
        imminence = -(minutes if minutes is not None else 1e9)
        blast = pi.factors.blast_radius
        return (pi.calibrated_priority, imminence, blast)

    scored.sort(key=_key, reverse=True)

    for pi in scored:
        sev = severity_for(pi.calibrated_priority, p1=p1_threshold, p2=p2_threshold)
        if mutate:
            pi.incident.severity = sev
            # keep the fused risk's calibrated confidence in step with the queue.
            pi.incident.risk.calibrated_confidence = round(
                min(1.0, max(0.0, pi.calibrated_priority)), 4
            )
    return scored


def triage_queue(
    incidents: Sequence[Incident],
    **kwargs,
) -> list[Incident]:
    """Convenience: return just the ordered list of :class:`Incident` objects."""
    return [pi.incident for pi in prioritize_incidents(incidents, **kwargs)]


__all__ = [
    "PrioritizedIncident",
    "score_incident",
    "prioritize_incidents",
    "triage_queue",
    "severity_for",
    "DEFAULT_P1_THRESHOLD",
    "DEFAULT_P2_THRESHOLD",
]
