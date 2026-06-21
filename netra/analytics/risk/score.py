"""Product-form risk score for an incident (research 07 A2.1).

We adopt a Risk-Based Alerting model: each *incident* (post-correlation, not each
raw alarm) gets a calculated risk from multiple weighted factors so operators
spend time on the few incidents most likely to cause real harm.

    Risk = AnomalyConfidence × TimeToImpactUrgency × BlastRadius × AssetCriticality

A **product** (geometric) form is used rather than a sum so that a near-zero
factor (e.g. zero blast radius) correctly *suppresses* the score — avoiding the
"high anomaly score but affects nothing" false urgency. Each factor is normalised
to [0,1]:

  * **AnomalyConfidence** — the calibrated ``FusedRisk.calibrated_confidence``
    (honest P(failure)); when absent, the raw ``risk_score``.
  * **TimeToImpactUrgency** — ``1 / (1 + minutes_to_impact)``; sooner ⇒ higher.
    A ``None`` ETA (no predicted crossing) yields a low-but-nonzero urgency.
  * **BlastRadius** — normalised #affected sites/SLAs/flows (from the
    deterministic graph computation).
  * **AssetCriticality** — business/role weight (DC-PE > hub-PE > branch-CE).

Returns a :class:`RiskFactors` breakdown (auditable) plus the combined score.
``calibrate.py`` maps the raw combined score to a calibrated probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from netra.contracts import BlastRadius, FusedRisk, Incident, TimeToImpact

from netra.analytics.correlation.blast_radius import blast_urgency_factor

# Per-factor exponents let the operator tune the relative pull of each term while
# keeping the product form. All default to 1.0 (equal weighting). A factor with a
# small floor (EPS) prevents a single missing signal from hard-zeroing risk.
EPS = 0.02


@dataclass
class RiskWeights:
    """Exponent weights on each product factor (all 1.0 = equal weighting)."""

    anomaly: float = 1.0
    urgency: float = 1.0
    blast: float = 1.0
    criticality: float = 1.0


@dataclass
class RiskFactors:
    """The four [0,1] factors plus the combined raw product score (auditable)."""

    anomaly_confidence: float
    time_to_impact_urgency: float
    blast_radius: float
    asset_criticality: float
    raw_score: float
    minutes_to_impact: float | None = None
    detail: dict[str, float] = field(default_factory=dict)


def time_to_impact_urgency(tti: TimeToImpact | None) -> tuple[float, float | None]:
    """Map a :class:`TimeToImpact` to a [0,1] urgency + the minutes-to-impact.

    ``urgency = 1 / (1 + minutes)``: 0 min ⇒ 1.0, 5 min ⇒ ~0.17, 60 min ⇒ ~0.016.
    A ``None`` ETA (no crossing predicted in horizon) returns a small floor (0.05)
    so a confidently-anomalous entity with no ETA still ranks above noise but below
    an imminent breach.
    """
    if tti is None or tti.eta_seconds is None:
        return 0.05, None
    minutes = max(0.0, tti.eta_seconds / 60.0)
    urgency = 1.0 / (1.0 + minutes)
    return round(urgency, 6), round(minutes, 3)


def anomaly_confidence_factor(risk: FusedRisk) -> float:
    """The calibrated-confidence factor, blended a little with raw risk score.

    Uses ``calibrated_confidence`` as the primary honest-probability term but
    multiplies in a mild dependence on ``risk_score`` so a high-confidence *low*
    risk does not rank like a high-confidence *high* risk.
    """
    conf = float(risk.calibrated_confidence)
    rs = float(risk.risk_score)
    return round(max(EPS, 0.5 * conf + 0.5 * (conf * rs) ** 0.5 if rs > 0 else conf * 0.5), 6)


def compute_risk_factors(
    incident: Incident,
    *,
    asset_criticality: float,
    weights: RiskWeights | None = None,
    blast: BlastRadius | None = None,
) -> RiskFactors:
    """Compute the four risk factors and their weighted product for an incident."""
    w = weights or RiskWeights()
    risk = incident.risk
    tti = risk.time_to_impact

    a = anomaly_confidence_factor(risk)
    u, minutes = time_to_impact_urgency(tti)
    b = blast_urgency_factor(blast if blast is not None else incident.blast_radius)
    c = float(max(EPS, min(1.0, asset_criticality)))

    # apply small floors so no factor can hard-zero the product.
    af, uf, bf, cf = max(a, EPS), max(u, EPS), max(b, EPS), max(c, EPS)
    raw = (
        (af ** w.anomaly)
        * (uf ** w.urgency)
        * (bf ** w.blast)
        * (cf ** w.criticality)
    )
    return RiskFactors(
        anomaly_confidence=round(a, 4),
        time_to_impact_urgency=round(u, 4),
        blast_radius=round(b, 4),
        asset_criticality=round(c, 4),
        raw_score=round(float(raw), 6),
        minutes_to_impact=minutes,
        detail={
            "weighted_anomaly": round(af ** w.anomaly, 4),
            "weighted_urgency": round(uf ** w.urgency, 4),
            "weighted_blast": round(bf ** w.blast, 4),
            "weighted_criticality": round(cf ** w.criticality, 4),
        },
    )


def geometric_mean_score(factors: RiskFactors) -> float:
    """The 4th-root of the product → a [0,1] score comparable across incidents.

    The raw product of four [0,1] numbers is heavily compressed toward 0; taking
    the geometric mean (n-th root) rescales it back to an interpretable [0,1]
    magnitude for display and as the pre-calibration score.
    """
    raw = max(factors.raw_score, 0.0)
    return round(raw ** 0.25, 6) if raw > 0 else 0.0


__all__ = [
    "RiskWeights",
    "RiskFactors",
    "compute_risk_factors",
    "time_to_impact_urgency",
    "anomaly_confidence_factor",
    "geometric_mean_score",
    "EPS",
]
