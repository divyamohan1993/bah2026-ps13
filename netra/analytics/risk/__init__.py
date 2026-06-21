"""netra.analytics.risk — calibrated alert prioritisation -> Incident (WS4).

Turns fused risk + correlation + blast radius into a ranked, deduplicated,
severity-bucketed ``netra.contracts.Incident`` triage queue using the product-form
risk ``AnomalyConfidence x TimeToImpactUrgency x BlastRadius x AssetCriticality``
(product so a zero factor suppresses false urgency), Platt-calibrated, with
BGP-style flap-penalty suppression to cut alert fatigue (Objective 4).

Builder: ``score.py`` (product-form risk), ``calibrate.py`` (Platt/isotonic),
``suppress.py`` (flap penalty + decay), ``prioritize.py`` (-> ordered Incident
queue). Report reliability diagram + Brier/ECE and the alarm compression ratio as
evidence.
"""

from __future__ import annotations

from .calibrate import (
    RiskCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)
from .prioritize import (
    DEFAULT_P1_THRESHOLD,
    DEFAULT_P2_THRESHOLD,
    PrioritizedIncident,
    prioritize_incidents,
    score_incident,
    severity_for,
    triage_queue,
)
from .score import (
    RiskFactors,
    RiskWeights,
    anomaly_confidence_factor,
    compute_risk_factors,
    geometric_mean_score,
    time_to_impact_urgency,
)
from .suppress import FlapSuppressor

__all__ = [
    # score
    "RiskWeights",
    "RiskFactors",
    "compute_risk_factors",
    "time_to_impact_urgency",
    "anomaly_confidence_factor",
    "geometric_mean_score",
    # calibrate
    "RiskCalibrator",
    "brier_score",
    "expected_calibration_error",
    "reliability_diagram",
    # suppress
    "FlapSuppressor",
    # prioritize
    "PrioritizedIncident",
    "score_incident",
    "prioritize_incidents",
    "triage_queue",
    "severity_for",
    "DEFAULT_P1_THRESHOLD",
    "DEFAULT_P2_THRESHOLD",
]
