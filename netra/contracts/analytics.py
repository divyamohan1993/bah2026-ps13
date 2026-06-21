"""Predictive analytics contracts — the outputs of the 30+ method ensemble.

These models carry the engine's answers to the operational questions:

  * :class:`Forecast`           — trajectory + uncertainty band (Q1 "and when").
  * :class:`AnomalyScore`       — per-method, normalised anomaly evidence.
  * :class:`TimeToImpact`       — calibrated lead time to an SLA threshold (Q1).
  * :class:`ContributingSignal` — SHAP-ranked "why" attribution (Q2).
  * :class:`FusedRisk`          — score-fusion + weighted-agreement + calibration.

The forecasting workstream produces ``Forecast``; the anomaly workstream
produces ``AnomalyScore``; the fusion workstream consumes both and produces
``TimeToImpact`` + ``FusedRisk``; the explain workstream produces
``ContributingSignal`` lists. All are joined on ``EntityRef.entity_id``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import EntityRef, NetraModel
from .enums import DetectorFamily, Direction, IssueType


class QuantilePoint(NetraModel):
    """A single forecast horizon step: predicted value + quantile bounds."""

    horizon_seconds: float = Field(
        ..., ge=0, description="Seconds ahead of forecast origin for this step."
    )
    predicted: float = Field(..., description="Point (median/mean) prediction.")
    lower: float | None = Field(
        default=None, description="Lower quantile bound (e.g. p10)."
    )
    upper: float | None = Field(
        default=None, description="Upper quantile bound (e.g. p90)."
    )


class Forecast(NetraModel):
    """A trajectory forecast for one entity+metric over a horizon.

    The basis for time-to-impact: the predicted curve plus its (preferably
    conformal-calibrated) uncertainty band, from which the first threshold
    crossing is read off. ``method`` and ``family`` identify the producing model
    so the ensemble can weight by recent backtest accuracy and report
    cross-model agreement as a confidence signal.
    """

    entity: EntityRef = Field(..., description="Entity forecast is about.")
    metric: str = Field(
        ..., description="Metric being forecast (prefer a MetricName value)."
    )
    origin: datetime = Field(..., description="Forecast origin (now) timestamp.")
    horizon_seconds: float = Field(
        ..., gt=0, description="Total forecast horizon length in seconds."
    )
    points: list[QuantilePoint] = Field(
        ..., min_length=1, description="Per-step predictions + bounds."
    )
    method: str = Field(
        ...,
        description="Producing model id.",
        examples=["lightgbm_global", "autoets", "kalman", "chronos_bolt_base"],
    )
    family: DetectorFamily = Field(
        default=DetectorFamily.FORECAST, description="Method family."
    )
    quantile_lower: float = Field(
        default=0.1, ge=0, le=1, description="Lower quantile level of `lower`."
    )
    quantile_upper: float = Field(
        default=0.9, ge=0, le=1, description="Upper quantile level of `upper`."
    )
    backtest_mase: float | None = Field(
        default=None,
        ge=0,
        description="Recent rolling MASE (for inverse-error ensemble weighting).",
    )


class AnomalyScore(NetraModel):
    """One detector's verdict on one entity+metric at one instant.

    Detectors output incomparable raw scores; ``normalized_score`` is the [0,1]
    value (z/min-max/unification over a rolling reference, or an EVT/SPOT tail
    probability) used by fusion so methods of different families are comparable.
    ``is_anomaly`` is the detector's own decision (often via an EVT-derived,
    risk-controlled threshold rather than a hand-set one).
    """

    entity: EntityRef = Field(..., description="Entity scored.")
    metric: str = Field(..., description="Metric or feature scored.")
    timestamp: datetime = Field(..., description="UTC instant scored.")
    method: str = Field(
        ...,
        description="Detector id.",
        examples=["half_space_trees", "isolation_forest", "copod", "tranad",
                  "page_hinkley", "spectral_residual", "dominant"],
    )
    family: DetectorFamily = Field(..., description="Detector family.")
    score: float = Field(..., description="Raw detector score (method-specific).")
    normalized_score: float = Field(
        ..., ge=0, le=1, description="[0,1] comparable score for fusion."
    )
    is_anomaly: bool = Field(
        ..., description="Detector's decision after its (EVT/SPOT) threshold."
    )
    threshold: float | None = Field(
        default=None, description="The (often EVT-derived) decision threshold used."
    )


class TimeToImpact(NetraModel):
    """Estimated lead time until a metric crosses an SLA/security threshold (Q1).

    The headline 'what fails next AND WHEN' number. Produced either by
    forecast-trajectory threshold-crossing (first time the predicted band crosses
    ``threshold``) or by a survival/hazard model. ``eta_seconds=None`` means no
    crossing is predicted within the horizon (healthy).
    """

    entity: EntityRef = Field(..., description="Entity at risk.")
    metric: str = Field(..., description="Metric whose threshold may be crossed.")
    origin: datetime = Field(..., description="Reference time the ETA counts from.")
    threshold: float = Field(..., description="SLA/security threshold value.")
    threshold_direction: Direction = Field(
        default=Direction.INCREASES_RISK,
        description="Whether crossing ABOVE or BELOW the threshold is the breach.",
    )
    eta_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Predicted seconds to crossing; None = no crossing in horizon.",
    )
    eta_lower_seconds: float | None = Field(
        default=None, ge=0, description="Lower CI bound on the ETA."
    )
    eta_upper_seconds: float | None = Field(
        default=None, ge=0, description="Upper CI bound on the ETA."
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Calibrated confidence in this ETA."
    )
    method: str = Field(
        default="trajectory_crossing",
        description="Estimator.",
        examples=["trajectory_crossing", "theil_sen_extrapolation", "cox_survival"],
    )


class ContributingSignal(NetraModel):
    """A single human-readable 'why' factor behind an elevated risk (Q2).

    Bridges the SHAP/attribution layer to the copilot: each signal pairs a
    machine attribution (``shap_value`` / ``direction``) with a one-line
    ``human_explanation`` the LLM can quote *grounded*, never invent. The copilot
    schema requires these so Q2 ("which signals contributed?") is always
    answered from real attributions.
    """

    signal: str = Field(
        ...,
        description="Signal/feature name.",
        examples=["if_util_pct:eth1", "bgp_flap_penalty", "tunnel_jitter_ms"],
    )
    shap_value: float | None = Field(
        default=None,
        description="Signed SHAP/attribution contribution to the risk score.",
    )
    direction: Direction = Field(
        ..., description="Whether this signal pushes risk up or down."
    )
    observation: str | None = Field(
        default=None,
        description="The concrete observed value/trend.",
        examples=["utilisation rising 4%/min, now 78%", "12 flaps in 5 min"],
    )
    human_explanation: str = Field(
        ...,
        description="One-line operator-readable explanation of the signal.",
        examples=["Hub-spoke uplink utilisation is trending toward saturation."],
    )
    entity: EntityRef | None = Field(
        default=None, description="Entity this signal pertains to, if specific."
    )


class MethodWeight(NetraModel):
    """A (method, family, weight) tuple contributing to a fused risk score."""

    method: str = Field(..., description="Detector/forecaster id.")
    family: DetectorFamily = Field(..., description="Method family.")
    normalized_score: float = Field(
        ..., ge=0, le=1, description="Method's normalised score that was fused."
    )
    weight: float = Field(
        ..., ge=0, description="Fusion weight applied to this method's score."
    )


class FusedRisk(NetraModel):
    """Fused, calibrated risk for an entity — the cross-verified verdict.

    Combines many detectors (score-normalisation + weighted-agreement across
    *independent* families) into one ``risk_score`` in [0,1], records exactly
    which methods contributed and with what weight (auditability), and exposes a
    ``calibrated_confidence`` (Platt/isotonic over the labelled fault scenarios)
    so the copilot's stated confidence is honest. ``agreement`` (fraction of
    independent families firing) is the robustness signal the 30+ method
    ensemble buys.
    """

    entity: EntityRef = Field(..., description="Entity the risk is about.")
    timestamp: datetime = Field(..., description="UTC instant of the assessment.")
    risk_score: float = Field(
        ..., ge=0, le=1, description="Fused risk in [0,1] (1 = certain failure)."
    )
    calibrated_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calibrated confidence the risk is real (reliability-diagram "
        "validated).",
    )
    predicted_issue: IssueType = Field(
        default=IssueType.NONE,
        description="Most likely fault class implied by the firing methods.",
    )
    agreement: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of independent detector families in agreement.",
    )
    contributing_methods: list[MethodWeight] = Field(
        default_factory=list,
        description="Methods fused into this score, with weights (auditable).",
    )
    time_to_impact: TimeToImpact | None = Field(
        default=None, description="Associated lead-time estimate, if any (Q1)."
    )

    @model_validator(mode="after")
    def _confidence_not_exceed_when_no_methods(self) -> FusedRisk:
        # A non-zero risk with zero contributing methods is an integration bug:
        # fusion must record provenance for anything it scores as risky.
        if self.risk_score > 0 and not self.contributing_methods:
            raise ValueError(
                "FusedRisk.risk_score > 0 requires at least one contributing method"
            )
        return self


__all__ = [
    "QuantilePoint",
    "Forecast",
    "AnomalyScore",
    "TimeToImpact",
    "ContributingSignal",
    "MethodWeight",
    "FusedRisk",
]
