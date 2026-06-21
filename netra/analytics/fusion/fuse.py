"""Risk fusion — many detectors -> one calibrated ``FusedRisk`` (#67-#69).

THE core fusion deliverable (research 04 §10-§13). No single detector is
trustworthy alone: each family has blind spots and its own false-alarm profile.
:class:`RiskFuser` combines, *for one entity at one instant*, the bank's
:class:`~netra.contracts.AnomalyScore`\\ s (and optionally a
:class:`~netra.contracts.Forecast` / :class:`~netra.contracts.TimeToImpact` and
the :class:`~netra.contracts.FeatureVector`) into a single
:class:`~netra.contracts.FusedRisk` that records exactly *which* methods fired and
with what weight (auditability) and exposes an *honest* calibrated confidence.

Fusion math (documented so it is auditable, not a black box)
------------------------------------------------------------
Let detector *i* report a normalised score ``s_i ∈ [0,1]`` in family ``f(i)``.

1. **EVT-adaptive evidence.** Rather than a fixed cutoff, each score is judged
   against an Extreme-Value-Theory (SPOT/DSPOT, :mod:`~.evt`) tail fit on the
   *population of this tick's scores*: a score in the GPD tail is "extreme" and
   earns an evidence multiplier ``e_i = 1 + β`` (β default 0.5); a score the
   detector itself flagged (``is_anomaly``) also earns it. This replaces hand-set
   thresholds with one risk knob ``q`` (research 04 §11).

2. **Per-method weight.** ``w_i = w_family(f(i)) · e_i``, where ``w_family``
   down-weights families that are *correlated* (so 6 statistical detectors don't
   out-vote one independent matrix-profile signal): each family's base weight is
   shared across its members (``w_family = prior_f / n_f``). This is the
   "independent-family" weighting the 30+-method ensemble is designed around.

3. **Weighted score.** The fused level is the weighted mean
   ``S = Σ w_i s_i / Σ w_i`` — a soft OR that rises when strong, independent
   evidence agrees.

4. **Cross-family AGREEMENT.** ``A`` = the (weighted) fraction of *independent
   families* whose evidence is elevated. Agreement is the robustness signal the
   ensemble buys: when many independent families concur the result is trustworthy;
   when a lone family fires it is "needs review". Agreement *sharpens* the score
   toward its extreme (``S' = S·(0.5 + 0.5·A)`` damped when ``A`` is low) and is
   the dominant term in confidence.

5. **Risk + confidence.** ``risk_score`` = optionally the calibrator applied to
   ``S'`` (else ``S'`` itself). ``calibrated_confidence`` blends agreement, the
   peak single-method evidence, and the calibrator's certainty — so a single weak
   detector yields LOW confidence even at a moderate score, while several agreeing
   independent families yield HIGH confidence (the property tests assert this).

Provenance: every contributing detector becomes a
:class:`~netra.contracts.MethodWeight` (method, family, its ``normalized_score``,
its fusion weight). A forecaster's agreement, if supplied, is folded in as
another independent voice and the earliest/best
:class:`~netra.contracts.TimeToImpact` is attached so downstream gets "and WHEN".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

import numpy as np

from netra.contracts import (
    AnomalyScore,
    DetectorFamily,
    EntityRef,
    FeatureVector,
    Forecast,
    FusedRisk,
    IssueType,
    MethodWeight,
    TimeToImpact,
)

from .calibrate import ProbabilityCalibrator
from .evt import ScoreStreamThresholder

# ---------------------------------------------------------------------------
# Family priors — relative trust for an *independent* family (normalised across
# the family's members so correlated detectors don't dominate the vote).
# Values are heuristics from research 04 §10; tune via backtesting offline.
# ---------------------------------------------------------------------------

_FAMILY_PRIOR: dict[DetectorFamily, float] = {
    DetectorFamily.FORECAST: 1.0,
    DetectorFamily.FORECAST_RESIDUAL: 1.1,
    DetectorFamily.STATISTICAL: 0.9,
    DetectorFamily.ML_UNSUPERVISED: 1.0,
    DetectorFamily.DEEP: 1.0,
    DetectorFamily.CHANGE_POINT: 1.1,
    DetectorFamily.MATRIX_PROFILE: 1.0,
    DetectorFamily.GRAPH: 1.0,
    DetectorFamily.ROUTING: 1.1,
    DetectorFamily.SURVIVAL: 1.0,
}
_DEFAULT_PRIOR = 1.0

# Map the firing-method families to the most likely fault class. Coarse; the RCA
# layer refines it. Only used to set FusedRisk.predicted_issue as a hint.
_FAMILY_ISSUE_HINT: dict[DetectorFamily, IssueType] = {
    DetectorFamily.CHANGE_POINT: IssueType.LATENCY_DRIFT,
    DetectorFamily.MATRIX_PROFILE: IssueType.LATENCY_DRIFT,
    DetectorFamily.ROUTING: IssueType.BGP_ROUTE_FLAP,
    DetectorFamily.FORECAST: IssueType.INTERFACE_CONGESTION,
}


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class RiskFuser:
    """Fuse detector + forecaster evidence into one calibrated ``FusedRisk``.

    Parameters
    ----------
    q:
        EVT tail probability (false-alarm risk) for the per-tick adaptive
        threshold over the score population (smaller ⇒ stricter "extreme" test).
    evidence_boost:
        Multiplier increment ``β`` applied to a score judged extreme by EVT or
        self-flagged by its detector (``e_i = 1 + β``). 0 disables boosting.
    elevated_score:
        Floor a normalised score must reach to count as an "elevated" family vote
        in the agreement fraction (independent of the EVT tail, which can be empty
        on a tiny score set).
    calibrator:
        Optional pre-fit :class:`~netra.analytics.fusion.calibrate.ProbabilityCalibrator`.
        When supplied (and fitted) the raw fused score is mapped through it to a
        calibrated probability; otherwise the raw agreement-sharpened score is used.

    The fuser is stateless per call: pass it the evidence for one (entity, time)
    and it returns the ``FusedRisk``. Construct one and reuse it across ticks.
    """

    method = "weighted_agreement_fusion"

    def __init__(
        self,
        *,
        q: float = 1e-3,
        evidence_boost: float = 0.5,
        elevated_score: float = 0.6,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> None:
        self.q = float(q)
        self.evidence_boost = float(evidence_boost)
        self.elevated_score = float(elevated_score)
        self.calibrator = calibrator

    # -- public API ---------------------------------------------------------

    def fuse(
        self,
        scores: Iterable[AnomalyScore],
        *,
        entity: EntityRef | None = None,
        timestamp: datetime | None = None,
        forecast: Forecast | None = None,
        time_to_impact: TimeToImpact | None = None,
        forecast_agreement: float | None = None,
        features: FeatureVector | None = None,
        predicted_issue: IssueType | None = None,
    ) -> FusedRisk:
        """Combine the supplied evidence into a :class:`FusedRisk`.

        Parameters
        ----------
        scores:
            The detector bank's :class:`AnomalyScore`\\ s for this entity+instant.
        entity, timestamp:
            Override the entity/instant (default: taken from the first score, or
            now). Required if ``scores`` is empty.
        forecast:
            Optional ensemble :class:`Forecast`; its ``backtest_mase`` and (via
            ``forecast_agreement``) cross-model agreement feed the fusion as an
            extra independent voice.
        time_to_impact:
            Optional :class:`TimeToImpact` to attach (the "and WHEN"). If several
            are available pass the best/earliest (see :meth:`pick_time_to_impact`).
        forecast_agreement:
            Cross-model agreement in [0,1] from the forecasting ensemble; folded
            into the overall agreement/confidence.
        features:
            Optional :class:`FeatureVector` (carried for context; ``triggered_drift``
            adds weak corroborating evidence to the agreement count).
        predicted_issue:
            Override the inferred fault class; otherwise inferred from the
            strongest firing family.
        """
        score_list = [s for s in scores if s is not None]

        ent = entity or (score_list[0].entity if score_list else None)
        if ent is None:
            raise ValueError("RiskFuser.fuse needs an entity (no scores supplied)")
        ts = timestamp or (score_list[0].timestamp if score_list else _utcnow())

        # No evidence at all -> a valid, zero-risk FusedRisk (contract allows it).
        if not score_list:
            return FusedRisk(
                entity=ent, timestamp=ts,
                risk_score=0.0, calibrated_confidence=0.0,
                predicted_issue=IssueType.NONE, agreement=0.0,
                contributing_methods=[], time_to_impact=time_to_impact,
            )

        norm = np.array([float(np.clip(s.normalized_score, 0.0, 1.0))
                         for s in score_list], dtype=float)
        flagged = np.array([bool(s.is_anomaly) for s in score_list], dtype=bool)
        families = [s.family for s in score_list]

        # (1) EVT-adaptive evidence: which scores are extreme for *this* tick?
        extreme = self._evt_extreme(norm)
        evidence_mult = np.where(extreme | flagged, 1.0 + self.evidence_boost, 1.0)

        # (2) per-method weights: family prior shared across its members × evidence
        fam_counts: dict[DetectorFamily, int] = {}
        for f in families:
            fam_counts[f] = fam_counts.get(f, 0) + 1
        base_w = np.array(
            [_FAMILY_PRIOR.get(f, _DEFAULT_PRIOR) / fam_counts[f] for f in families],
            dtype=float,
        )
        weights = base_w * evidence_mult

        # (3) weighted score (soft OR of agreeing evidence)
        wsum = float(weights.sum())
        fused = float((weights * norm).sum() / wsum) if wsum > 0 else float(norm.mean())

        # (4) cross-family agreement: weighted fraction of INDEPENDENT families
        #     whose evidence is elevated. Forecaster + drift features add voices.
        agreement = self._family_agreement(
            norm, families, forecast_agreement=forecast_agreement,
            features=features,
        )

        # agreement sharpens the score: high agreement pushes toward the extreme,
        # low agreement damps it (a lone detector is not yet "certain failure").
        sharpened = float(np.clip(fused * (0.5 + 0.5 * agreement), 0.0, 1.0))

        # (5) risk + confidence
        if self.calibrator is not None and self.calibrator.is_fitted:
            risk_score = float(np.clip(self.calibrator.transform(sharpened), 0.0, 1.0))
        else:
            risk_score = sharpened
        confidence = self._confidence(
            norm, weights, agreement, fused,
            forecast_agreement=forecast_agreement,
        )

        # provenance: one MethodWeight per contributing detector
        contributing = [
            MethodWeight(
                method=s.method, family=s.family,
                normalized_score=float(np.clip(s.normalized_score, 0.0, 1.0)),
                weight=float(w),
            )
            for s, w in zip(score_list, weights)
        ]

        issue = predicted_issue or self._infer_issue(norm, families, risk_score)

        # ``score_list`` is non-empty here, so ``contributing`` is always
        # populated — satisfying the contract rule that a risk_score>0 must carry
        # provenance (we keep provenance even at zero risk for auditability).
        return FusedRisk(
            entity=ent,
            timestamp=ts,
            risk_score=risk_score,
            calibrated_confidence=confidence,
            predicted_issue=issue,
            agreement=float(np.clip(agreement, 0.0, 1.0)),
            contributing_methods=contributing,
            time_to_impact=time_to_impact,
        )

    # convenience: alias so callers can use the fuser as a callable
    __call__ = fuse

    @staticmethod
    def pick_time_to_impact(
        ttis: Sequence[TimeToImpact | None],
    ) -> TimeToImpact | None:
        """Pick the best (earliest credible) :class:`TimeToImpact` of several.

        Prefers the soonest predicted crossing (smallest finite ``eta_seconds``),
        breaking ties by higher confidence; a ``None`` ETA (no crossing) only wins
        if nothing predicts a crossing, in which case the most confident "healthy"
        verdict is returned. This is how a caller collapses per-estimator TTIs
        (trajectory / Theil-Sen / survival) into the one attached to the risk.
        """
        cand = [t for t in ttis if t is not None]
        if not cand:
            return None
        crossing = [t for t in cand if t.eta_seconds is not None]
        if crossing:
            return min(crossing, key=lambda t: (t.eta_seconds, -t.confidence))
        return max(cand, key=lambda t: t.confidence)

    # -- internals ----------------------------------------------------------

    def _evt_extreme(self, norm: np.ndarray) -> np.ndarray:
        """Boolean mask of scores in the EVT (SPOT/DSPOT) tail for this tick.

        Fits an adaptive tail on the *population of this instant's normalised
        scores* (warm up on the lower body, test each score) so "extreme" is
        defined relative to the current detector consensus rather than a hand-set
        cutoff. Degrades to a high-quantile rule when there are too few scores to
        fit a GPD tail.
        """
        n = norm.size
        if n == 0:
            return np.zeros(0, dtype=bool)
        if n < 6:
            # too few to fit a tail; call the top scores extreme via a robust rule
            med = float(np.median(norm))
            mad = float(np.median(np.abs(norm - med))) * 1.4826
            cut = med + 2.5 * mad if mad > 1e-9 else max(self.elevated_score, med)
            return norm >= max(cut, self.elevated_score * 0.5)
        thr = ScoreStreamThresholder(q=self.q, depth=max(4, n // 3))
        # warm up on the benign body (exclude the top quartile so true extremes
        # aren't absorbed into the normal tail), then test every score.
        body = np.sort(norm)[: max(2, int(0.75 * n))]
        thr.warmup(body)
        out = np.zeros(n, dtype=bool)
        for i, x in enumerate(norm):
            is_ext, _ = thr.update(float(x))
            out[i] = bool(is_ext)
        # ensure the single largest, clearly-elevated score is considered extreme
        if not out.any() and float(norm.max()) >= self.elevated_score:
            out[int(np.argmax(norm))] = True
        return out

    def _family_agreement(
        self,
        norm: np.ndarray,
        families: list[DetectorFamily],
        *,
        forecast_agreement: float | None,
        features: FeatureVector | None,
    ) -> float:
        """Cross-family agreement in [0,1] = elevated-fraction × concurrence-breadth.

        Collapses members to their family's *max* normalised score (so a family
        votes once), then combines two factors so agreement means *cross-family
        concurrence*, not a single family agreeing with itself:

          * **elevated fraction** — share of independent voices clearing
            ``elevated_score`` (penalises dissenting/quiet families);
          * **breadth** — how *many* independent voices are elevated, saturating
            via ``1 - exp(-k·n_elevated)``. A lone elevated family is therefore
            capped well below 1.0; concurrence across ≥3 independent families
            approaches full agreement. This is the robustness the 30+-method
            ensemble buys — one detector is "needs review", many independent
            families agreeing is "trustworthy".

        A supplied forecaster (high cross-model agreement) and any
        ``triggered_drift`` features count as extra independent voices. Returns
        [0,1].
        """
        per_family: dict[DetectorFamily, float] = {}
        for f, s in zip(families, norm):
            per_family[f] = max(per_family.get(f, 0.0), float(s))

        votes = list(per_family.values())
        # forecaster as an independent voice: treat its agreement as a "score"
        if forecast_agreement is not None:
            votes.append(float(np.clip(forecast_agreement, 0.0, 1.0)))
        # drift features corroborate (each fired drift detector is weak evidence)
        if features is not None and getattr(features, "triggered_drift", None):
            votes.append(min(1.0, 0.5 + 0.1 * len(features.triggered_drift)))

        if not votes:
            return 0.0
        n_elevated = sum(1 for v in votes if v >= self.elevated_score)
        if n_elevated == 0:
            return 0.0
        fraction = n_elevated / len(votes)
        # breadth saturates with the COUNT of agreeing independent voices:
        # n=1 -> ~0.55, n=2 -> ~0.80, n=3 -> ~0.91, n>=4 -> ~0.95+.
        breadth = 1.0 - np.exp(-0.8 * n_elevated)
        return float(np.clip(fraction * breadth, 0.0, 1.0))

    def _confidence(
        self,
        norm: np.ndarray,
        weights: np.ndarray,
        agreement: float,
        fused: float,
        *,
        forecast_agreement: float | None,
    ) -> float:
        """Calibrated confidence in [0,1].

        Blends three honest signals so confidence tracks *evidence quality*, not
        just score magnitude:

          * **agreement** (dominant) — independent families concurring;
          * **peak evidence** — the single strongest normalised score (a clearly
            extreme detector is informative even before others corroborate);
          * **calibrator certainty** — if a fitted calibrator is present, how far
            its mapped probability is from the indecisive 0.5.

        Consequently a lone weak detector → low confidence; several agreeing
        independent families → high confidence (asserted by the property tests).
        """
        if norm.size == 0:
            return 0.0
        peak = float(norm.max())
        # base: agreement-led, with peak evidence as support
        conf = 0.65 * float(np.clip(agreement, 0.0, 1.0)) + 0.35 * peak
        # a single contributing family caps confidence (no cross-verification yet)
        if np.count_nonzero(weights > 0) <= 1:
            conf = min(conf, 0.5 * peak + 0.1)
        # calibrator certainty, when available
        if self.calibrator is not None and self.calibrator.is_fitted:
            p = float(self.calibrator.transform(fused))
            conf = 0.7 * conf + 0.3 * (2.0 * abs(p - 0.5))
        if forecast_agreement is not None:
            conf = 0.85 * conf + 0.15 * float(np.clip(forecast_agreement, 0.0, 1.0))
        return float(np.clip(conf, 0.0, 1.0))

    @staticmethod
    def _infer_issue(norm: np.ndarray, families: list[DetectorFamily],
                     risk_score: float) -> IssueType:
        """Best-guess fault class from the strongest-firing family (a hint only)."""
        if risk_score <= 0.0 or norm.size == 0:
            return IssueType.NONE
        # the family carrying the single highest score sets the hint
        top_family = families[int(np.argmax(norm))]
        return _FAMILY_ISSUE_HINT.get(top_family, IssueType.NONE)


__all__ = ["RiskFuser"]
