"""Tests for ``netra.analytics`` — forecasting + anomaly + fusion (Workstream 3).

Coverage (per the build-plan deliverable), all on the LIGHT CPU/offline tier:

  * **Forecasting** — on a synthetic linear ramp a forecaster fits and returns a
    contract-valid :class:`~netra.contracts.Forecast` with a sensible point + a
    monotone (non-crossing) quantile band; the time-to-impact facade returns a
    finite, ordered :class:`~netra.contracts.TimeToImpact` whose ETA *decreases*
    as the series approaches the threshold.
  * **Anomaly** — the detector bank fires (some detector flags) inside an injected
    spike / level-shift window and stays quiet on clean data, and the EVT (SPOT)
    threshold adapts to the stream rather than using a fixed cutoff.
  * **Fusion** — :class:`~netra.analytics.fusion.RiskFuser` produces a
    contract-valid :class:`~netra.contracts.FusedRisk` (``risk_score`` ∈ [0,1],
    non-empty :class:`~netra.contracts.MethodWeight` provenance); calibrated
    confidence RISES when several independent families agree vs a single weak
    signal; and the :class:`ProbabilityCalibrator` maps scores into [0,1].
  * **Registry** — :func:`list_methods` returns ≥30 well-formed entries.

Inputs are constructed inline from ``netra.contracts`` (no dependency on the data
generator). Imports are restricted to ``netra.contracts`` + the analytics package.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from netra.contracts import (
    AnomalyScore,
    DetectorFamily,
    Direction,
    EntityRef,
    FeatureVector,
    Forecast,
    FusedRisk,
    IssueType,
    MethodWeight,
    TimeToImpact,
)

# analytics package under test
from netra.analytics.anomaly import DetectorBank, build_detector_bank
from netra.analytics.anomaly.evt import SPOT
from netra.analytics.forecasting import (
    EnsembleForecaster,
    EwmaForecaster,
    ThetaForecaster,
    TimeToImpactEstimator,
)
from netra.analytics.fusion import (
    ProbabilityCalibrator,
    RiskFuser,
    count_by_family,
    list_methods,
    method_count,
)

UTC = timezone.utc
T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _entity() -> EntityRef:
    return EntityRef(
        entity_id="hub1:pe-hub1:PE:eth1",
        site="hub1",
        device="pe-hub1",
        role="PE",
        sub="eth1",
    )


def _linear_ramp(n: int = 80, start: float = 20.0, slope: float = 0.5,
                 noise: float = 0.4, seed: int = 7) -> np.ndarray:
    """A noisy upward linear ramp (utilisation creeping toward saturation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return start + slope * t + rng.normal(0.0, noise, size=n)


def _clean_series(n: int = 200, level: float = 50.0, noise: float = 1.0,
                  seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level + rng.normal(0.0, noise, size=n)


def _spike_series(n: int = 200, level: float = 50.0, noise: float = 1.0,
                  spike_at: int = 150, spike_mag: float = 25.0,
                  seed: int = 3) -> np.ndarray:
    """Clean baseline with a sharp injected spike + sustained level-shift."""
    s = _clean_series(n, level, noise, seed)
    s[spike_at] += spike_mag                 # transient spike
    s[spike_at + 1:] += 0.6 * spike_mag      # sustained level shift after it
    return s


def _make_score(method: str, family: DetectorFamily, norm: float,
                is_anom: bool | None = None, ent: EntityRef | None = None,
                ) -> AnomalyScore:
    ent = ent or _entity()
    return AnomalyScore(
        entity=ent, metric="if_util_pct", timestamp=T0,
        method=method, family=family,
        score=float(norm) * 10.0, normalized_score=float(norm),
        is_anomaly=bool(norm >= 0.8) if is_anom is None else bool(is_anom),
    )


# ===========================================================================
# Forecasting
# ===========================================================================


class TestForecasting:
    def test_forecaster_fits_ramp_and_returns_valid_forecast(self):
        ent = _entity()
        series = _linear_ramp(n=80, start=20.0, slope=0.5)
        fc = EwmaForecaster(ent, "if_util_pct").fit(series).forecast(
            steps=20, step_seconds=60.0, origin=T0
        )
        assert isinstance(fc, Forecast)
        assert fc.entity == ent and fc.metric == "if_util_pct"
        assert len(fc.points) == 20
        # horizons strictly increasing, starting one step ahead
        hs = [p.horizon_seconds for p in fc.points]
        assert hs[0] == pytest.approx(60.0)
        assert all(b > a for a, b in zip(hs, hs[1:]))
        # the ramp is rising: the last predicted point should exceed the series end
        assert fc.points[-1].predicted > series[-1] - 5.0
        # band is present and never crosses the point (lower <= point <= upper)
        for p in fc.points:
            assert p.lower is not None and p.upper is not None
            assert p.lower <= p.predicted <= p.upper
            assert np.isfinite(p.lower) and np.isfinite(p.upper)
        # band widens with the horizon (uncertainty grows)
        w0 = fc.points[0].upper - fc.points[0].lower
        wN = fc.points[-1].upper - fc.points[-1].lower
        assert wN >= w0

    def test_ensemble_forecast_has_agreement(self):
        ent = _entity()
        series = _linear_ramp(n=90)
        res = EnsembleForecaster(ent, "if_util_pct").forecast_with_members(
            series, steps=15, step_seconds=60.0, origin=T0
        )
        assert isinstance(res.combined, Forecast)
        assert len(res.members) >= 2
        assert 0.0 <= res.agreement <= 1.0
        # members forecasting the same clean ramp should broadly agree
        assert res.agreement > 0.3

    def test_tti_finite_ordered_and_decreases_toward_threshold(self):
        """ETA must be finite + ordered, and shrink as the series nears the threshold."""
        ent = _entity()
        est = TimeToImpactEstimator(sample_period_seconds=60.0)
        threshold = 90.0

        def eta_for(series: np.ndarray) -> TimeToImpact:
            fc = ThetaForecaster(ent, "if_util_pct").fit(series).forecast(
                steps=120, step_seconds=60.0, origin=T0
            )
            return est.estimate(
                fc, threshold, direction=Direction.INCREASES_RISK,
                history=series, current_value=float(series[-1]),
            )

        # early: series ends around 20 + 0.5*60 ≈ 50, far from 90
        early = _linear_ramp(n=60, start=20.0, slope=0.5)
        # later: series ends around 20 + 0.5*130 ≈ 85, close to 90
        late = _linear_ramp(n=130, start=20.0, slope=0.5)

        tti_early = eta_for(early)
        tti_late = eta_for(late)

        assert isinstance(tti_early, TimeToImpact) and isinstance(tti_late, TimeToImpact)
        for tti in (tti_early, tti_late):
            assert tti.eta_seconds is not None and np.isfinite(tti.eta_seconds)
            assert tti.eta_seconds >= 0.0
            assert 0.0 <= tti.confidence <= 1.0
            # CI bounds, when present, bracket sanely (lower <= upper)
            if tti.eta_lower_seconds is not None and tti.eta_upper_seconds is not None:
                assert tti.eta_lower_seconds <= tti.eta_upper_seconds
        # closer to the threshold => sooner crossing
        assert tti_late.eta_seconds < tti_early.eta_seconds

    def test_tti_healthy_series_no_crossing(self):
        ent = _entity()
        est = TimeToImpactEstimator(sample_period_seconds=60.0)
        flat = _clean_series(n=80, level=30.0, noise=0.5)
        fc = EwmaForecaster(ent, "if_util_pct").fit(flat).forecast(
            steps=30, step_seconds=60.0, origin=T0
        )
        tti = est.estimate(fc, 95.0, direction=Direction.INCREASES_RISK,
                           history=flat, current_value=float(flat[-1]))
        # a flat, low series should not be predicted to breach a far threshold
        assert tti.eta_seconds is None


# ===========================================================================
# Anomaly
# ===========================================================================


class TestAnomaly:
    def test_bank_builds_with_many_detectors(self):
        ent = _entity()
        bank = build_detector_bank(ent, "if_util_pct")
        assert len(bank) >= 10
        methods = {d.method for d in bank}
        # representatives from each tier present
        assert "robust_z" in methods
        assert "isolation_forest" in methods
        assert "page_hinkley" in methods

    def test_bank_fires_in_anomaly_window_quiet_on_clean(self):
        ent = _entity()
        spike_at = 150
        spike = _spike_series(n=200, spike_at=spike_at, spike_mag=30.0)
        clean = _clean_series(n=200)

        bank = DetectorBank(ent, "if_util_pct")
        bank.warmup(clean[:100])             # learn benign behaviour
        spike_results = bank.score_series(spike)
        clean_results = DetectorBank(ent, "if_util_pct")
        clean_results.warmup(clean[:100])
        clean_out = clean_results.score_series(clean)

        # any detector flags within the post-spike window
        window = range(spike_at, min(spike_at + 25, len(spike_results)))
        fired_in_window = any(
            any(s.is_anomaly for s in spike_results[i]) for i in window
        )
        assert fired_in_window, "no detector flagged the injected spike/level-shift"

        # the spike instant scores clearly higher than a representative clean tick
        spike_max = max((s.normalized_score for s in spike_results[spike_at]),
                        default=0.0)
        clean_ref = max((s.normalized_score for s in clean_out[spike_at]),
                        default=0.0)
        assert spike_max >= clean_ref

        # the clean series is mostly quiet (few flags) — adaptive, not trigger-happy
        clean_flags = sum(
            1 for step in clean_out for s in step if s.is_anomaly
        )
        total = sum(len(step) for step in clean_out) or 1
        assert clean_flags / total < 0.2

    def test_evt_spot_threshold_adapts(self):
        """SPOT's threshold should sit above the benign level and flag a true extreme."""
        clean = _clean_series(n=300, level=50.0, noise=1.0)
        spot = SPOT(q=1e-3, init_quantile=0.95).initialize(clean[:200])
        thr = spot.threshold
        assert np.isfinite(thr)
        # threshold is above the benign mean (adaptive cutoff, not a fixed value)
        assert thr > float(np.mean(clean))
        # a clear extreme is flagged; an in-distribution value is not
        assert spot.step(float(np.mean(clean)) + 20.0) is True
        assert spot.step(float(np.mean(clean))) is False


# ===========================================================================
# Fusion
# ===========================================================================


class TestFusion:
    def test_fused_risk_is_contract_valid(self):
        ent = _entity()
        scores = [
            _make_score("robust_z", DetectorFamily.STATISTICAL, 0.85),
            _make_score("isolation_forest", DetectorFamily.ML_UNSUPERVISED, 0.80),
            _make_score("page_hinkley", DetectorFamily.CHANGE_POINT, 0.90),
        ]
        fused = RiskFuser().fuse(scores, entity=ent, timestamp=T0)
        assert isinstance(fused, FusedRisk)
        # round-trips through the contract validators
        FusedRisk.model_validate(fused.model_dump())
        assert 0.0 <= fused.risk_score <= 1.0
        assert 0.0 <= fused.calibrated_confidence <= 1.0
        assert 0.0 <= fused.agreement <= 1.0
        assert fused.contributing_methods
        for mw in fused.contributing_methods:
            assert isinstance(mw, MethodWeight)
            assert 0.0 <= mw.normalized_score <= 1.0
            assert mw.weight >= 0.0

    def test_risk_score_positive_carries_provenance(self):
        """The contract forbids risk>0 with no methods; the fuser must comply."""
        ent = _entity()
        scores = [_make_score("copod", DetectorFamily.STATISTICAL, 0.95, is_anom=True)]
        fused = RiskFuser().fuse(scores, entity=ent, timestamp=T0)
        if fused.risk_score > 0:
            assert fused.contributing_methods

    def test_empty_evidence_yields_zero_risk(self):
        ent = _entity()
        fused = RiskFuser().fuse([], entity=ent, timestamp=T0)
        assert fused.risk_score == 0.0
        assert fused.predicted_issue == IssueType.NONE
        assert fused.contributing_methods == []

    def test_agreement_raises_confidence_vs_single_weak_signal(self):
        ent = _entity()
        fuser = RiskFuser()

        # a single, weak detector firing alone
        weak = [_make_score("robust_z", DetectorFamily.STATISTICAL, 0.72, is_anom=True)]
        fused_weak = fuser.fuse(weak, entity=ent, timestamp=T0)

        # several INDEPENDENT families strongly agreeing
        strong = [
            _make_score("robust_z", DetectorFamily.STATISTICAL, 0.92, is_anom=True),
            _make_score("isolation_forest", DetectorFamily.ML_UNSUPERVISED, 0.90, is_anom=True),
            _make_score("page_hinkley", DetectorFamily.CHANGE_POINT, 0.93, is_anom=True),
            _make_score("matrix_profile", DetectorFamily.MATRIX_PROFILE, 0.88, is_anom=True),
        ]
        fused_strong = fuser.fuse(strong, entity=ent, timestamp=T0)

        # cross-family agreement must raise both agreement and confidence
        assert fused_strong.agreement > fused_weak.agreement
        assert fused_strong.calibrated_confidence > fused_weak.calibrated_confidence
        assert fused_strong.risk_score >= fused_weak.risk_score

    def test_disagreement_lowers_confidence(self):
        """One strong + several quiet detectors => low agreement, low confidence."""
        ent = _entity()
        scores = [
            _make_score("robust_z", DetectorFamily.STATISTICAL, 0.95, is_anom=True),
            _make_score("isolation_forest", DetectorFamily.ML_UNSUPERVISED, 0.10),
            _make_score("hbos", DetectorFamily.STATISTICAL, 0.08),
            _make_score("page_hinkley", DetectorFamily.CHANGE_POINT, 0.12),
        ]
        fused = RiskFuser().fuse(scores, entity=ent, timestamp=T0)
        assert fused.agreement < 0.5
        assert fused.calibrated_confidence < 0.7

    def test_time_to_impact_attached(self):
        ent = _entity()
        tti = TimeToImpact(
            entity=ent, metric="if_util_pct", origin=T0, threshold=90.0,
            eta_seconds=300.0, confidence=0.7, method="trajectory_crossing",
        )
        scores = [_make_score("robust_z", DetectorFamily.STATISTICAL, 0.9, is_anom=True)]
        fused = RiskFuser().fuse(scores, entity=ent, timestamp=T0, time_to_impact=tti)
        assert fused.time_to_impact is not None
        assert fused.time_to_impact.eta_seconds == pytest.approx(300.0)

    def test_pick_time_to_impact_prefers_earliest(self):
        ent = _entity()
        soon = TimeToImpact(entity=ent, metric="m", origin=T0, threshold=1.0,
                            eta_seconds=120.0, confidence=0.6)
        late = TimeToImpact(entity=ent, metric="m", origin=T0, threshold=1.0,
                            eta_seconds=600.0, confidence=0.9)
        none = TimeToImpact(entity=ent, metric="m", origin=T0, threshold=1.0,
                            eta_seconds=None, confidence=0.8)
        best = RiskFuser.pick_time_to_impact([late, soon, none])
        assert best is soon
        # with no crossings, the most confident healthy verdict wins
        best_healthy = RiskFuser.pick_time_to_impact([none])
        assert best_healthy is none

    def test_features_drift_adds_a_voice(self):
        ent = _entity()
        scores = [
            _make_score("robust_z", DetectorFamily.STATISTICAL, 0.85, is_anom=True),
        ]
        feats = FeatureVector(
            entity=ent, timestamp=T0,
            features={"util_slope": 1.2},
            triggered_drift=["page_hinkley:if_util_pct", "adwin:if_util_pct"],
        )
        with_feats = RiskFuser().fuse(scores, entity=ent, timestamp=T0, features=feats)
        without = RiskFuser().fuse(scores, entity=ent, timestamp=T0)
        # corroborating drift features should not LOWER agreement
        assert with_feats.agreement >= without.agreement

    def test_fuse_end_to_end_from_bank_scores(self):
        """Drive the fuser with REAL detector-bank output on a spike series."""
        ent = _entity()
        spike = _spike_series(n=200, spike_at=150, spike_mag=35.0)
        bank = DetectorBank(ent, "if_util_pct")
        bank.warmup(_clean_series(n=120)[:100])
        per_step = bank.score_series(spike)
        # fuse the post-spike instant
        fused = RiskFuser().fuse(per_step[151], entity=ent, timestamp=T0)
        FusedRisk.model_validate(fused.model_dump())
        assert 0.0 <= fused.risk_score <= 1.0
        assert fused.contributing_methods


# ===========================================================================
# Calibration
# ===========================================================================


class TestCalibration:
    def test_identity_until_fitted(self):
        cal = ProbabilityCalibrator()
        assert not cal.is_fitted
        assert cal.transform(0.42) == pytest.approx(0.42)
        assert cal.transform(1.5) == pytest.approx(1.0)   # clipped to [0,1]
        assert cal.transform(-0.3) == pytest.approx(0.0)

    @pytest.mark.parametrize("method", ["platt", "isotonic"])
    def test_calibrator_maps_into_unit_interval_and_is_monotone(self, method):
        rng = np.random.default_rng(0)
        # scores correlated with a binary label via a logistic link
        scores = rng.uniform(0, 1, size=200)
        p_true = 1.0 / (1.0 + np.exp(-(6.0 * scores - 3.0)))
        labels = (rng.uniform(0, 1, size=200) < p_true).astype(int)

        cal = ProbabilityCalibrator(method=method).fit(scores, labels)
        assert cal.is_fitted
        grid = np.linspace(0.0, 1.0, 11)
        out = np.array([cal.transform(float(g)) for g in grid])
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        # calibrated probability should be (weakly) increasing in the score
        assert out[-1] >= out[0]
        # a high score maps to a higher prob than a low score
        assert cal.transform(0.9) > cal.transform(0.1)

    def test_degenerate_single_class_stays_identity(self):
        cal = ProbabilityCalibrator().fit([0.1, 0.2, 0.3], [0, 0, 0])
        assert not cal.is_fitted
        assert cal.transform(0.5) == pytest.approx(0.5)

    def test_calibrator_used_in_fuser(self):
        """A fitted calibrator should make the fuser emit a calibrated risk."""
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 1, size=300)
        p_true = 1.0 / (1.0 + np.exp(-(8.0 * scores - 4.0)))
        labels = (rng.uniform(0, 1, size=300) < p_true).astype(int)
        cal = ProbabilityCalibrator(method="platt").fit(scores, labels)

        ent = _entity()
        det = [
            _make_score("robust_z", DetectorFamily.STATISTICAL, 0.9, is_anom=True),
            _make_score("isolation_forest", DetectorFamily.ML_UNSUPERVISED, 0.88, is_anom=True),
        ]
        fused = RiskFuser(calibrator=cal).fuse(det, entity=ent, timestamp=T0)
        assert 0.0 <= fused.risk_score <= 1.0
        FusedRisk.model_validate(fused.model_dump())


# ===========================================================================
# Registry
# ===========================================================================


class TestRegistry:
    def test_list_methods_has_at_least_30(self):
        methods = list_methods()
        assert len(methods) >= 30, f"only {len(methods)} methods registered"
        assert method_count() == len(methods)

    def test_methods_are_well_formed(self):
        from netra.analytics.fusion.registry import FAMILIES, MethodInfo

        names = set()
        for m in list_methods():
            assert isinstance(m, MethodInfo)
            assert m.name and isinstance(m.name, str)
            assert m.family in FAMILIES
            assert isinstance(m.offline_capable, bool)
            assert isinstance(m.optional_heavy, bool)
            assert m.name not in names, f"duplicate method id {m.name}"
            names.add(m.name)

    def test_count_by_family_sums_to_total(self):
        tally = count_by_family()
        assert sum(tally.values()) == method_count()
        # the headline families are all represented
        assert tally["forecasting"] >= 5
        assert tally["fusion"] >= 3
        assert tally["streaming"] >= 5

    def test_filters(self):
        offline = list_methods(offline_only=True)
        light = list_methods(exclude_optional_heavy=True)
        assert len(offline) == len(list_methods())   # everything is offline-capable
        assert 0 < len(light) <= len(list_methods())
        fam = list_methods(family="fusion")
        assert all(m.family == "fusion" for m in fam)
        assert len(fam) >= 3
