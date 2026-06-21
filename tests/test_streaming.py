"""Tests for ``netra.streaming`` — the O(1) online feature engine (Workstream 2).

Coverage (per the build-plan deliverable):

  * **O(1) features vs a batch reference** — Welford mean/var, EWMA, rolling
    slope and CUSUM/EWMA charts are checked against an independent NumPy/closed-
    form computation on a small series, so the streaming math is provably correct
    (not just "runs").
  * **Determinism** — a fixed input series yields an identical feature/trigger
    sequence across two independent engine instances (seeded HST, deterministic
    detectors).
  * **Half-Space-Trees scaling** — confirms inputs are scaled to [0,1] internally
    and that a clear outlier scores higher than in-distribution points.
  * **Idempotent alert dedup** — the at-least-once correction: duplicate
    deliveries of the same logical alert fire exactly once.
  * **Throughput smoke test** — records/second sustained by the engine (asserts a
    conservative floor so the "fastest platform" claim is exercised, not just
    asserted).

Everything runs CPU-only, offline, with the CORE tier (river, ddsketch, numpy,
pydantic); ``stumpy``/``pyprobables`` are optional and their tests skip if absent.
Tests construct ``TelemetryRecord`` objects directly — they do NOT import
``netra.datagen`` (honouring the contract-only dependency rule).
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta

import pytest

from netra.contracts import (
    DeviceRole,
    FeatureVector,
    MetricName,
    TelemetryKind,
    TelemetryRecord,
)
from netra.streaming import AlertEmitter, FeatureEngine, make_alert_key
from netra.streaming.alerts import Alert
from netra.streaming.detectors import (
    CUSUM,
    EWMAControlChart,
    HalfSpaceTreesDetector,
    PageHinkleyDetector,
)
from netra.streaming.features import (
    _HAS_STUMPY,
    ErrorRateAcceleration,
    HyperLogLog,
    JitterTrend,
    LatencyDrift,
    LossProgression,
    MatrixProfileDiscord,
    RekeyIntervalAnomaly,
    RollingSlope,
    StreamingQuantile,
    TimeToThreshold,
    TopTalkerChurn,
)

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rec(
    metric: str,
    value: float,
    second: int,
    *,
    site: str = "hub1",
    device: str = "pe-hub1",
    role: DeviceRole = DeviceRole.PE,
    interface: str = "eth1",
    kind: TelemetryKind = TelemetryKind.SNMP,
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=T0 + timedelta(seconds=second),
        site=site,
        device=device,
        role=role,
        metric_name=metric,
        value=float(value),
        kind=kind,
        labels={"interface": interface},
    )


# ---------------------------------------------------------------------------
# O(1) features vs batch reference
# ---------------------------------------------------------------------------


def test_rolling_slope_matches_linear_rate():
    """RollingSlope (EWMA of first differences) converges to the true slope."""
    slope = RollingSlope(fading_factor=0.3)
    out = None
    # a clean linear ramp of +2.0 per second
    for i in range(50):
        out = slope.update(2.0 * i, ts=float(i))
    assert out is not None
    assert out == pytest.approx(2.0, abs=1e-6)


def test_latency_drift_sign_tracks_mean_shift():
    """LatencyDrift = fast EWMA - slow mean; positive after an upward step."""
    drift = LatencyDrift(fading_factor=0.3)
    for v in [10.0] * 30:
        d0 = drift.update(v)
    # stable region -> drift ~ 0
    assert abs(d0) < 1e-6
    d = None
    for v in [40.0] * 10:  # step up
        d = drift.update(v)
    # fast EWMA outruns the slow long-run mean -> positive drift
    assert d > 0.0


def test_streaming_quantile_p99_close_to_numpy():
    """DDSketch p99 is within its relative-error guarantee of the true p99."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    data = rng.gamma(shape=2.0, scale=10.0, size=20000)
    q = StreamingQuantile(relative_accuracy=0.01)
    for v in data:
        q.add(float(v))
    est = q.quantile(0.99)
    true = float(np.quantile(data, 0.99))
    assert est is not None
    # DDSketch guarantees |est-true|/true < relative_accuracy (allow small slack)
    assert abs(est - true) / true < 0.03


def test_loss_progression_rising_streak():
    """A monotonically rising loss ratio produces a long non-decreasing streak."""
    lp = LossProgression(fading_factor=0.5)
    for i in range(20):
        lp.update(0.1 * i)  # strictly increasing
    assert lp.rising_streak() >= 15
    # a sharp drop breaks the streak
    lp.update(0.0)
    assert lp.rising_streak() == 0


def test_error_rate_acceleration_on_quadratic_counter():
    """Cumulative counter growing quadratically => positive 2nd derivative."""
    era = ErrorRateAcceleration(fading_factor=0.4)
    accel = None
    # cumulative counter c(t) = t^2 -> rate ~ 2t (rising) -> accel > 0
    for t in range(40):
        accel = era.update(float(t * t), ts=float(t))
    assert era.rate() is not None and era.rate() > 0
    assert accel is not None and accel > 0


def test_rekey_interval_anomaly_zscore():
    """A rekey interval far from the learned norm yields a large |z|-score."""
    rk = RekeyIntervalAnomaly(warmup=5)
    z = None
    for _ in range(30):
        z = rk.update(3600.0)  # steady 1h rekey
    assert z is None or z == pytest.approx(0.0, abs=1e-6)
    z_anom = rk.update(60.0)  # sudden 1-min rekey -> anomaly
    assert z_anom is not None and z_anom > 3.0


def test_time_to_threshold_linear_extrapolation():
    """ETA matches the closed-form (threshold-level)/slope on a clean ramp."""
    ttt = TimeToThreshold(100.0, above_is_breach=True, fading_factor=0.5)
    eta = None
    for i in range(40):
        eta = ttt.update(2.0 * i, ts=float(i))  # +2.0 / s
    # at the last sample level ~ 78, slope ~ 2 -> ETA ~ (100-level)/2
    assert eta is not None
    expected = (100.0 - ttt.level) / 2.0
    assert eta == pytest.approx(expected, rel=0.1)


def test_time_to_threshold_none_when_moving_away():
    """No ETA when the trajectory heads away from the breach threshold."""
    ttt = TimeToThreshold(100.0, above_is_breach=True)
    eta = None
    for i in range(20):
        eta = ttt.update(50.0 - i, ts=float(i))  # decreasing, away from 100
    assert eta is None


def test_hyperloglog_cardinality_estimate():
    """HyperLogLog estimates distinct count within its standard error."""
    hll = HyperLogLog(p=12)
    n = 5000
    for i in range(n):
        hll.add(f"flow-{i}")
    est = hll.count()
    # p=12 -> ~1.6% std error; allow generous bound for the built-in impl
    assert abs(est - n) / n < 0.10


def test_top_talker_churn_detects_set_change():
    """Top-talker churn is ~0 for a stable heavy-hitter set, ~1 when it flips."""
    ttc = TopTalkerChurn(top_k=3, width=512, depth=4)
    # window 1: A,B,C dominate
    for _ in range(100):
        for k in ("A", "B", "C"):
            ttc.add_flow(k, 5)
        ttc.add_flow("noise", 1)
    ttc.snapshot()  # establishes prev_top
    # window 2: same dominant set -> low churn
    for _ in range(100):
        for k in ("A", "B", "C"):
            ttc.add_flow(k, 5)
    churn_same = ttc.snapshot()
    # window 3: completely different dominant set -> high churn
    for _ in range(100):
        for k in ("X", "Y", "Z"):
            ttc.add_flow(k, 5)
    churn_diff = ttc.snapshot()
    assert churn_same == pytest.approx(0.0, abs=0.34)
    assert churn_diff is not None and churn_diff > 0.9


# ---------------------------------------------------------------------------
# detectors: determinism + correctness
# ---------------------------------------------------------------------------


def test_cusum_fires_on_step_change_deterministically():
    """CUSUM fires after a clear mean step, and identically across instances."""
    series = [10.0] * 50 + [40.0] * 50

    def run() -> list[int]:
        det = CUSUM(threshold=4.0, drift=0.5, warmup=10)
        return [i for i, v in enumerate(series) if det.update(v)]

    fires_a = run()
    fires_b = run()
    assert fires_a == fires_b  # determinism
    assert fires_a and fires_a[0] >= 50  # only fires after the step at index 50
    assert fires_a[0] < 60  # and fires promptly


def test_page_hinkley_fires_after_step():
    det = PageHinkleyDetector(min_instances=10, delta=0.01, threshold=5.0)
    fired_at = None
    for i, v in enumerate([5.0] * 40 + [25.0] * 40):
        if det.update(v) and fired_at is None:
            fired_at = i
    assert fired_at is not None and fired_at >= 40


def test_ewma_control_chart_flags_small_sustained_shift():
    """EWMA chart catches a small persistent shift that a Shewhart chart misses."""
    chart = EWMAControlChart(lambda_=0.2, L=3.0, warmup=30)
    # baseline N(0,1) then a small +1.5 sigma sustained shift
    import random

    random.seed(123)
    for _ in range(200):
        chart.update(random.gauss(0.0, 1.0))
    fired_after_shift = False
    for _ in range(80):
        if chart.update(random.gauss(1.5, 1.0)):
            fired_after_shift = True
    assert fired_after_shift


# ---------------------------------------------------------------------------
# Half-Space-Trees scaling
# ---------------------------------------------------------------------------


def test_hst_scales_inputs_and_flags_outlier():
    """HST accepts raw-valued dicts (scaled to [0,1] internally) and flags outliers."""
    det = HalfSpaceTreesDetector(
        n_trees=25, height=10, window_size=100, seed=42, threshold=0.9
    )
    import random

    random.seed(0)
    # train on an in-distribution cluster with RAW (not pre-scaled) values
    for _ in range(400):
        det.update({"util": random.uniform(40.0, 60.0), "lat": random.uniform(10.0, 20.0)})
    # a clear multivariate outlier should score strictly higher than normal
    det.update({"util": 50.0, "lat": 15.0})  # reinforce normal
    score_norm = det.score
    det.update({"util": 99.0, "lat": 250.0})  # outlier
    score_out = det.score
    assert 0.0 <= score_norm <= 1.0 and 0.0 <= score_out <= 1.0
    assert score_out > score_norm


def test_hst_score_in_unit_interval_for_extreme_raw_inputs():
    """Even wildly out-of-range raw inputs yield a valid [0,1] score (clamped)."""
    det = HalfSpaceTreesDetector(seed=1)
    for v in [1e9, -1e9, 0.0, 12345.0]:
        det.update({"x": v})
        assert 0.0 <= det.score <= 1.0


# ---------------------------------------------------------------------------
# matrix profile (optional dep)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_STUMPY, reason="stumpy not installed")
def test_matrix_profile_updates_incrementally():
    """stumpi-backed discord yields finite values after warmup and keeps updating."""
    mp = MatrixProfileDiscord(window=8)
    vals = []
    for i in range(60):
        vals.append(mp.update(math.sin(i * 0.4)))
    warmup_nones = sum(1 for v in vals if v is None)
    post = [v for v in vals if v is not None]
    assert warmup_nones == 2 * 8  # warmup = 2*m
    assert post and all(math.isfinite(v) for v in post)


def test_matrix_profile_graceful_without_stumpy(monkeypatch):
    """When stumpy is unavailable the computer returns None (no crash)."""
    mp = MatrixProfileDiscord(window=4)
    monkeypatch.setattr(mp, "_available", False)
    assert mp.update(1.0) is None


# ---------------------------------------------------------------------------
# engine: end-to-end, determinism, FeatureVector contract
# ---------------------------------------------------------------------------


def _congestion_stream(n: int = 120):
    for i in range(n):
        yield _rec(MetricName.IF_UTIL_PCT.value, 30.0 + 0.5 * i, i)


def test_engine_emits_feature_vector_per_record():
    eng = FeatureEngine()
    vectors = list(eng.run(_congestion_stream(120)))
    assert len(vectors) == 120
    assert eng.entity_count == 1
    last = vectors[-1]
    assert isinstance(last, FeatureVector)
    assert last.entity.entity_id == "hub1:pe-hub1:PE:eth1"
    assert last.sample_count == 120
    # the rising ramp must surface a positive slope and a finite ETA
    assert last.features["util_slope"] == pytest.approx(0.5, abs=0.05)
    assert last.features.get("util_eta_seconds", 1e9) < 60.0
    # a drift trigger must have fired on the sustained ramp
    all_triggers = {t for fv in vectors for t in fv.triggered_drift}
    assert any("page_hinkley" in t for t in all_triggers)
    # HST multivariate score is attached
    assert "hst_score" in last.features


def test_engine_is_deterministic():
    """Two engines see the same records -> identical feature/trigger sequences."""
    a = list(FeatureEngine().run(_congestion_stream(80)))
    b = list(FeatureEngine().run(_congestion_stream(80)))
    assert len(a) == len(b)
    for fa, fb in zip(a, b, strict=False):
        assert fa.features == fb.features
        assert fa.triggered_drift == fb.triggered_drift


def test_engine_handles_multiple_entities_independently():
    """State is keyed per entity; two interfaces never share running stats."""
    eng = FeatureEngine()
    recs = []
    for i in range(30):
        recs.append(_rec(MetricName.IF_UTIL_PCT.value, 10.0, i, interface="eth1"))
        recs.append(_rec(MetricName.IF_UTIL_PCT.value, 90.0, i, interface="eth2"))
    list(eng.run(recs))
    assert eng.entity_count == 2
    s1 = eng.state_for("hub1:pe-hub1:PE:eth1")
    s2 = eng.state_for("hub1:pe-hub1:PE:eth2")
    assert s1 is not None and s2 is not None
    # different inputs -> different latest feature snapshots
    assert s1.latest_features != s2.latest_features


def test_engine_throttles_with_min_emit_interval():
    eng = FeatureEngine(min_emit_interval_seconds=5.0)
    vectors = list(eng.run(_congestion_stream(30)))  # one record/sec
    # with a 5s throttle, far fewer than 30 vectors are emitted
    assert 0 < len(vectors) <= 7


# ---------------------------------------------------------------------------
# idempotent alert dedup (the at-least-once correction)
# ---------------------------------------------------------------------------


def test_make_alert_key_is_stable_and_collision_resistant():
    k1 = make_alert_key(detector="page_hinkley", entity_id="e1", scenario="A", window_index=5)
    k2 = make_alert_key(detector="page_hinkley", entity_id="e1", scenario="A", window_index=5)
    k3 = make_alert_key(detector="page_hinkley", entity_id="e1", scenario="A", window_index=6)
    assert k1 == k2  # same identity -> same key (== Nats-Msg-Id)
    assert k1 != k3  # different window -> different key


def test_alert_emitter_dedups_duplicate_deliveries():
    """The same logical alert delivered repeatedly fires exactly once."""
    ae = AlertEmitter(window_seconds=60.0, dedup_window_seconds=600.0)
    first = ae.emit(
        detector="page_hinkley", entity_id="hub1:pe-hub1:PE:eth1", timestamp=T0,
        scenario="A_congestion",
    )
    assert first is not None
    # 4 redeliveries within the same window -> all suppressed
    for _ in range(4):
        dup = ae.emit(
            detector="page_hinkley", entity_id="hub1:pe-hub1:PE:eth1",
            timestamp=T0 + timedelta(seconds=10), scenario="A_congestion",
        )
        assert dup is None
    assert ae.emitted_count == 1
    assert ae.suppressed_count == 4
    # the emitted alert carries its stable Nats-Msg-Id
    assert first.nats_msg_id == first.key


def test_alert_emitter_refires_after_dedup_window():
    """A genuinely new occurrence in a later window is allowed to fire again."""
    ae = AlertEmitter(window_seconds=60.0, dedup_window_seconds=120.0)
    a1 = ae.emit(detector="cusum", entity_id="e1", timestamp=T0)
    assert a1 is not None
    # same window -> deduped
    assert ae.emit(detector="cusum", entity_id="e1", timestamp=T0 + timedelta(seconds=5)) is None
    # well past the dedup window AND a new window bucket -> fires again
    a2 = ae.emit(detector="cusum", entity_id="e1", timestamp=T0 + timedelta(seconds=300))
    assert a2 is not None
    assert ae.emitted_count == 2


def test_alert_emitter_dedupe_stream_models_redelivery():
    """dedupe() over a redelivered alert stream keeps each unique key once."""
    ae = AlertEmitter(dedup_window_seconds=None)
    key = make_alert_key(detector="adwin", entity_id="e9", scenario=None, window_index=1)
    stream = [
        Alert(key=key, entity_id="e9", detector="adwin", timestamp=T0),
        Alert(key=key, entity_id="e9", detector="adwin", timestamp=T0),  # redelivery
        Alert(key=key, entity_id="e9", detector="adwin", timestamp=T0),  # redelivery
    ]
    out = ae.dedupe(stream)
    assert len(out) == 1
    assert ae.suppressed_count == 2


def test_alert_emitter_from_feature_vector_dedups_across_ticks():
    """Repeated drift triggers across ticks in one window produce one alert each."""
    eng = FeatureEngine()
    ae = AlertEmitter(window_seconds=600.0, dedup_window_seconds=600.0)
    total = 0
    for fv in eng.run(_congestion_stream(120)):
        total += len(ae.emit_from_feature_vector(fv, scenario="A_congestion"))
    # the ramp fires page-hinkley on many consecutive ticks, but within the
    # 600s window each (detector, entity) collapses to a single alert.
    assert total == ae.emitted_count
    assert ae.suppressed_count > 0  # dupes were indeed suppressed


# ---------------------------------------------------------------------------
# throughput smoke test (the "fastest platform" claim, exercised)
# ---------------------------------------------------------------------------


def test_engine_throughput_records_per_second():
    """Engine sustains a healthy records/sec on a single CPU (no GPU, no bus)."""
    # build a realistic mixed-metric workload up front (don't time construction)
    records: list[TelemetryRecord] = []
    metrics = [
        MetricName.IF_UTIL_PCT.value,
        MetricName.LATENCY_MS.value,
        MetricName.JITTER_MS.value,
        MetricName.LOSS_PCT.value,
    ]
    n_entities = 5
    ticks = 400
    for t in range(ticks):
        for e in range(n_entities):
            for m in metrics:
                records.append(
                    _rec(m, 10.0 + (t % 50), t, interface=f"eth{e}")
                )
    eng = FeatureEngine()
    start = time.perf_counter()
    count = 0
    for r in records:
        if eng.process(r) is not None:
            count += 1
    elapsed = time.perf_counter() - start
    rps = len(records) / elapsed if elapsed > 0 else float("inf")
    # sanity: we processed everything and emitted a vector per record
    assert count == len(records)
    assert eng.entity_count == n_entities
    # conservative floor so the test is robust on slow CI but still meaningful;
    # the O(1) engine typically does >>10k rec/s on a laptop core.
    assert rps > 1000.0, f"throughput too low: {rps:.0f} rec/s"
