"""``FeatureEngine`` — the O(1) online feature engine (Workstream 2 core).

Consumes a stream of :class:`~netra.contracts.TelemetryRecord` (an iterable, a
NATS subscription, or a direct ``TelemetrySource``) and emits a
:class:`~netra.contracts.FeatureVector` per entity per tick. Every telemetry
record is folded into the entity's running state in **O(1)** time and constant
memory, so the engine sustains high record/second throughput and the precursor
features are always current — the lead-time win (research
``02-telemetry-pipeline.md`` §3).

Design:

  * **Contract-only dependency.** This module imports *only* from
    ``netra.contracts`` (plus ``netra.streaming`` siblings). It never imports
    ``netra.datagen`` — tests and callers construct ``TelemetryRecord`` objects
    directly or pass any iterable of them, honouring the dual-source abstraction.
  * **Per-entity state.** State is keyed by ``EntityRef.entity_id`` (the
    universal join key) and, within an entity, by metric. Memory is O(entities ×
    features), independent of stream length.
  * **Pluggable feature registry.** A :class:`FeatureRegistry` maps a metric name
    to the set of O(1) computers/detectors to run on it. Callers can register
    custom features without modifying the engine — extensibility the
    ``FeatureVector.features`` free-form dict was designed for.
  * **One tick = one emitted vector.** By default the engine emits a
    ``FeatureVector`` for the record's entity on *every* record (streaming-first,
    no window-boundary latency). A ``min_emit_interval`` can throttle emission per
    entity if a consumer prefers coarser ticks.

The engine output is the contract boundary to Phase 3: the predictive ensemble
(``netra.analytics``) consumes these ``FeatureVector``s (slope/drift/anomaly
features + ``triggered_drift`` votes) to produce ``Forecast`` / ``AnomalyScore``
/ ``FusedRisk``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from netra.contracts import (
    EntityRef,
    FeatureVector,
    MetricName,
    TelemetryRecord,
)

from .detectors import (
    ADWINDetector,
    HalfSpaceTreesDetector,
    PageHinkleyDetector,
)
from .features import (
    ErrorRateAcceleration,
    JitterTrend,
    LatencyDrift,
    LossProgression,
    RekeyIntervalAnomaly,
    RollingSlope,
    TimeToThreshold,
)

__all__ = [
    "FeatureSpec",
    "FeatureRegistry",
    "EntityState",
    "FeatureEngine",
    "default_registry",
    "DEFAULT_SLA_THRESHOLDS",
]


# Default SLA/security thresholds per metric, used by the streaming
# time-to-threshold helper. These are deliberately conservative lab defaults;
# the full analytics layer refines them with EVT/SPOT adaptive thresholds.
DEFAULT_SLA_THRESHOLDS: dict[str, float] = {
    MetricName.IF_UTIL_PCT.value: 90.0,       # % utilisation
    MetricName.LATENCY_MS.value: 150.0,        # ms RTT
    MetricName.JITTER_MS.value: 30.0,          # ms jitter
    MetricName.LOSS_PCT.value: 2.0,            # % loss
    MetricName.TUNNEL_LOSS_PCT.value: 2.0,     # % tunnel loss
    MetricName.TUNNEL_JITTER_MS.value: 30.0,   # ms tunnel jitter
    MetricName.QUEUE_DEPTH.value: 0.8,         # normalised queue occupancy
}


@dataclass
class FeatureSpec:
    """One named O(1) feature/detector to run on a metric stream.

    ``factory`` builds a fresh stateful computer per ``(entity, metric)`` (state
    is never shared across entities). ``reader`` extracts the float feature value
    to record after :meth:`update`; if ``None`` the value returned by ``update``
    is used directly. ``is_detector`` marks boolean drift/change-point triggers
    whose firing is recorded in ``FeatureVector.triggered_drift`` rather than as a
    continuous feature.
    """

    key: str
    factory: Callable[[], object]
    reader: Callable[[object], float | None] | None = None
    is_detector: bool = False


@dataclass
class FeatureRegistry:
    """Maps a metric name -> the list of :class:`FeatureSpec` to run on it.

    A metric stream (e.g. ``latency_ms`` for one interface) is processed by every
    spec registered for that metric. ``register`` is additive so callers can
    extend the default set without subclassing the engine.
    """

    specs: dict[str, list[FeatureSpec]] = field(default_factory=dict)

    def register(self, metric: str, spec: FeatureSpec) -> FeatureRegistry:
        self.specs.setdefault(metric, []).append(spec)
        return self

    def for_metric(self, metric: str) -> list[FeatureSpec]:
        return self.specs.get(metric, [])

    def metrics(self) -> Iterable[str]:
        return self.specs.keys()


def default_registry(
    sla_thresholds: Mapping[str, float] | None = None,
) -> FeatureRegistry:
    """Build the default feature registry wiring metrics to O(1) computers.

    The wiring mirrors the precursor table in ``ARCHITECTURE.md`` §6 / research §5:

      * utilisation  -> rolling slope, time-to-threshold, EWMA-drift trigger.
      * latency      -> latency drift (level + drift), Page-Hinkley trigger,
                        time-to-threshold.
      * jitter       -> jitter variance trend (+ p99 tail), ADWIN trigger.
      * loss         -> loss progression, Page-Hinkley trigger, time-to-threshold.
      * errors       -> error-rate acceleration.
      * rekey        -> rekey-interval anomaly.

    The multivariate Half-Space-Trees anomaly score is computed once per entity
    over the whole feature vector (see :meth:`FeatureEngine._score_multivariate`),
    not per metric, so it captures cross-metric anomalies.
    """
    th = dict(DEFAULT_SLA_THRESHOLDS)
    if sla_thresholds:
        th.update(sla_thresholds)
    reg = FeatureRegistry()

    def _ttt(metric: str, above: bool = True) -> Callable[[], object]:
        thr = th.get(metric, 100.0)
        return lambda: TimeToThreshold(thr, above_is_breach=above)

    # --- interface utilisation -------------------------------------------------
    util = MetricName.IF_UTIL_PCT.value
    reg.register(util, FeatureSpec("util_slope", lambda: RollingSlope()))
    reg.register(util, FeatureSpec("util_eta_seconds", _ttt(util)))
    reg.register(
        util,
        FeatureSpec(
            "util_drift", lambda: PageHinkleyDetector(threshold=8.0), is_detector=True
        ),
    )

    # --- latency ---------------------------------------------------------------
    lat = MetricName.LATENCY_MS.value
    reg.register(lat, FeatureSpec("latency_drift", lambda: LatencyDrift()))
    reg.register(
        lat,
        FeatureSpec(
            "latency_ph", lambda: PageHinkleyDetector(threshold=20.0), is_detector=True
        ),
    )
    reg.register(lat, FeatureSpec("latency_eta_seconds", _ttt(lat)))

    # --- jitter ----------------------------------------------------------------
    jit = MetricName.JITTER_MS.value
    reg.register(jit, FeatureSpec("jitter_ewvar", lambda: JitterTrend()))
    reg.register(
        jit, FeatureSpec("jitter_adwin", lambda: ADWINDetector(), is_detector=True)
    )

    # --- loss ------------------------------------------------------------------
    loss = MetricName.LOSS_PCT.value
    reg.register(loss, FeatureSpec("loss_progression", lambda: LossProgression()))
    reg.register(
        loss,
        FeatureSpec(
            "loss_ph", lambda: PageHinkleyDetector(threshold=3.0), is_detector=True
        ),
    )
    reg.register(loss, FeatureSpec("loss_eta_seconds", _ttt(loss)))

    # --- tunnel loss / jitter (scenario C) -------------------------------------
    tloss = MetricName.TUNNEL_LOSS_PCT.value
    reg.register(tloss, FeatureSpec("tunnel_loss_progression", lambda: LossProgression()))
    reg.register(
        tloss,
        FeatureSpec(
            "tunnel_loss_ph", lambda: PageHinkleyDetector(threshold=3.0), is_detector=True
        ),
    )
    tjit = MetricName.TUNNEL_JITTER_MS.value
    reg.register(tjit, FeatureSpec("tunnel_jitter_ewvar", lambda: JitterTrend()))

    # --- interface errors ------------------------------------------------------
    err = MetricName.IF_IN_ERRORS.value
    reg.register(err, FeatureSpec("err_accel", lambda: ErrorRateAcceleration()))

    # --- IPSec rekey interval --------------------------------------------------
    rekey = MetricName.TUNNEL_REKEY_INTERVAL_S.value
    reg.register(rekey, FeatureSpec("rekey_anomaly", lambda: RekeyIntervalAnomaly()))

    return reg


@dataclass
class EntityState:
    """Per-entity running state: one computer instance per ``(metric, feature)``.

    Holds the live O(1) computers plus the latest feature snapshot and the
    multivariate Half-Space-Trees scorer. Memory is bounded by the number of
    metrics × features for the entity — independent of how many samples flow.
    """

    entity: EntityRef
    computers: dict[tuple[str, str], object] = field(default_factory=dict)
    latest_features: dict[str, float] = field(default_factory=dict)
    hst: HalfSpaceTreesDetector | None = None
    sample_count: int = 0
    last_emit_ts: float | None = None


class FeatureEngine:
    """Online feature engine: ``TelemetryRecord`` stream -> ``FeatureVector`` stream.

    Usage (CPU-only, no bus — the smoke-test path)::

        engine = FeatureEngine()
        for record in some_iterable_of_telemetry_records:
            fv = engine.process(record)        # O(1); FeatureVector or None
            if fv is not None:
                consume(fv)

    or in bulk::

        for fv in engine.run(stream_of_records):
            consume(fv)

    The engine is deliberately transport-agnostic: ``stream_of_records`` may come
    from an in-memory list, a ``netra.datagen`` ``TelemetrySource`` (passed by the
    caller — the engine does not import it), or a NATS JetStream subscription that
    deserialises ``telemetry.>`` messages into ``TelemetryRecord``. ``sources.py``
    (the NATS / TelemetrySource adapters) wires those transports to this engine.

    Parameters
    ----------
    registry:
        The feature wiring; defaults to :func:`default_registry`.
    sla_thresholds:
        Optional per-metric SLA thresholds for the time-to-threshold helper
        (merged into :data:`DEFAULT_SLA_THRESHOLDS`).
    enable_hst:
        Run the multivariate Half-Space-Trees anomaly score per entity.
    min_emit_interval_seconds:
        If set, emit at most one ``FeatureVector`` per entity per interval
        (throttling); ``None`` (default) emits on every record.
    """

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        *,
        sla_thresholds: Mapping[str, float] | None = None,
        enable_hst: bool = True,
        min_emit_interval_seconds: float | None = None,
        hst_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.registry = registry or default_registry(sla_thresholds)
        self.enable_hst = enable_hst
        self.min_emit_interval = min_emit_interval_seconds
        self._hst_kwargs = dict(hst_kwargs) if hst_kwargs else {}
        self._states: dict[str, EntityState] = {}
        self._records_seen = 0

    # -- introspection ---------------------------------------------------------
    @property
    def entity_count(self) -> int:
        return len(self._states)

    @property
    def records_processed(self) -> int:
        return self._records_seen

    def state_for(self, entity_id: str) -> EntityState | None:
        return self._states.get(entity_id)

    # -- core ------------------------------------------------------------------
    def _get_state(self, entity: EntityRef) -> EntityState:
        st = self._states.get(entity.entity_id)
        if st is None:
            st = EntityState(entity=entity)
            if self.enable_hst:
                st.hst = HalfSpaceTreesDetector(**self._hst_kwargs)
            self._states[entity.entity_id] = st
        return st

    @staticmethod
    def _read_feature(spec: FeatureSpec, computer: object, returned: float | None) -> float | None:  # noqa: E501
        if spec.reader is not None:
            return spec.reader(computer)
        return returned

    def process(self, record: TelemetryRecord) -> FeatureVector | None:
        """Fold one telemetry record in (O(1)); return a ``FeatureVector`` or None.

        Returns ``None`` only when emission is throttled by
        ``min_emit_interval_seconds``; otherwise always returns the entity's
        current feature snapshot.
        """
        self._records_seen += 1
        entity = record.entity()
        st = self._get_state(entity)
        st.sample_count += 1
        ts_epoch = record.timestamp.timestamp()
        metric = record.metric_name
        value = float(record.value)

        triggered: list[str] = []
        for spec in self.registry.for_metric(metric):
            key = (metric, spec.key)
            computer = st.computers.get(key)
            if computer is None:
                computer = spec.factory()
                st.computers[key] = computer
            returned = computer.update(value)  # type: ignore[attr-defined]
            if spec.is_detector:
                if bool(returned):
                    triggered.append(f"{getattr(computer, 'name', spec.key)}:{metric}")
            else:
                fval = self._read_feature(spec, computer, returned)
                if fval is not None and _finite(fval):
                    st.latest_features[spec.key] = float(fval)
                # surface auxiliary readings (p99 tail, levels) where available
                self._read_aux(spec.key, computer, st.latest_features)

        # multivariate anomaly score over the entity's current feature snapshot
        if self.enable_hst and st.hst is not None and st.latest_features:
            is_anom = st.hst.update(st.latest_features)
            st.latest_features["hst_score"] = st.hst.score
            if is_anom:
                triggered.append(f"half_space_trees:{entity.entity_id}")

        # emission throttling
        if self.min_emit_interval is not None:
            if (
                st.last_emit_ts is not None
                and (ts_epoch - st.last_emit_ts) < self.min_emit_interval
            ):
                return None
        st.last_emit_ts = ts_epoch

        return FeatureVector(
            entity=entity,
            timestamp=record.timestamp,
            features=dict(st.latest_features),
            triggered_drift=triggered,
            sample_count=st.sample_count,
        )

    @staticmethod
    def _read_aux(
        key: str, computer: object, into: dict[str, float]
    ) -> None:
        """Surface secondary readings a computer exposes (p99 tail, level, etc.)."""
        # jitter p99 tail
        p99 = getattr(computer, "p99", None)
        if callable(p99):
            v = p99()
            if v is not None and _finite(v):
                into[f"{key}_p99"] = float(v)
        # latency level (EWMA)
        level = getattr(computer, "level", None)
        if level is not None and not callable(level) and _finite(level):
            into[f"{key}_level"] = float(level)
        # error first-derivative rate alongside the acceleration
        rate = getattr(computer, "rate", None)
        if callable(rate):
            v = rate()
            if v is not None and _finite(v):
                into[f"{key}_rate"] = float(v)

    def run(self, records: Iterable[TelemetryRecord]) -> Iterator[FeatureVector]:
        """Process an iterable/stream of records, yielding each emitted vector."""
        for record in records:
            fv = self.process(record)
            if fv is not None:
                yield fv


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))  # not NaN, not inf
