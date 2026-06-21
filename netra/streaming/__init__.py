"""netra.streaming — O(1) online feature engine (Workstream 2).

Consumes raw telemetry (off NATS JetStream or directly from a TelemetrySource)
and folds each sample into constant-memory running statistics
(Welford/EWMA/DDSketch/Page-Hinkley/Half-Space-Trees/stumpi), emitting a
``netra.contracts.FeatureVector`` per entity per tick — the leading-indicator
features the predictive ensemble consumes.

Public surface
--------------
- :mod:`~netra.streaming.features`  — O(1) precursor feature computers
  (rolling slope, latency drift, jitter trend, loss progression, error-rate
  acceleration, BGP churn, adjacency flaps, rekey anomaly, path asymmetry,
  top-talker churn, streaming time-to-threshold) + streaming sketches
  (DDSketch quantiles, stumpi matrix-profile discord, Count-Min, HyperLogLog).
- :mod:`~netra.streaming.detectors` — O(1) change/anomaly detectors
  (ADWIN, Page-Hinkley, KSWIN, CUSUM, EWMA control chart, Half-Space-Trees).
- :mod:`~netra.streaming.engine`    — :class:`FeatureEngine`: TelemetryRecord
  stream -> FeatureVector stream, with a pluggable feature registry, O(1)/record.
- :mod:`~netra.streaming.sources`   — TelemetrySource + NATS JetStream adapters.
- :mod:`~netra.streaming.alerts`    — idempotent, dedup-keyed alert emitter
  (implements the at-least-once correction: stable key / Nats-Msg-Id).

Everything is import-light: ``features``/``detectors``/``engine``/``alerts`` need
only the CORE tier (river, ddsketch, numpy, pydantic). Heavy/extra deps
(``stumpy``, ``pyprobables``, ``nats-py``) are import-guarded with working
fallbacks so the CPU-only path always runs.
"""

from __future__ import annotations

from .alerts import Alert, AlertEmitter, make_alert_key
from .engine import (
    DEFAULT_SLA_THRESHOLDS,
    EntityState,
    FeatureEngine,
    FeatureRegistry,
    FeatureSpec,
    default_registry,
)

__all__ = [
    "FeatureEngine",
    "FeatureRegistry",
    "FeatureSpec",
    "EntityState",
    "default_registry",
    "DEFAULT_SLA_THRESHOLDS",
    "Alert",
    "AlertEmitter",
    "make_alert_key",
]
