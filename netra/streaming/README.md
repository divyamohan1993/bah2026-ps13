# `netra/streaming/` — O(1) online feature engine (Workstream 2)

The **"fastest platform" core**: consumes a stream of
`netra.contracts.TelemetryRecord` and folds each sample into constant-memory
running statistics in **O(1) per record**, emitting a
`netra.contracts.FeatureVector` per entity per tick. Computing precursor features
*online* (not in batches) is the lead-time win — the anomaly/drift/ETA score is
always current, so the NOC sees a warning at the earliest possible instant
(research [`../../research/02-telemetry-pipeline.md`](../../research/02-telemetry-pipeline.md) §3).

```
TelemetrySource / NATS telemetry.>  ─▶  FeatureEngine  ─▶  FeatureVector (features.>)
   (TelemetryRecord stream)              O(1)/record          ▼
                                                         predictive ensemble (Phase 3)
```

## Modules

| Module | What it provides |
|---|---|
| [`features.py`](features.py) | O(1) precursor **feature computers** + streaming sketches. |
| [`detectors.py`](detectors.py) | O(1) change / anomaly **detectors** (drift triggers). |
| [`engine.py`](engine.py) | `FeatureEngine` — record→vector, pluggable registry, per-entity state. |
| [`sources.py`](sources.py) | `TelemetrySource` + NATS JetStream **adapters** feeding the engine. |
| [`alerts.py`](alerts.py) | **Idempotent**, dedup-keyed `AlertEmitter` (at-least-once correction). |

### `features.py` — precursor features (each O(1)/incremental)

Built on `river.stats` (Welford `Mean`/`Var`, `EWMean`/`EWVar`), `ddsketch`
(streaming quantiles), `stumpy.stumpi` (incremental Matrix Profile), and
Count-Min / HyperLogLog sketches. Mapping to the failure modes in
`ARCHITECTURE.md` §6:

| Computer | Precursor | Scenario |
|---|---|---|
| `RollingSlope` | rolling utilisation slope (EWMA of Δ) | A congestion |
| `LatencyDrift` | latency mean-shift (fast EWMA vs slow mean) | A / C |
| `JitterTrend` | jitter variance trend + DDSketch p99 tail | C tunnel |
| `LossProgression` | monotonic loss-ratio rise + rising-streak | C tunnel |
| `ErrorRateAcceleration` | 2nd derivative of error counters | faulty optics |
| `BGPChurnRate` | BGP UPDATE/withdraw event rate | B bgp flap |
| `AdjacencyFlapCount` | adjacency up/down rate (sliding window) | B bgp flap |
| `RekeyIntervalAnomaly` | IPSec rekey-interval \|z\|-score | C tunnel |
| `PathAsymmetry` | fwd vs rev path divergence (EWMA) | B reroute |
| `TopTalkerChurn` | heavy-hitter set Jaccard churn (Count-Min) | A traffic shift |
| `TimeToThreshold` | streaming seconds-to-SLA-crossing | headline "when" |

Plus reusable sketches: `StreamingQuantile` (DDSketch p95/p99),
`MatrixProfileDiscord` (stumpi), `CountMinSketch`, `HyperLogLog`.

### `detectors.py` — O(1) drift / anomaly triggers

`ADWINDetector`, `PageHinkleyDetector`, `KSWINDetector` (`river.drift`), a
dependency-free `CUSUM` and `EWMAControlChart`, and `HalfSpaceTreesDetector`
(`river.anomaly`). **HST inputs are scaled to [0,1] internally** (the classic
HST footgun) via an online `MinMaxScaler`, and the wrapper **scores before
learning** each point. Each detector exposes a uniform `update(value) -> bool`
(fired this tick) plus a stable `.name`.

### `engine.py` — `FeatureEngine`

```python
from netra.streaming import FeatureEngine

engine = FeatureEngine()                       # default feature registry
for record in telemetry_records:               # any iterable of TelemetryRecord
    fv = engine.process(record)                # O(1); FeatureVector or None
    if fv is not None:
        consume(fv)                            # -> Phase 3 ensemble
```

- **Contract-only dependency.** Imports only `netra.contracts` (+ siblings).
  **Never imports `netra.datagen`** — sources are dependency-injected, honouring
  the dual-source abstraction. Tests construct `TelemetryRecord`s directly.
- **Per-entity O(1) state** keyed by `EntityRef.entity_id`; memory is
  O(entities × features), independent of stream length.
- **Pluggable `FeatureRegistry`** maps a metric → the O(1) computers/detectors to
  run; `default_registry()` wires the precursor table above. Register custom
  features without subclassing.
- **One tick = one vector** by default (streaming-first, no window-boundary
  latency); `min_emit_interval_seconds` throttles emission per entity if desired.

### `sources.py` — transport adapters

- `drive_engine(engine, source)` / `iter_telemetry_source(source)` — CPU-only
  path: drive the engine from a `netra.datagen` `TelemetrySource` (injected, not
  imported) or any iterable of records. No NATS, no sim.
- `NatsTelemetrySource` — optional JetStream consumer (`telemetry.>` →
  engine → `features.>`); import-guarded so the module loads without `nats-py`.
- `routing_event_to_records(event)` — maps discrete `RoutingEvent`s onto the
  rate metrics the churn/flap computers consume.

### `alerts.py` — idempotent alert emitter (the at-least-once correction)

Implements the **PR-review correction (P2)** directly. The bus is
**at-least-once**; "exactly-once-effective" needs **(a)** publisher dedup via
`Nats-Msg-Id` **and (b)** confirmed ack (`AckSync`) — otherwise consumers must be
**idempotent on a stable key**.

```python
from netra.streaming import AlertEmitter

ae = AlertEmitter(window_seconds=60, dedup_window_seconds=600)
for fv in engine.run(records):
    for alert in ae.emit_from_feature_vector(fv, scenario="A_congestion"):
        publish("alerts.precursor", alert, nats_msg_id=alert.nats_msg_id)  # (a)
# duplicate deliveries / repeated triggers within the window -> suppressed
```

`make_alert_key(detector, entity_id, scenario, window_index)` is the **stable
key == `Nats-Msg-Id`** (`scenario+entity+window+detector`, sha256-digested).
`AlertEmitter` is a bounded-LRU + time-windowed dedupe, so a redelivered alert
never double-fires (`emitted_count` vs `suppressed_count` expose the effect).

## FeatureVector output contract (the boundary to Phase 3)

Each emitted `FeatureVector` carries:

- `entity: EntityRef` — the universal join key (`site:device:role[:sub]`).
- `timestamp: datetime` — the tick instant.
- `features: dict[str, float]` — named O(1) features. Keys the default engine
  populates include: `util_slope`, `util_eta_seconds`, `latency_drift`,
  `latency_drift_level`, `jitter_ewvar`, `jitter_ewvar_p99`, `loss_progression`,
  `err_accel`, `err_accel_rate`, `rekey_anomaly`, `hst_score` (multivariate),
  and per-metric `*_eta_seconds`. (Free-form by contract — extensible.)
- `triggered_drift: list[str]` — names of detectors that fired this tick (e.g.
  `["page_hinkley:latency_ms", "half_space_trees:hub1:pe-hub1:PE:eth1"]`); the
  fusion layer treats these as votes.
- `sample_count: int` — samples folded so far (warm-up awareness).

## Integration (how the engine plugs in)

1. **Source → engine.** Live: `NatsTelemetrySource` consumes `telemetry.>`.
   CPU-only: `drive_engine(engine, datagen_source)`. The engine is agnostic to
   which produced the records.
2. **Engine → analytics.** Emitted `FeatureVector`s are published to `features.>`
   (or handed in-process to `netra.analytics`). The ensemble consumes
   slope/drift/anomaly features + `triggered_drift` votes to produce `Forecast` /
   `AnomalyScore` / `FusedRisk` / `TimeToImpact`. `util_eta_seconds` etc. are the
   streaming precursor of the calibrated `TimeToImpact` the analytics layer
   refines.
3. **Engine/analytics → alerts.** `AlertEmitter` turns triggers into deduped
   `alerts.>` messages (stamp `Nats-Msg-Id`), consumed idempotently by the
   copilot/incident pipeline.

## Dependencies

Runs on the **CORE tier** ([`../../requirements-core.txt`](../../requirements-core.txt)):
`river`, `stumpy`, `ddsketch`, `nats-py`, `numpy`, `pydantic`. The only **extra**
dep is optional and import-guarded — see
[`requirements-streaming.txt`](requirements-streaming.txt) (`pyprobables` for
Count-Min/HyperLogLog; a built-in fallback runs if it is absent). `stumpy`
(Matrix Profile) and `nats-py` (bus) are also import-guarded so the CPU-only path
runs even if they are missing.

## Tests

[`../../tests/test_streaming.py`](../../tests/test_streaming.py) — 27 tests:
O(1) features vs batch/closed-form reference (Welford, EWMA, slope, DDSketch
p99, CUSUM/EWMA charts), determinism, HST [0,1] scaling, idempotent alert
dedup, and a records/sec throughput smoke test. Run in an isolated venv:

```bash
python -m venv /tmp/venv && /tmp/venv/bin/pip install \
  "pydantic>=2.6,<3" "numpy>=1.26,<2.2" "river>=0.21,<0.22" \
  "ddsketch>=2.0,<3" "stumpy>=1.12,<1.14" pytest
PYTHONPATH=. /tmp/venv/bin/python -m pytest tests/test_streaming.py -q
```
