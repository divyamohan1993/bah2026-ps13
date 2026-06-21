# `netra/analytics/` — Predictive analytics engine

The 50+ method predictive engine and the correlation/RCA/risk/explain layer.
Split across two workstreams:

**Workstream 3 — predictive ensemble:**
- `forecasting/` — trajectory forecasters (M1–M24 + foundation) → `Forecast`;
  time-to-impact (trajectory-crossing + survival) → `TimeToImpact`.
- `anomaly/` — tiered detector bank (#19–#60: statistical / ML / deep /
  change-point / matrix-profile / graph) → `AnomalyScore`; EVT/SPOT/DSPOT
  adaptive thresholds.
- `fusion/` — score-normalisation + weighted-agreement across independent
  families + stacking → `FusedRisk` (with `MethodWeight` provenance); Platt/
  isotonic calibration trained on `ScenarioLabel`s.

**Workstream 4 — correlation / RCA / risk / explain:**
- `correlation/` — NetworkX digital twin; temporal + topological event
  correlation (WCC/SCC); blast-radius via BFS reachability ∩ NetFlow.
- `risk/` — product-form risk + severity bucketing + flap suppression →
  `Incident`.
- `explain/` — TreeSHAP/ECOD attributions → `ContributingSignal` (Q2).

**Contracts in/out:** consumes `FeatureVector`, `TelemetryRecord`,
`RoutingEvent`, `FlowRecord`, `ScenarioLabel`; produces `Forecast`,
`AnomalyScore`, `FusedRisk`, `TimeToImpact`, `Incident`, `ContributingSignal`.

**Degradation:** the CPU statistical + gradient-boosted + change-point + graph +
survival members always run; deep/foundation members are feature-flagged on when
a GPU/time is available. See [`../../docs/BUILD_PLAN.md`](../../docs/BUILD_PLAN.md)
WS3–WS4.
