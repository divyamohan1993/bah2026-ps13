"""netra.streaming — O(1) online feature engine (Workstream 2).

Consumes raw telemetry (off NATS JetStream or directly from a TelemetrySource)
and folds each sample into constant-memory running statistics
(Welford/EWMA/DDSketch/Page-Hinkley/Half-Space-Trees/stumpi), emitting a
``netra.contracts.FeatureVector`` per entity per tick — the leading-indicator
features the predictive ensemble consumes.

Builder: implement ``features.py`` (the O(1) operators), ``engine.py`` (NATS
consumer loop), ``sources.py`` (direct TelemetrySource adapter for the CPU-only
path). Every feature must be O(1)/amortised-O(1); scale to [0,1] before HST.
"""
