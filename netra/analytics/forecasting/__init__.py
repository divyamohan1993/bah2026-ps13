"""netra.analytics.forecasting — trajectory forecasting + time-to-impact (WS3).

Tiered forecasters (always-on CPU classical/state-space/GBDT + feature-flagged
deep/foundation members) producing ``netra.contracts.Forecast``, and the
time-to-impact estimators (forecast-trajectory threshold-crossing + Theil-Sen
extrapolation + Cox/RSF survival) producing ``netra.contracts.TimeToImpact`` —
the engine's answer to Q1 ("what fails next AND WHEN").

Builder: ``ensemble.py`` (forecasters), ``timeimpact.py`` (TTI). Feature-flag
deep/foundation members; the CPU ensemble alone must yield usable forecasts.
"""
