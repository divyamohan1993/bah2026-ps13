"""netra.analytics.anomaly — tiered detector bank + EVT thresholds (WS3).

The non-forecasting detectors (#19-#60: statistical/streaming, ML unsupervised,
deep, change-point/drift, matrix-profile, graph) each producing a
``netra.contracts.AnomalyScore`` with a [0,1] ``normalized_score`` for fusion,
plus EVT/SPOT/DSPOT (#68) adaptive, risk-controlled thresholds that replace
hand-set ones.

Builder: ``detectors.py`` (the tiered bank), ``evt.py`` (SPOT/DSPOT). Deep/graph
members feature-flagged; the streaming + batch-ML tiers always run on CPU.
"""
