"""netra.analytics.fusion — score fusion + calibration -> FusedRisk (WS3).

Combines the many detectors (score-normalisation + weighted-agreement across
*independent* families + optional stacking) into one calibrated
``netra.contracts.FusedRisk`` (recording ``MethodWeight`` provenance for every
contributing method) and attaches the ``TimeToImpact``. Calibration (Platt/
isotonic) is trained on the labelled ``ScenarioLabel`` fault scenarios so the
copilot's confidence is honest.

Builder: ``fuse.py`` (fusion -> FusedRisk), ``calibrate.py`` (calibration).
Remember: ``FusedRisk.risk_score>0`` MUST carry ``contributing_methods``.
"""
