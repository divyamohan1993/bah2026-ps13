"""netra.analytics — predictive ensemble + correlation/RCA/risk/explain.

Sub-packages:
    forecasting/  M1-M24 + foundation -> Forecast; time-to-impact -> TimeToImpact (WS3)
    anomaly/      #19-#60 detectors + EVT/SPOT -> AnomalyScore (WS3)
    fusion/       score-fusion + weighted-agreement + calibration -> FusedRisk (WS3)
    correlation/  graph event-correlation + blast-radius (WS4)
    risk/         calibrated prioritisation -> Incident (WS4)
    explain/      SHAP -> ContributingSignal (WS4)

See netra/analytics/README.md and docs/BUILD_PLAN.md (WS3-WS4).
"""
