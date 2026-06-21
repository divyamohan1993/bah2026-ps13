"""netra.analytics.risk — calibrated alert prioritisation -> Incident (WS4).

Turns fused risk + correlation + blast radius into a ranked, deduplicated,
severity-bucketed ``netra.contracts.Incident`` triage queue using the product-form
risk ``AnomalyConfidence x TimeToImpactUrgency x BlastRadius x AssetCriticality``
(product so a zero factor suppresses false urgency), Platt-calibrated, with
BGP-style flap-penalty suppression to cut alert fatigue (Objective 4).

Builder: ``prioritize.py`` (-> Incident). Report reliability diagram + Brier/ECE
and the alarm compression ratio as evidence.
"""
