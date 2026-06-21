"""netra.analytics.explain — SHAP attributions -> ContributingSignal (WS4, Q2).

Computes feature attributions (TreeSHAP over the tree detectors — exact + fast;
ECOD/COPOD per-dimension tails; violated-PCA loadings) and renders them as
``netra.contracts.ContributingSignal`` (signed value + direction + a grounded
one-line human explanation) — the engine's answer to Q2 ("why / which signals").

Builder: ``shap_explain.py`` (-> ContributingSignal). Pure-Python ``shap``,
CPU-only. These attributions are what the copilot quotes — grounded, never
invented.
"""
