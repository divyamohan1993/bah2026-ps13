"""netra.analytics.explain — SHAP attributions -> ContributingSignal (WS4, Q2).

Computes feature attributions (TreeSHAP over the tree detectors — exact + fast;
ECOD/COPOD per-dimension tails; violated-PCA loadings) and renders them as
``netra.contracts.ContributingSignal`` (signed value + direction + a grounded
one-line human explanation) — the engine's answer to Q2 ("why / which signals").

Builder: ``shap_explain.py`` (attributions; shap optional + deterministic
fallback), ``signals.py`` (-> ContributingSignal). CPU-only. These attributions
are what the copilot quotes — grounded, never invented.
"""

from __future__ import annotations

from .shap_explain import (
    Attribution,
    attribute_fused_risk,
    permutation_importance_fallback,
    shap_available,
)
from .signals import attributions_to_signals, explain_fused_risk

__all__ = [
    "Attribution",
    "attribute_fused_risk",
    "permutation_importance_fallback",
    "shap_available",
    "attributions_to_signals",
    "explain_fused_risk",
]
