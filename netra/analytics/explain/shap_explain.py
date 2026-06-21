"""Feature attribution for an elevated risk — SHAP (optional) + deterministic fallback.

The engine's answer to Q2 ("why is risk elevated, which signals contributed?")
starts here: compute a signed contribution per feature behind a
:class:`netra.contracts.FusedRisk`, which :mod:`signals` renders into
:class:`netra.contracts.ContributingSignal`.

Two paths (research 04 §9 #65):
  * **SHAP** (TreeSHAP exact + fast for tree detectors; KernelSHAP general) — used
    when the ``shap`` library and a model are available. **Import-guarded**: shap is
    optional/heavy, so its absence must never break this module.
  * **Deterministic fallback** — normalised feature-contribution / permutation
    importance computed in pure NumPy. Reproducible across audits (no model, no
    randomness by default), so the CPU-only path always answers Q2.

The fallback works directly from the ``MethodWeight`` provenance carried on every
``FusedRisk`` (which method fired on which feature, with what weight × score),
mirroring the per-dimension ECOD/COPOD contribution idea — no model object needed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from netra.contracts import FusedRisk

# Optional SHAP — heavy, import-guarded so the module loads without it.
try:  # pragma: no cover - environment-dependent
    import shap as _shap  # type: ignore

    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _shap = None  # type: ignore[assignment]
    _HAVE_SHAP = False


@dataclass
class Attribution:
    """A signed contribution of one feature to the risk score."""

    feature: str
    value: float  # signed contribution magnitude
    base_observation: float | None = None  # the observed feature value, if known
    method: str = "fallback"  # "treeshap" | "kernelshap" | "fallback"
    detail: dict[str, float] = field(default_factory=dict)


def shap_available() -> bool:
    """Whether the optional ``shap`` library imported successfully."""
    return _HAVE_SHAP


# ---------------------------------------------------------------------------
# Primary entry point — attributions for a FusedRisk
# ---------------------------------------------------------------------------
def attribute_fused_risk(
    risk: FusedRisk,
    *,
    feature_values: Mapping[str, float] | None = None,
    model: object | None = None,
    background: np.ndarray | None = None,
    instance: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    prefer_shap: bool = True,
) -> list[Attribution]:
    """Compute signed feature attributions for a fused risk.

    Resolution order:
      1. If ``prefer_shap`` and SHAP + a usable ``model`` (+ instance) are present,
         use TreeSHAP/KernelSHAP.
      2. Otherwise the deterministic fallback over the ``contributing_methods``
         provenance (and ``feature_values`` if supplied).

    Always returns a non-empty list when the risk has any provenance, sorted by
    descending absolute contribution.
    """
    if prefer_shap and _HAVE_SHAP and model is not None and instance is not None:
        attrs = _shap_attributions(model, instance, background, feature_names)
        if attrs:
            return attrs
    return _fallback_attributions(risk, feature_values=feature_values)


# ---------------------------------------------------------------------------
# SHAP path (optional)
# ---------------------------------------------------------------------------
def _shap_attributions(
    model: object,
    instance: np.ndarray,
    background: np.ndarray | None,
    feature_names: Sequence[str] | None,
) -> list[Attribution]:  # pragma: no cover - requires shap + a model
    inst = np.asarray(instance, dtype=float).reshape(1, -1)
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(inst.shape[1])]
    method = "kernelshap"
    values: np.ndarray | None = None
    # try the fast unified Explainer (TreeSHAP when the model is tree-based).
    try:
        explainer = _shap.Explainer(model, background) if background is not None else _shap.Explainer(model)
        sv = explainer(inst)
        values = np.asarray(sv.values).reshape(-1)
        method = "treeshap" if "Tree" in type(explainer).__name__ else "shap"
    except Exception:
        try:
            predict = getattr(model, "decision_function", None) or getattr(
                model, "predict", None
            )
            bg = background if background is not None else inst
            explainer = _shap.KernelExplainer(predict, bg)
            values = np.asarray(explainer.shap_values(inst, silent=True)).reshape(-1)
            method = "kernelshap"
        except Exception:
            return []
    if values is None or values.size != len(names):
        return []
    attrs = [
        Attribution(feature=names[i], value=float(values[i]), method=method)
        for i in range(len(names))
    ]
    attrs.sort(key=lambda a: abs(a.value), reverse=True)
    return attrs


# ---------------------------------------------------------------------------
# Deterministic fallback — works from MethodWeight provenance + feature values
# ---------------------------------------------------------------------------
def _fallback_attributions(
    risk: FusedRisk,
    *,
    feature_values: Mapping[str, float] | None = None,
) -> list[Attribution]:
    """Normalised feature-contribution attribution from the firing detectors.

    Each contributing method votes ``weight × normalized_score`` onto the feature
    it fired on (the method's ``feature``/metric, if recoverable, else the method
    name). Contributions are aggregated per feature and normalised to sum to the
    risk score, giving a signed, reproducible attribution with no model needed.
    """
    contrib: dict[str, float] = {}
    obs: dict[str, float] = {}

    for mw in risk.contributing_methods:
        feat = _feature_for_method(mw.method)
        weight_score = float(mw.weight) * float(mw.normalized_score)
        contrib[feat] = contrib.get(feat, 0.0) + weight_score

    # fold in externally-supplied feature observations (e.g. from FeatureVector):
    # a high observed value reinforces that feature's contribution.
    if feature_values:
        for name, val in feature_values.items():
            obs[name] = float(val)
            if name not in contrib:
                contrib[name] = abs(float(val))

    if not contrib:
        # nothing to attribute — surface the predicted issue as a single signal.
        return [
            Attribution(
                feature=risk.predicted_issue.value,
                value=float(risk.risk_score),
                method="fallback",
            )
        ]

    total = sum(abs(v) for v in contrib.values())
    scale = (risk.risk_score / total) if total > 0 else 0.0
    attrs: list[Attribution] = []
    for feat, val in contrib.items():
        attrs.append(
            Attribution(
                feature=feat,
                value=round(val * scale, 6) if scale else round(val, 6),
                base_observation=obs.get(feat),
                method="fallback",
                detail={"raw_contribution": round(val, 6)},
            )
        )
    attrs.sort(key=lambda a: abs(a.value), reverse=True)
    return attrs


def permutation_importance_fallback(
    predict: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    feature_names: Sequence[str],
    *,
    background: np.ndarray | None = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> list[Attribution]:
    """Model-agnostic permutation-importance attribution (no shap needed).

    For each feature, measure how much the model's score changes when that feature
    is replaced by background-sampled values; the mean absolute change is its
    importance. Deterministic given ``seed``. Used when a callable model is present
    but ``shap`` is not.
    """
    rng = np.random.default_rng(seed)
    inst = np.asarray(instance, dtype=float).reshape(1, -1)
    n_features = inst.shape[1]
    names = list(feature_names)[:n_features]
    base = float(np.asarray(predict(inst)).reshape(-1)[0])

    bg = (
        np.asarray(background, dtype=float)
        if background is not None
        else np.tile(inst, (8, 1))
    )
    attrs: list[Attribution] = []
    for j in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            perturbed = inst.copy()
            perturbed[0, j] = bg[rng.integers(0, bg.shape[0]), j]
            score = float(np.asarray(predict(perturbed)).reshape(-1)[0])
            deltas.append(base - score)
        imp = float(np.mean(deltas))
        attrs.append(
            Attribution(
                feature=names[j] if j < len(names) else f"f{j}",
                value=round(imp, 6),
                base_observation=float(inst[0, j]),
                method="permutation",
            )
        )
    attrs.sort(key=lambda a: abs(a.value), reverse=True)
    return attrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _feature_for_method(method: str) -> str:
    """Best-effort feature name a detector method fired on.

    Many detector ids encode the metric (e.g. ``page_hinkley:latency_ms``); split
    on ':' to recover it. Otherwise the method name itself is the feature label.
    """
    if ":" in method:
        return method.split(":", 1)[1]
    return method


__all__ = [
    "Attribution",
    "attribute_fused_risk",
    "permutation_importance_fallback",
    "shap_available",
]
