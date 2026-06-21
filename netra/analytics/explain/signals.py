"""Attributions → :class:`ContributingSignal` list (the data behind Q2).

Converts the signed feature :class:`~netra.analytics.explain.shap_explain.Attribution`
objects into :class:`netra.contracts.ContributingSignal` entries — each pairing a
machine attribution (``shap_value`` + ``direction``) with an operator-readable
``human_explanation`` and an ``observation``. This is exactly what the copilot
quotes for "why is risk elevated / which signals contributed", grounded — never
invented (architecture Q2 row).

The human explanations are generated from a small, metric-aware template table so
they are deterministic and faithful to the computed attribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from netra.contracts import (
    ContributingSignal,
    Direction,
    EntityRef,
    FusedRisk,
)

from .shap_explain import Attribution, attribute_fused_risk

# ---------------------------------------------------------------------------
# Metric-aware human explanation templates. Keyed by a substring of the feature
# name; the first match wins. Each entry: (rising_phrase, falling_phrase).
# ---------------------------------------------------------------------------
_EXPLANATION_TEMPLATES: list[tuple[str, tuple[str, str]]] = [
    ("if_util", (
        "Interface utilisation is climbing toward saturation, the leading "
        "congestion precursor.",
        "Interface utilisation is easing, reducing congestion risk.",
    )),
    ("util", (
        "Link utilisation is trending up toward its capacity ceiling.",
        "Link utilisation is trending down.",
    )),
    ("latency", (
        "Latency is drifting upward, a precursor to SLA degradation.",
        "Latency is recovering toward baseline.",
    )),
    ("jitter", (
        "Jitter is rising, consistent with tunnel/underlay instability.",
        "Jitter is settling back toward baseline.",
    )),
    ("loss", (
        "Packet loss is increasing, directly threatening the SLA.",
        "Packet loss is decreasing.",
    )),
    ("rekey", (
        "IPSec rekey interval is deviating from baseline — an overlay-tunnel "
        "instability signature.",
        "IPSec rekey behaviour is normalising.",
    )),
    ("flap", (
        "Route/adjacency flap penalty is rising, indicating a flap cascade is "
        "building.",
        "Flap penalty is decaying as the link stabilises.",
    )),
    ("bgp", (
        "BGP churn (updates/withdrawals) is elevated, a route-instability "
        "precursor.",
        "BGP churn is subsiding.",
    )),
    ("ospf", (
        "OSPF LSA/SPF activity is elevated, indicating convergence stress.",
        "OSPF convergence activity is returning to normal.",
    )),
    ("adjacency", (
        "Adjacency flaps are increasing, a routing-instability precursor.",
        "Adjacency stability is improving.",
    )),
    ("queue", (
        "Egress queue depth/drops are building, an early congestion indicator.",
        "Queue pressure is easing.",
    )),
    ("config_drift", (
        "Configuration drift from the golden baseline detected — a controller "
        "policy-drift signature.",
        "Configuration is converging back to the golden baseline.",
    )),
    ("path_asymmetry", (
        "Forward/reverse path asymmetry is increasing, often accompanying reroute "
        "events.",
        "Path symmetry is being restored.",
    )),
    ("err", (
        "Interface error counters are accelerating.",
        "Interface error rate is falling.",
    )),
]

_DEFAULT_EXPLANATION = (
    "This signal's contribution is pushing the assessed risk upward.",
    "This signal's contribution is pulling the assessed risk downward.",
)


def _direction_for(value: float | None) -> Direction:
    if value is None:
        return Direction.NEUTRAL
    if value > 1e-9:
        return Direction.INCREASES_RISK
    if value < -1e-9:
        return Direction.DECREASES_RISK
    return Direction.NEUTRAL


def _human_explanation(feature: str, value: float | None) -> str:
    rising = value is None or value >= 0
    low = feature.lower()
    for needle, (up, down) in _EXPLANATION_TEMPLATES:
        if needle in low:
            return up if rising else down
    return _DEFAULT_EXPLANATION[0] if rising else _DEFAULT_EXPLANATION[1]


def _observation_for(attr: Attribution) -> str | None:
    if attr.base_observation is not None:
        return f"observed value {attr.base_observation:.3g} (contribution {attr.value:+.3f})"
    return f"attribution {attr.value:+.3f}"


def attributions_to_signals(
    attributions: Sequence[Attribution],
    *,
    entity: EntityRef | None = None,
    entity_by_feature: Mapping[str, EntityRef] | None = None,
    top_k: int | None = None,
    min_abs_value: float = 0.0,
) -> list[ContributingSignal]:
    """Render attributions as :class:`ContributingSignal` (sorted, optionally top-k).

    Parameters
    ----------
    attributions:
        Signed feature attributions (already ranked by magnitude is fine).
    entity:
        Default entity to attach to each signal (the risk's entity).
    entity_by_feature:
        Optional per-feature entity override (when a signal pertains to a specific
        correlated entity rather than the root).
    top_k:
        Keep only the strongest ``top_k`` signals.
    min_abs_value:
        Drop signals whose |attribution| is below this floor.
    """
    ranked = sorted(attributions, key=lambda a: abs(a.value), reverse=True)
    out: list[ContributingSignal] = []
    for attr in ranked:
        if abs(attr.value) < min_abs_value:
            continue
        ent = None
        if entity_by_feature and attr.feature in entity_by_feature:
            ent = entity_by_feature[attr.feature]
        elif entity is not None:
            ent = entity
        out.append(
            ContributingSignal(
                signal=attr.feature,
                shap_value=round(float(attr.value), 6),
                direction=_direction_for(attr.value),
                observation=_observation_for(attr),
                human_explanation=_human_explanation(attr.feature, attr.value),
                entity=ent,
            )
        )
        if top_k is not None and len(out) >= top_k:
            break
    return out


def explain_fused_risk(
    risk: FusedRisk,
    *,
    feature_values: Mapping[str, float] | None = None,
    model: object | None = None,
    instance=None,
    background=None,
    feature_names: Sequence[str] | None = None,
    entity: EntityRef | None = None,
    top_k: int | None = 8,
) -> list[ContributingSignal]:
    """One-shot: a :class:`FusedRisk` → ranked :class:`ContributingSignal` list (Q2).

    Computes attributions (SHAP if available + model supplied, else the
    deterministic fallback) and renders them as contributing signals attached to
    the risk's entity. This is the function the copilot/risk layer calls to fill
    ``Incident.contributing_signals`` / ``CopilotResponse.contributing_signals``.
    """
    attrs = attribute_fused_risk(
        risk,
        feature_values=feature_values,
        model=model,
        instance=instance,
        background=background,
        feature_names=feature_names,
    )
    return attributions_to_signals(
        attrs,
        entity=entity or risk.entity,
        top_k=top_k,
    )


__all__ = [
    "attributions_to_signals",
    "explain_fused_risk",
]
