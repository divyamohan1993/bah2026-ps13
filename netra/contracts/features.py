"""Streaming feature contract — output of the O(1) online feature engine.

The streaming workstream (``netra.streaming``) consumes raw telemetry off the
bus and folds each sample into running, constant-memory statistics (Welford /
EWMA / DDSketch / Page-Hinkley / Half-Space-Trees / stumpy). The result, per
entity per tick, is a :class:`FeatureVector` — the leading-indicator features
the predictive ensemble and the copilot consume.

Keeping features in a free-form ``dict[str, float]`` (rather than a fixed schema)
lets the streaming team add/remove online features without an interface change,
while ``triggered_drift`` surfaces the discrete precursor triggers (e.g. a
Page-Hinkley change-point firing) that the fusion layer treats as votes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import EntityRef, NetraModel


class FeatureVector(NetraModel):
    """O(1) streaming features for one entity at one instant.

    Standard feature keys the streaming engine SHOULD populate where applicable
    (free-form so it is extensible; documented here for cross-team alignment):

      * ``util_ewma`` / ``util_slope``      — rolling utilisation level & slope
      * ``latency_ewma`` / ``latency_p99``  — latency mean & DDSketch tail
      * ``jitter_ewvar``                     — jitter variance trend
      * ``loss_ewma``                        — loss-ratio EWMA
      * ``err_accel``                        — 2nd-derivative of error counters
      * ``bgp_churn_rate`` / ``flap_penalty``— routing instability features
      * ``adj_flap_rate``                    — adjacency flap rate
      * ``rekey_anomaly``                    — IPSec rekey-interval deviation
      * ``path_asymmetry``                   — fwd/rev path divergence
      * ``hst_score``                        — Half-Space-Trees anomaly score
      * ``mp_discord``                       — stumpy matrix-profile discord
    """

    entity: EntityRef = Field(..., description="Entity these features describe.")
    timestamp: datetime = Field(..., description="UTC instant of the feature snapshot.")
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Named O(1) streaming features (see class docstring).",
    )
    triggered_drift: list[str] = Field(
        default_factory=list,
        description="Names of drift/change-point detectors that fired this tick "
        "(e.g. ['page_hinkley:latency_ms', 'adwin:bgp_update_rate']).",
    )
    window_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Effective look-back window backing rolling features, if any.",
    )
    sample_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of samples folded into these running stats so far.",
    )


__all__ = ["FeatureVector"]
