"""Method registry — the auditable census of NETRA's predictive ensemble.

The problem statement asks for "30+ methods"; this module is the single,
machine-readable place that *substantiates* that claim rather than asserting it.
It enumerates **every deployed method** across the analytics engine — the
forecasting members, the anomaly detector bank, the fusion/threshold operators —
*and* references the streaming-layer's O(1) online detectors/sketches (the
:mod:`netra.streaming` engine), each tagged with:

  * ``name``            — the stable method id (matches ``Forecast.method`` /
    ``AnomalyScore.method`` where the method emits a contract);
  * ``family``          — a coarse grouping (see :data:`FAMILIES`);
  * ``offline_capable`` — runs CPU-only, air-gapped, with no network/GPU;
  * ``optional_heavy``  — needs an optional/heavy backend (torch, stumpy,
    lifelines, …) and degrades to a fallback when that backend is absent.

Where the forecasting/anomaly subpackages already expose their member ids (via
the ensemble/detector-bank constructors) we keep this list aligned with them; the
entries below are the authoritative catalogue used by the docs/UI to render the
"methods deployed" panel and by tests to assert the ≥30 floor.

Use :func:`list_methods` for the flat catalogue and :func:`count_by_family` for
the per-family tally.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Coarse method families (a superset of ``contracts.DetectorFamily`` plus the
#: cross-cutting fusion/streaming buckets that aren't a single detector family).
FAMILIES: tuple[str, ...] = (
    "forecasting",
    "statistical-anomaly",
    "ml-anomaly",
    "deep-anomaly",
    "changepoint",
    "matrixprofile",
    "graph",
    "survival",
    "fusion",
    "streaming",
)


@dataclass(frozen=True)
class MethodInfo:
    """One deployed method in the ensemble census.

    Attributes
    ----------
    name:
        Stable method id (e.g. ``"isolation_forest"``, ``"ewma_holt"``).
    family:
        One of :data:`FAMILIES`.
    offline_capable:
        True if the method runs on the light CPU/air-gapped tier with no network.
    optional_heavy:
        True if the method's *preferred* implementation needs an optional heavy
        dependency (it still runs via a fallback when that dep is absent).
    description:
        One-line human summary for the docs/UI methods panel.
    """

    name: str
    family: str
    offline_capable: bool = True
    optional_heavy: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# The catalogue. Kept aligned with:
#   * forecasting member `.method` ids (netra.analytics.forecasting)
#   * anomaly detector `.method` ids  (netra.analytics.anomaly)
#   * fusion operators                (this subpackage)
#   * streaming detectors/sketches    (netra.streaming)
# ---------------------------------------------------------------------------

_FORECASTING: tuple[MethodInfo, ...] = (
    MethodInfo("ewma_holt", "forecasting", description="EWMA / Holt damped-trend level forecaster."),
    MethodInfo("holt_winters", "forecasting", description="Holt-Winters (triple ES) seasonal forecaster."),
    MethodInfo("theta", "forecasting", description="Theta method (M3-winning decomposition forecaster)."),
    MethodInfo("stl_ets", "forecasting", description="STL decomposition + ETS on the deseasonalised series."),
    MethodInfo("online_arima", "forecasting", description="Online/streaming ARIMA state-space forecaster."),
    MethodInfo("gbm_lag", "forecasting", optional_heavy=True,
               description="Gradient-boosted lag regressor (LightGBM/HistGBR/RF backends)."),
    MethodInfo("chronos_bolt", "forecasting", offline_capable=True, optional_heavy=True,
               description="Chronos-Bolt foundation forecaster (optional local weights)."),
    MethodInfo("forecast_ensemble", "forecasting",
               description="Heterogeneous forecast ensemble (inverse-error + median pool)."),
    MethodInfo("trajectory_crossing", "forecasting",
               description="Time-to-impact from the forecast band's first threshold crossing."),
    MethodInfo("theil_sen_extrapolation", "forecasting",
               description="Robust Theil-Sen slope extrapolation to threshold (TTI cross-check)."),
    MethodInfo("cox_survival", "survival", optional_heavy=True,
               description="Cox proportional-hazards time-to-event (lifelines, MIT)."),
)

_ANOMALY: tuple[MethodInfo, ...] = (
    # statistical
    MethodInfo("robust_z", "statistical-anomaly", description="Robust z / median-MAD outlier score."),
    MethodInfo("ewma_control", "statistical-anomaly", description="EWMA control-chart deviation score."),
    MethodInfo("hbos", "statistical-anomaly", optional_heavy=True,
               description="Histogram-Based Outlier Score (pyod, NumPy fallback)."),
    MethodInfo("copod", "statistical-anomaly", optional_heavy=True,
               description="Copula-Based Outlier Detection (pyod, NumPy fallback)."),
    MethodInfo("ecod", "statistical-anomaly", optional_heavy=True,
               description="Empirical-CDF Outlier Detection (pyod, NumPy fallback)."),
    # ml / unsupervised
    MethodInfo("half_space_trees", "ml-anomaly", optional_heavy=True,
               description="Streaming Half-Space-Trees (river, NumPy fallback)."),
    MethodInfo("isolation_forest", "ml-anomaly", optional_heavy=True,
               description="Isolation Forest (scikit-learn/pyod, fallback)."),
    MethodInfo("lof", "ml-anomaly", optional_heavy=True,
               description="Local Outlier Factor (scikit-learn, fallback)."),
    MethodInfo("forecast_residual", "ml-anomaly",
               description="Predict-then-flag detector on forecast residuals."),
    MethodInfo("pca_recon", "ml-anomaly", optional_heavy=True,
               description="PCA reconstruction-error detector (scikit-learn, fallback)."),
    # change-point / drift
    MethodInfo("page_hinkley", "changepoint", optional_heavy=True,
               description="Page-Hinkley sequential change detector (river, fallback)."),
    MethodInfo("adwin", "changepoint", optional_heavy=True,
               description="ADWIN adaptive-windowing drift detector (river, fallback)."),
    MethodInfo("kswin", "changepoint", optional_heavy=True,
               description="KSWIN Kolmogorov-Smirnov windowed drift detector (river, fallback)."),
    MethodInfo("ruptures_pelt", "changepoint", optional_heavy=True,
               description="Offline PELT/BinSeg change-point segmentation (ruptures, fallback)."),
    # matrix profile
    MethodInfo("matrix_profile", "matrixprofile", optional_heavy=True,
               description="Matrix-profile discord detector (stumpy, NumPy fallback)."),
    # deep
    MethodInfo("autoencoder", "deep-anomaly", offline_capable=True, optional_heavy=True,
               description="Reconstruction autoencoder detector (torch; skipped if absent)."),
)

_FUSION: tuple[MethodInfo, ...] = (
    MethodInfo("weighted_agreement_fusion", "fusion",
               description="Weighted cross-family agreement fusion → FusedRisk.risk_score."),
    MethodInfo("score_unification", "fusion",
               description="Per-method [0,1] score normalisation (rank/z/unify)."),
    MethodInfo("evt_spot_threshold", "fusion",
               description="EVT/SPOT/DSPOT adaptive thresholding of (fused) score streams."),
    MethodInfo("probability_calibration", "fusion", optional_heavy=True,
               description="Platt/isotonic calibration of risk (scikit-learn, NumPy fallback)."),
)

# Streaming-layer methods (netra.streaming) — referenced so the census reflects
# the full deployed surface, not only the offline analytics engine.
_STREAMING: tuple[MethodInfo, ...] = (
    MethodInfo("adwin_stream", "streaming", optional_heavy=True,
               description="O(1) ADWIN drift detector in the live feature engine."),
    MethodInfo("page_hinkley_stream", "streaming", optional_heavy=True,
               description="O(1) Page-Hinkley change detector in the live feature engine."),
    MethodInfo("kswin_stream", "streaming", optional_heavy=True,
               description="O(1) KSWIN drift detector in the live feature engine."),
    MethodInfo("cusum_stream", "streaming",
               description="O(1) CUSUM cumulative-sum change detector."),
    MethodInfo("ewma_control_chart_stream", "streaming",
               description="O(1) EWMA control-chart detector in the feature engine."),
    MethodInfo("half_space_trees_stream", "streaming", optional_heavy=True,
               description="O(1) Half-Space-Trees scorer in the feature engine."),
    MethodInfo("ddsketch_quantile", "streaming", optional_heavy=True,
               description="DDSketch relative-error streaming quantiles (latency p99)."),
    MethodInfo("matrix_profile_stream", "streaming", optional_heavy=True,
               description="stumpi incremental matrix-profile discord in-stream."),
    MethodInfo("count_min_sketch", "streaming",
               description="Count-Min sketch for heavy-hitter / top-talker churn."),
    MethodInfo("hyperloglog", "streaming",
               description="HyperLogLog distinct-flow cardinality estimation."),
)

# Graph-family detectors deployed by the correlation/RCA workstream (WS4). Listed
# here for a complete ensemble census; they consume the same FeatureVector join.
_GRAPH: tuple[MethodInfo, ...] = (
    MethodInfo("correlation_graph", "graph",
               description="Cross-entity correlation-graph anomaly (networkx)."),
    MethodInfo("centrality_shift", "graph",
               description="Betweenness/eigenvector centrality-shift detector."),
)

_ALL_METHODS: tuple[MethodInfo, ...] = (
    _FORECASTING + _ANOMALY + _FUSION + _STREAMING + _GRAPH
)


def list_methods(
    *,
    family: str | None = None,
    offline_only: bool = False,
    exclude_optional_heavy: bool = False,
) -> list[MethodInfo]:
    """Return the catalogue of deployed methods (optionally filtered).

    Parameters
    ----------
    family:
        If given, return only methods in this family (see :data:`FAMILIES`).
    offline_only:
        If True, drop methods that cannot run on the air-gapped CPU tier.
    exclude_optional_heavy:
        If True, drop methods whose preferred backend is an optional heavy dep
        (useful to count the "always-on light tier" only).
    """
    out = list(_ALL_METHODS)
    if family is not None:
        out = [m for m in out if m.family == family]
    if offline_only:
        out = [m for m in out if m.offline_capable]
    if exclude_optional_heavy:
        out = [m for m in out if not m.optional_heavy]
    return out


def count_by_family(methods: list[MethodInfo] | None = None) -> dict[str, int]:
    """Return a ``{family: count}`` tally over ``methods`` (default: all)."""
    src = _ALL_METHODS if methods is None else methods
    tally: dict[str, int] = {f: 0 for f in FAMILIES}
    for m in src:
        tally[m.family] = tally.get(m.family, 0) + 1
    return tally


def method_count() -> int:
    """Total number of deployed methods in the census."""
    return len(_ALL_METHODS)


def method_names() -> list[str]:
    """Just the stable method ids (handy for assertions / docs tables)."""
    return [m.name for m in _ALL_METHODS]


__all__ = [
    "MethodInfo",
    "FAMILIES",
    "list_methods",
    "count_by_family",
    "method_count",
    "method_names",
]
