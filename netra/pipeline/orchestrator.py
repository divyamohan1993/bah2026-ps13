"""``NetraPipeline`` — the end-to-end NETRA chain wired into one runnable object.

This is the integration layer (ARCHITECTURE.md §2 data-flow): it threads the seven
module workstreams into a single offline, CPU-only pipeline and reconciles the
interface gaps between them. The wired chain, in order, is::

    TelemetrySource (netra.datagen)
      └─> FeatureEngine                         # streaming O(1) FeatureVectors
            └─> per-(entity,metric) history      # rolling buffers kept HERE (the
                                                 #   gap: forecasters want a series,
                                                 #   the engine emits per-tick vectors)
                  ├─> streaming detectors/tick   # robust-z/EWMA/Page-Hinkley/ADWIN
                  │     + batch ML at assembly    #   ECOD/LOF/PCA/MP/HST -> AnomalyScore
                  └─> EnsembleForecaster on the series
                        └─> TimeToImpactEstimator -> TimeToImpact          (Q1 "when")
            └─> RiskFuser.fuse(scores, forecast, tti) -> FusedRisk(+TTI)    (per entity)
                  └─> correlate_to_incidents(graph, anomalies=, fused=, flows=)
                        └─> explain_fused_risk -> ContributingSignal[]      (Q2 "why")
                              └─> prioritize_incidents -> ranked Incident[]
                                    └─> Copilot.answer(req, analytics_context)
                                          └─> CopilotResponse               (Q1/Q2/Q3)

Two entry points:

  * :meth:`run_scenario` — batch: consume a whole :class:`TelemetrySource` (or
    scenario id), run the full chain, return a :class:`SituationReport` with the
    ranked incidents, the FusedRisk timeline, the copilot answers and the
    per-scenario evaluation (predicted? lead time vs the ground-truth label?).
  * :meth:`process` — incremental/streaming: fold one ``TelemetryRecord`` at a
    time (O(1) features + online detectors), so the same pipeline drives a live
    stream; call :meth:`assemble` to materialise the current incidents/report.

Design notes
------------
* **Adapter, not invasive edit.** The pipeline keeps the per-(entity,metric)
  history buffers the forecasters need (the engine is intentionally streaming-only
  and stateless across metrics), and builds the correlation graph from the datagen
  reference topology (see :mod:`~netra.pipeline.topology_adapter`). No module was
  modified to wire this.
* **Import-light + graceful degradation.** Only ``netra.*`` is imported at module
  load; the heavy ensemble/LLM members each degrade to CPU/template fallbacks, so
  the whole pipeline runs on the core tier with no GPU/model/sim/internet.
* **Bounded work.** Detectors + forecasters are run only on the *precursor*
  metrics that carry SLA risk (utilisation, latency, jitter, loss, tunnel
  loss/jitter/rekey, BGP churn/flap, config-drift) and only on entities that
  actually emit them, so a full 5-site run stays CPU-fast.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from netra.analytics.anomaly import (
    AdwinDetector,
    Detector,
    EwmaControlChart,
    PageHinkleyDetector,
    RobustZDetector,
)
from netra.analytics.correlation import (
    TopologyGraph,
    correlate_to_incidents,
)
from netra.analytics.explain import explain_fused_risk
from netra.analytics.forecasting import EnsembleForecaster
from netra.analytics.forecasting.tti import TimeToImpactEstimator
from netra.analytics.fusion import RiskFuser
from netra.analytics.risk import prioritize_incidents
from netra.contracts import (
    AnomalyScore,
    ContributingSignal,
    CopilotRequest,
    CopilotResponse,
    EntityRef,
    FlowRecord,
    FusedRisk,
    Incident,
    IssueType,
    MetricName,
    ScenarioId,
    ScenarioLabel,
    TelemetryKind,
    TelemetryRecord,
    TunnelStat,
)
from netra.copilot import AnalyticsContext, Copilot
from netra.datagen import (
    SyntheticSource,
    TelemetrySource,
    record_timestamp,
)
from netra.streaming import FeatureEngine

from .report import RiskPoint, ScenarioEval, SituationReport
from .topology_adapter import build_pipeline_graph

# --------------------------------------------------------------------------- #
# Which metrics carry SLA/precursor risk and the breach threshold + direction. #
# (Mirrors streaming.DEFAULT_SLA_THRESHOLDS + the precursor table in           #
#  ARCHITECTURE.md §6; kept here so the pipeline can read off a TimeToImpact.)  #
# --------------------------------------------------------------------------- #
# A stream whose peak normalised detector score is below this is "healthy" this
# tick — we skip fusing/recording it to keep the per-tick path O(streams_active),
# not O(streams_total). Tuned low so precursor onsets are still captured.
_TIMELINE_FLOOR = 0.20

# Genuine-excursion gate (see ``NetraPipeline._is_genuine_excursion``): a stream
# becomes an incident candidate only if its peak is a robust z >= this from its
# baseline, OR reaches this fraction of its SLA threshold. Stops detector noise on
# near-flat baselines from spawning spurious incidents.
_EXCURSION_Z = 6.0
_EXCURSION_THRESHOLD_FRAC = 0.5
# the absolute peak excursion must also clear this fraction of the SLA threshold
# (so a tiny-but-high-z blip on a flat baseline is not mistaken for a precursor).
# Set above the synthetic baseline noise (which peaks at ~40% of the loss/discards
# thresholds) so only genuine, large excursions qualify via the z-path.
_EXCURSION_MIN_ABS_FRAC = 0.50


@dataclass(frozen=True)
class _MetricSpec:
    threshold: float
    above_is_breach: bool = True  # True: crossing ABOVE breaches (rising risk)


# Metrics the pipeline forecasts + scores (the leading indicators). Everything
# else (octets, counts) is still folded into features but not separately fused.
# NOTE: thresholds double as the SLA-breach level for the time-to-impact AND the
# scale for the genuine-excursion gate. They are set above the synthetic-baseline
# noise floor (loss/discards/jitter baselines peak around 2-3; injected faults
# reach 10-225) so normal diurnal variation does not spawn spurious incidents
# while the labeled faults trigger cleanly.
_PRECURSOR_METRICS: dict[str, _MetricSpec] = {
    MetricName.IF_UTIL_PCT.value: _MetricSpec(90.0),
    MetricName.LATENCY_MS.value: _MetricSpec(150.0),
    MetricName.JITTER_MS.value: _MetricSpec(30.0),
    MetricName.LOSS_PCT.value: _MetricSpec(5.0),
    MetricName.IF_OUT_DISCARDS.value: _MetricSpec(80.0),
    MetricName.TUNNEL_LOSS_PCT.value: _MetricSpec(5.0),
    MetricName.TUNNEL_JITTER_MS.value: _MetricSpec(30.0),
    MetricName.TUNNEL_REKEY_INTERVAL_S.value: _MetricSpec(2500.0, above_is_breach=False),
    MetricName.BGP_UPDATE_RATE.value: _MetricSpec(30.0),
    MetricName.BGP_FLAP_PENALTY.value: _MetricSpec(2000.0),
    MetricName.ADJ_FLAP_COUNT.value: _MetricSpec(0.5),
    MetricName.OSPF_LSA_RATE.value: _MetricSpec(20.0),
    MetricName.CONFIG_DRIFT_SCORE.value: _MetricSpec(0.5),
}

# Map a precursor metric to the fault class it most directly evidences, so the
# fused risk / incident carries a meaningful predicted_issue (the fusion layer's
# family-hint is coarse; this is metric-precise and used as an override).
_METRIC_ISSUE: dict[str, IssueType] = {
    MetricName.IF_UTIL_PCT.value: IssueType.INTERFACE_CONGESTION,
    MetricName.IF_OUT_DISCARDS.value: IssueType.INTERFACE_CONGESTION,
    MetricName.LATENCY_MS.value: IssueType.LATENCY_DRIFT,
    MetricName.JITTER_MS.value: IssueType.LATENCY_DRIFT,
    MetricName.LOSS_PCT.value: IssueType.INTERFACE_CONGESTION,
    MetricName.TUNNEL_LOSS_PCT.value: IssueType.TUNNEL_DEGRADATION,
    MetricName.TUNNEL_JITTER_MS.value: IssueType.TUNNEL_DEGRADATION,
    MetricName.TUNNEL_REKEY_INTERVAL_S.value: IssueType.TUNNEL_DEGRADATION,
    MetricName.BGP_UPDATE_RATE.value: IssueType.BGP_ROUTE_FLAP,
    MetricName.BGP_FLAP_PENALTY.value: IssueType.BGP_ROUTE_FLAP,
    MetricName.ADJ_FLAP_COUNT.value: IssueType.BGP_ROUTE_FLAP,
    MetricName.OSPF_LSA_RATE.value: IssueType.OSPF_CONVERGENCE_STRESS,
    MetricName.CONFIG_DRIFT_SCORE.value: IssueType.POLICY_DRIFT,
}


@dataclass
class PipelineConfig:
    """Tunables for a :class:`NetraPipeline` run (all have sensible defaults).

    The ``profile`` selects a coarse speed/fidelity trade-off and, in ``__post_init__``,
    seeds the forecasting/enrichment knobs accordingly (any field the caller sets
    explicitly still wins — the profile only fills in fields left at their sentinel).

      * ``"fast"`` (the demo/test default) — runs ONLY the lightweight O(n)
        forecasters (EWMA/Holt linear trend, Theta, damped Holt) and SKIPS the slow
        members (river SNARIMAX online-ARIMA, statsmodels SARIMAX/STL, gradient
        boosting). It also PRE-SCREENS with the cheap streaming anomaly detectors and
        runs the forecast + time-to-impact branch on only the top-``forecast_top_k``
        most-anomalous (entity, metric) streams, downsampling long series and capping
        the horizon. Output contract is unchanged — the precursor entity is still
        detected with lead time, it is just not the full cartesian fan-out.
      * ``"full"`` — the original heterogeneous ensemble across every qualifying
        stream (highest cross-model agreement, slower).
    """

    #: speed/fidelity profile: ``"fast"`` (default) or ``"full"``. See class docstring.
    profile: str = "fast"
    #: rolling history length per (entity, metric) feeding the forecasters.
    history_len: int = 240
    #: min samples before a forecaster/TTI is run on a series.
    min_history_for_forecast: int = 12
    #: forecast horizon (steps) and per-step seconds (set from the source cadence).
    forecast_steps: int = 24
    step_seconds: float = 10.0
    #: a fused risk at/above this is an "elevated risk" alert (lead-time credit).
    alert_risk_threshold: float = 0.30
    #: only run the (costlier) forecast + time-to-impact branch on streams whose
    #: peak detector score clears this higher bar (a forecast on a barely-noisy
    #: stream is neither meaningful nor worth its cost).
    forecast_risk_floor: float = 0.55
    #: a fused risk must clear this to become a correlation event (an "incident
    #: candidate"). The genuine-excursion gate (``_is_genuine_excursion``) is the
    #: primary noise filter; this is a secondary floor that simply drops near-zero
    #: residual risk on streams that passed the gate but fused weakly.
    incident_risk_floor: float = 0.40
    #: correlation sliding-window width (seconds). Kept moderate so temporally
    #: distinct scenarios (whose onsets are minutes apart) form separate incidents
    #: rather than chaining into one through the densely-connected topology.
    correlation_window_seconds: float = 240.0
    #: how many top incidents get a copilot answer.
    copilot_top_n: int = 3
    #: run the heavier ensemble forecaster (vs a single fast member). CPU-cheap
    #: either way on these short series; the ensemble adds cross-model agreement.
    use_ensemble_forecaster: bool = True
    #: FAST-mode forecasting: restrict the ensemble to the lightweight O(n) members
    #: (EWMA/Holt linear, Theta, damped Holt) and skip the slow ones (river
    #: SNARIMAX, statsmodels SARIMAX/STL, gradient boosting). ``None`` = let the
    #: ``profile`` decide (fast ⇒ True, full ⇒ False).
    lightweight_forecasters: bool | None = None
    #: FAST-mode pre-screen: after the cheap anomaly detectors rank the streams,
    #: run the (costlier) forecast + time-to-impact branch on only the top-K
    #: most-anomalous (entity, metric) streams rather than the full cartesian
    #: product. 0 = unbounded (the ``full`` profile). Correctness is preserved
    #: because the scenario precursor is, by construction, among the highest-scoring
    #: streams; lead-time detection itself comes from the per-tick risk timeline
    #: (which is unaffected), not from the forecast.
    forecast_top_k: int = 0
    #: FAST-mode downsample factor for the series handed to the forecasters (keep 1
    #: point in N). 1 = no downsampling. The TTI estimator is told the effective
    #: per-step seconds so the eta stays in real time.
    forecast_downsample: int = 1
    #: cap on distinct (entity, metric) detector streams (safety valve for huge
    #: topologies); 0 = unbounded.
    max_streams: int = 0
    #: The per-tick path runs only the cheap streaming detectors
    #: (:func:`_build_pertick_detectors`: robust-z, EWMA control chart,
    #: Page-Hinkley, ADWIN — 3 independent families). The heavier batch ML
    #: detectors (iForest/COPOD/ECOD/LOF/PCA/matrix-profile/Half-Space-Trees) RE-FIT
    #: or maintain trees and so run ONCE as a batch enrichment at assembly (see
    #: ``batch_enrich_*``) — the architecture's "Tier-1 streaming / Tier-2 batch"
    #: split (research 04 §12).
    #: at assembly, run the heavier tier-2 batch detectors ONCE over the recent
    #: window of the highest-risk streams, to add independent-family agreement.
    batch_enrich: bool = True
    #: how many top-risk streams get the batch tier-2 enrichment pass.
    batch_enrich_top_streams: int = 12
    #: how many trailing samples of each enriched stream the batch pass scores.
    batch_enrich_window: int = 120

    def __post_init__(self) -> None:
        """Seed the speed/fidelity knobs from ``profile`` (explicit caller wins).

        Only fields left at their sentinel (``None`` for the tristate flag, ``0``
        for the caps) are filled in, so a caller that sets e.g. ``forecast_top_k=5``
        keeps it regardless of profile.
        """
        prof = (self.profile or "fast").lower()
        if prof not in ("fast", "full"):
            prof = "fast"
        self.profile = prof
        fast = prof == "fast"
        if self.lightweight_forecasters is None:
            self.lightweight_forecasters = fast
        if fast:
            # only seed when left at the unbounded sentinel so callers can override
            if self.forecast_top_k == 0:
                self.forecast_top_k = 10
            if self.forecast_downsample <= 1:
                # ~24 samples/min at step=10s → keep every other point for the
                # forecast (the trend/threshold-crossing is unchanged at this rate).
                self.forecast_downsample = 2


def _build_pertick_detectors(entity: EntityRef, metric: str) -> list[Detector]:
    """The cheap O(1) streaming detectors run on EVERY tick of EVERY stream.

    Three *independent families* — statistical (robust-z + EWMA control chart) and
    change-point/drift (Page-Hinkley + ADWIN) — which are the precursor-firing
    members that earn lead time, and which cost microseconds per sample. The
    heavier Half-Space-Trees / Isolation-Forest / COPOD / matrix-profile members
    (which re-fit or maintain trees) are deferred to the once-per-run batch
    enrichment pass so the per-tick path stays fast on a full 5-site topology.
    """
    return [
        RobustZDetector(entity, metric),
        EwmaControlChart(entity, metric),
        PageHinkleyDetector(entity, metric),
        AdwinDetector(entity, metric),
    ]


def _build_batch_detectors(
    entity: EntityRef, metric: str, *, lightweight: bool = False
) -> list[Detector]:
    """Curated, *fast-to-prime* batch detectors for the once-per-run enrichment.

    Adds further independent families on top of the per-tick streaming set —
    ML-unsupervised (ECOD + LOF + PCA-reconstruction), matrix-profile (discord) and
    Half-Space-Trees — so the fused risk reflects cross-family agreement across
    statistical + change-point + ML + matrix-profile evidence. The heavier-to-prime
    members (Isolation Forest, HBOS, COPOD) are intentionally omitted here: their
    ``fit`` rescans the whole reference window and dominates runtime while adding
    little beyond ECOD/PCA for these univariate streams.

    ``lightweight`` (the FAST profile) additionally drops the matrix-profile member:
    it is backed by ``stumpy``/``numba`` whose one-time JIT compilation dominates
    the whole run (~50 s of warm-up), yet on these short univariate streams it adds
    little beyond ECOD/PCA. Four independent families (statistical + change-point +
    ML-unsupervised + Half-Space-Trees) still agree, so the cross-verification claim
    and the ``methods_fired >= 3`` contract hold.
    """
    from netra.analytics.anomaly import (
        EcodDetector,
        HalfSpaceTreesDetector,
        LofDetector,
        PcaReconstructionDetector,
    )

    dets: list[Detector] = [
        EcodDetector(entity, metric),
        LofDetector(entity, metric),
        PcaReconstructionDetector(entity, metric),
        HalfSpaceTreesDetector(entity, metric),
    ]
    if not lightweight:
        from netra.analytics.anomaly import MatrixProfileDiscordDetector

        dets.append(MatrixProfileDiscordDetector(entity, metric))
    return dets


class _PerTickBank:
    """A minimal detector bank: run a fixed list of streaming detectors per sample."""

    def __init__(self, detectors: list[Detector]) -> None:
        self.detectors = detectors

    def update(self, value: float, timestamp: datetime | None = None) -> list[AnomalyScore]:
        out: list[AnomalyScore] = []
        for d in self.detectors:
            try:
                out.append(d.update(value, timestamp=timestamp))
            except Exception:
                continue
        return out


@dataclass
class _Stream:
    """Per-(entity, metric) online state: history + detector bank + last scores."""

    entity: EntityRef
    metric: str
    spec: _MetricSpec
    values: deque = field(default_factory=lambda: deque(maxlen=240))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=240))
    bank: _PerTickBank | None = None
    last_scores: list[AnomalyScore] = field(default_factory=list)
    #: snapshot of the detector scores AT the tick of highest risk (so correlation
    #: receives onset-stamped events spread over time, not all at the final tick).
    peak_scores: list[AnomalyScore] = field(default_factory=list)
    peak_value: float = 0.0
    #: values/timestamps captured up to the peak tick (history for the peak forecast).
    peak_history: list[float] = field(default_factory=list)
    peak_timestamps: list[datetime] = field(default_factory=list)


class NetraPipeline:
    """The wired end-to-end NETRA pipeline (streaming + batch).

    Construct once (it builds the topology graph, the fuser, the TTI estimator and
    — lazily — the copilot), then either :meth:`run_scenario` for a batch or feed
    records through :meth:`process` and call :meth:`assemble`.

    Parameters
    ----------
    config:
        A :class:`PipelineConfig` (defaults are CPU-fast and demo-ready).
    topology:
        A pre-built correlation :class:`TopologyGraph`; defaults to the datagen
        reference topology via :func:`~netra.pipeline.topology_adapter.build_pipeline_graph`.
    copilot:
        A pre-built :class:`~netra.copilot.Copilot`; defaults to a lazily-built one
        on the template-fallback path (no model/RAG heavy deps).
    prefer_models:
        Pass through to the copilot (try llama.cpp/bge/HHEM); default False so the
        run is fully offline on the core tier.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        topology: TopologyGraph | None = None,
        copilot: Copilot | None = None,
        prefer_models: bool = False,
    ) -> None:
        self.config = config or PipelineConfig()
        self.topology = topology or build_pipeline_graph()
        self._prefer_models = prefer_models
        self._copilot = copilot  # built lazily so import stays light
        # The per-stream DetectorBank already runs Half-Space-Trees per
        # (entity,metric); the FeatureEngine's own multivariate HST would be
        # redundant here and is the single most expensive per-tick op, so disable
        # it — we consume the engine's O(1) ``features`` dict, not its HST trigger.
        self.engine = FeatureEngine(enable_hst=False)
        self.fuser = RiskFuser()
        self.tti = TimeToImpactEstimator(sample_period_seconds=self.config.step_seconds)

        # online state
        self._streams: dict[tuple[str, str], _Stream] = {}
        self._latest_feature: dict[str, dict] = {}  # entity_id -> features dict
        self._latest_drift: dict[str, list[str]] = {}
        self._flows: list[FlowRecord] = []
        # full fused-risk trajectory per entity (for the timeline + eval)
        self._risk_history: dict[str, list[RiskPoint]] = defaultdict(list)
        self._records_seen = 0
        self._first_ts: datetime | None = None
        self._last_ts: datetime | None = None
        self._labels: list[ScenarioLabel] = []

    # ------------------------------------------------------------------ #
    # Copilot (lazy)                                                      #
    # ------------------------------------------------------------------ #
    @property
    def copilot(self) -> Copilot:
        if self._copilot is None:
            self._copilot = Copilot(prefer_models=self._prefer_models)
        return self._copilot

    # ================================================================== #
    # Batch entry point                                                  #
    # ================================================================== #
    def run_scenario(
        self,
        scenario: TelemetrySource | ScenarioId | None = None,
        *,
        labels: list[ScenarioLabel] | None = None,
        seed: int = 1337,
        duration_s: float = 1800.0,
        step_s: float | None = None,
        answer_copilot: bool = True,
    ) -> SituationReport:
        """Run the full pipeline over a telemetry source and return a report.

        ``scenario`` may be:
          * a :class:`~netra.datagen.TelemetrySource` (used directly, with its
            ``labels()`` as ground truth) — the usual path;
          * a :class:`~netra.contracts.ScenarioId` (a fresh
            :class:`~netra.datagen.SyntheticSource` is built for *only* that
            scenario, so the eval isolates one fault morphology);
          * ``None`` (a fresh all-four-scenario :class:`SyntheticSource`).
        """
        source = self._resolve_source(scenario, seed=seed, duration_s=duration_s, step_s=step_s)
        self._labels = list(labels) if labels is not None else list(source.labels())

        # feed the whole stream through the incremental path
        for rec in source.iter_records():
            self.process(rec)

        return self.assemble(answer_copilot=answer_copilot)

    def _resolve_source(
        self,
        scenario: TelemetrySource | ScenarioId | None,
        *,
        seed: int,
        duration_s: float,
        step_s: float | None,
    ) -> TelemetrySource:
        if isinstance(scenario, TelemetrySource):
            return scenario
        step = step_s if step_s is not None else self.config.step_seconds
        if isinstance(scenario, ScenarioId):
            scenarios = () if scenario == ScenarioId.BASELINE else (scenario,)
            return SyntheticSource(
                seed=seed, duration_s=duration_s, step_s=step, scenarios=scenarios
            )
        return SyntheticSource(seed=seed, duration_s=duration_s, step_s=step)

    # ================================================================== #
    # Incremental entry point                                            #
    # ================================================================== #
    def process(self, record: object) -> None:
        """Fold one telemetry record into the online state (O(1) amortised).

        Routes by type: only :class:`TelemetryRecord` drives the feature engine +
        detectors + history buffers; :class:`FlowRecord` is retained for the blast
        radius; routing/syslog events are carried implicitly through the numeric
        ``BGP_*``/``ADJ_*`` TelemetryRecords the generator emits alongside them.
        """
        self._records_seen += 1
        ts = record_timestamp(record) if hasattr(record, "timestamp") else None
        if ts is not None:
            if self._first_ts is None:
                self._first_ts = ts
            self._last_ts = ts

        if isinstance(record, FlowRecord):
            self._flows.append(record)
            return
        if isinstance(record, TunnelStat):
            # Tunnel health (scenario C) arrives as its own record type; project its
            # loss/jitter/rekey fields onto the same per-(entity,metric) streaming
            # path as everything else (the entity id matches the generator's
            # ``Topology.tunnel_entity_id`` so it joins the topology correctly).
            for tr in _tunnelstat_to_records(record):
                self._process_telemetry(tr)
            return
        if not isinstance(record, TelemetryRecord):
            return  # SyslogEvent / RoutingEvent: not separately scored here
        self._process_telemetry(record)

    def _process_telemetry(self, record: TelemetryRecord) -> None:
        """Fold one :class:`TelemetryRecord` into features + history + detectors."""

        # 1) streaming O(1) features (per entity per tick)
        fv = self.engine.process(record)
        if fv is not None:
            self._latest_feature[fv.entity.entity_id] = dict(fv.features)
            if fv.triggered_drift:
                self._latest_drift[fv.entity.entity_id] = list(fv.triggered_drift)

        # 2) per-(entity, metric) history + detector bank for precursor metrics
        metric = record.metric_name
        spec = _PRECURSOR_METRICS.get(metric)
        if spec is None:
            return
        entity = record.entity()
        key = (entity.entity_id, metric)
        st = self._streams.get(key)
        if st is None:
            if self.config.max_streams and len(self._streams) >= self.config.max_streams:
                return
            st = _Stream(
                entity=entity,
                metric=metric,
                spec=spec,
                values=deque(maxlen=self.config.history_len),
                timestamps=deque(maxlen=self.config.history_len),
                bank=_PerTickBank(_build_pertick_detectors(entity, metric)),
            )
            self._streams[key] = st

        value = float(record.value)
        st.values.append(value)
        st.timestamps.append(record.timestamp)
        # online detector scoring (each detector folds the sample in O(1)/amortised)
        st.last_scores = st.bank.update(value, timestamp=record.timestamp)  # type: ignore[union-attr]

        # capture the onset snapshot at the tick of highest risk so the correlation
        # layer can separate temporally-distinct incidents (otherwise every event
        # would carry the final timestamp and collapse into one incident).
        peak = max((float(s.normalized_score) for s in st.last_scores), default=0.0)
        if peak > st.peak_value:
            st.peak_value = peak
            st.peak_scores = list(st.last_scores)
            st.peak_history = list(st.values)
            st.peak_timestamps = list(st.timestamps)

        # 3) per-tick fused risk for this entity+metric → trajectory point
        self._record_risk_point(st)

    # ------------------------------------------------------------------ #
    # Per-tick fusion (drives the risk timeline + the lead-time eval)    #
    # ------------------------------------------------------------------ #
    def _record_risk_point(self, st: _Stream) -> None:
        """Fuse the current detector scores for one stream into a RiskPoint.

        Healthy ticks (all detector scores below ``_TIMELINE_FLOOR``) are skipped:
        they carry no risk and fusing every stream on every tick would dominate the
        runtime. The timeline therefore records points only once a stream starts to
        light up — exactly the precursor onset we want to visualise.
        """
        if not st.last_scores:
            return
        peak = max((float(s.normalized_score) for s in st.last_scores), default=0.0)
        if peak < _TIMELINE_FLOOR:
            return
        fused = self._fuse_stream(st, with_tti=False)
        if fused is None:
            return
        issue = _METRIC_ISSUE.get(st.metric, fused.predicted_issue)
        self._risk_history[st.entity.entity_id].append(
            RiskPoint(
                timestamp=fused.timestamp,
                risk_score=fused.risk_score,
                calibrated_confidence=fused.calibrated_confidence,
                predicted_issue=issue if fused.risk_score > 0 else IssueType.NONE,
                agreement=fused.agreement,
            )
        )

    def _fuse_stream(
        self, st: _Stream, *, with_tti: bool, at_peak: bool = False
    ) -> FusedRisk | None:
        """Fuse one stream's scores (+ optional forecast/TTI) → FusedRisk.

        ``at_peak`` selects the onset snapshot (the scores/history/timestamp at the
        tick of highest risk) rather than the most-recent tick — used when building
        the correlation events so temporally-distinct incidents stay separate.
        """
        if at_peak and st.peak_scores:
            scores = st.peak_scores
            history = st.peak_history
            ts = st.peak_timestamps[-1] if st.peak_timestamps else scores[0].timestamp
        else:
            scores = st.last_scores
            history = list(st.values)
            ts = scores[0].timestamp if scores else None
        if not scores:
            return None
        features = self._feature_vector_for(st.entity.entity_id, ts)
        forecast = None
        tti = None
        agreement = None
        if with_tti and len(history) >= self.config.min_history_for_forecast:
            forecast, agreement = self._forecast_stream(st, history=history)
            if forecast is not None:
                tti = self._tti_for(st, forecast, agreement, history=history)
        issue = _METRIC_ISSUE.get(st.metric)
        # only stamp a concrete issue when there is real risk in the scores
        peak = max((float(s.normalized_score) for s in scores), default=0.0)
        predicted_issue = issue if (issue is not None and peak > 0.0) else None
        return self.fuser.fuse(
            scores,
            entity=st.entity,
            timestamp=ts,
            forecast=forecast,
            time_to_impact=tti,
            forecast_agreement=agreement,
            features=features,
            predicted_issue=predicted_issue,
        )

    def _is_genuine_excursion(self, st: _Stream) -> bool:
        """True if the stream genuinely deviated from its own baseline.

        Streaming detectors (robust-z, EWMA, Page-Hinkley) can register elevated
        normalised scores on the micro-variation of a near-flat baseline — a
        normaliser artefact, not a real precursor. We accept a stream as an incident
        candidate only when its peak value is a real excursion: either a robust
        z-distance from its early-baseline median exceeds a floor, OR the peak is a
        meaningful fraction of the metric's SLA threshold (so a metric climbing
        toward its breach counts even if its baseline was already non-trivial).
        """
        import numpy as np

        # Use the FULL stream history (not the peak-score snapshot): the detector
        # score peaks at the *onset* of change (drift detectors fire on the
        # derivative), which can be well before the metric reaches its maximum
        # value, so the peak-score snapshot would truncate the excursion.
        hist = list(st.values)
        if len(hist) < self.config.min_history_for_forecast:
            return False
        arr = np.asarray(hist, dtype=float)
        # Use the true max (or min, for below-breach) excursion of the whole series.
        excursion_val = float(np.max(arr)) if st.spec.above_is_breach else float(np.min(arr))
        # baseline = the early portion of the series (before any excursion).
        n_base = max(5, len(arr) // 3)
        base = arr[:n_base]
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med))) * 1.4826
        abs_dev = abs(excursion_val - med)

        # A move's significance needs BOTH a robust z (shape) AND a minimum ABSOLUTE
        # excursion (magnitude): on a near-flat baseline a 2-unit blip yields a huge
        # z but is not a real precursor, so require the absolute move to also clear a
        # fraction of the SLA threshold.
        min_abs = _EXCURSION_MIN_ABS_FRAC * st.spec.threshold
        if mad > 1e-9:
            z = abs_dev / mad
            if z >= _EXCURSION_Z and abs_dev >= min_abs:
                return True
        # OR the metric reached a meaningful fraction of its breach threshold
        # outright (a strong, unambiguous excursion regardless of baseline shape).
        if st.spec.above_is_breach:
            return excursion_val >= _EXCURSION_THRESHOLD_FRAC * st.spec.threshold
        # below-is-breach metrics (e.g. IPSec rekey interval): a breach is going
        # BELOW the threshold, so a genuine excursion is the value DROPPING to near
        # (or below) the threshold — within a small margin ABOVE it. A healthy
        # baseline well above the threshold must NOT pass.
        return excursion_val <= st.spec.threshold * (1.0 + (1.0 - _EXCURSION_THRESHOLD_FRAC) * 0.4)

    def _feature_vector_for(self, entity_id: str, ts: datetime):
        """Build a light FeatureVector for fusion's drift-corroboration input."""
        feats = self._latest_feature.get(entity_id)
        drift = self._latest_drift.get(entity_id, [])
        if not feats and not drift:
            return None
        from netra.contracts import FeatureVector  # local import: keep load light

        return FeatureVector(
            entity=self.topology.entity_ref(self.topology.map_to_node(entity_id) or entity_id),
            timestamp=ts,
            features=feats or {},
            triggered_drift=drift,
        )

    def _forecast_stream(self, st: _Stream, *, history: list[float] | None = None):
        """Forecast the stream's history → (Forecast, cross-model agreement).

        ``EnsembleForecaster.forecast_with_members(history, steps, ...)`` fits every
        member internally and returns an :class:`EnsembleResult` (``.combined`` +
        ``.agreement``); the single-member path uses the ``Forecaster`` ABC
        (``fit`` then ``forecast``).
        """
        series = history if history is not None else list(st.values)
        # FAST profile: downsample the series (keep 1 point in N) and tell the
        # forecaster the effective per-step seconds, so the projected trajectory and
        # its threshold-crossing time stay in real seconds while the fit is cheaper.
        ds = max(1, int(self.config.forecast_downsample))
        eff_series = series[::ds] if ds > 1 else series
        eff_step = self.config.step_seconds * ds
        # cap the horizon so the projection covers the same real-time window with
        # fewer (downsampled) steps.
        steps = max(1, self.config.forecast_steps // ds) if ds > 1 else self.config.forecast_steps
        try:
            if self.config.use_ensemble_forecaster:
                ens = EnsembleForecaster(
                    st.entity,
                    st.metric,
                    enable_gbm=False,
                    lightweight=bool(self.config.lightweight_forecasters),
                )
                res = ens.forecast_with_members(
                    eff_series,
                    steps,
                    step_seconds=eff_step,
                )
                return res.combined, res.agreement
            from netra.analytics.forecasting import HoltWintersForecaster

            fc = HoltWintersForecaster(st.entity, st.metric)
            fc.fit(eff_series)
            return (
                fc.forecast(steps, step_seconds=eff_step),
                None,
            )
        except Exception:
            return None, None

    def _tti_for(self, st: _Stream, forecast, agreement, *, history: list[float] | None = None):
        """Read a TimeToImpact off the forecast vs the metric's SLA threshold."""
        from netra.contracts import Direction

        series = history if history is not None else list(st.values)
        direction = (
            Direction.INCREASES_RISK if st.spec.above_is_breach else Direction.DECREASES_RISK
        )
        try:
            return self.tti.estimate(
                forecast,
                st.spec.threshold,
                direction=direction,
                history=series,
                current_value=float(series[-1]) if series else None,
                agreement=agreement,
            )
        except Exception:
            return None

    # ================================================================== #
    # Assembly: streams → FusedRisk[] → incidents → copilot → report     #
    # ================================================================== #
    def assemble(self, *, answer_copilot: bool = True) -> SituationReport:
        """Materialise the current state into a :class:`SituationReport`.

        Runs the *final* fusion (with forecasts + time-to-impact) per entity,
        correlates into incidents, explains + prioritises them, and answers the
        copilot for the top incident(s). Safe to call repeatedly while streaming.
        """
        now = self._last_ts or datetime.now(timezone.utc)

        # 0) batch enrichment: run the heavier tier-2 detectors ONCE over the
        #    recent window of the highest-risk streams (independent-family agreement
        #    without paying the per-tick re-fit cost).
        if self.config.batch_enrich:
            self._batch_enrich()

        # 1) final per-(entity,metric) fusion at the ONSET (peak) snapshot, so each
        #    fused event carries the timestamp at which the stream actually lit up —
        #    this is what lets correlation keep temporally-distinct scenarios apart.
        #    Run the EXPENSIVE forecast+TTI branch only on genuinely-trending streams.
        fused_by_metric: list[FusedRisk] = []
        anomalies: list[AnomalyScore] = []
        # excursion fraction (value/threshold) per fused event, so the per-entity
        # issue can be driven by the most-breached metric rather than whichever
        # metric merely fused highest (e.g. congestion's util should win over the
        # incidental latency drift it induces).
        excursion_frac: dict[int, float] = {}

        # PRE-SCREEN (anomaly-first, top-K): the cheap per-tick detectors have
        # already scored every stream; rank the genuine-excursion streams by their
        # onset peak and run the EXPENSIVE forecast + time-to-impact branch only on
        # the top-K (``forecast_top_k``; 0 = all, the ``full`` profile). This caps
        # the forecasting fan-out from O(entities × metrics) to O(K) without losing
        # the precursor — by construction the scenario's breaching metric is the
        # highest-scoring stream, and lead-time detection comes from the per-tick
        # risk timeline (already recorded), not from this forecast pass.
        screened = [
            st
            for st in self._streams.values()
            if st.peak_value >= _TIMELINE_FLOOR and self._is_genuine_excursion(st)
        ]
        screened.sort(key=lambda s: s.peak_value, reverse=True)
        top_k = self.config.forecast_top_k
        forecast_set = (
            set(id(s) for s in screened[:top_k]) if top_k and top_k > 0 else None
        )

        for st in screened:
            peak = st.peak_value
            # The forecast/TTI branch runs only for streams that clear the higher
            # forecast floor AND (in fast mode) are in the top-K screened set.
            in_top_k = forecast_set is None or id(st) in forecast_set
            with_tti = peak >= self.config.forecast_risk_floor and in_top_k
            fr = self._fuse_stream(st, with_tti=with_tti, at_peak=True)
            if fr is None or fr.risk_score < self.config.incident_risk_floor:
                continue
            fused_by_metric.append(fr)
            excursion_frac[id(fr)] = self._excursion_fraction(st)

        # 2) collapse to one FusedRisk per ENTITY — keep the event with the strongest
        #    EXCURSION (most-breached metric) so the entity's predicted_issue and TTI
        #    reflect the dominant fault signature, not an incidental side-effect.
        fused_by_entity = self._collapse_per_entity(fused_by_metric, excursion_frac)

        # 3) correlate → incidents (graph event-correlation + RCA + blast radius)
        incidents = correlate_to_incidents(
            self.topology,
            anomalies=anomalies,
            fused=fused_by_entity,
            flows=self._flows or None,
            window_seconds=self.config.correlation_window_seconds,
            now=now,
        )

        # 4) enrich each incident's Q2 signals with SHAP-style attributions
        for inc in incidents:
            self._enrich_signals(inc)

        # 5) prioritise (calibrated product-form risk + severity + flap suppression)
        prioritized = prioritize_incidents(incidents, topology=self.topology, now=now)
        ranked = [pi.incident for pi in prioritized]

        # 6) attach the scenario ground-truth label to each incident (eval/demo)
        self._tag_scenarios(ranked)

        report = SituationReport(
            generated_at=now,
            incidents=ranked,
            risk_history={k: list(v) for k, v in self._risk_history.items()},
            labels=list(self._labels),
            window_start=self._first_ts,
            window_end=self._last_ts,
            stats={
                "records_processed": float(self._records_seen),
                "entities_tracked": float(len({k[0] for k in self._streams})),
                "streams_tracked": float(len(self._streams)),
                "incidents": float(len(ranked)),
                "flows": float(len(self._flows)),
            },
        )

        # 7) copilot Q1/Q2/Q3 for the top incident(s)
        if answer_copilot and ranked:
            for inc in ranked[: self.config.copilot_top_n]:
                report.copilot_answers[inc.incident_id] = self._answer(inc)

        # 8) per-scenario evaluation against the labels
        report.scenario_evals = self._evaluate(ranked)
        return report

    # ------------------------------------------------------------------ #
    # batch tier-2 enrichment (independent-family agreement, run once)    #
    # ------------------------------------------------------------------ #
    def _batch_enrich(self) -> None:
        """Score the recent window of the top-risk streams through tier-2 detectors.

        The per-tick path runs only the cheap streaming (tier-1) family; here, once,
        we run the curated batch detectors (ECOD, LOF, PCA-recon, matrix-profile,
        Half-Space-Trees — see :func:`_build_batch_detectors`) over the window UP TO
        the onset (peak) tick of the highest-risk streams, and append their
        onset-sample :class:`AnomalyScore` to that stream's ``peak_scores`` (the
        snapshot fusion/correlation consume). Fusion then sees multiple *independent
        families* agreeing, lifting the calibrated confidence and the cross-family
        agreement that the 30+-method ensemble is built around.
        """
        # rank streams by their onset (peak) streaming score; enrich the top-N.
        candidates = [
            st
            for st in self._streams.values()
            if len(st.peak_history) >= self.config.min_history_for_forecast
            and st.peak_value > 0.0
        ]
        candidates.sort(key=lambda s: s.peak_value, reverse=True)
        for st in candidates[: self.config.batch_enrich_top_streams]:
            window = list(st.peak_history)[-self.config.batch_enrich_window :]
            ts_window = list(st.peak_timestamps)[-self.config.batch_enrich_window :]
            onset_ts = ts_window[-1] if ts_window else None
            extra: list[AnomalyScore] = []
            for det in _build_batch_detectors(
                st.entity, st.metric, lightweight=bool(self.config.lightweight_forecasters)
            ):
                try:
                    # warm on the benign body, then take the score on the onset sample
                    det.fit(window[: max(2, len(window) - 1)])
                    scored = det.update(float(window[-1]), timestamp=onset_ts)
                    extra.append(scored)
                except Exception:
                    continue
            if extra:
                # de-dup by method so a re-run doesn't double-count
                have = {s.method for s in st.peak_scores}
                st.peak_scores = list(st.peak_scores) + [
                    s for s in extra if s.method not in have
                ]

    # ------------------------------------------------------------------ #
    # helpers for assembly                                               #
    # ------------------------------------------------------------------ #
    def _collapse_per_entity(
        self, fused: list[FusedRisk], excursion_frac: dict[int, float] | None = None
    ) -> list[FusedRisk]:
        """Keep one representative FusedRisk per entity.

        Ranks an entity's competing per-metric fused risks by their excursion
        fraction first (how far the metric breached its threshold — the dominant
        fault signature), then by risk score, so the surviving event carries the
        right ``predicted_issue`` and time-to-impact.
        """
        ef = excursion_frac or {}

        def _rank(fr: FusedRisk) -> tuple[float, float]:
            return (ef.get(id(fr), 0.0), fr.risk_score)

        best: dict[str, FusedRisk] = {}
        for fr in fused:
            cur = best.get(fr.entity.entity_id)
            if cur is None or _rank(fr) > _rank(cur):
                best[fr.entity.entity_id] = fr
        return list(best.values())

    def _excursion_fraction(self, st: _Stream) -> float:
        """How far the stream breached its SLA threshold, in [0, inf).

        For an above-breach metric this is ``max(series)/threshold``; for a
        below-breach metric (rekey interval) it is ``threshold/min(series)`` so a
        deeper drop scores higher. Used to pick the dominant fault metric per
        entity.
        """
        import numpy as np

        arr = np.asarray(list(st.values), dtype=float)
        if arr.size == 0 or st.spec.threshold == 0:
            return 0.0
        if st.spec.above_is_breach:
            return float(np.max(arr)) / float(st.spec.threshold)
        lo = float(np.min(arr))
        return float(st.spec.threshold) / lo if lo > 1e-9 else 0.0

    def _enrich_signals(self, incident: Incident) -> None:
        """Merge SHAP-style attributions into the incident's contributing signals.

        Correlation already produced correlation-level signals; here we add the
        analytics ``explain_fused_risk`` attributions (grounded, ranked) so the
        copilot's Q2 quotes real per-feature contributions. We keep the union,
        de-duplicated by signal name, root-cause signals first.
        """
        root_id = (
            incident.root_cause_entity.entity_id if incident.root_cause_entity else None
        )
        feats = self._latest_feature.get(root_id or "", {})
        try:
            attr_signals = explain_fused_risk(
                incident.risk,
                feature_values=feats or None,
                entity=incident.root_cause_entity,
                top_k=8,
            )
        except Exception:
            attr_signals = []
        merged: dict[str, ContributingSignal] = {}
        for s in list(attr_signals) + list(incident.contributing_signals):
            merged.setdefault(s.signal, s)
        incident.contributing_signals = list(merged.values())[:10]

    def _tag_scenarios(self, incidents: list[Incident]) -> None:
        """Stamp ``Incident.scenario_label`` by matching root/correlated entities
        to the ground-truth label target entities (so the demo/eval can join)."""
        if not self._labels:
            return
        for inc in incidents:
            ent_ids = {e.entity_id for e in inc.correlated_entities}
            if inc.root_cause_entity:
                ent_ids.add(inc.root_cause_entity.entity_id)
            node_ids = {self.topology.map_to_node(e) for e in ent_ids}
            for label in self._labels:
                tgt_node = self.topology.map_to_node(label.target_entity_id)
                if tgt_node in node_ids or label.target_entity_id in ent_ids:
                    inc.scenario_label = label.scenario
                    break

    def _answer(self, incident: Incident) -> CopilotResponse:
        """Run the copilot for one incident → grounded Q1/Q2/Q3 CopilotResponse."""
        req = CopilotRequest(
            request_id=f"pipeline-{incident.incident_id}",
            created_at=incident.created_at,
            auto_trigger=True,
            incident_ref=incident.incident_id,
            entity_refs=(
                [incident.root_cause_entity.entity_id]
                if incident.root_cause_entity
                else []
            ),
        )
        ctx = AnalyticsContext(
            incident=incident,
            fused_risk=incident.risk,
            time_to_impact=incident.risk.time_to_impact,
            contributing_signals=list(incident.contributing_signals),
            blast_radius=incident.blast_radius,
            playbook=incident.recommended_playbook,
            root_cause_entity=incident.root_cause_entity,
        )
        try:
            return self.copilot.answer(req, analytics_context=ctx)
        except Exception:
            # graceful: synthesize a minimal, contract-valid response from the
            # incident if the copilot stack errors for any reason.
            return _fallback_response_from_incident(req.request_id, incident)

    # ================================================================== #
    # Evaluation against the synthetic ground-truth labels               #
    # ================================================================== #
    def _evaluate(self, incidents: list[Incident]) -> list[ScenarioEval]:
        """Score the run against each ScenarioLabel: detected? lead time? method?"""
        evals: list[ScenarioEval] = []
        for label in self._labels:
            evals.append(self._evaluate_one(label, incidents))
        return evals

    def _evaluate_one(
        self, label: ScenarioLabel, incidents: list[Incident]
    ) -> ScenarioEval:
        tgt_node = self.topology.map_to_node(label.target_entity_id)
        precursor_start = _as_utc(label.precursor_window_start)
        fault_start = _as_utc(label.fault_window_start)
        fault_end = _as_utc(label.fault_window_end)

        ev = ScenarioEval(
            scenario=label.scenario,
            expected_issue=label.expected_issue,
            target_entity_id=label.target_entity_id,
            precursor_window_start=precursor_start,
            fault_window_start=fault_start,
            fault_window_end=fault_end,
            expected_lead_time_seconds=label.expected_lead_time_seconds,
        )

        # Gather the fused-risk trajectory of the target entity (+ entities mapping
        # to the same device node) within the eval horizon.
        first_alert: datetime | None = None
        peak = 0.0
        for entity_id, points in self._risk_history.items():
            if self.topology.map_to_node(entity_id) != tgt_node and entity_id != label.target_entity_id:
                continue
            for p in points:
                pts = _as_utc(p.timestamp)
                if pts > fault_end:
                    continue
                peak = max(peak, p.risk_score)
                # an elevated-risk alert strictly BEFORE the breach earns lead time
                if (
                    p.risk_score >= self.config.alert_risk_threshold
                    and precursor_start <= pts < fault_start
                ):
                    if first_alert is None or pts < first_alert:
                        first_alert = pts

        ev.peak_risk = round(peak, 4)
        if first_alert is not None:
            ev.detected = True
            ev.first_alert_at = first_alert
            ev.lead_time_seconds = round((fault_start - first_alert).total_seconds(), 1)

        # which detector families fired on the target during the window + the
        # incident the pipeline raised for this scenario (+ its issue/eta).
        ev.methods_fired = self._methods_for_target(tgt_node, label.target_entity_id, fault_end)
        inc = self._incident_for_scenario(label, tgt_node, incidents)
        if inc is not None:
            ev.incident_id = inc.incident_id
            ev.predicted_issue_correct = inc.predicted_issue == label.expected_issue
            tti = inc.risk.time_to_impact
            if tti is not None and tti.eta_seconds is not None:
                ev.eta_seconds_at_alert = round(float(tti.eta_seconds), 1)
        else:
            # even without a correlated incident, the per-entity fused issue may match
            ev.predicted_issue_correct = self._issue_seen_for_target(
                tgt_node, label.target_entity_id, label.expected_issue, fault_end
            )
        return ev

    def _methods_for_target(
        self, tgt_node: str | None, target_id: str, horizon: datetime
    ) -> list[str]:
        """Distinct detector/forecaster methods that flagged the target in-window."""
        fired: dict[str, int] = defaultdict(int)
        for (entity_id, _metric), st in self._streams.items():
            if self.topology.map_to_node(entity_id) != tgt_node and entity_id != target_id:
                continue
            for s in st.last_scores:
                if s.is_anomaly or s.normalized_score >= self.config.alert_risk_threshold:
                    fired[s.method] += 1
        # most-firing methods first (a crude salience ranking)
        return [m for m, _ in sorted(fired.items(), key=lambda kv: kv[1], reverse=True)]

    def _issue_seen_for_target(
        self, tgt_node: str | None, target_id: str, expected: IssueType, horizon: datetime
    ) -> bool:
        for entity_id, points in self._risk_history.items():
            if self.topology.map_to_node(entity_id) != tgt_node and entity_id != target_id:
                continue
            for p in points:
                if p.predicted_issue == expected:
                    return True
        return False

    def _incident_for_scenario(
        self, label: ScenarioLabel, tgt_node: str | None, incidents: list[Incident]
    ) -> Incident | None:
        # first prefer an incident already tagged with this scenario
        for inc in incidents:
            if inc.scenario_label == label.scenario:
                return inc
        # else any incident whose root/correlated entity maps to the target node
        for inc in incidents:
            ids = {e.entity_id for e in inc.correlated_entities}
            if inc.root_cause_entity:
                ids.add(inc.root_cause_entity.entity_id)
            if tgt_node in {self.topology.map_to_node(i) for i in ids}:
                return inc
        return None


# --------------------------------------------------------------------------- #
# module helpers                                                              #
# --------------------------------------------------------------------------- #
def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _tunnelstat_to_records(ts: TunnelStat) -> list[TelemetryRecord]:
    """Project a :class:`TunnelStat` onto the per-metric ``TelemetryRecord`` streams.

    The synthetic generator emits overlay-tunnel health (scenario C: loss/jitter
    progression + IPSec rekey anomaly) as a dedicated ``TunnelStat`` record; the
    streaming/analytics path keys everything on ``(entity, metric)`` numeric
    ``TelemetryRecord``s. We fan the relevant fields out, reusing the generator's
    ``site:device:role:tunnel`` entity-id convention so the resulting streams join
    the topology and correlation graph exactly like a native metric.
    """
    out: list[TelemetryRecord] = []
    labels = {"tunnel": ts.tunnel_id}
    common = {
        "timestamp": ts.timestamp,
        "site": ts.site,
        "device": ts.device,
        "role": ts.role,
        "kind": TelemetryKind.TUNNEL,
        "labels": labels,
    }
    out.append(
        TelemetryRecord(metric_name=MetricName.TUNNEL_LOSS_PCT.value, value=float(ts.loss_pct), unit="pct", **common)
    )
    out.append(
        TelemetryRecord(metric_name=MetricName.TUNNEL_JITTER_MS.value, value=float(ts.jitter_ms), unit="ms", **common)
    )
    if ts.rekey_interval_s is not None:
        out.append(
            TelemetryRecord(
                metric_name=MetricName.TUNNEL_REKEY_INTERVAL_S.value,
                value=float(ts.rekey_interval_s),
                unit="s",
                **common,
            )
        )
    return out


def _fallback_response_from_incident(request_id: str, inc: Incident) -> CopilotResponse:
    """A last-resort, contract-valid CopilotResponse derived from the incident.

    Mirrors the deterministic template shape (``used_fallback=True``) so the demo
    and API always render Q1/Q2/Q3 even if the copilot stack raised.
    """
    from netra.contracts import (
        AffectedScope,
        CopilotAction,
        CopilotSignal,
        Urgency,
    )

    tti = inc.risk.time_to_impact
    tti_min = (
        round(tti.eta_seconds / 60.0, 1) if tti and tti.eta_seconds is not None else None
    )
    scope = AffectedScope(
        sites=list(inc.blast_radius.affected_sites),
        devices=list(inc.blast_radius.affected_devices),
        services_or_vpns=list(inc.blast_radius.affected_services_or_vpns),
    )
    signals = [
        CopilotSignal(
            signal=s.signal,
            observation=s.observation or s.human_explanation,
            shap_contribution=s.shap_value,
        )
        for s in inc.contributing_signals[:6]
    ]
    actions: list[CopilotAction] = []
    citations: list[str] = []
    if inc.recommended_playbook:
        citations.append(
            inc.recommended_playbook.source_ref or inc.recommended_playbook.playbook_id
        )
        for a in inc.recommended_playbook.actions:
            actions.append(
                CopilotAction(
                    step=a.description,
                    runbook_ref=a.runbook_ref,
                    urgency=a.urgency,
                    requires_approval=a.requires_approval,
                )
            )
    if not actions:
        actions.append(
            CopilotAction(
                step=(
                    "Collect current interface/queue/routing/tunnel statistics for the "
                    "affected entities and correlate against the predicted issue before "
                    "any state-changing action."
                ),
                runbook_ref=None,
                urgency=Urgency.IMMEDIATE,
                requires_approval=False,
            )
        )
    citations.append(f"telemetry:{inc.incident_id}:{inc.window_start.isoformat()}")
    citations = list(dict.fromkeys(citations))
    return CopilotResponse(
        request_id=request_id,
        predicted_issue=inc.predicted_issue,
        confidence_score=inc.risk.calibrated_confidence,
        time_to_impact_minutes=tti_min,
        root_cause_hypothesis=inc.root_cause_hypothesis or "See contributing signals.",
        contributing_signals=signals,
        affected_scope=scope,
        recommended_actions=actions,
        citations=citations,
        insufficient_context=False,
        used_fallback=True,
        model_id="template-fallback",
    )


__all__ = ["NetraPipeline", "PipelineConfig", "SituationReport"]
