"""Per-scenario signal models: diurnal baselines + injected precursor ramps.

This module is the *fidelity core* of the synthetic generator. It defines, with
no I/O and no global state:

  1. **Diurnal baselines** — realistic time-of-day seasonality for every metric
     family (utilisation, latency, jitter, loss, routing churn, tunnel health),
     with per-site weighting (DC > hub > branch), business-hours peak, lunch dip,
     overnight trough, plus weekly and small multiplicative noise. This is the
     "normal" the forecasters learn (``research/01`` §3.1).

  2. **Precursor injectors** — for each of the four validation scenarios, a
     deterministic, parameterised perturbation that begins **measurably before**
     the labeled fault window so a forecaster/drift detector has lead time. Each
     injector returns a multiplicative/additive delta for a given metric at a
     given time, expressed via simple, *statistically detectable* shapes:
       * a monotonic **trend** ramp (Mann-Kendall / Theil-Sen / forecast slope),
       * a **variance** inflation (EWMA-variance / Half-Space-Trees),
       * a **regime shift** step (Page-Hinkley / BOCPD / PELT change-point),
       * bursty **churn** (CUSUM / ADWIN on event-rate counters).

  3. **Scenario specs** — the metadata (target entity, expected ``IssueType``,
     precursor/fault window offsets, severity, playbook id) used to emit the
     ground-truth :class:`netra.contracts.ScenarioLabel`.

Everything is driven by a single integer seed plus the absolute timestamp, so a
given (seed, scenario, time) tuple is byte-for-byte reproducible. The math uses
only ``numpy`` (core tier); there is a tiny pure-python fallback hash so the
*shapes* remain deterministic even if numpy's RNG stream ever changes.

The contract surface here is the closed vocabulary in ``netra.contracts``:
``ScenarioId``, ``IssueType``, ``MetricName``, ``Severity`` — imported, never
redefined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from netra.contracts import IssueType, MetricName, ScenarioId, Severity

# --------------------------------------------------------------------------- #
# Deterministic per-stream RNG                                                #
# --------------------------------------------------------------------------- #


def stream_rng(seed: int, *parts: object) -> np.random.Generator:
    """Build a numpy ``Generator`` deterministically keyed by ``seed`` + parts.

    Each (entity, metric) stream gets its *own* independent, reproducible noise
    sequence by folding the human-readable ``parts`` (e.g. entity id, metric
    name) into a stable 64-bit hash combined with the global ``seed``. Same
    inputs -> same generator -> same numbers, on any machine.
    """
    h = np.uint64(1469598103934665603)  # FNV-1a offset basis
    prime = np.uint64(1099511628211)
    payload = "|".join([str(seed), *(str(p) for p in parts)])
    with np.errstate(over="ignore"):
        for ch in payload.encode("utf-8"):
            h ^= np.uint64(ch)
            h *= prime
    return np.random.default_rng(int(h))


def _stable_unit(seed: int, *parts: object) -> float:
    """A deterministic value in [0,1) from a seed+parts (numpy-free fallback)."""
    payload = "|".join([str(seed), *(str(p) for p in parts)])
    h = 1469598103934665603
    for ch in payload.encode("utf-8"):
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (h % 1_000_000) / 1_000_000.0


# --------------------------------------------------------------------------- #
# Diurnal baseline model                                                      #
# --------------------------------------------------------------------------- #


def _seconds_of_day(ts: datetime) -> float:
    ts = ts.astimezone(UTC)
    return ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6


def diurnal_multiplier(ts: datetime, *, peak_hour: float = 15.0) -> float:
    """A smooth [~0.18 .. ~1.0] time-of-day load multiplier.

    Models a realistic working-day curve: overnight trough, morning ramp, a
    mid-day peak near ``peak_hour`` with a small lunch dip, and an evening
    decline. Deterministic (pure function of the timestamp).
    """
    h = _seconds_of_day(ts) / 3600.0
    # Primary daily sinusoid (peak at peak_hour).
    primary = 0.55 + 0.42 * math.sin((h - peak_hour + 6.0) / 24.0 * 2 * math.pi - math.pi / 2)
    # Lunch dip around 13:00.
    lunch = -0.10 * math.exp(-((h - 13.0) ** 2) / (2 * 0.8**2))
    # Overnight floor.
    val = max(0.18, primary + lunch)
    # Weekly modulation: weekends ~30% quieter.
    weekday = ts.astimezone(UTC).weekday()  # 0=Mon
    if weekday >= 5:
        val *= 0.7
    return float(min(1.05, val))


@dataclass(frozen=True)
class MetricBaseline:
    """Baseline parameters for one metric on one class of entity.

    The synthetic value at time ``t`` is::

        base * diurnal_weight(t) [if seasonal]  + gaussian_noise(sigma)
        clamped to [floor, ceil]

    plus any scenario precursor delta layered on top.
    """

    base: float
    sigma: float  # std-dev of multiplicative/additive noise (see ``additive``)
    floor: float = 0.0
    ceil: float = float("inf")
    seasonal: bool = True  # follow the diurnal curve?
    additive_noise: bool = False  # noise added (True) vs multiplicative (False)
    diurnal_gain: float = 1.0  # how strongly the diurnal curve modulates this metric
    scale_by_site: bool = True  # multiply base by per-site load weight?

    def sample(self, ts: datetime, rng: np.random.Generator, *, site_weight: float = 1.0) -> float:
        """Draw a baseline value for ``ts`` using per-stream ``rng``.

        ``site_weight`` scales *load-like* metrics (utilisation, queue, flow
        volume) by site population (DC > hub > branch). Physical or protocol
        metrics (latency, jitter, IPSec rekey interval, error/flap counters) set
        ``scale_by_site=False`` so a branch tunnel's rekey period is still ~3600s,
        not 3600s × the branch's small load weight.
        """
        level = self.base * (site_weight if self.scale_by_site else 1.0)
        if self.seasonal:
            dm = diurnal_multiplier(ts)
            # diurnal_gain blends between flat (0) and fully seasonal (1).
            level *= (1.0 - self.diurnal_gain) + self.diurnal_gain * dm
        if self.additive_noise:
            val = level + rng.normal(0.0, self.sigma)
        else:
            val = level * (1.0 + rng.normal(0.0, self.sigma))
        return float(min(self.ceil, max(self.floor, val)))


# Site weighting: DC carries the most load, branches the least.
SITE_WEIGHT: dict[str, float] = {
    "dc": 1.0,
    "hub": 0.8,
    "br1": 0.45,
    "br2": 0.4,
    "br3": 0.5,
    "core": 0.9,
}


# Default per-metric baselines (the "healthy normal"). Interface utilisation is
# the seasonal workhorse; latency/jitter/loss are low and mostly flat; routing
# churn and tunnel health sit near zero until a scenario perturbs them.
DEFAULT_BASELINES: dict[str, MetricBaseline] = {
    MetricName.IF_UTIL_PCT.value: MetricBaseline(
        base=42.0, sigma=0.06, floor=0.5, ceil=100.0, diurnal_gain=0.9
    ),
    MetricName.IF_OUT_DISCARDS.value: MetricBaseline(
        base=0.3, sigma=0.8, floor=0.0, ceil=1e6, additive_noise=True, diurnal_gain=0.7
    ),
    MetricName.IF_IN_ERRORS.value: MetricBaseline(
        base=0.05, sigma=0.3, floor=0.0, ceil=1e6, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.LATENCY_MS.value: MetricBaseline(
        base=18.0, sigma=0.04, floor=1.0, ceil=2000.0, diurnal_gain=0.35,
        scale_by_site=False,
    ),
    MetricName.JITTER_MS.value: MetricBaseline(
        base=2.5, sigma=0.12, floor=0.1, ceil=500.0, diurnal_gain=0.4,
        scale_by_site=False,
    ),
    MetricName.LOSS_PCT.value: MetricBaseline(
        base=0.05, sigma=0.5, floor=0.0, ceil=100.0, additive_noise=True, diurnal_gain=0.3,
        scale_by_site=False,
    ),
    MetricName.QUEUE_DEPTH.value: MetricBaseline(
        base=6.0, sigma=0.15, floor=0.0, ceil=1e5, diurnal_gain=0.8
    ),
    # routing-instability metrics (near zero baseline; scenario B drives them).
    # Per-session quantities — not scaled by site population.
    MetricName.BGP_UPDATE_RATE.value: MetricBaseline(
        base=1.2, sigma=0.9, floor=0.0, ceil=1e5, additive_noise=True, diurnal_gain=0.5,
        scale_by_site=False,
    ),
    MetricName.BGP_WITHDRAW_RATE.value: MetricBaseline(
        base=0.2, sigma=0.6, floor=0.0, ceil=1e5, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.BGP_FLAP_PENALTY.value: MetricBaseline(
        base=0.0, sigma=0.0, floor=0.0, ceil=1e4, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.ADJ_FLAP_COUNT.value: MetricBaseline(
        base=0.0, sigma=0.0, floor=0.0, ceil=1e4, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.OSPF_LSA_RATE.value: MetricBaseline(
        base=0.8, sigma=0.7, floor=0.0, ceil=1e4, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.OSPF_SPF_RATE.value: MetricBaseline(
        base=0.3, sigma=0.5, floor=0.0, ceil=1e4, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    MetricName.PATH_ASYMMETRY.value: MetricBaseline(
        base=0.02, sigma=0.4, floor=0.0, ceil=1.0, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
    # tunnel-health metrics (scenario C drives them) — per-tunnel, not site-scaled.
    MetricName.TUNNEL_LOSS_PCT.value: MetricBaseline(
        base=0.08, sigma=0.5, floor=0.0, ceil=100.0, additive_noise=True, diurnal_gain=0.3,
        scale_by_site=False,
    ),
    MetricName.TUNNEL_JITTER_MS.value: MetricBaseline(
        base=3.0, sigma=0.14, floor=0.1, ceil=500.0, diurnal_gain=0.4,
        scale_by_site=False,
    ),
    MetricName.TUNNEL_REKEY_INTERVAL_S.value: MetricBaseline(
        base=3600.0, sigma=0.01, floor=60.0, ceil=86400.0, seasonal=False,
        scale_by_site=False,
    ),
    # controller / policy drift (scenario D) — single controller, not site-scaled.
    MetricName.CONFIG_DRIFT_SCORE.value: MetricBaseline(
        base=0.0, sigma=0.0, floor=0.0, ceil=1.0, additive_noise=True, seasonal=False,
        scale_by_site=False,
    ),
}


def baseline_for(metric: str) -> MetricBaseline:
    """Return the baseline params for ``metric`` (sensible default if unknown)."""
    return DEFAULT_BASELINES.get(
        metric,
        MetricBaseline(base=1.0, sigma=0.1, floor=0.0, ceil=1e9, additive_noise=True),
    )


# --------------------------------------------------------------------------- #
# Precursor shape primitives                                                  #
# --------------------------------------------------------------------------- #


def _ramp01(t: float, t0: float, t1: float) -> float:
    """Linear ramp from 0 at ``t0`` to 1 at ``t1`` (clamped outside)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return float(min(1.0, max(0.0, (t - t0) / (t1 - t0))))


def _smoothstep(x: float) -> float:
    """Smooth 0->1 Hermite interpolation of a clamped [0,1] input."""
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


@dataclass(frozen=True)
class ScenarioSpec:
    """Static description of one validation scenario instance.

    Offsets are seconds relative to the dataset's ``start`` time, so a generator
    run is fully described by (start, spec, seed). The precursor window is the
    lead-time region: an alert firing inside
    ``[precursor_start, fault_start)`` earns lead-time credit.
    """

    scenario: ScenarioId
    expected_issue: IssueType
    #: entity-id of the primary injected-fault entity (ground-truth root cause)
    target_entity_id: str
    #: seconds from dataset start to the precursor onset (ramp begins)
    precursor_offset_s: float
    #: seconds from dataset start to the fault/breach onset
    fault_offset_s: float
    #: seconds from dataset start to fault clearance
    fault_end_offset_s: float
    severity: Severity = Severity.P2
    playbook_id: str | None = None
    target_sites: tuple[str, ...] = ()
    target_vpns: tuple[str, ...] = ()
    #: free-form knobs recorded into ScenarioLabel.params and used by injectors
    params: dict[str, float] = field(default_factory=dict)

    @property
    def precursor_window_s(self) -> float:
        return self.fault_offset_s - self.precursor_offset_s


# --------------------------------------------------------------------------- #
# Precursor injectors — one per scenario                                      #
# --------------------------------------------------------------------------- #
#
# Each injector is a pure function: given the scenario spec, the relative time
# ``t`` (seconds since dataset start), the entity-id, the metric and a per-stream
# rng, it returns ADDITIVE deltas to apply to the baseline sample. Returning 0.0
# for an untouched (entity, metric) keeps every other stream perfectly healthy.
#
# Design rule (lead time): the precursor delta becomes *statistically* non-zero
# at ``precursor_offset_s`` — strictly before ``fault_offset_s`` — and grows so a
# trend/variance/CP detector can fire while the metric is still below its SLA
# threshold. The fault window then makes the breach explicit and severe.


def _is_target(spec: ScenarioSpec, entity_id: str) -> bool:
    return entity_id == spec.target_entity_id


def inject_congestion(
    spec: ScenarioSpec, t: float, entity_id: str, metric: str, rng: np.random.Generator
) -> float:
    """Scenario A — progressive congestion buildup on a hub-spoke link.

    Precursor: a monotonically rising **utilisation slope** with growing queue
    depth and creeping discards/latency/jitter — all *before* loss starts. Fault:
    utilisation saturates, queue/discards spike, loss crosses SLA.
    """
    if not _is_target(spec, entity_id):
        return 0.0
    pre = _ramp01(t, spec.precursor_offset_s, spec.fault_offset_s)  # 0->1 precursor
    fault = _ramp01(t, spec.fault_offset_s, spec.fault_offset_s + 90)  # quick breach
    after = _ramp01(t, spec.fault_end_offset_s, spec.fault_end_offset_s + 60)
    active = (1.0 - after)  # decays to 0 after clearance
    peak_util = spec.params.get("peak_util_pct", 55.0)
    if metric == MetricName.IF_UTIL_PCT.value:
        # smooth precursor ramp + sharper fault saturation
        delta = (_smoothstep(pre) * peak_util + fault * 30.0) * active
        return delta * (1.0 + rng.normal(0, 0.02))
    if metric == MetricName.QUEUE_DEPTH.value:
        return (_smoothstep(pre) ** 2 * 80.0 + fault * 120.0) * active
    if metric == MetricName.IF_OUT_DISCARDS.value:
        # discards stay near zero through early precursor, then accelerate
        return (max(0.0, pre - 0.4) * 40.0 + fault * 200.0) * active
    if metric == MetricName.LATENCY_MS.value:
        return (_smoothstep(pre) * 22.0 + fault * 60.0) * active
    if metric == MetricName.JITTER_MS.value:
        # variance inflation: noise amplitude grows with the precursor
        return (_smoothstep(pre) * 4.0 + fault * 12.0) * active + rng.normal(
            0, 1.5 * pre
        )
    if metric == MetricName.LOSS_PCT.value:
        # loss only really appears at/after the fault boundary
        return (max(0.0, pre - 0.85) * 6.0 + fault * 9.0) * active
    return 0.0


def inject_bgp_flap(
    spec: ScenarioSpec, t: float, entity_id: str, metric: str, rng: np.random.Generator
) -> float:
    """Scenario B — BGP route flap + downstream reroute cascade.

    Precursor: rising **flap penalty** and bursty UPDATE/withdraw **churn** with
    occasional adjacency flaps and small path-asymmetry — before mass reachability
    loss. Fault: rapid up/down flapping, withdraw storms, SPF/LSA churn.
    """
    if not _is_target(spec, entity_id):
        return 0.0
    pre = _ramp01(t, spec.precursor_offset_s, spec.fault_offset_s)
    after = _ramp01(t, spec.fault_end_offset_s, spec.fault_end_offset_s + 45)
    active = 1.0 - after
    in_fault = spec.fault_offset_s <= t < spec.fault_end_offset_s
    flap_period = spec.params.get("flap_period_s", 80.0)
    # square-wave flap state during the fault (60s up / 20s down style)
    flap_on = in_fault and ((t - spec.fault_offset_s) % flap_period) < flap_period * 0.35
    if metric == MetricName.BGP_FLAP_PENALTY.value:
        # RFD-style decaying penalty that trends up during precursor, spikes in fault
        return (_smoothstep(pre) * 800.0 + (1500.0 if flap_on else 400.0 * in_fault)) * active
    if metric == MetricName.BGP_UPDATE_RATE.value:
        churn = _smoothstep(pre) * 15.0 + (90.0 if flap_on else 10.0 * in_fault)
        return max(0.0, churn + rng.normal(0, 4.0 * (pre + in_fault))) * active
    if metric == MetricName.BGP_WITHDRAW_RATE.value:
        return ((25.0 if flap_on else 2.0 * in_fault) + max(0.0, pre - 0.5) * 8.0) * active
    if metric == MetricName.ADJ_FLAP_COUNT.value:
        return (1.0 if flap_on else 0.0) * active
    if metric == MetricName.OSPF_SPF_RATE.value:
        return ((6.0 if flap_on else 1.0 * in_fault) + max(0.0, pre - 0.6) * 3.0) * active
    if metric == MetricName.OSPF_LSA_RATE.value:
        return ((10.0 if flap_on else 1.5 * in_fault) + max(0.0, pre - 0.6) * 4.0) * active
    if metric == MetricName.PATH_ASYMMETRY.value:
        return min(1.0, _smoothstep(pre) * 0.4 + (0.5 if flap_on else 0.0)) * active
    return 0.0


def inject_tunnel_degradation(
    spec: ScenarioSpec, t: float, entity_id: str, metric: str, rng: np.random.Generator
) -> float:
    """Scenario C — intermittent MPLS underlay failure / tunnel degradation.

    Precursor: a rising **loss/jitter trend** on the overlay tunnel plus
    **IPSec rekey-interval anomalies** (rekey period shrinking erratically) and
    occasional micro-bursts — before SLA loss. Fault: intermittent loss spikes,
    high jitter, BFD-style flaps.
    """
    if not _is_target(spec, entity_id):
        return 0.0
    pre = _ramp01(t, spec.precursor_offset_s, spec.fault_offset_s)
    after = _ramp01(t, spec.fault_end_offset_s, spec.fault_end_offset_s + 60)
    active = 1.0 - after
    in_fault = spec.fault_offset_s <= t < spec.fault_end_offset_s
    burst_period = spec.params.get("burst_period_s", 40.0)
    # intermittent spike: short bad window every burst_period seconds during fault
    bursting = in_fault and ((t - spec.fault_offset_s) % burst_period) < burst_period * 0.25
    if metric == MetricName.TUNNEL_LOSS_PCT.value:
        trend = _smoothstep(pre) * 1.8
        spike = 12.0 if bursting else (2.0 if in_fault else 0.0)
        return max(0.0, (trend + spike) + rng.normal(0, 0.6 * (pre + in_fault))) * active
    if metric == MetricName.TUNNEL_JITTER_MS.value:
        trend = _smoothstep(pre) * 8.0
        spike = 25.0 if bursting else (6.0 if in_fault else 0.0)
        return (trend + spike + rng.normal(0, 2.0 * (pre + in_fault))) * active
    if metric == MetricName.TUNNEL_REKEY_INTERVAL_S.value:
        # rekey interval shrinks erratically as the SA rebuilds under stress —
        # a NEGATIVE delta from the 3600s baseline (anomaly the detectors catch)
        anomaly = _smoothstep(pre) * -900.0
        jitter = rng.normal(0, 120.0 * pre) if pre > 0 else 0.0
        burstdrop = -1500.0 if bursting else 0.0
        return (anomaly + jitter + burstdrop) * active
    if metric == MetricName.LATENCY_MS.value:
        return ((_smoothstep(pre) * 10.0) + (40.0 if bursting else 0.0)) * active
    return 0.0


def inject_policy_drift(
    spec: ScenarioSpec, t: float, entity_id: str, metric: str, rng: np.random.Generator
) -> float:
    """Scenario D — controller misconfig -> policy drift.

    The discriminator (research/01 §4.3 D): a **step regime change** that appears
    at many entities **simultaneously** with no physical fault. Precursor is the
    config-change event itself (earliest signal); the drift then fans out as a
    sustained step in config-drift score and a slow divergence of QoS/flow
    behaviour, but NO loss/error spike.

    Note: scenario D is multi-target — see ``DRIFT_FANOUT_ENTITIES`` applied by
    the generator — so ``_is_target`` is relaxed to "entity participates in drift".
    """
    fanout: tuple[str, ...] = tuple(spec.params.get("_fanout_ids", ()))  # type: ignore[arg-type]
    participates = entity_id == spec.target_entity_id or entity_id in fanout
    if not participates:
        return 0.0
    # config-change is a near-instant step at the fault boundary; the precursor
    # window is short (the change push is the earliest event).
    step = _smoothstep(_ramp01(t, spec.fault_offset_s - 15, spec.fault_offset_s + 15))
    after = _ramp01(t, spec.fault_end_offset_s, spec.fault_end_offset_s + 30)
    active = 1.0 - after
    if metric == MetricName.CONFIG_DRIFT_SCORE.value:
        return min(1.0, step * 0.9) * active + (rng.normal(0, 0.01) if step > 0 else 0.0)
    if metric == MetricName.IF_UTIL_PCT.value:
        # mis-marked QoS class slowly shifts utilisation distribution (no breach)
        drift = _ramp01(t, spec.fault_offset_s, spec.fault_offset_s + 240)
        return step * 12.0 * drift * active
    if metric == MetricName.PATH_ASYMMETRY.value:
        # wrong RT import -> leaked routes -> rising path asymmetry
        return min(1.0, step * 0.35) * active
    return 0.0


# Registry: scenario -> its injector function.
INJECTORS = {
    ScenarioId.A_CONGESTION: inject_congestion,
    ScenarioId.B_BGP_FLAP: inject_bgp_flap,
    ScenarioId.C_TUNNEL_DEGRADATION: inject_tunnel_degradation,
    ScenarioId.D_POLICY_DRIFT: inject_policy_drift,
}


def apply_injection(
    spec: ScenarioSpec,
    rel_t: float,
    entity_id: str,
    metric: str,
    rng: np.random.Generator,
) -> float:
    """Dispatch to the right injector for ``spec.scenario`` and return the delta."""
    fn = INJECTORS.get(spec.scenario)
    if fn is None:
        return 0.0
    return fn(spec, rel_t, entity_id, metric, rng)


__all__ = [
    "stream_rng",
    "diurnal_multiplier",
    "MetricBaseline",
    "DEFAULT_BASELINES",
    "baseline_for",
    "SITE_WEIGHT",
    "ScenarioSpec",
    "apply_injection",
    "INJECTORS",
    "inject_congestion",
    "inject_bgp_flap",
    "inject_tunnel_degradation",
    "inject_policy_drift",
]
