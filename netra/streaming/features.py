"""O(1) online feature computers — the "fastest platform" core (Workstream 2).

Every operator in this module folds *one* new telemetry sample into a constant
amount of running state in **O(1)** (or amortised-O(1)) time and **constant
memory**, then emits the current value of a *precursor feature* — a leading
indicator that trends *before* a threshold breach. Computing these online (not in
batches) is the lead-time win: the score is always current, so the NOC sees the
warning at the earliest possible instant (research ``02-telemetry-pipeline.md``
§3).

The building blocks (all O(1)/amortised, all CPU, all offline):

  * ``river.stats.Mean`` / ``Var``           — Welford running mean/variance.
  * ``river.stats.EWMean`` / ``EWVar``       — exponentially-weighted mean/var.
  * ``ddsketch.DDSketch``                    — streaming quantiles (p95/p99) with
                                                a relative-error guarantee, mergeable.
  * ``stumpy.stumpi``                        — incremental Matrix Profile discord
                                                (amortised-O(1) ``.update()``).
  * Count-Min Sketch + HyperLogLog           — sublinear heavy-hitter frequency and
                                                distinct-cardinality (``pyprobables``
                                                if present, else a small built-in).

Optional heavy/extra deps (``stumpy``, ``pyprobables``) are import-guarded: if a
library is missing the corresponding computer falls back to a lighter exact or
approximate implementation so the CPU-only path always runs.

Precursor features implemented here (each O(1)/incremental), mapping to the
network failure modes in ``ARCHITECTURE.md`` §6:

  =============================  ====================================  =================
  Feature computer               Precursor it surfaces                 Fails-of scenario
  =============================  ====================================  =================
  ``RollingSlope``               rolling utilisation slope             A congestion
  ``LatencyDrift``               latency drift (mean shift)            A / C
  ``JitterTrend``                jitter variance trend + p99 tail      C tunnel
  ``LossProgression``            monotonic loss-ratio rise             C tunnel
  ``ErrorRateAcceleration``      2nd-derivative of error counters      faulty optics
  ``BGPChurnRate``               BGP update/withdraw rate              B bgp flap
  ``AdjacencyFlapCount``         adjacency up/down rate                B bgp flap
  ``RekeyIntervalAnomaly``       IPSec rekey-interval deviation        C tunnel
  ``PathAsymmetry``              fwd vs rev path divergence            B reroute
  ``TopTalkerChurn``             heavy-hitter set churn                A traffic shift
  ``TimeToThreshold``            streaming seconds-to-SLA-crossing     headline "when"
  =============================  ====================================  =================

All computers share the :class:`FeatureComputer` protocol: construct once per
``(entity, signal)``, then call :meth:`update` per sample. ``update`` returns the
current feature value (``float`` or ``None`` while warming up) so the engine can
read it without reaching into internal state.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

# --- core deps (always present in the CORE tier) ---------------------------
from river import stats

try:  # ddsketch is a CORE dep; guard anyway so a partial install still imports.
    from ddsketch import DDSketch

    _HAS_DDSKETCH = True
except Exception:  # pragma: no cover - exercised only on a broken install
    DDSketch = None  # type: ignore[assignment]
    _HAS_DDSKETCH = False

# --- optional/heavy deps (import-guarded; fallbacks below) -----------------
try:
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    import stumpy as _stumpy  # heavy: pulls Numba. amortised-O(1) .update().

    _HAS_STUMPY = True
except Exception:  # pragma: no cover - CPU-only path must still run without it
    _stumpy = None  # type: ignore[assignment]
    _HAS_STUMPY = False

try:
    # pyprobables ships CountMinSketch + HyperLogLog. Extra dep (see
    # requirements-streaming.txt); fall back to the small impls below if absent.
    from probables import CountMinSketch as _PPCountMinSketch
    from probables import HyperLogLog as _PPHyperLogLog

    _HAS_PYPROBABLES = True
except Exception:
    _PPCountMinSketch = None  # type: ignore[assignment]
    _PPHyperLogLog = None  # type: ignore[assignment]
    _HAS_PYPROBABLES = False


__all__ = [
    "FeatureComputer",
    "RollingSlope",
    "LatencyDrift",
    "JitterTrend",
    "LossProgression",
    "ErrorRateAcceleration",
    "BGPChurnRate",
    "AdjacencyFlapCount",
    "RekeyIntervalAnomaly",
    "PathAsymmetry",
    "TopTalkerChurn",
    "TimeToThreshold",
    "StreamingQuantile",
    "MatrixProfileDiscord",
    "CountMinSketch",
    "HyperLogLog",
]


@runtime_checkable
class FeatureComputer(Protocol):
    """Protocol every O(1) feature computer satisfies.

    Construct once per ``(entity, signal)``; call :meth:`update` per sample.
    Implementations MUST keep per-update cost and memory constant.
    """

    name: str

    def update(self, value: float, ts: float | None = None) -> float | None:
        """Fold one sample in (O(1)) and return the current feature value.

        ``ts`` is an optional absolute timestamp in seconds (epoch). Returns
        ``None`` while the computer is still warming up.
        """
        ...


# ---------------------------------------------------------------------------
# Small, dependency-free streaming sketches (used as fallbacks AND directly).
# ---------------------------------------------------------------------------


class CountMinSketch:
    """Count-Min Sketch — approximate per-key frequency in sublinear space.

    O(d) per update/query where ``d`` (the number of hash rows) is a small
    constant, hence O(1). Over-estimates only (never under-counts), with error
    bounded by ``epsilon * N`` with probability ``1 - delta``. Used for
    heavy-hitter / top-talker tracking on flow keys (research §3 D).

    Prefers ``pyprobables`` when available (battle-tested), else uses this exact
    pure-Python implementation so the feature works with zero extra deps.
    """

    def __init__(self, width: int = 1024, depth: int = 4, *, seed: int = 1469598103) -> None:
        self.width = int(width)
        self.depth = int(depth)
        self._seed = seed
        self._backend = None
        if _HAS_PYPROBABLES:
            try:
                self._backend = _PPCountMinSketch(width=self.width, depth=self.depth)
            except Exception:  # pragma: no cover - defensive
                self._backend = None
        if self._backend is None:
            self._rows: list[list[int]] = [
                [0] * self.width for _ in range(self.depth)
            ]

    def _hashes(self, key: str) -> Iterable[int]:
        # Two independent hashes combined into ``depth`` via the Kirsch-Mitzenmacher
        # double-hashing trick: h_i = h1 + i*h2 (mod width). O(depth), constant.
        h1 = hash((self._seed, key)) & 0xFFFFFFFF
        h2 = (hash((self._seed ^ 0x9E3779B9, key)) & 0xFFFFFFFF) | 1
        for i in range(self.depth):
            yield (h1 + i * h2) % self.width

    def add(self, key: str, count: int = 1) -> None:
        """Increment the estimated count of ``key`` by ``count`` (O(1))."""
        if self._backend is not None:
            self._backend.add(key, count)
            return
        for row, col in enumerate(self._hashes(key)):
            self._rows[row][col] += count

    def estimate(self, key: str) -> int:
        """Return the (over-)estimated count of ``key`` (O(1))."""
        if self._backend is not None:
            return int(self._backend.check(key))
        return min(self._rows[row][col] for row, col in enumerate(self._hashes(key)))


class HyperLogLog:
    """HyperLogLog — approximate distinct-element cardinality in O(1) per add.

    Tracks the number of *unique* items (e.g. unique source IPs / flow 5-tuples)
    in constant memory, with a standard relative error of ~``1.04/sqrt(2^p)``.
    Prefers ``pyprobables`` if present, else a compact built-in estimator.
    """

    def __init__(self, p: int = 12) -> None:
        self.p = int(p)
        self._backend = None
        if _HAS_PYPROBABLES:
            try:
                self._backend = _PPHyperLogLog(p=self.p)
            except Exception:  # pragma: no cover - defensive
                self._backend = None
        if self._backend is None:
            self._m = 1 << self.p
            self._registers = bytearray(self._m)
            # bias-correction constant alpha_m (Flajolet et al.)
            if self._m == 16:
                self._alpha = 0.673
            elif self._m == 32:
                self._alpha = 0.697
            elif self._m == 64:
                self._alpha = 0.709
            else:
                self._alpha = 0.7213 / (1.0 + 1.079 / self._m)

    def add(self, item: str) -> None:
        """Register one observation of ``item`` (O(1))."""
        if self._backend is not None:
            self._backend.add(item)
            return
        x = hash((0xC2B2AE3D, item)) & 0xFFFFFFFFFFFFFFFF
        idx = x & (self._m - 1)
        w = x >> self.p
        # rank = position of leftmost 1-bit (+1) in the remaining 64-p bits.
        rank = (64 - self.p) - w.bit_length() + 1 if w else (64 - self.p) + 1
        if rank > self._registers[idx]:
            self._registers[idx] = rank

    def count(self) -> float:
        """Return the estimated number of distinct items seen (O(m))."""
        if self._backend is not None:
            return float(len(self._backend))
        m = self._m
        registers = self._registers
        # raw HLL estimate
        z = sum(2.0 ** (-r) for r in registers)
        est = self._alpha * m * m / z if z else 0.0
        if est <= 2.5 * m:  # small-range linear-counting correction
            zeros = registers.count(0)
            if zeros:
                est = m * math.log(m / zeros)
        return est


class StreamingQuantile:
    """Streaming p95/p99 tail via DDSketch (O(1) insert, relative-error bound).

    Latency and jitter are long-tailed; DDSketch gives a relative-error guarantee
    on any quantile, is mergeable across devices, and inserts in O(1) (research §3
    C). Falls back to a bounded reservoir + exact quantile if ddsketch is absent
    (still O(1) insert; the read is O(k log k) over the bounded sample).
    """

    def __init__(self, relative_accuracy: float = 0.01, *, reservoir_size: int = 2048) -> None:
        self.relative_accuracy = relative_accuracy
        self._reservoir_size = reservoir_size
        if _HAS_DDSKETCH:
            self._sketch = DDSketch(relative_accuracy=relative_accuracy)
            self._reservoir = None
        else:  # pragma: no cover - ddsketch is a core dep
            self._sketch = None
            self._reservoir = []  # type: ignore[var-annotated]
            self._n = 0

    def add(self, value: float) -> None:
        """Insert ``value`` (O(1))."""
        if self._sketch is not None:
            # DDSketch (non-extended) only handles values > 0; clamp tiny/neg.
            self._sketch.add(value if value > 0 else 1e-9)
            return
        # pragma: no cover - reservoir fallback
        self._n += 1
        if len(self._reservoir) < self._reservoir_size:
            self._reservoir.append(value)
        else:
            import random

            j = random.randint(0, self._n - 1)
            if j < self._reservoir_size:
                self._reservoir[j] = value

    def quantile(self, q: float) -> float | None:
        """Return the value at quantile ``q`` in [0,1], or None if empty."""
        if self._sketch is not None:
            try:
                return self._sketch.get_quantile_value(q)
            except Exception:
                return None
        if not self._reservoir:  # pragma: no cover
            return None
        s = sorted(self._reservoir)  # pragma: no cover
        idx = min(len(s) - 1, int(q * len(s)))  # pragma: no cover
        return s[idx]  # pragma: no cover


class MatrixProfileDiscord:
    """Streaming Matrix-Profile discord via ``stumpy.stumpi`` (amortised-O(1)).

    The Matrix Profile of a time series records, for every subsequence, the
    distance to its nearest neighbour; a *large* value = a subsequence unlike any
    other = a **discord** (shape anomaly). ``stumpy.stumpi`` maintains this
    incrementally: seed once over ``m`` points, then ``.update(x)`` per new sample
    is amortised-O(1). We expose the latest profile value ``P_[-1]``.

    If ``stumpy``/``numpy`` are unavailable the computer degrades to ``None``
    (the engine simply omits ``mp_discord``); the rest of the feature set is
    unaffected — graceful degradation per the build plan.
    """

    def __init__(self, window: int = 16, *, warmup: int | None = None) -> None:
        self.window = int(window)
        # stumpi needs >= 2*m seed points to be meaningful; default warmup 2*m+1.
        self.warmup = int(warmup) if warmup is not None else 2 * self.window + 1
        self._seed: deque[float] = deque(maxlen=self.warmup)
        self._stream = None
        self._available = _HAS_STUMPY and _HAS_NUMPY

    def update(self, value: float, ts: float | None = None) -> float | None:
        if not self._available:
            return None
        if self._stream is None:
            self._seed.append(float(value))
            if len(self._seed) < self.warmup:
                return None
            try:
                seed = _np.asarray(self._seed, dtype=float)
                self._stream = _stumpy.stumpi(seed, m=self.window, egress=False)
            except Exception:  # pragma: no cover - defensive; disable on failure
                self._available = False
                return None
            return float(self._stream.P_[-1]) if len(self._stream.P_) else None
        try:
            self._stream.update(float(value))
            p = self._stream.P_
            val = float(p[-1]) if len(p) else None
            if val is not None and (math.isinf(val) or math.isnan(val)):
                return None
            return val
        except Exception:  # pragma: no cover - defensive
            return None


# ---------------------------------------------------------------------------
# Precursor feature computers (the network leading indicators).
# ---------------------------------------------------------------------------


class RollingSlope:
    """Rolling utilisation slope d(value)/dt — congestion buildup precursor.

    Estimates the rate of change per second with an **EWMA of first differences**
    — strictly O(1) (constant state, no window buffer). A sustained positive
    slope on interface utilisation is the earliest sign of progressive congestion
    (scenario A): the metric is still below threshold but trending toward it.

    The returned value is in *units per second* (e.g. %/s for utilisation). The
    engine typically also surfaces ``per-minute`` by multiplying by 60.
    """

    name = "rolling_slope"

    def __init__(self, fading_factor: float = 0.2) -> None:
        self._ew = stats.EWMean(fading_factor=fading_factor)
        self._prev_value: float | None = None
        self._prev_ts: float | None = None
        self._slope: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        if self._prev_value is not None:
            dv = value - self._prev_value
            dt = 1.0
            if ts is not None and self._prev_ts is not None:
                dt = ts - self._prev_ts
                if dt <= 0:
                    dt = 1.0
            self._ew.update(dv / dt)
            self._slope = self._ew.get()
        self._prev_value = value
        self._prev_ts = ts
        return self._slope


class LatencyDrift:
    """Latency drift — an upward shift in mean RTT (path/queue degradation).

    Pairs an O(1) EWMA *level* with the *deviation* of the latest sample from a
    slow running baseline (Welford mean). A positive ``drift`` (current EWMA above
    the long-run mean) is a precursor of queue buildup / path degradation before
    loss appears. The discrete change-point *trigger* is handled separately by
    the drift detectors in ``detectors.py`` (Page-Hinkley on the same stream).
    """

    name = "latency_drift"

    def __init__(self, fading_factor: float = 0.1) -> None:
        self._fast = stats.EWMean(fading_factor=fading_factor)
        self._slow = stats.Mean()
        self._level: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        self._fast.update(value)
        self._slow.update(value)
        fast = self._fast.get()
        slow = self._slow.get()
        self._level = fast
        if fast is None or slow is None:
            return None
        return fast - slow  # signed drift of current level vs long-run baseline

    @property
    def level(self) -> float | None:
        """Current EWMA latency level (the ``latency_ewma`` feature)."""
        return self._level


class JitterTrend:
    """Jitter trend — rising variance of inter-packet delay (tunnel instability).

    Tracks an O(1) exponentially-weighted *variance* (``river.stats.EWVar``) as
    the trend signal and a DDSketch p99 tail. Growing jitter variance precedes
    tunnel SLA loss (scenario C). Returns the current EWVar; the p99 tail is read
    via :meth:`p99`.
    """

    name = "jitter_trend"

    def __init__(self, fading_factor: float = 0.1) -> None:
        self._ewvar = stats.EWVar(fading_factor=fading_factor)
        self._p99 = StreamingQuantile(relative_accuracy=0.01)
        self._var: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        self._ewvar.update(value)
        self._p99.add(value)
        self._var = self._ewvar.get()
        return self._var

    def p99(self) -> float | None:
        """Streaming p99 of jitter (the ``jitter_p99`` feature)."""
        return self._p99.quantile(0.99)


class LossProgression:
    """Loss progression — monotonic rise in loss ratio (underlay/tunnel health).

    EWMA of the loss ratio plus a *monotonic-rise* indicator: an O(1) count of
    consecutive non-decreasing EWMA steps, normalised. A steadily climbing loss
    EWMA (rather than spiky noise) is the strong precursor of MPLS underlay /
    tunnel failure (scenario C). Returns the EWMA level; :meth:`rising_streak`
    exposes how persistently it has been climbing.
    """

    name = "loss_progression"

    def __init__(self, fading_factor: float = 0.15) -> None:
        self._ew = stats.EWMean(fading_factor=fading_factor)
        self._prev: float | None = None
        self._streak = 0
        self._level: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        self._ew.update(value)
        cur = self._ew.get()
        if cur is not None and self._prev is not None:
            if cur >= self._prev - 1e-12:
                self._streak += 1
            else:
                self._streak = 0
        self._prev = cur
        self._level = cur
        return cur

    def rising_streak(self) -> int:
        """Number of consecutive non-decreasing EWMA steps (rise persistence)."""
        return self._streak


class ErrorRateAcceleration:
    """Error-rate acceleration — 2nd derivative of CRC/discard counters.

    Counters are cumulative, so we difference once for the *rate* (errors/s) and
    again for the *acceleration* (errors/s²), each smoothed by an O(1) EWMA. A
    positive acceleration means the error rate itself is increasing — the
    signature of a degrading interface/optic before hard failure. Returns the
    acceleration; :meth:`rate` exposes the first derivative.
    """

    name = "error_rate_acceleration"

    def __init__(self, fading_factor: float = 0.2) -> None:
        self._rate_ew = stats.EWMean(fading_factor=fading_factor)
        self._accel_ew = stats.EWMean(fading_factor=fading_factor)
        self._prev_count: float | None = None
        self._prev_ts: float | None = None
        self._prev_rate: float | None = None
        self._rate: float | None = None
        self._accel: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        if self._prev_count is not None:
            dt = 1.0
            if ts is not None and self._prev_ts is not None:
                dt = ts - self._prev_ts
                if dt <= 0:
                    dt = 1.0
            # cumulative counters can reset (reboot); clamp negative deltas to 0.
            d = max(0.0, value - self._prev_count)
            self._rate_ew.update(d / dt)
            self._rate = self._rate_ew.get()
            if self._prev_rate is not None and self._rate is not None:
                self._accel_ew.update((self._rate - self._prev_rate) / dt)
                self._accel = self._accel_ew.get()
            self._prev_rate = self._rate
        self._prev_count = value
        self._prev_ts = ts
        return self._accel

    def rate(self) -> float | None:
        """Current smoothed error rate (errors per second)."""
        return self._rate


class BGPChurnRate:
    """BGP churn rate — UPDATE/withdraw events per second (flap precursor).

    Routing events arrive as discrete occurrences; this is an O(1) *event-rate*
    estimator using an EWMA over inter-arrival gaps (or per-bucket counts). A
    rising churn rate precedes a full route-flap cascade (scenario B). Call
    :meth:`event` once per BGP UPDATE/withdraw with its timestamp; call
    :meth:`tick` to decay the rate when no event occurred in an interval.
    """

    name = "bgp_churn_rate"

    def __init__(self, fading_factor: float = 0.1) -> None:
        self._ew = stats.EWMean(fading_factor=fading_factor)
        self._last_ts: float | None = None
        self._rate: float | None = None

    def event(self, ts: float | None = None) -> float | None:
        """Register one churn event (UPDATE/withdraw) and return current rate."""
        if ts is not None and self._last_ts is not None:
            gap = ts - self._last_ts
            inst_rate = 1.0 / gap if gap > 1e-9 else 1.0
            self._ew.update(inst_rate)
            self._rate = self._ew.get()
        elif ts is not None:
            self._ew.update(0.0)  # seed
            self._rate = self._ew.get()
        if ts is not None:
            self._last_ts = ts
        return self._rate

    def update(self, value: float, ts: float | None = None) -> float | None:
        """FeatureComputer-compatible: treat ``value`` as a per-tick event count.

        Folds ``value`` events (a count over the tick) into the rate EWMA. Use
        :meth:`event` for true per-event arrival timing.
        """
        self._ew.update(float(value))
        self._rate = self._ew.get()
        if ts is not None:
            self._last_ts = ts
        return self._rate

    def rate(self) -> float | None:
        return self._rate


class AdjacencyFlapCount:
    """Adjacency flap count/rate — OSPF/BGP up↓down transitions per window.

    A bounded ring buffer (``deque(maxlen=w)``) of recent transition timestamps
    gives an O(1) sliding *count* and *rate*; pair it with the ADWIN detector on
    the rate for a change-point trigger. Rising adjacency flaps signal convergence
    stress / link instability (scenario B). Call :meth:`flap` on each transition.
    """

    name = "adjacency_flap_count"

    def __init__(self, window_seconds: float = 300.0, maxlen: int = 1024) -> None:
        self.window_seconds = window_seconds
        self._events: deque[float] = deque(maxlen=maxlen)
        self._count = 0

    def flap(self, ts: float) -> int:
        """Register one adjacency up/down transition; return current window count."""
        self._events.append(float(ts))
        self._evict(ts)
        return len(self._events)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def update(self, value: float, ts: float | None = None) -> float | None:
        """FeatureComputer-compatible: ``value`` flaps occurred at ``ts``."""
        now = ts if ts is not None else (self._events[-1] if self._events else 0.0)
        for _ in range(int(value)):
            self._events.append(float(now))
        self._evict(now)
        return float(len(self._events))

    def count(self, now: float | None = None) -> int:
        if now is not None:
            self._evict(now)
        return len(self._events)


class RekeyIntervalAnomaly:
    """IPSec rekey-interval anomaly — deviation from the expected rekey period.

    Learns the normal rekey interval online (Welford mean + std) and scores each
    new observed interval as a |z|-score — O(1), constant memory. A rekey
    interval that suddenly shortens/lengthens is a control-plane misbehaviour
    precursor (scenario C). Returns the absolute z-score of the latest interval.
    """

    name = "rekey_interval_anomaly"

    def __init__(self, warmup: int = 5, flat_baseline_score: float = 6.0) -> None:
        self._mean = stats.Mean()
        self._var = stats.Var()
        self._n = 0
        self.warmup = warmup
        # score assigned when a steady (zero-variance) baseline is first broken
        self.flat_baseline_score = float(flat_baseline_score)

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        z: float | None = None
        if self._n >= self.warmup:
            mu = self._mean.get()
            var = self._var.get()
            sd = math.sqrt(var) if var and var > 0 else 0.0
            if mu is not None:
                dev = abs(value - mu)
                if sd > 1e-9:
                    z = dev / sd
                elif dev > 1e-9:
                    # Degenerate baseline (zero variance) but the new interval
                    # differs from the constant mean: that IS the anomaly. Use a
                    # relative deviation floor so a sudden change from a perfectly
                    # steady rekey period still scores high (avoids 0/0 -> None).
                    rel = dev / (abs(mu) + 1e-9)
                    z = max(self.flat_baseline_score, rel * 100.0)
                else:
                    z = 0.0  # identical to a flat baseline -> not anomalous
        self._mean.update(value)
        self._var.update(value)
        self._n += 1
        return z


class PathAsymmetry:
    """Path asymmetry — divergence between forward and reverse path metrics.

    Maintains an O(1) EWMA of the *signed difference* between a forward and a
    reverse directional metric (e.g. AS-path length, RTT, or hop count per
    direction). A growing asymmetry indicates asymmetric routing / a reroute
    cascade (scenario B). Feed paired samples via :meth:`update_pair`; the scalar
    :meth:`update` treats ``value`` as a pre-computed fwd-minus-rev difference.
    """

    name = "path_asymmetry"

    def __init__(self, fading_factor: float = 0.2) -> None:
        self._ew = stats.EWMean(fading_factor=fading_factor)
        self._fwd: float | None = None
        self._rev: float | None = None
        self._asym: float | None = None

    def update_pair(
        self, forward: float, reverse: float, ts: float | None = None
    ) -> float | None:
        self._fwd = float(forward)
        self._rev = float(reverse)
        self._ew.update(self._fwd - self._rev)
        self._asym = self._ew.get()
        return self._asym

    def update(self, value: float, ts: float | None = None) -> float | None:
        self._ew.update(float(value))
        self._asym = self._ew.get()
        return self._asym


class TopTalkerChurn:
    """Top-talker churn — change in the heavy-hitter flow set over time.

    Uses a Count-Min Sketch to track approximate per-flow byte/packet frequency
    in O(1), snapshots the current top-K heavy hitters each window, and reports
    the **Jaccard churn** (``1 - |A∩B|/|A∪B|``) versus the previous window's set.
    A high churn = a shifting traffic matrix / micro-burst onset (scenario A).
    All operations are O(1) per flow plus O(K) per window snapshot.
    """

    name = "top_talker_churn"

    def __init__(self, top_k: int = 10, width: int = 2048, depth: int = 4) -> None:
        self.top_k = top_k
        self._cms = CountMinSketch(width=width, depth=depth)
        self._seen: set[str] = set()  # candidate keys this window (bounded by reset)
        self._prev_top: set[str] | None = None
        self._churn: float | None = None

    def add_flow(self, key: str, weight: int = 1) -> None:
        """Record ``weight`` units of traffic for flow ``key`` (O(1))."""
        self._cms.add(key, weight)
        # keep a bounded candidate set so the top-K snapshot stays cheap
        if len(self._seen) < 4096:
            self._seen.add(key)

    def snapshot(self) -> float | None:
        """Close the window: compute top-K, return churn vs previous, then reset."""
        if not self._seen:
            return self._churn
        ranked = sorted(self._seen, key=self._cms.estimate, reverse=True)
        top = set(ranked[: self.top_k])
        if self._prev_top is not None:
            union = top | self._prev_top
            inter = top & self._prev_top
            self._churn = 1.0 - (len(inter) / len(union)) if union else 0.0
        self._prev_top = top
        # reset candidate set + sketch for the next window (constant memory)
        self._seen = set()
        self._cms = CountMinSketch(width=self._cms.width, depth=self._cms.depth)
        return self._churn

    def update(self, value: float, ts: float | None = None) -> float | None:
        """FeatureComputer-compatible: returns the last computed churn value."""
        return self._churn


class TimeToThreshold:
    """Streaming time-to-threshold helper — the headline "when" (O(1)).

    Linear-extrapolates the current level + slope to the first time the metric is
    predicted to cross ``threshold``. Level and slope come from O(1) EWMA
    estimators, so the ETA is recomputed every sample with microsecond latency —
    the streaming precursor of :class:`~netra.contracts.TimeToImpact` (the full
    analytics layer refines this with forecast bands + survival models).

    Returns seconds-to-crossing (``>= 0``) or ``None`` when the trajectory is not
    heading toward the threshold (healthy / moving away).
    """

    name = "time_to_threshold"

    def __init__(
        self,
        threshold: float,
        *,
        above_is_breach: bool = True,
        fading_factor: float = 0.2,
        max_eta_seconds: float = 86400.0,
    ) -> None:
        self.threshold = float(threshold)
        self.above_is_breach = above_is_breach
        self.max_eta_seconds = max_eta_seconds
        self._level_ew = stats.EWMean(fading_factor=fading_factor)
        self._slope = RollingSlope(fading_factor=fading_factor)
        self._level: float | None = None
        self._eta: float | None = None

    def update(self, value: float, ts: float | None = None) -> float | None:
        value = float(value)
        self._level_ew.update(value)
        level = self._level_ew.get()
        slope = self._slope.update(value, ts)  # units per second
        self._level = level
        self._eta = None
        if level is None or slope is None or abs(slope) < 1e-12:
            return None
        gap = self.threshold - level
        if self.above_is_breach:
            # need level to rise to threshold: gap>0 and slope>0
            if gap > 0 and slope > 0:
                self._eta = min(gap / slope, self.max_eta_seconds)
            elif gap <= 0:
                self._eta = 0.0  # already breached
        else:
            # breach is crossing BELOW threshold: gap<0 and slope<0
            if gap < 0 and slope < 0:
                self._eta = min(gap / slope, self.max_eta_seconds)
            elif gap >= 0:
                self._eta = 0.0
        return self._eta

    @property
    def level(self) -> float | None:
        return self._level
