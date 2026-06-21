"""Streaming change / anomaly detectors — O(1) precursor triggers (Workstream 2).

Where ``features.py`` computes *continuous* leading indicators, this module
fires the *discrete* triggers: the instant a stream's distribution shifts or a
sample is anomalous. These are the "precursor firing" signals the fusion layer
treats as votes (``FeatureVector.triggered_drift``), and they are exactly the
change-point / drift / streaming-AD members of the Phase-3 catalogue that belong
in the O(1) real-time path (research ``02-telemetry-pipeline.md`` §3 B, E):

  * ``river.drift.ADWIN``         — adaptive-windowing distribution change (#39).
  * ``river.drift.PageHinkley``   — CUSUM-family mean-shift detector (#38).
  * ``river.drift.KSWIN``         — Kolmogorov-Smirnov windowed change (#40).
  * ``CUSUM``                     — classic two-sided cumulative-sum detector (#37).
  * ``EWMAControlChart``          — EWMA control chart (Shewhart/EWMA, #20/#21).
  * ``river.anomaly.HalfSpaceTrees`` — always-on unsupervised multivariate AD (#26).

Every detector exposes a uniform interface::

    det = SomeDetector(...)
    fired: bool = det.update(value)      # O(1); True on this sample's trigger
    det.name                              # stable id, e.g. "page_hinkley"

so the engine can iterate a registry of detectors per signal and collect the
names that fired this tick. The Half-Space-Trees wrapper additionally exposes a
[0,1] ``score`` and **scales every feature to [0,1] internally** (River's HST
requires inputs in the unit cube — the single most common HST footgun).

Determinism: all detectors are seedable / parameter-deterministic so a fixed
input series yields the same trigger sequence (tested in ``tests/test_streaming``).
"""

from __future__ import annotations

import math
from typing import Mapping

from river import anomaly, drift, preprocessing

__all__ = [
    "ADWINDetector",
    "PageHinkleyDetector",
    "KSWINDetector",
    "CUSUM",
    "EWMAControlChart",
    "HalfSpaceTreesDetector",
]


class ADWINDetector:
    """ADWIN adaptive-windowing drift detector (O(1)-amortised).

    Maintains a variable-length window and signals when the means of two
    sub-windows differ beyond a Hoeffding bound — i.e. the stream's distribution
    has changed. Ideal for non-stationary rates (BGP update rate, adjacency flap
    rate) where the *change* is the precursor, not an absolute level.
    """

    name = "adwin"

    def __init__(self, delta: float = 0.002) -> None:
        self._d = drift.ADWIN(delta=delta)
        self._fired = False

    def update(self, value: float) -> bool:
        self._d.update(float(value))
        self._fired = bool(self._d.drift_detected)
        return self._fired

    @property
    def drift_detected(self) -> bool:
        return self._fired


class PageHinkleyDetector:
    """Page-Hinkley test — CUSUM-family mean-shift detector (O(1)).

    Keeps only a handful of running scalars (cumulative deviation + running min),
    so it is O(1) per sample and ideal for high-rate per-interface streams. Fires
    when the cumulative deviation from the running mean exceeds ``threshold`` —
    the canonical latency-drift / loss-progression trigger (scenario A/C).
    """

    name = "page_hinkley"

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 1e-4,
    ) -> None:
        self._d = drift.PageHinkley(
            min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha
        )
        self._fired = False

    def update(self, value: float) -> bool:
        self._d.update(float(value))
        self._fired = bool(self._d.drift_detected)
        return self._fired

    @property
    def drift_detected(self) -> bool:
        return self._fired


class KSWINDetector:
    """KSWIN — Kolmogorov-Smirnov windowing change detector.

    Compares the empirical distributions of a recent sub-window vs a reference
    sample via the KS statistic; fires when they diverge at significance
    ``alpha``. Per-check cost is O(window) (a bounded constant), so the amortised
    per-sample cost stays constant. Good at distribution-*shape* changes that a
    mean-only test (Page-Hinkley) can miss.
    """

    name = "kswin"

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = 42,
    ) -> None:
        self._d = drift.KSWIN(
            alpha=alpha, window_size=window_size, stat_size=stat_size, seed=seed
        )
        self._fired = False

    def update(self, value: float) -> bool:
        self._d.update(float(value))
        self._fired = bool(self._d.drift_detected)
        return self._fired

    @property
    def drift_detected(self) -> bool:
        return self._fired


class CUSUM:
    """Two-sided CUSUM (cumulative sum) change detector — O(1), constant memory.

    Tracks upward and downward cumulative sums of standardised deviations from a
    running mean and fires when either exceeds ``threshold`` (in std units). A
    small, transparent, dependency-free complement to Page-Hinkley that we own
    end-to-end (useful for the determinism test). The running mean/variance are
    Welford-updated so the detector is self-calibrating.

    Parameters
    ----------
    threshold:
        Decision threshold ``h`` in standard-deviation units (typical 4-5).
    drift:
        Allowance/slack ``k`` in std units (typical 0.5) — the half-shift the
        chart is tuned to detect; larger ``k`` = less sensitive to small drifts.
    warmup:
        Samples to accumulate before the detector may fire (lets mean/var settle).
    """

    name = "cusum"

    def __init__(
        self, threshold: float = 5.0, drift: float = 0.5, warmup: int = 20
    ) -> None:
        self.threshold = float(threshold)
        self.k = float(drift)
        self.warmup = int(warmup)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared deviations (Welford)
        self._g_pos = 0.0
        self._g_neg = 0.0
        self._fired = False

    def update(self, value: float) -> bool:
        value = float(value)
        # Welford online mean/variance (O(1)).
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)
        self._fired = False
        if self._n <= self.warmup:
            return False
        var = self._m2 / (self._n - 1) if self._n > 1 else 0.0
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd < 1e-12:
            # no variability yet -> cannot standardise; hold steady.
            return False
        z = (value - self._mean) / sd
        # standardised CUSUM recursion with slack k
        self._g_pos = max(0.0, self._g_pos + z - self.k)
        self._g_neg = min(0.0, self._g_neg + z + self.k)
        if self._g_pos > self.threshold or self._g_neg < -self.threshold:
            self._fired = True
            # reset accumulators after a detection (standard practice)
            self._g_pos = 0.0
            self._g_neg = 0.0
        return self._fired

    @property
    def drift_detected(self) -> bool:
        return self._fired


class EWMAControlChart:
    """EWMA control chart — detects small persistent shifts in the mean (O(1)).

    Maintains an exponentially-weighted mean ``z`` and flags when it leaves the
    control band ``mu ± L * sigma_z`` where ``sigma_z = sigma * sqrt(lambda/(2-lambda))``
    is the (asymptotic) EWMA standard error. EWMA charts are more sensitive than
    Shewhart charts to *small* sustained drifts — exactly the sub-threshold
    creep that buys lead time. Mean/variance are learned online (Welford), so the
    chart self-calibrates without a separate training phase.

    Parameters
    ----------
    lambda_:
        Smoothing constant in (0,1]; smaller = longer memory, more sensitive to
        small shifts (typical 0.1-0.3).
    L:
        Control-limit width in sigma units (typical 2.7-3.0).
    warmup:
        Samples before the chart may fire (mean/var settling period).
    """

    name = "ewma_control"

    def __init__(self, lambda_: float = 0.2, L: float = 3.0, warmup: int = 20) -> None:
        self.lmbda = float(lambda_)
        self.L = float(L)
        self.warmup = int(warmup)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._z: float | None = None
        self._fired = False

    def update(self, value: float) -> bool:
        value = float(value)
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (value - self._mean)
        # EWMA statistic seeded at the running mean
        if self._z is None:
            self._z = self._mean
        else:
            self._z = self.lmbda * value + (1 - self.lmbda) * self._z
        self._fired = False
        if self._n <= self.warmup:
            return False
        var = self._m2 / (self._n - 1) if self._n > 1 else 0.0
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd < 1e-12:
            return False
        sigma_z = sd * math.sqrt(self.lmbda / (2 - self.lmbda))
        ucl = self._mean + self.L * sigma_z
        lcl = self._mean - self.L * sigma_z
        if self._z > ucl or self._z < lcl:
            self._fired = True
        return self._fired

    @property
    def statistic(self) -> float | None:
        """Current EWMA statistic ``z`` (for charting)."""
        return self._z

    @property
    def drift_detected(self) -> bool:
        return self._fired


class HalfSpaceTreesDetector:
    """Half-Space-Trees streaming anomaly detector (O(1) per update).

    River's HST is an always-on, unsupervised, multivariate anomaly detector: an
    ensemble of random half-space trees scoring a point by the "mass" of the leaf
    it lands in. **Requirement (the classic footgun): inputs must be in [0,1].**
    This wrapper therefore runs every feature through an online
    ``preprocessing.MinMaxScaler`` first, so callers may pass raw-valued feature
    dicts and still get a valid score.

    Scoring order matters and is handled here: we **score before learning** each
    point (so a point is judged against the model built from its predecessors),
    matching River's recommended streaming-AD usage. ``score`` is already in
    [0,1] (higher = more anomalous) and so is directly usable as a fusion input /
    HST feature.

    Parameters mirror ``river.anomaly.HalfSpaceTrees`` (``n_trees``, ``height``,
    ``window_size``, ``seed``). ``threshold`` sets the boolean ``is_anomaly``
    decision on the [0,1] score.
    """

    name = "half_space_trees"

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 10,
        window_size: int = 250,
        seed: int | None = 42,
        threshold: float = 0.9,
        scale_inputs: bool = True,
    ) -> None:
        self.threshold = float(threshold)
        self._scale_inputs = scale_inputs
        self._scaler = preprocessing.MinMaxScaler() if scale_inputs else None
        self._hst = anomaly.HalfSpaceTrees(
            n_trees=n_trees, height=height, window_size=window_size, seed=seed
        )
        self._score = 0.0
        self._fired = False

    def _prep(self, x: Mapping[str, float]) -> dict[str, float]:
        xd = {k: float(v) for k, v in x.items()}
        if self._scaler is None:
            # caller asserts inputs already in [0,1]; clamp defensively.
            return {k: min(1.0, max(0.0, v)) for k, v in xd.items()}
        # online scaling: learn the running min/max, then transform.
        self._scaler.learn_one(xd)
        scaled = self._scaler.transform_one(xd)
        # MinMaxScaler can emit slightly-out-of-range values on new extremes;
        # clamp to keep HST's [0,1] contract.
        return {k: min(1.0, max(0.0, float(v))) for k, v in scaled.items()}

    def update(self, x: Mapping[str, float] | float) -> bool:
        """Fold one (multivariate) point in (O(1)); return ``is_anomaly``.

        Accepts a feature ``dict`` or a bare ``float`` (wrapped as ``{"v": x}``).
        """
        xd = {"v": float(x)} if isinstance(x, (int, float)) else dict(x)
        scaled = self._prep(xd)
        # score THEN learn (judge against predecessors).
        self._score = float(self._hst.score_one(scaled))
        self._hst.learn_one(scaled)
        self._fired = self._score >= self.threshold
        return self._fired

    @property
    def score(self) -> float:
        """Latest [0,1] anomaly score (higher = more anomalous)."""
        return self._score

    @property
    def is_anomaly(self) -> bool:
        return self._fired
