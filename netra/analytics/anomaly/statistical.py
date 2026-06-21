"""Statistical / streaming anomaly detectors (Family 1, #19-#29).

The always-on, O(1)-per-point first tier (research 04 §2). Cheap, interpretable,
and instantly available — they give lead-time signal at negligible cost:

  * :class:`RobustZDetector` (#19)   — rolling median + MAD robust z-score; a few
    existing anomalies don't poison the baseline.
  * :class:`EwmaControlChart` (#20)  — EWMA ± L·σ_EWMA control limits; catches
    slow drifts (congestion buildup) earlier than a Shewhart chart.
  * :class:`HbosDetector` (#24)      — histogram-based outlier score (``pyod``),
    very fast multivariate-friendly density.
  * :class:`CopodDetector` (#25)     — copula-based tail probability (``pyod``).
  * :class:`EcodDetector` (#25)      — empirical-CDF tail probability (``pyod``);
    parameter-free, deterministic, per-dimension tail contributions = explanation.

The pyod-backed members import lazily and guarded; if ``pyod`` is unavailable
they degrade to a robust-z surrogate so the tier still runs. All are streaming:
they maintain a rolling window and refit the (cheap) batch model periodically.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from netra.contracts import DetectorFamily

from .base import Detector


class RobustZDetector(Detector):
    """Robust z-score on a rolling median + MAD (#19).

    Score = ``|x - median| / (1.4826·MAD)`` over the trailing window — the same
    median/MAD robustness Twitter's S-H-ESD uses. Fires when the robust z exceeds
    ``k`` (default 3.5). The cheapest meaningful detector on every SNMP counter.
    """

    method = "robust_z"
    family = DetectorFamily.STATISTICAL
    higher_is_anomalous = True

    def __init__(self, *args, window: int = 50, k: float = 3.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window = int(window)
        self.k = float(k)
        self._buf: deque[float] = deque(maxlen=self.window)

    def _fit(self, series: np.ndarray) -> None:
        for v in series[-self.window:]:
            self._buf.append(float(v))

    def _score_one(self, value: object) -> float:
        x = float(value)
        if len(self._buf) >= 5:
            arr = np.fromiter(self._buf, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            z = abs(x - med) / mad if mad > 1e-12 else (0.0 if x == med else 6.0)
        else:
            z = 0.0
        self._buf.append(x)
        return float(z)

    def _decide(self, raw: float, norm: float):
        return raw >= self.k, self.k


class EwmaControlChart(Detector):
    """EWMA control chart (#20) — detects small persistent mean shifts early.

    Textbook EWMA SPC: smooth the series ``z = λx + (1-λ)z`` toward an in-control
    *target* (the baseline mean) and flag when ``z`` leaves the control limits
    ``target ± L·σ·sqrt(λ/(2-λ))``, where ``σ`` is the **baseline** process std
    (estimated robustly from the warm-up / a rolling reference and held stable —
    crucially *not* instantaneously adapted, or a real shift would be normalised
    away). The reported raw score is the standardised EWMA deviation
    ``|z - target| / limit_sigma`` so it is comparable to ``L``. EWMA reacts to
    small persistent shifts faster than a Shewhart chart — ideal early warning for
    gradual congestion/loss creep.

    ``target`` adapts only slowly (via a long rolling baseline) so a sustained
    excursion keeps firing through the transition rather than being absorbed in
    one sample.
    """

    method = "ewma_control"
    family = DetectorFamily.STATISTICAL
    higher_is_anomalous = True

    def __init__(self, *args, lam: float = 0.25, L: float = 3.0,
                 baseline_window: int = 80, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lam = float(lam)
        self.L = float(L)
        self.baseline_window = int(baseline_window)
        self._z: float | None = None
        self._target: float | None = None
        self._sigma = 1.0
        self._baseline: deque[float] = deque(maxlen=self.baseline_window)
        self._n = 0

    def _fit(self, series: np.ndarray) -> None:
        for v in series:
            self._update_state(float(v))

    def _refresh_baseline(self) -> None:
        if len(self._baseline) >= 5:
            arr = np.fromiter(self._baseline, dtype=float)
            self._target = float(np.median(arr))
            mad = float(np.median(np.abs(arr - self._target))) * 1.4826
            self._sigma = mad if mad > 1e-9 else (float(np.std(arr)) or 1.0)

    def _update_state(self, x: float) -> float:
        if self._z is None:
            self._z = x
            self._target = x
            self._baseline.append(x)
            self._n = 1
            return 0.0
        # During warm-up (until the baseline buffer is reasonably full) always
        # learn the baseline so target/σ stabilise from real data; afterwards only
        # learn when in-control so a sustained shift keeps firing.
        warming = len(self._baseline) < min(self.baseline_window, 30)
        # refresh target/σ from the current baseline BEFORE scoring this point
        if self._n % 5 == 0 or warming:
            self._refresh_baseline()
        # EWMA statistic toward the in-control target
        self._z = self.lam * x + (1 - self.lam) * self._z
        limit_sigma = self._sigma * np.sqrt(self.lam / (2 - self.lam))
        if limit_sigma <= 0:
            limit_sigma = 1e-9
        target = self._target if self._target is not None else x
        stat = abs(self._z - target) / limit_sigma
        if warming or stat < self.L:
            self._baseline.append(x)
        self._n += 1
        return float(stat)

    def _score_one(self, value: object) -> float:
        return self._update_state(float(value))

    def _decide(self, raw: float, norm: float):
        # warm-up guard: don't fire until we have a baseline
        return (self._n > 8 and raw >= self.L), self.L


class _PyodUnivariateDetector(Detector):
    """Base for pyod batch detectors run in a streaming window (internal).

    Maintains a rolling window of recent values; periodically refits the wrapped
    pyod model and scores the latest point with ``decision_function``. Falls back
    to a robust-z surrogate when ``pyod`` is unavailable so the tier never breaks.
    Subclasses set :attr:`pyod_name` and :meth:`_make_model`.
    """

    higher_is_anomalous = True

    def __init__(self, *args, window: int = 120, refit_every: int = 20,
                 lags: int = 1, norm_threshold: float = 0.9, **kwargs) -> None:
        super().__init__(*args, norm_threshold=norm_threshold, **kwargs)
        self.window = int(window)
        self.refit_every = int(refit_every)
        self.lags = max(int(lags), 1)
        self._buf: deque[float] = deque(maxlen=self.window)
        self._model = None
        self._since_fit = 0
        self._fallback = False
        self._raw_cut: float | None = None
        self._last_raw: float = 0.0

    def _make_model(self):
        raise NotImplementedError

    def _embed(self, arr: np.ndarray) -> np.ndarray:
        """Optionally lag-embed a 1-D window into (n, lags) for multivariate feel."""
        if self.lags == 1:
            return arr.reshape(-1, 1)
        n = arr.size - self.lags + 1
        if n <= 0:
            return arr.reshape(-1, 1)
        return np.stack([arr[i:i + self.lags] for i in range(n)])

    def _refit(self) -> None:
        if len(self._buf) < max(10, self.lags + 5):
            return
        arr = np.fromiter(self._buf, dtype=float)
        X = self._embed(arr)
        try:
            model = self._make_model()
            model.fit(X)
            self._model = model
            self._fallback = False
            # remember the model's own high-quantile cutoff on the training pool,
            # used as a second gate so the decision isn't purely percentile-rank.
            try:
                scores = np.asarray(model.decision_scores_, dtype=float)
                med = float(np.median(scores))
                mad = float(np.median(np.abs(scores - med))) * 1.4826 or float(scores.std()) or 1.0
                # robust upper fence: max of the 99th pct and median+4·MAD
                self._raw_cut = float(max(np.quantile(scores, 0.99), med + 4.0 * mad))
            except Exception:
                self._raw_cut = None
        except Exception:
            self._model = None
            self._fallback = True

    def _fit(self, series: np.ndarray) -> None:
        for v in series[-self.window:]:
            self._buf.append(float(v))
        self._refit()

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._buf.append(x)
        self._since_fit += 1
        if self._model is None or self._since_fit >= self.refit_every:
            self._refit()
            self._since_fit = 0
        if self._model is not None and not self._fallback:
            arr = np.fromiter(self._buf, dtype=float)
            tail = arr[-self.lags:] if self.lags > 1 else arr[-1:]
            row = tail.reshape(1, -1)
            try:
                self._last_raw = float(self._model.decision_function(row)[0])
                return self._last_raw
            except Exception:
                pass
        # robust-z surrogate
        arr = np.fromiter(self._buf, dtype=float)
        if arr.size >= 5:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826
            self._last_raw = float(abs(x - med) / mad) if mad > 1e-12 else 0.0
        else:
            self._last_raw = 0.0
        return self._last_raw

    def _decide(self, raw: float, norm: float):
        """Fire only when BOTH the rolling rank is high AND the raw score clears
        the model's own contamination cutoff — cuts the percentile-rank's inherent
        ~10% false-positive floor on noisy flat data.
        """
        rank_hit = norm >= self.norm_threshold
        raw_hit = True if self._raw_cut is None else (raw >= self._raw_cut)
        return (rank_hit and raw_hit), (self._raw_cut if self._raw_cut is not None
                                        else self.norm_threshold)


class HbosDetector(_PyodUnivariateDetector):
    """Histogram-Based Outlier Score (#24) via ``pyod.models.hbos`` — very fast."""

    method = "hbos"
    family = DetectorFamily.STATISTICAL

    def _make_model(self):
        from pyod.models.hbos import HBOS

        return HBOS(contamination=0.1)


class CopodDetector(_PyodUnivariateDetector):
    """COPOD copula tail-probability detector (#25) via ``pyod.models.copod``."""

    method = "copod"
    family = DetectorFamily.STATISTICAL

    def _make_model(self):
        from pyod.models.copod import COPOD

        return COPOD(contamination=0.1)


class EcodDetector(_PyodUnivariateDetector):
    """ECOD empirical-CDF tail detector (#25) via ``pyod.models.ecod``.

    Parameter-free and deterministic — same telemetry yields identical scores
    across audits (valued by the offline-compliance dimension). Its per-dimension
    tail contributions double as a 'why' signal for the explain layer.
    """

    method = "ecod"
    family = DetectorFamily.STATISTICAL

    def _make_model(self):
        from pyod.models.ecod import ECOD

        return ECOD(contamination=0.1)


__all__ = [
    "RobustZDetector",
    "EwmaControlChart",
    "HbosDetector",
    "CopodDetector",
    "EcodDetector",
]
