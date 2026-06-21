"""Change-point / drift detectors (Family 4, #37-#43).

These target *regime change* — the step shifts of scenario D (controller
misconfig → policy drift) and sudden BGP/OSPF regime changes — and mark the
boundaries fusion uses to reset baselines and the copilot uses to say "what
changed when". Most are streaming O(1):

  * :class:`PageHinkleyDetector` (#38) — sequential CUSUM variant for online
    abrupt mean-change (``river.drift.PageHinkley``).
  * :class:`AdwinDetector` (#39)       — ADaptive WINdowing; cuts the window when
    old vs new sub-window statistics diverge, with FP bounds
    (``river.drift.ADWIN``). Detects a *distribution* shift, threshold-free.
  * :class:`KswinDetector` (#40)       — Kolmogorov-Smirnov windowing; non-
    parametric shape-change detection (``river.drift.KSWIN``).
  * :class:`RupturesChangePointDetector` (#43) — offline PELT/BinSeg/Window over a
    rolling buffer (``ruptures``); retrospective segmentation answering *when
    exactly* the regime shifted.

The river members emit a discrete change event; we convert that to an
:class:`~netra.contracts.AnomalyScore` whose ``normalized_score`` pulses to ~1 at
the change and decays, so fusion treats a change-point firing as a strong, decaying
vote. All backends are guarded; on absence the detector reports a quiet stream
rather than failing.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from netra.contracts import DetectorFamily, EntityRef

from .base import Detector


class _RiverDriftDetector(Detector):
    """Base wrapper turning a river drift detector into AnomalyScores.

    A river drift detector exposes ``update(x)`` and a ``drift_detected`` flag.
    On a change we set the raw score to 1.0 and let it decay geometrically over
    subsequent samples (so the 'regime just changed' vote persists briefly), which
    matches how fusion treats a change-point as a decaying vote.
    """

    family = DetectorFamily.CHANGE_POINT
    higher_is_anomalous = True
    decay: float = 0.6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._impl = None
        self._fallback = False
        self._pulse = 0.0
        self._last_changed = False
        self._n = 0

    def _make_impl(self):
        raise NotImplementedError

    def _ensure(self) -> None:
        if self._impl is not None or self._fallback:
            return
        try:
            self._impl = self._make_impl()
        except Exception:
            self._impl = None
            self._fallback = True

    def _fit(self, series: np.ndarray) -> None:
        self._ensure()
        for v in series:
            self._score_one(float(v))
        # warming the detector on benign data is fine; reset the pulse afterward
        self._pulse = 0.0
        self._last_changed = False

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._n += 1
        self._ensure()
        changed = False
        if self._impl is not None and not self._fallback:
            try:
                self._impl.update(x)
                changed = bool(getattr(self._impl, "drift_detected", False))
            except Exception:
                self._fallback = True
        self._last_changed = changed
        if changed:
            self._pulse = 1.0
        else:
            self._pulse *= self.decay
        return float(self._pulse)

    def _decide(self, raw: float, norm: float):
        # The change event itself is the decision; a fresh/strong pulse = anomaly.
        return (self._last_changed or raw >= 0.5), 0.5


class PageHinkleyDetector(_RiverDriftDetector):
    """Page-Hinkley online change-point detector (#38)."""

    method = "page_hinkley"

    def __init__(self, *args, delta: float = 0.005, threshold: float = 50.0,
                 min_instances: int = 30, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.min_instances = int(min_instances)

    def _make_impl(self):
        from river.drift import PageHinkley

        return PageHinkley(min_instances=self.min_instances,
                           delta=self.delta, threshold=self.threshold)


class AdwinDetector(_RiverDriftDetector):
    """ADWIN adaptive-windowing distribution-drift detector (#39)."""

    method = "adwin"

    def __init__(self, *args, delta: float = 0.002, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.delta = float(delta)

    def _make_impl(self):
        from river.drift import ADWIN

        return ADWIN(delta=self.delta)


class KswinDetector(_RiverDriftDetector):
    """KSWIN Kolmogorov-Smirnov windowing drift detector (#40)."""

    method = "kswin"

    def __init__(self, *args, alpha: float = 0.005, window_size: int = 100,
                 stat_size: int = 30, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.stat_size = int(stat_size)

    def _make_impl(self):
        from river.drift import KSWIN

        return KSWIN(alpha=self.alpha, window_size=self.window_size,
                     stat_size=self.stat_size, seed=1337)


class RupturesChangePointDetector(Detector):
    """Offline PELT/BinSeg/Window change-point over a rolling buffer (#43).

    Runs an exact/approximate penalised segmentation (``ruptures``) on the trailing
    window every ``refit_every`` samples; the score pulses to 1.0 when a new change
    point appears near the end of the buffer (i.e. the regime just shifted) and
    decays otherwise. ``algo`` selects ``"pelt"`` (exact, near-linear, default),
    ``"binseg"`` or ``"window"``. Degrades to a CUSUM-style surrogate if ruptures
    is unavailable.
    """

    method = "ruptures_pelt"
    family = DetectorFamily.CHANGE_POINT
    higher_is_anomalous = True

    def __init__(self, *args, algo: str = "pelt", model: str = "l2",
                 penalty: float = 8.0, window: int = 120, refit_every: int = 15,
                 recent_frac: float = 0.2, decay: float = 0.6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.algo = str(algo).lower()
        self.model = str(model)
        self.penalty = float(penalty)
        self.window = int(window)
        self.refit_every = int(refit_every)
        self.recent_frac = float(recent_frac)
        self.decay = float(decay)
        self._buf: deque[float] = deque(maxlen=self.window)
        self._since = 0
        self._pulse = 0.0
        self._last_changed = False
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._cusum_mean: float | None = None

    def _fit(self, series: np.ndarray) -> None:
        for v in series[-self.window:]:
            self._buf.append(float(v))

    def _detect(self) -> bool:
        if len(self._buf) < 20:
            return False
        arr = np.fromiter(self._buf, dtype=float)
        try:
            import ruptures as rpt

            if self.algo == "binseg":
                algo = rpt.Binseg(model=self.model).fit(arr)
            elif self.algo == "window":
                algo = rpt.Window(width=max(10, len(arr) // 6), model=self.model).fit(arr)
            else:
                algo = rpt.Pelt(model=self.model, min_size=5).fit(arr)
            bkps = algo.predict(pen=self.penalty)
            bkps = [b for b in bkps if b < len(arr)]    # drop the sentinel end index
            if not bkps:
                return False
            recent_cut = len(arr) * (1.0 - self.recent_frac)
            return any(b >= recent_cut for b in bkps)
        except Exception:
            # CUSUM surrogate on the buffer
            return self._cusum_recent(arr)

    def _cusum_recent(self, arr: np.ndarray) -> bool:
        mean = float(np.mean(arr[:-1])) if arr.size > 1 else float(arr[-1])
        sd = float(np.std(arr[:-1])) or 1.0
        x = float(arr[-1])
        k = 0.5 * sd
        self._cusum_pos = max(0.0, self._cusum_pos + (x - mean) - k)
        self._cusum_neg = max(0.0, self._cusum_neg - (x - mean) - k)
        h = 4.0 * sd
        if self._cusum_pos > h or self._cusum_neg > h:
            self._cusum_pos = self._cusum_neg = 0.0
            return True
        return False

    def _score_one(self, value: object) -> float:
        self._buf.append(float(value))
        self._since += 1
        changed = False
        if self._since >= self.refit_every:
            changed = self._detect()
            self._since = 0
        self._last_changed = changed
        if changed:
            self._pulse = 1.0
        else:
            self._pulse *= self.decay
        return float(self._pulse)

    def _decide(self, raw: float, norm: float):
        return (self._last_changed or raw >= 0.5), 0.5


__all__ = [
    "PageHinkleyDetector",
    "AdwinDetector",
    "KswinDetector",
    "RupturesChangePointDetector",
]
