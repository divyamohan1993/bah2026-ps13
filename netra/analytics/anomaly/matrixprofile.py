"""Matrix-profile discord detector (Family 5, #44).

For each subsequence of length ``m``, the matrix profile is the distance to its
nearest neighbour elsewhere in the series; **maxima are discords** (anomalous
shapes). ``stumpy.stumpi`` ("STUMP incremental") updates the profile in
~O(1)-amortised per new sample — purpose-built for continuously-arriving
telemetry — so this is the streaming *shape*-anomaly member, catching subtle
intermittent waveforms (e.g. scenario C tunnel degradation) that single-point
detectors miss.

Caveat (research 04 §14): matrix profile is for subsequence discords, not single
spikes — so it is deployed *alongside* the point detectors (#19/#26/#59), never
alone. Degrades to a rolling-window shape-distance surrogate if ``stumpy`` is
unavailable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from netra.contracts import DetectorFamily

from .base import Detector


class MatrixProfileDiscordDetector(Detector):
    """Streaming matrix-profile discord detector via ``stumpy.stumpi`` (#44).

    Maintains an incremental matrix profile over the stream; the live anomaly
    score is the newest subsequence's profile value (distance to its closest
    match) — large when the recent shape is unlike anything seen before. Needs a
    warm-up of at least ``m`` points before it produces non-trivial scores.

    Parameters
    ----------
    m:
        Subsequence (window) length in samples — the shape granularity.
    warmup:
        Minimum points to seed ``stumpi`` (defaults to ``4*m``).
    """

    method = "matrix_profile"
    family = DetectorFamily.MATRIX_PROFILE
    higher_is_anomalous = True

    def __init__(self, *args, m: int = 12, warmup: int | None = None,
                 surrogate_window: int = 200, k: float = 3.0,
                 norm_threshold: float = 0.9, **kwargs) -> None:
        super().__init__(*args, norm_threshold=norm_threshold, **kwargs)
        self.m = int(m)
        self.k = float(k)
        self.warmup = int(warmup) if warmup is not None else 4 * self.m
        self._stream = None          # stumpy.stumpi object
        self._seed: list[float] = []
        self._fallback = False
        self._buf: deque[float] = deque(maxlen=int(surrogate_window))
        # frozen reference distribution of discord scores from benign warm-up,
        # used for a robust-z decision (discord scores on noisy flat data are
        # naturally jumpy, so a rolling percentile rank over-fires).
        self._ref_scores: deque[float] = deque(maxlen=int(surrogate_window))
        self._ref_med: float | None = None
        self._ref_mad: float = 1.0
        self._last_raw: float = 0.0

    def _fit(self, series: np.ndarray) -> None:
        for v in series:
            raw = self._score_one(float(v))
            self._ref_scores.append(raw)
        self._freeze_reference()

    def _freeze_reference(self) -> None:
        if len(self._ref_scores) >= 8:
            arr = np.fromiter(self._ref_scores, dtype=float)
            self._ref_med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - self._ref_med))) * 1.4826
            self._ref_mad = mad if mad > 1e-9 else (float(arr.std()) or 1.0)

    def _start_stream(self) -> None:
        try:
            import stumpy

            seed = np.asarray(self._seed, dtype=float)
            self._stream = stumpy.stumpi(seed, m=self.m, egress=False)
        except Exception:
            self._stream = None
            self._fallback = True

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._buf.append(x)
        if self._fallback:
            self._last_raw = self._surrogate(x)
            return self._last_raw
        if self._stream is None:
            self._seed.append(x)
            if len(self._seed) >= max(self.warmup, self.m + 2):
                self._start_stream()
            self._last_raw = 0.0
            return 0.0
        try:
            self._stream.update(x)
            P = np.asarray(self._stream.P_, dtype=float)
            val = float(P[-1]) if P.size else 0.0
            self._last_raw = val if np.isfinite(val) else 0.0
            return self._last_raw
        except Exception:
            self._fallback = True
            self._last_raw = self._surrogate(x)
            return self._last_raw

    def _decide(self, raw: float, norm: float):
        """Robust-z of the discord score against the frozen benign reference.

        Fires when the current discord is ``k`` robust-sigmas above the warm-up
        discord level — a stable cutoff that beats a rolling percentile rank on
        the naturally-jumpy matrix-profile score.
        """
        if self._ref_med is None:
            return norm >= self.norm_threshold, self.norm_threshold
        z = (raw - self._ref_med) / self._ref_mad
        return z >= self.k, float(self._ref_med + self.k * self._ref_mad)

    def _surrogate(self, x: float) -> float:
        """Rolling shape-distance surrogate when stumpy is unavailable.

        Compares the latest length-``m`` window to all earlier windows in the
        buffer and returns the minimum Euclidean distance (z-normalised) — a crude
        but serviceable discord score.
        """
        arr = np.fromiter(self._buf, dtype=float)
        if arr.size < 2 * self.m:
            return 0.0
        m = self.m
        last = arr[-m:]
        last = (last - last.mean()) / (last.std() + 1e-9)
        best = np.inf
        for i in range(0, arr.size - m - m + 1):
            w = arr[i:i + m]
            w = (w - w.mean()) / (w.std() + 1e-9)
            d = float(np.linalg.norm(last - w))
            if d < best:
                best = d
        return 0.0 if not np.isfinite(best) else best


__all__ = ["MatrixProfileDiscordDetector"]
