"""Extreme-Value-Theory thresholding — POT / SPOT / DSPOT (#68).

Replaces hand-set thresholds (which the problem statement criticises as reactive)
with thresholds derived from **Extreme Value Theory**: fit a Generalized Pareto
Distribution (GPD) to the tail of excesses over a high quantile
(Pickands–Balkema–de Haan), then solve for the value whose exceedance probability
equals a target false-alarm risk ``q``. One tunable knob (``q``) controls the
false-positive rate fleet-wide (research 04 §11; Siffer et al., KDD'17).

Three variants, all pure-NumPy/SciPy (no heavy deps):

  * :class:`POT`   — offline/batch: fit on historical benign data, freeze a
    production threshold.
  * :class:`SPOT`  — streaming: keep the GPD tail updated online, adapt the
    threshold continuously as new normal data arrives.
  * :class:`DSPOT` — SPOT **with drift**: apply SPOT to residuals of a local
    moving average so the threshold tracks a non-stationary baseline (essential
    for diurnal traffic and slowly-drifting links).

Apply EVT to each metric's residual stream *and* to each detector's anomaly-score
stream, so even the fused ensemble score gets a principled, self-calibrating
cutoff. The GPD fit uses the method of moments by default (robust, closed-form),
optionally MLE via ``scipy.stats.genpareto``.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _gpd_fit_mom(excesses: np.ndarray) -> tuple[float, float]:
    """Method-of-moments GPD fit → ``(gamma, sigma)`` (shape, scale).

    Closed-form and robust for small tails: from the excess mean ``m`` and
    variance ``v``, ``gamma = 0.5*(1 - m²/v)``, ``sigma = 0.5*m*(m²/v + 1)``.
    Falls back to an exponential tail (``gamma=0``) when the variance is
    degenerate.
    """
    e = np.asarray(excesses, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 2:
        m = float(e.mean()) if e.size else 1.0
        return 0.0, max(m, 1e-6)
    m = float(e.mean())
    v = float(e.var(ddof=1))
    if v <= 1e-12 or m <= 0:
        return 0.0, max(m, 1e-6)
    ratio = (m * m) / v
    gamma = 0.5 * (1.0 - ratio)
    sigma = 0.5 * m * (ratio + 1.0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(m, 1e-6)
    # keep shape in a sane range for invertibility
    gamma = float(np.clip(gamma, -0.5, 0.9))
    return gamma, float(sigma)


def _gpd_quantile(t: float, gamma: float, sigma: float, q: float,
                  n: int, nt: int) -> float:
    """Threshold ``z`` with tail exceedance probability ≈ ``q`` (POT formula).

    ``z = t + (sigma/gamma) * ((q*n/nt)^(-gamma) - 1)`` for ``gamma != 0``; the
    ``gamma→0`` limit is the exponential ``z = t - sigma*ln(q*n/nt)``. ``t`` is the
    initial high quantile, ``n`` total samples, ``nt`` number of excesses.
    """
    if nt <= 0 or q <= 0:
        return t
    r = q * n / nt
    r = min(max(r, 1e-12), 1.0)
    if abs(gamma) < 1e-8:
        return float(t - sigma * np.log(r))
    return float(t + (sigma / gamma) * (r ** (-gamma) - 1.0))


class POT:
    """Peaks-Over-Threshold offline EVT threshold (batch).

    Fit on a benign reference array; :attr:`threshold` is then the value whose
    exceedance probability equals ``q`` under the fitted GPD tail. :meth:`detect`
    classifies new values against that frozen threshold.

    Parameters
    ----------
    q:
        Target tail probability (false-alarm risk), e.g. 1e-3.
    init_quantile:
        Quantile defining the start of the tail to fit the GPD on (e.g. 0.95).
    """

    def __init__(self, q: float = 1e-3, init_quantile: float = 0.95) -> None:
        self.q = float(q)
        self.init_quantile = float(init_quantile)
        self.t = 0.0
        self.threshold = np.inf
        self.gamma = 0.0
        self.sigma = 1.0
        self._fitted = False

    def fit(self, data: object) -> POT:
        arr = np.asarray(list(data), dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            # too little data: fall back to a simple mean+kσ threshold
            self.t = float(np.quantile(arr, self.init_quantile)) if arr.size else 0.0
            self.threshold = (float(arr.mean() + 4 * arr.std())
                              if arr.size else np.inf)
            self._fitted = True
            return self
        self.t = float(np.quantile(arr, self.init_quantile))
        excesses = arr[arr > self.t] - self.t
        if excesses.size < 2:
            self.threshold = float(arr.max())
            self._fitted = True
            return self
        self.gamma, self.sigma = _gpd_fit_mom(excesses)
        self.threshold = _gpd_quantile(self.t, self.gamma, self.sigma, self.q,
                                       n=arr.size, nt=excesses.size)
        self._fitted = True
        return self

    def detect(self, value: float) -> bool:
        """True if ``value`` exceeds the EVT threshold (an anomaly)."""
        return bool(float(value) > self.threshold)


class SPOT:
    """Streaming Peaks-Over-Threshold — online, self-calibrating threshold.

    Initialise on a benign warm-up window (sets the initial tail), then feed live
    values via :meth:`step`. Non-anomalous values that still exceed the tail level
    ``t`` are absorbed into the running excess pool and the threshold is recomputed,
    so the cutoff adapts to the stream without a hand-set value. Anomalies (values
    above the current ``threshold``) are reported and *not* learned (so a real
    spike doesn't inflate the normal tail).
    """

    def __init__(self, q: float = 1e-3, init_quantile: float = 0.95,
                 max_excess: int = 400) -> None:
        self.q = float(q)
        self.init_quantile = float(init_quantile)
        self.max_excess = int(max_excess)
        self.t = 0.0
        self.threshold = np.inf
        self.gamma = 0.0
        self.sigma = 1.0
        self.n = 0
        self._excess: deque[float] = deque(maxlen=self.max_excess)
        self._initialized = False

    def initialize(self, warmup: object) -> SPOT:
        """Seed the tail from a benign warm-up window."""
        arr = np.asarray(list(warmup), dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            self.t = float(np.quantile(arr, self.init_quantile)) if arr.size else 0.0
            self.threshold = (float(arr.mean() + 4 * arr.std())
                              if arr.size else np.inf)
            self.n = int(arr.size)
            self._initialized = True
            return self
        self.t = float(np.quantile(arr, self.init_quantile))
        exc = arr[arr > self.t] - self.t
        for e in exc:
            self._excess.append(float(e))
        self.n = int(arr.size)
        self._recompute()
        self._initialized = True
        return self

    def _recompute(self) -> None:
        if len(self._excess) >= 2:
            self.gamma, self.sigma = _gpd_fit_mom(np.fromiter(self._excess, dtype=float))
            self.threshold = _gpd_quantile(self.t, self.gamma, self.sigma, self.q,
                                           n=max(self.n, 1), nt=len(self._excess))
        elif self.n:
            self.threshold = self.t

    def step(self, value: float) -> bool:
        """Process one streamed value; return whether it is an EVT anomaly."""
        x = float(value)
        self.n += 1
        if not self._initialized:
            # lazy init on the first value alone -> permissive until warmed
            self.t = x
            self.threshold = np.inf
            self._initialized = True
            return False
        if x > self.threshold:
            return True                      # anomaly: do NOT learn it
        if x > self.t:
            self._excess.append(x - self.t)  # normal exceedance -> update tail
            self._recompute()
        return False


class DSPOT:
    """Drift SPOT — SPOT on residuals of a local moving average (#68).

    Tracks a moving average ``mu`` of the last ``depth`` values and runs
    :class:`SPOT` on the *residuals* ``x - mu``, so the threshold follows a drifting
    baseline (diurnal traffic, slowly-saturating links) instead of firing on the
    drift itself. This is the recommended EVT variant for NETRA's non-stationary
    metrics and is what the fusion layer uses on residual / ensemble-score streams.

    Parameters
    ----------
    q, init_quantile:
        Passed through to the inner SPOT.
    depth:
        Window length of the drift-removing moving average.
    """

    def __init__(self, q: float = 1e-3, init_quantile: float = 0.95,
                 depth: int = 20) -> None:
        self.depth = int(depth)
        self._spot = SPOT(q=q, init_quantile=init_quantile)
        self._window: deque[float] = deque(maxlen=self.depth)
        self._initialized = False

    @property
    def threshold(self) -> float:
        """Current EVT threshold on the *residual* scale (add ``mu`` for raw)."""
        return self._spot.threshold

    def _mu(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.0

    def initialize(self, warmup: object) -> DSPOT:
        """Seed the drift window + the inner SPOT on benign residuals."""
        arr = np.asarray(list(warmup), dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            self._initialized = True
            return self
        # build residuals against a trailing moving average
        resid = []
        win: deque[float] = deque(maxlen=self.depth)
        for x in arr:
            mu = float(np.mean(win)) if win else x
            resid.append(x - mu)
            win.append(x)
        self._window = win
        self._spot.initialize(resid)
        self._initialized = True
        return self

    def step(self, value: float) -> bool:
        """Process one streamed value against the drift-adjusted EVT threshold."""
        x = float(value)
        mu = self._mu() if self._window else x
        resid = x - mu
        is_anom = self._spot.step(resid)
        if not is_anom:
            self._window.append(x)     # only update baseline on non-anomalies
        return is_anom

    def current_raw_threshold(self) -> float:
        """The EVT threshold expressed on the raw value scale (``mu + thr``)."""
        thr = self._spot.threshold
        return float(self._mu() + thr) if np.isfinite(thr) else np.inf


__all__ = ["POT", "SPOT", "DSPOT"]
