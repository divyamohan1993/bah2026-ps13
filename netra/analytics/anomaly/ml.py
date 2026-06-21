"""ML unsupervised anomaly detectors (Family 2/streaming, #26, #29, #30, #60).

The batch-ML / streaming-ensemble confirmation tier (research 04 §3, §5, §7):

  * :class:`HalfSpaceTreesDetector` (#26) — the reference *online* detector;
    constant time & memory over a sliding window (``river.anomaly.HalfSpaceTrees``).
    Top pick for the always-on streaming tier.
  * :class:`IsolationForestDetector` (#30) — the popular general detector; short
    isolation path = anomaly (``sklearn.ensemble.IsolationForest`` or
    ``pyod.models.iforest``). Backbone batch detector + TreeSHAP-friendly.
  * :class:`LofDetector` (#29) — Local Outlier Factor density-ratio
    (``sklearn.neighbors.LocalOutlierFactor``); good when outliers sit in
    locally-sparse regions.
  * :class:`ForecastResidualDetector` (#60) — predict-then-flag: feed it the
    forecaster's prediction and the actual, score the residual robustly. The
    canonical *precursor* mechanism — it fires while the metric is still
    *trending* toward breach, maximising lead time, and is the cleanest bridge to
    the forecasting half.

ML members lag-embed the univariate stream so they behave multivariate-ish, and
periodically refit. Backends are imported lazily/guarded; HST falls back to a
robust streaming z-score, the tree/LOF members to robust-z, so the tier always
produces scores.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from netra.contracts import DetectorFamily

from .base import Detector
from .statistical import _PyodUnivariateDetector


class HalfSpaceTreesDetector(Detector):
    """Half-Space Trees streaming anomaly detector (#26).

    An ensemble of random half-space trees with mass profiles over a sliding
    window — constant per-point time and memory, designed for evolving streams.
    Inputs are min-max scaled into [0,1] (HST expects bounded features) using a
    rolling range. Uses ``river.anomaly.HalfSpaceTrees``; falls back to a robust
    streaming z-score if river is unavailable.
    """

    method = "half_space_trees"
    family = DetectorFamily.ML_UNSUPERVISED
    higher_is_anomalous = True

    def __init__(self, *args, n_trees: int = 25, height: int = 8,
                 window_size: int = 100, seed: int = 1337,
                 scale_window: int = 200, norm_threshold: float = 0.85,
                 **kwargs) -> None:
        super().__init__(*args, norm_threshold=norm_threshold, **kwargs)
        self.n_trees = int(n_trees)
        self.height = int(height)
        self.window_size = int(window_size)
        self.seed = int(seed)
        self._model = None
        self._scale_buf: deque[float] = deque(maxlen=int(scale_window))
        self._scale_lo: float | None = None   # frozen baseline range from warm-up
        self._scale_hi: float | None = None
        self._fallback = False
        # frozen reference of HST scores on benign warm-up -> robust-z decision
        # (HST mass scores on low-dim data don't separate well under a rolling
        # percentile rank, so we z-score against the warm-up score level).
        self._ref_med: float | None = None
        self._ref_mad: float = 1.0
        self._ref_scores: deque[float] = deque(maxlen=int(scale_window))
        self.k = 3.5

    def _ensure_model(self) -> None:
        if self._model is not None or self._fallback:
            return
        try:
            from river.anomaly import HalfSpaceTrees

            self._model = HalfSpaceTrees(
                n_trees=self.n_trees, height=self.height,
                window_size=self.window_size, seed=self.seed,
                limits={"x": (0.0, 1.0)},
            )
        except Exception:
            self._model = None
            self._fallback = True

    def _scaled(self, x: float) -> float:
        """Min-max scale into [0,1] using a baseline range frozen at warm-up.

        Scaling against a *frozen* benign range (rather than a range that absorbs
        the shift) is what lets a persistent step actually push the scaled value
        toward the [0,1] extreme so HST scores it as anomalous, instead of the
        rolling range silently re-centring it.
        """
        self._scale_buf.append(x)
        if self._scale_lo is None or self._scale_hi is None:
            arr = np.fromiter(self._scale_buf, dtype=float)
            lo, hi = float(arr.min()), float(arr.max())
        else:
            lo, hi = self._scale_lo, self._scale_hi
        span = hi - lo
        if span < 1e-12:
            span = abs(hi) * 0.1 + 1e-6
        # allow the scaled value to exceed [0,1] a little so excursions register,
        # then clip to HST's declared limit.
        return float(np.clip((x - lo) / span, 0.0, 1.0))

    def _freeze_scale(self) -> None:
        if len(self._scale_buf) >= 5:
            arr = np.fromiter(self._scale_buf, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826 or float(arr.std()) or 1.0
            # robust baseline band ~ median ± 4·MAD (so normal noise maps inside)
            self._scale_lo = med - 4 * mad
            self._scale_hi = med + 4 * mad

    def _fit(self, series: np.ndarray) -> None:
        self._ensure_model()
        # First pass: learn the trees AND establish the frozen scale on benign
        # data; we then re-score the warm-up to capture the reference score level.
        for v in series:
            self._score_one(float(v))
        self._freeze_scale()
        for v in series:
            self._ref_scores.append(self._score_one(float(v)))
        if len(self._ref_scores) >= 8:
            arr = np.fromiter(self._ref_scores, dtype=float)
            self._ref_med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - self._ref_med))) * 1.4826
            self._ref_mad = mad if mad > 1e-9 else (float(arr.std()) or 1e-3)

    def _decide(self, raw: float, norm: float):
        """Robust-z of the HST score vs the frozen benign score level."""
        if self._ref_med is None:
            return norm >= self.norm_threshold, self.norm_threshold
        z = (raw - self._ref_med) / self._ref_mad
        return z >= self.k, float(self._ref_med + self.k * self._ref_mad)

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._ensure_model()
        xs = self._scaled(x)
        if self._model is not None and not self._fallback:
            try:
                score = self._model.score_one({"x": xs})
                self._model.learn_one({"x": xs})
                return float(score)
            except Exception:
                self._fallback = True
        # fallback: robust streaming z on the scaled value
        arr = np.fromiter(self._scale_buf, dtype=float)
        if arr.size >= 5:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826
            return float(abs(xs - med) / mad) if mad > 1e-12 else 0.0
        return 0.0


class IsolationForestDetector(_PyodUnivariateDetector):
    """Isolation Forest (#30) — random partitioning; short path = anomaly.

    Uses ``sklearn.ensemble.IsolationForest`` (preferred — TreeSHAP-explainable)
    via the rolling-window refit machinery, lag-embedding the stream so isolation
    sees local shape, not just level. Inherits the robust-z fallback.
    """

    method = "isolation_forest"
    family = DetectorFamily.ML_UNSUPERVISED

    def __init__(self, *args, lags: int = 4, random_state: int = 1337, **kwargs) -> None:
        super().__init__(*args, lags=lags, **kwargs)
        self.random_state = int(random_state)

    def _make_model(self):
        from sklearn.ensemble import IsolationForest

        # Wrap so decision_function polarity matches "higher = more anomalous".
        return _SklearnScorer(
            IsolationForest(n_estimators=150, random_state=self.random_state),
            invert=True,   # sklearn: higher score_samples = MORE normal
        )


class LofDetector(_PyodUnivariateDetector):
    """Local Outlier Factor (#29) via ``sklearn.neighbors.LocalOutlierFactor``."""

    method = "lof"
    family = DetectorFamily.ML_UNSUPERVISED

    def __init__(self, *args, lags: int = 4, n_neighbors: int = 20, **kwargs) -> None:
        super().__init__(*args, lags=lags, **kwargs)
        self.n_neighbors = int(n_neighbors)

    def _make_model(self):
        from sklearn.neighbors import LocalOutlierFactor

        n = max(5, min(self.n_neighbors, len(self._buf) - 1))
        return _SklearnScorer(
            LocalOutlierFactor(n_neighbors=n, novelty=True),
            invert=True,
        )


class _SklearnScorer:
    """Adapt an sklearn outlier estimator to a pyod-like ``decision_function``.

    sklearn's ``score_samples``/``decision_function`` are higher-for-normal; we
    negate so the convention matches "higher = more anomalous" used everywhere in
    NETRA. Supports IsolationForest and novelty-mode LocalOutlierFactor.
    """

    def __init__(self, est, invert: bool = True) -> None:
        self.est = est
        self.invert = invert

    def fit(self, X):
        self.est.fit(X)
        return self

    def decision_function(self, X):
        if hasattr(self.est, "score_samples"):
            s = self.est.score_samples(X)
        else:
            s = self.est.decision_function(X)
        return -np.asarray(s) if self.invert else np.asarray(s)


class ForecastResidualDetector(Detector):
    """Forecast-residual / predict-then-flag detector (#60).

    The cleanest tie-in to the forecasting half and the primary congestion
    precursor: given the forecaster's prediction and the realised value, score
    ``actual - predicted`` robustly (rolling MAD of residuals). It fires while the
    metric is still *trending* toward breach (the residual grows before the level
    does), so it maximises lead time. The residual magnitude is also a natural
    severity score.

    Call :meth:`update_residual(actual, predicted)` (or feed a tuple to
    :meth:`update`) at each step.
    """

    method = "forecast_residual"
    family = DetectorFamily.FORECAST_RESIDUAL
    higher_is_anomalous = True

    def __init__(self, *args, window: int = 80, k: float = 3.5,
                 signed: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.window = int(window)
        self.k = float(k)
        self.signed = bool(signed)
        self._res: deque[float] = deque(maxlen=self.window)

    def update_residual(self, actual: float, predicted: float,
                        timestamp=None):
        """Score one (actual, predicted) pair → :class:`AnomalyScore`."""
        return self.update((float(actual), float(predicted)), timestamp=timestamp)

    def _score_one(self, value: object) -> float:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            actual, predicted = float(value[0]), float(value[1])
            resid = actual - predicted
        else:
            # if a bare value is passed, treat it as the residual directly
            resid = float(value)
        # robust z of the residual against recent residual scale
        if len(self._res) >= 5:
            arr = np.fromiter(self._res, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            scale = mad if mad > 1e-12 else (float(np.std(arr)) or 1.0)
            z = (resid - med) / scale
        else:
            z = 0.0
        self._res.append(resid)
        return float(z if self.signed else abs(z))

    def _decide(self, raw: float, norm: float):
        mag = abs(raw)
        return mag >= self.k, self.k


__all__ = [
    "HalfSpaceTreesDetector",
    "IsolationForestDetector",
    "LofDetector",
    "ForecastResidualDetector",
]
