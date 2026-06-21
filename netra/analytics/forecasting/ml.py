"""Gradient-boosted lag-feature forecaster (global ML tier).

The accuracy workhorse of research 03 §D: build lag + rolling-statistic +
calendar-ish features from the history and learn a regressor that maps them to
the next value, then roll the prediction forward recursively to a multi-step
trajectory. Quantile bounds come from two extra quantile regressors (p10/p90)
when the backend supports a pinball/quantile objective, else from a
horizon-growing residual band.

**Optional heavy backend, graceful fallback (a hard requirement of this build):**

  1. **LightGBM** (``lightgbm.LGBMRegressor``) — preferred; native quantile loss.
  2. **sklearn ``HistGradientBoostingRegressor``** — quantile loss too; ships in
     the light tier (scikit-learn), so this is the realistic CPU default.
  3. **sklearn ``RandomForestRegressor``** — final fallback; bounds from the
     spread of per-tree predictions.

All three are imported lazily under ``try/except``; if *none* is importable the
member raises at ``fit`` time and the ensemble simply drops it (it never breaks
the always-on classical tier).
"""

from __future__ import annotations

import numpy as np

from netra.contracts import DetectorFamily

from .base import Forecaster, residual_std, z_for_quantile


def _make_lag_features(series: np.ndarray, n_lags: int, roll: int):
    """Build a supervised (X, y) table predicting series[t] from its past.

    Features per row t: the last ``n_lags`` values, plus rolling mean/std/min/max
    and the local slope over the trailing ``roll`` window — the standard global
    gradient-boosting recipe. Returns ``(X, y)`` with ``X`` shaped
    ``(n_rows, n_features)``.
    """
    n = series.size
    start = max(n_lags, roll)
    rows_X, rows_y = [], []
    for t in range(start, n):
        lags = series[t - n_lags:t][::-1]                      # most-recent first
        window = series[t - roll:t]
        feats = np.concatenate([
            lags,
            [window.mean(), window.std(), window.min(), window.max(),
             float(window[-1] - window[0]) / max(roll - 1, 1)],
        ])
        rows_X.append(feats)
        rows_y.append(series[t])
    if not rows_X:
        return np.empty((0, n_lags + 5)), np.empty((0,))
    return np.asarray(rows_X, dtype=float), np.asarray(rows_y, dtype=float)


def _featurize_tail(series: np.ndarray, n_lags: int, roll: int) -> np.ndarray:
    """Feature row for forecasting one step past the end of ``series``."""
    lags = series[-n_lags:][::-1]
    window = series[-roll:]
    return np.concatenate([
        lags,
        [window.mean(), window.std(), window.min(), window.max(),
         float(window[-1] - window[0]) / max(roll - 1, 1)],
    ]).reshape(1, -1)


class GradientBoostedForecaster(Forecaster):
    """Recursive multi-step forecaster on lag/rolling features.

    Trains a point regressor (LightGBM → HistGBR → RandomForest) and, where the
    backend supports it, two quantile regressors for the band. Forecasts are
    produced by recursively appending each predicted value and re-featurising —
    the standard global-model rollout. With very short histories it falls back to
    a Holt-style extrapolation (handled by the ensemble, which only includes this
    member when enough history exists).
    """

    method = "gbm_lag"
    family = DetectorFamily.FORECAST

    def __init__(self, *args, n_lags: int = 6, roll: int = 6,
                 quantile_band: bool = True, random_state: int = 1337,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_lags = int(n_lags)
        self.roll = int(roll)
        self.quantile_band = bool(quantile_band)
        self.random_state = int(random_state)
        self.min_history = max(self.n_lags, self.roll) + 4
        self._backend = "none"
        self._point = None
        self._q_lo = None
        self._q_hi = None
        self._sigma = 1.0

    # -- backend construction ----------------------------------------------

    def _build_models(self):
        """Return ``(point_model, lo_model, hi_model, backend_name)``.

        ``lo_model``/``hi_model`` may be ``None`` when the backend can't do
        quantiles; in that case a residual band is synthesised at predict time.
        """
        ql, qu = self.quantile_lower, self.quantile_upper
        # 1) LightGBM
        try:
            from lightgbm import LGBMRegressor

            common = dict(n_estimators=200, num_leaves=15, min_child_samples=5,
                          learning_rate=0.05, random_state=self.random_state,
                          verbose=-1)
            point = LGBMRegressor(objective="regression", **common)
            lo = hi = None
            if self.quantile_band:
                lo = LGBMRegressor(objective="quantile", alpha=ql, **common)
                hi = LGBMRegressor(objective="quantile", alpha=qu, **common)
            return point, lo, hi, "lightgbm"
        except Exception:
            pass
        # 2) sklearn HistGradientBoostingRegressor (quantile loss supported)
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor

            common = dict(max_iter=200, max_depth=4, learning_rate=0.06,
                          min_samples_leaf=5, random_state=self.random_state)
            point = HistGradientBoostingRegressor(loss="squared_error", **common)
            lo = hi = None
            if self.quantile_band:
                lo = HistGradientBoostingRegressor(loss="quantile", quantile=ql, **common)
                hi = HistGradientBoostingRegressor(loss="quantile", quantile=qu, **common)
            return point, lo, hi, "hist_gbr"
        except Exception:
            pass
        # 3) RandomForest (bounds from tree spread)
        try:
            from sklearn.ensemble import RandomForestRegressor

            point = RandomForestRegressor(n_estimators=200, max_depth=8,
                                          min_samples_leaf=3,
                                          random_state=self.random_state, n_jobs=1)
            return point, None, None, "random_forest"
        except Exception:
            pass
        return None, None, None, "none"

    def _fit(self, series: np.ndarray) -> None:
        self._sigma = residual_std(series)
        X, y = _make_lag_features(series, self.n_lags, self.roll)
        point, lo, hi, backend = self._build_models()
        if backend == "none" or X.shape[0] < 4:
            raise RuntimeError("GradientBoostedForecaster: no usable backend/data")
        point.fit(X, y)
        if lo is not None and hi is not None:
            try:
                lo.fit(X, y)
                hi.fit(X, y)
            except Exception:
                lo = hi = None
        self._point, self._q_lo, self._q_hi, self._backend = point, lo, hi, backend
        # method id reflects the backend actually used, for honest provenance.
        self.method = f"gbm_lag_{backend}"

    def _predict(self, steps: int):
        assert self._history is not None and self._point is not None
        series = self._history.astype(float).copy()
        point_out = np.empty(steps)
        lo_out = np.empty(steps)
        hi_out = np.empty(steps)
        have_q = self._q_lo is not None and self._q_hi is not None
        rf_spread = None
        for i in range(steps):
            feat = _featurize_tail(series, self.n_lags, self.roll)
            yhat = float(self._point.predict(feat)[0])
            point_out[i] = yhat
            if have_q:
                lo_out[i] = float(self._q_lo.predict(feat)[0])
                hi_out[i] = float(self._q_hi.predict(feat)[0])
            elif self._backend == "random_forest":
                # spread of per-tree predictions -> a data-driven band
                try:
                    preds = np.array([est.predict(feat)[0]
                                      for est in self._point.estimators_])
                    rf_spread = float(preds.std())
                except Exception:
                    rf_spread = self._sigma
                z = z_for_quantile(self.quantile_upper)
                grow = np.sqrt(i + 1)
                lo_out[i] = yhat - z * max(rf_spread, self._sigma) * grow
                hi_out[i] = yhat + z * max(rf_spread, self._sigma) * grow
            series = np.append(series, yhat)
        if not have_q and self._backend != "random_forest":
            lo_out, hi_out = self._symmetric_band(point_out, self._sigma, grow=1.0)
        # enforce ordering lo<=point<=hi (quantile crossing can happen)
        lo_out = np.minimum(lo_out, point_out)
        hi_out = np.maximum(hi_out, point_out)
        return point_out, lo_out, hi_out


__all__ = ["GradientBoostedForecaster"]
