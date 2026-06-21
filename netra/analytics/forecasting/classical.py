"""Classical / state-space / decomposition forecasters (CPU, always-on).

The always-on forecasting tier — pure CPU, light dependencies, fast — covering
the smooth-trend and seasonal fault morphologies (notably scenario A progressive
congestion). Members, each a :class:`~netra.analytics.forecasting.base.Forecaster`:

  * :class:`EwmaForecaster`     — EWMA level + slope (Holt linear-trend, O(1)).
  * :class:`HoltWintersForecaster` — additive/multiplicative Holt-Winters
    (level+trend+season) via ``river.time_series.HoltWinters`` with a hand-rolled
    NumPy fallback so it works even on a minimal install.
  * :class:`ThetaForecaster`    — the Theta method (M3-competition winner; a
    famously strong simple baseline) implemented in NumPy.
  * :class:`StlEtsForecaster`   — STL decomposition + ETS/Holt on the
    seasonally-adjusted series (``statsmodels``), recomposed with the seasonal
    naive — the interpretable multi-component member.
  * :class:`OnlineArimaForecaster` — online ARIMA/SNARIMAX
    (``river.time_series.SNARIMAX``) updated sample-by-sample (O(1)-ish), with a
    statsmodels SARIMAX fallback and finally an AR(1) fallback.

Every member emits a horizon-growing quantile band (research 03 §F): the bound
is what the time-to-impact estimator extrapolates to a threshold crossing, so an
honest, widening band is mandatory — a point alone is not enough.

``river`` and ``statsmodels`` are in the light/core tier, but each backend is
still imported lazily and guarded so a member degrades to a NumPy fallback rather
than failing the whole ensemble.
"""

from __future__ import annotations

import warnings

import numpy as np

from netra.contracts import DetectorFamily

from .base import Forecaster, residual_std

# ---------------------------------------------------------------------------
# Holt linear-trend EWMA (O(1), no backend) — the cheapest always-on member
# ---------------------------------------------------------------------------


class EwmaForecaster(Forecaster):
    """Double-exponential smoothing (Holt linear trend) — O(1), backend-free.

    Maintains a smoothed *level* and *slope*; the forecast is a straight-line
    extrapolation ``level + h*slope``. This is the canonical cheap precursor
    forecaster: on a ramping metric the slope is non-zero immediately, so the
    threshold-crossing lead time is available from the very first samples. The
    band grows like a random walk around the projected line.

    Parameters
    ----------
    alpha, beta:
        Level and trend smoothing factors in (0, 1].
    damping:
        Multiplicative damping on the projected slope (``<1`` flattens long
        horizons, guarding against runaway linear extrapolation).
    """

    method = "ewma_holt"
    family = DetectorFamily.FORECAST
    min_history = 2

    def __init__(self, *args, alpha: float = 0.4, beta: float = 0.2,
                 damping: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.damping = float(damping)
        self._level = 0.0
        self._slope = 0.0
        self._sigma = 1.0

    def _fit(self, series: np.ndarray) -> None:
        level = float(series[0])
        slope = float(series[1] - series[0]) if series.size > 1 else 0.0
        fitted = [level]
        for i in range(1, series.size):
            prev_level = level
            level = self.alpha * series[i] + (1 - self.alpha) * (level + slope)
            slope = self.beta * (level - prev_level) + (1 - self.beta) * slope
            fitted.append(prev_level + slope)
        self._level, self._slope = level, slope
        self._sigma = residual_std(series, np.asarray(fitted))

    def _predict(self, steps: int):
        h = np.arange(1, steps + 1)
        if self.damping == 1.0:
            trend = self._slope * h
        else:
            # geometric (damped) cumulative slope
            d = self.damping
            trend = self._slope * (d * (1 - d ** h) / (1 - d)) if d != 1 else self._slope * h
        point = self._level + trend
        lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        return point, lower, upper


# ---------------------------------------------------------------------------
# Holt-Winters (level + trend + seasonal)
# ---------------------------------------------------------------------------


class HoltWintersForecaster(Forecaster):
    """Triple-exponential smoothing (Holt-Winters) with seasonality.

    Captures the daily/business-hour cycles in traffic so a congestion forecast
    is not fooled by the normal diurnal ramp. Uses ``river.time_series.HoltWinters``
    when available (online, O(1) per update); otherwise a compact NumPy additive
    Holt-Winters is used so the member still runs on a minimal install.

    ``seasonality`` is the period in samples (0 disables the seasonal term).
    """

    method = "holt_winters"
    family = DetectorFamily.FORECAST
    min_history = 4

    def __init__(self, *args, alpha: float = 0.3, beta: float = 0.1,
                 gamma: float = 0.1, seasonality: int = 0,
                 multiplicative: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.seasonality = int(seasonality)
        self.multiplicative = bool(multiplicative)
        self._impl = None          # river model if available
        self._np_state = None      # (level, slope, season[]) fallback
        self._sigma = 1.0

    def _fit(self, series: np.ndarray) -> None:
        self._sigma = residual_std(series)
        # Prefer river's online Holt-Winters.
        try:
            from river.time_series import HoltWinters

            season = self.seasonality if self.seasonality and series.size > 2 * self.seasonality else 0
            model = HoltWinters(
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma if season else None,
                seasonality=season,
                multiplicative=self.multiplicative,
            )
            for v in series:
                model.learn_one(float(v))
            self._impl = model
            return
        except Exception:
            self._impl = None
        # NumPy additive Holt-Winters fallback.
        self._np_state = _np_holt_winters_fit(
            series, self.alpha, self.beta, self.gamma,
            self.seasonality if series.size > 2 * max(self.seasonality, 1) else 0,
        )

    def _predict(self, steps: int):
        if self._impl is not None:
            try:
                point = np.asarray(self._impl.forecast(horizon=steps), dtype=float)
                if point.size == steps and np.all(np.isfinite(point)):
                    lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
                    return point, lower, upper
            except Exception:
                pass
        # fallback path
        if self._np_state is None:
            level = float(self._history[-1]) if self._history is not None else 0.0
            point = np.full(steps, level)
        else:
            point = _np_holt_winters_forecast(self._np_state, steps)
        lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        return point, lower, upper


def _np_holt_winters_fit(series, alpha, beta, gamma, m):
    """Fit a minimal additive Holt-Winters; returns (level, slope, season list)."""
    n = series.size
    if m and n >= 2 * m:
        season = [float(series[i] - np.mean(series[:m])) for i in range(m)]
        level = float(np.mean(series[:m]))
        slope = float((np.mean(series[m:2 * m]) - np.mean(series[:m])) / m)
    else:
        m = 0
        season = []
        level = float(series[0])
        slope = float(series[1] - series[0]) if n > 1 else 0.0
    for i in range(n):
        s = season[i % m] if m else 0.0
        prev_level = level
        level = alpha * (series[i] - s) + (1 - alpha) * (level + slope)
        slope = beta * (level - prev_level) + (1 - beta) * slope
        if m:
            season[i % m] = gamma * (series[i] - level) + (1 - gamma) * s
    return (level, slope, season, m)


def _np_holt_winters_forecast(state, steps):
    level, slope, season, m = state
    out = np.empty(steps)
    for h in range(1, steps + 1):
        s = season[(h - 1) % m] if m else 0.0
        out[h - 1] = level + h * slope + s
    return out


# ---------------------------------------------------------------------------
# Theta method (NumPy)
# ---------------------------------------------------------------------------


class ThetaForecaster(Forecaster):
    """The Theta method — decompose into theta-lines and recombine.

    A deceptively strong, near-parameter-free baseline (won the M3 competition).
    We implement the classic theta=0/theta=2 decomposition: the long-term trend
    (OLS line, theta=0) plus SES on the theta=2 line (which doubles local
    curvature), combined 50/50. Excellent cheap diversity in the ensemble and a
    robust fallback when seasonal models over-fit short histories.
    """

    method = "theta"
    family = DetectorFamily.FORECAST
    min_history = 3

    def __init__(self, *args, ses_alpha: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ses_alpha = float(ses_alpha)
        self._intercept = 0.0
        self._slope = 0.0
        self._ses_level = 0.0
        self._n = 0
        self._sigma = 1.0

    def _fit(self, series: np.ndarray) -> None:
        n = series.size
        t = np.arange(n, dtype=float)
        # OLS trend (theta line 0)
        slope, intercept = np.polyfit(t, series, 1)
        self._slope, self._intercept, self._n = float(slope), float(intercept), n
        # theta=2 line emphasises curvature: 2*y - trend_line
        trend_line = intercept + slope * t
        theta2 = 2.0 * series - trend_line
        # SES on theta2
        lvl = float(theta2[0])
        for v in theta2[1:]:
            lvl = self.ses_alpha * v + (1 - self.ses_alpha) * lvl
        self._ses_level = lvl
        self._sigma = residual_std(series, trend_line)

    def _predict(self, steps: int):
        h = np.arange(1, steps + 1)
        trend_future = self._intercept + self._slope * (self._n - 1 + h)
        # combine the extrapolated trend (theta0) with the SES level of theta2,
        # averaged back to the original scale.
        point = 0.5 * trend_future + 0.5 * self._ses_level
        lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        return point, lower, upper


# ---------------------------------------------------------------------------
# STL decomposition + ETS/Holt on the deseasonalised series
# ---------------------------------------------------------------------------


class StlEtsForecaster(Forecaster):
    """STL decomposition + Holt on the seasonally-adjusted series.

    Splits the metric into trend + seasonal + remainder (LOESS), forecasts the
    seasonally-adjusted series with a Holt linear trend, then re-adds the
    seasonal-naive component. The explicit trend slope is itself a congestion
    precursor, and the clean residual is what the forecast-residual anomaly
    detector consumes. Falls back to a plain :class:`EwmaForecaster` when
    ``statsmodels`` STL is unavailable or the history is too short to decompose.
    """

    method = "stl_ets"
    family = DetectorFamily.FORECAST
    min_history = 6

    def __init__(self, *args, period: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.period = int(period)
        self._seasonal_tail: np.ndarray | None = None
        self._period_used = 0
        self._trend_model: EwmaForecaster | None = None
        self._sigma = 1.0
        self._fallback = False

    def _fit(self, series: np.ndarray) -> None:
        self._sigma = residual_std(series)
        period = self.period
        # Need >= 2 full periods to decompose meaningfully.
        if period >= 2 and series.size >= 2 * period:
            try:
                from statsmodels.tsa.seasonal import STL

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = STL(series, period=period, robust=True).fit()
                seasonal = np.asarray(res.seasonal, dtype=float)
                deseason = series - seasonal
                self._seasonal_tail = seasonal[-period:]
                self._period_used = period
                tm = EwmaForecaster(self.entity, self.metric,
                                    quantile_lower=self.quantile_lower,
                                    quantile_upper=self.quantile_upper)
                tm.fit(deseason)
                self._trend_model = tm
                self._sigma = residual_std(np.asarray(res.resid, dtype=float)
                                           + 0.0) or self._sigma
                return
            except Exception:
                pass
        # fallback: no usable seasonality -> Holt on raw series
        self._fallback = True
        tm = EwmaForecaster(self.entity, self.metric,
                            quantile_lower=self.quantile_lower,
                            quantile_upper=self.quantile_upper)
        tm.fit(series)
        self._trend_model = tm

    def _predict(self, steps: int):
        base = self._trend_model.forecast(steps, step_seconds=1.0)
        point = np.array([p.predicted for p in base.points], dtype=float)
        if not self._fallback and self._seasonal_tail is not None and self._period_used:
            seas = np.array([self._seasonal_tail[(i) % self._period_used]
                             for i in range(steps)], dtype=float)
            point = point + seas
        lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        return point, lower, upper


# ---------------------------------------------------------------------------
# Online ARIMA / SNARIMAX
# ---------------------------------------------------------------------------


class OnlineArimaForecaster(Forecaster):
    """Online ARIMA via ``river.time_series.SNARIMAX`` (O(1)-ish per sample).

    Streaming ARIMA: each new sample updates the model in (near) constant time,
    so the engine adapts to regime changes live without a full refit — the
    online forecasting mode of research 03 §"Online/incremental". Falls back to a
    batch ``statsmodels`` SARIMAX, then to a trivial AR(1), so it always returns
    a forecast even on a minimal install or a pathological series.

    Parameters ``p, d, q`` are the ARIMA orders; ``m, sp, sd, sq`` add a seasonal
    component when ``m > 1``.
    """

    method = "online_arima"
    family = DetectorFamily.FORECAST
    min_history = 5

    def __init__(self, *args, p: int = 2, d: int = 1, q: int = 1,
                 m: int = 1, sp: int = 0, sd: int = 0, sq: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p, self.d, self.q = int(p), int(d), int(q)
        self.m, self.sp, self.sd, self.sq = int(m), int(sp), int(sd), int(sq)
        self._impl = None
        self._mode = "ar1"
        self._ar1 = (0.0, 0.0, 0.0)   # (phi, intercept, last)
        self._sigma = 1.0

    def _fit(self, series: np.ndarray) -> None:
        self._sigma = residual_std(series)
        # 1) river SNARIMAX online
        try:
            from river import linear_model, preprocessing
            from river.time_series import SNARIMAX

            model = SNARIMAX(
                p=self.p, d=self.d, q=self.q,
                m=self.m, sp=self.sp, sd=self.sd, sq=self.sq,
                regressor=(preprocessing.StandardScaler() | linear_model.LinearRegression()),
            )
            for v in series:
                model.learn_one(float(v))
            self._impl = model
            self._mode = "snarimax"
            return
        except Exception:
            self._impl = None
        # 2) statsmodels SARIMAX batch
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                order = (self.p, self.d, self.q)
                seasonal = (self.sp, self.sd, self.sq, self.m) if self.m > 1 else (0, 0, 0, 0)
                res = SARIMAX(series, order=order, seasonal_order=seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False)
            self._impl = res
            self._mode = "sarimax"
            return
        except Exception:
            self._impl = None
        # 3) AR(1) fallback
        self._fit_ar1(series)
        self._mode = "ar1"

    def _fit_ar1(self, series: np.ndarray) -> None:
        if series.size > 2:
            x, y = series[:-1], series[1:]
            denom = np.dot(x - x.mean(), x - x.mean())
            phi = float(np.dot(x - x.mean(), y - y.mean()) / denom) if denom > 0 else 0.0
            phi = float(np.clip(phi, -0.999, 0.999))
            intercept = float(y.mean() - phi * x.mean())
        else:
            phi, intercept = 0.0, float(series[-1])
        self._ar1 = (phi, intercept, float(series[-1]))

    def _predict(self, steps: int):
        point = None
        if self._mode == "snarimax" and self._impl is not None:
            try:
                fc = self._impl.forecast(horizon=steps)
                point = np.asarray(fc, dtype=float)
            except Exception:
                point = None
        elif self._mode == "sarimax" and self._impl is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    point = np.asarray(self._impl.forecast(steps), dtype=float)
            except Exception:
                point = None
        if point is None or point.size != steps or not np.all(np.isfinite(point)):
            if self._mode != "ar1" and self._history is not None:
                self._fit_ar1(self._history)
            phi, intercept, last = self._ar1
            out = np.empty(steps)
            cur = last
            for i in range(steps):
                cur = intercept + phi * cur
                out[i] = cur
            point = out
        lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        return point, lower, upper


__all__ = [
    "EwmaForecaster",
    "HoltWintersForecaster",
    "ThetaForecaster",
    "StlEtsForecaster",
    "OnlineArimaForecaster",
]
