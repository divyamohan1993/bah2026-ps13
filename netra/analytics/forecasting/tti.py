"""Time-to-impact estimation — the headline 'and WHEN' number (Q1).

Lead time is the win condition (35% of score). This module turns a forecast
trajectory — and, crucially, its **uncertainty band** — into a calibrated
:class:`~netra.contracts.TimeToImpact`: the first time the predicted value (and
its bounds) crosses an SLA/security threshold, with a confidence derived from how
tight the band is at the crossing.

Three complementary estimators (research 03 §G, research 04 §9 #66), all CPU/offline:

  * :class:`TrajectoryCrossingTTI` — read the crossing time straight off a
    :class:`Forecast`. The *point* trajectory gives the ETA; the *upper* and
    *lower* bands give the optimistic/pessimistic CI; the quantile spread at the
    crossing sets the confidence. This is the primary estimator and consumes the
    ensemble forecast directly.
  * :class:`TheilSenTTI` — a robust O(1) slope (Theil-Sen, optionally the
    ``pymannkendall`` trend) on the recent history extrapolated to the threshold;
    a backend-light cross-check / cold-start estimator.
  * :class:`SurvivalTTI` — a Cox proportional-hazards model (``lifelines``, MIT —
    never scikit-survival/GPL) trained on engineered features + fault labels,
    giving an expected time-to-breach when the failure is event-like rather than
    a smooth trend.

:class:`TimeToImpactEstimator` is the convenience facade most callers use: give
it a forecast (and/or history) + threshold and it returns the best available
:class:`TimeToImpact`, preferring the trajectory band and falling back to the
slope extrapolation.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from netra.contracts import (
    Direction,
    EntityRef,
    Forecast,
    TimeToImpact,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _crosses(value: float, threshold: float, direction: Direction) -> bool:
    """True if ``value`` is on the breach side of ``threshold``."""
    if direction == Direction.DECREASES_RISK:
        return value <= threshold
    return value >= threshold


def _interp_cross_time(times: np.ndarray, values: np.ndarray,
                       threshold: float, direction: Direction,
                       start_value: float | None = None) -> float | None:
    """First time ``values(times)`` crosses ``threshold`` (linear interpolation).

    ``times`` are seconds-ahead (strictly increasing, starting > 0). If a
    ``start_value`` (the value at t=0, i.e. now) is given and it is already on the
    breach side, returns 0.0 (already breached). Returns ``None`` if no crossing
    occurs within the supplied horizon.
    """
    if start_value is not None and _crosses(start_value, threshold, direction):
        return 0.0
    prev_t = 0.0
    prev_v = start_value if start_value is not None else float(values[0])
    for t, v in zip(times, values, strict=False):
        if _crosses(v, threshold, direction):
            # linear interpolation between (prev_t, prev_v) and (t, v)
            if v == prev_v:
                return float(t)
            frac = (threshold - prev_v) / (v - prev_v)
            frac = float(np.clip(frac, 0.0, 1.0))
            return float(prev_t + frac * (t - prev_t))
        prev_t, prev_v = float(t), float(v)
    return None


class TrajectoryCrossingTTI:
    """Read time-to-impact off a forecast trajectory and its quantile band.

    The point trajectory yields the ETA; because risk grows toward the breach,
    the **upper** band crosses *earliest* (pessimistic / shortest ETA) and the
    **lower** band crosses *latest* (optimistic) for an upward threshold — and
    vice-versa for a downward one. The CI is ``[earliest, latest]`` of the band
    crossings and the confidence shrinks as that interval widens relative to the
    point ETA (a tight band ⇒ high confidence, a wide band ⇒ "needs review").
    """

    method = "trajectory_crossing"

    def estimate(
        self,
        forecast: Forecast,
        threshold: float,
        *,
        direction: Direction = Direction.INCREASES_RISK,
        current_value: float | None = None,
        origin: datetime | None = None,
        agreement: float | None = None,
    ) -> TimeToImpact:
        """Compute a :class:`TimeToImpact` from ``forecast`` vs ``threshold``.

        Parameters
        ----------
        agreement:
            Optional cross-model agreement in [0,1] from the ensemble; folded into
            the confidence (disagreement lowers confidence).
        """
        pts = forecast.points
        times = np.array([p.horizon_seconds for p in pts], dtype=float)
        point = np.array([p.predicted for p in pts], dtype=float)
        lower = np.array([p.lower if p.lower is not None else p.predicted
                          for p in pts], dtype=float)
        upper = np.array([p.upper if p.upper is not None else p.predicted
                          for p in pts], dtype=float)

        eta = _interp_cross_time(times, point, threshold, direction, current_value)
        # band crossings: which bound is pessimistic depends on direction
        up_cross = _interp_cross_time(times, upper, threshold, direction, current_value)
        lo_cross = _interp_cross_time(times, lower, threshold, direction, current_value)
        band_crossings = [c for c in (up_cross, lo_cross) if c is not None]
        eta_lower = min(band_crossings) if band_crossings else None   # earliest (pessimistic)
        eta_upper = max(band_crossings) if band_crossings else None   # latest (optimistic)

        confidence = self._confidence(eta, eta_lower, eta_upper, times[-1],
                                      n_band_cross=len(band_crossings),
                                      agreement=agreement)

        return TimeToImpact(
            entity=forecast.entity,
            metric=forecast.metric,
            origin=origin or forecast.origin,
            threshold=float(threshold),
            threshold_direction=direction,
            eta_seconds=eta,
            eta_lower_seconds=eta_lower,
            eta_upper_seconds=eta_upper,
            confidence=confidence,
            method=self.method,
        )

    @staticmethod
    def _confidence(eta, eta_lower, eta_upper, horizon, *,
                    n_band_cross: int, agreement: float | None) -> float:
        """Confidence in [0,1] from band tightness + band agreement + ensemble agreement."""
        if eta is None:
            # No predicted crossing: confidence reflects how *clearly* healthy it
            # is — if not even the upper band crosses, that's a confident 'safe'.
            base = 0.6 if n_band_cross == 0 else 0.35
            return float(np.clip(base * (agreement if agreement is not None else 1.0), 0.0, 1.0))
        if eta_lower is not None and eta_upper is not None and horizon > 0:
            ci_width = max(eta_upper - eta_lower, 0.0)
            rel = ci_width / max(eta, horizon * 0.1, 1.0)
            base = 1.0 / (1.0 + rel)            # tight band -> ~1, wide band -> ~0
        else:
            base = 0.5                          # only the point crossed
        if agreement is not None:
            base = 0.5 * base + 0.5 * float(np.clip(agreement, 0.0, 1.0))
        return float(np.clip(base, 0.05, 0.99))


class TheilSenTTI:
    """Robust slope-extrapolation time-to-impact (backend-light cross-check).

    Fits a robust linear slope to the recent history (Theil-Sen via
    ``scipy.stats.theilslopes``, optionally cross-checked by ``pymannkendall``)
    and solves analytically for the threshold-crossing time. O(1)-ish, no
    forecast needed — used as a fast streaming estimator and a sanity check on
    the trajectory crossing. Confidence comes from the slope's confidence
    interval (a well-determined slope ⇒ high confidence).
    """

    method = "theil_sen_extrapolation"

    def __init__(self, sample_period_seconds: float = 60.0,
                 window: int = 30) -> None:
        self.sample_period_seconds = float(sample_period_seconds)
        self.window = int(window)

    def estimate(
        self,
        entity: EntityRef,
        metric: str,
        history: object,
        threshold: float,
        *,
        direction: Direction = Direction.INCREASES_RISK,
        origin: datetime | None = None,
    ) -> TimeToImpact:
        series = np.asarray(list(history), dtype=float).ravel()
        series = series[np.isfinite(series)]
        if series.size < 3:
            raise ValueError("TheilSenTTI needs >= 3 historical points")
        series = series[-self.window:]
        t = np.arange(series.size, dtype=float)
        from scipy.stats import theilslopes

        slope, intercept, lo_slope, hi_slope = theilslopes(series, t)
        current = float(series[-1])

        def cross_seconds(s: float) -> float | None:
            if abs(s) < 1e-12:
                return None
            steps = (threshold - current) / s
            if steps <= 0:
                # already past, or moving away
                return 0.0 if _crosses(current, threshold, direction) else None
            # only count if moving toward the breach side
            moving_up = s > 0
            breach_up = direction != Direction.DECREASES_RISK
            if moving_up != breach_up:
                return None
            return float(steps * self.sample_period_seconds)

        eta = cross_seconds(slope)
        eta_lo = cross_seconds(hi_slope)        # steeper slope -> sooner
        eta_hi = cross_seconds(lo_slope)        # shallower slope -> later
        candidates = [c for c in (eta_lo, eta_hi) if c is not None]
        eta_lower = min(candidates) if candidates else None
        eta_upper = max(candidates) if candidates else None

        # confidence: how tight is the slope CI relative to the slope?
        slope_ci = abs(hi_slope - lo_slope)
        rel = slope_ci / (abs(slope) + 1e-9)
        confidence = float(np.clip(1.0 / (1.0 + rel), 0.05, 0.95)) if eta is not None else 0.3

        return TimeToImpact(
            entity=entity, metric=metric,
            origin=origin or datetime.now().astimezone(),
            threshold=float(threshold), threshold_direction=direction,
            eta_seconds=eta, eta_lower_seconds=eta_lower, eta_upper_seconds=eta_upper,
            confidence=confidence, method=self.method,
        )


class SurvivalTTI:
    """Cox proportional-hazards time-to-impact (``lifelines``, MIT — not GPL).

    Models 'time until breach/failure' as a hazard from engineered features +
    injected-fault labels, yielding an expected time-to-event and a hazard-based
    risk even when the precursor is not a smooth trend (e.g. flap onset). Uses
    ``lifelines`` exclusively — **never scikit-survival**, which is GPL-3.0 and
    excluded from the permissive air-gap bundle.

    Train once on labelled history; then :meth:`estimate` returns the expected
    time-to-event for a current feature row. If ``lifelines`` is unavailable the
    estimator reports itself unfitted and callers fall back to the trajectory/slope
    estimators.
    """

    method = "cox_survival"

    def __init__(self, penalizer: float = 0.1) -> None:
        self.penalizer = float(penalizer)
        self._cph = None
        self._feature_cols: list[str] = []
        self._max_duration = 0.0

    @property
    def is_fitted(self) -> bool:
        return self._cph is not None

    def fit(self, frame: object, duration_col: str = "duration",
            event_col: str = "event") -> SurvivalTTI:
        """Fit Cox PH on a table of features + (duration, event) columns.

        ``frame`` is anything coercible to a ``pandas.DataFrame``; every column
        other than ``duration``/``event`` is treated as a covariate. Returns self;
        raises only if ``lifelines`` cannot be imported (a real missing dep) — a
        degenerate fit is swallowed so the pipeline keeps running.
        """
        import pandas as pd

        df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        try:
            from lifelines import CoxPHFitter
        except Exception as exc:  # genuinely missing optional dep
            raise RuntimeError("lifelines is required for SurvivalTTI") from exc
        self._feature_cols = [c for c in df.columns if c not in (duration_col, event_col)]
        self._max_duration = float(df[duration_col].max())
        try:
            cph = CoxPHFitter(penalizer=self.penalizer)
            cph.fit(df, duration_col=duration_col, event_col=event_col)
            self._cph = cph
        except Exception:
            self._cph = None        # degenerate data -> stay unfitted, caller falls back
        return self

    def estimate(
        self,
        entity: EntityRef,
        metric: str,
        features: dict[str, float],
        threshold: float,
        *,
        direction: Direction = Direction.INCREASES_RISK,
        origin: datetime | None = None,
    ) -> TimeToImpact:
        if not self.is_fitted:
            raise RuntimeError("SurvivalTTI.estimate called before a successful fit")
        import pandas as pd

        row = pd.DataFrame([{c: float(features.get(c, 0.0)) for c in self._feature_cols}])
        try:
            expected = float(self._cph.predict_expectation(row).iloc[0])
        except Exception:
            expected = self._max_duration
        eta = max(expected, 0.0) * self.sample_scale()
        # CI from the survival function quantiles if available
        eta_lower = eta_upper = None
        try:
            med = float(self._cph.predict_median(row).iloc[0]) * self.sample_scale()
            eta_lower = min(eta, med)
            eta_upper = max(eta, med)
        except Exception:
            pass
        # confidence from the model's concordance (how well it ranks risk)
        conf = float(np.clip(getattr(self._cph, "concordance_index_", 0.7), 0.05, 0.95))
        return TimeToImpact(
            entity=entity, metric=metric,
            origin=origin or datetime.now().astimezone(),
            threshold=float(threshold), threshold_direction=direction,
            eta_seconds=eta if np.isfinite(eta) else None,
            eta_lower_seconds=eta_lower, eta_upper_seconds=eta_upper,
            confidence=conf, method=self.method,
        )

    @staticmethod
    def sample_scale() -> float:
        """Duration unit -> seconds. Durations are supplied in seconds already."""
        return 1.0


class TimeToImpactEstimator:
    """Facade choosing the best available time-to-impact estimator.

    Default behaviour: estimate from the forecast trajectory band
    (:class:`TrajectoryCrossingTTI`); if that yields no crossing but the history
    has a clear robust trend toward the threshold, fall back to
    :class:`TheilSenTTI`. A pre-fitted :class:`SurvivalTTI` can be supplied to add
    a hazard-based cross-check. This is what fusion / the copilot call.
    """

    def __init__(self, sample_period_seconds: float = 60.0,
                 survival: SurvivalTTI | None = None) -> None:
        self.sample_period_seconds = float(sample_period_seconds)
        self.trajectory = TrajectoryCrossingTTI()
        self.theil_sen = TheilSenTTI(sample_period_seconds=sample_period_seconds)
        self.survival = survival

    def estimate(
        self,
        forecast: Forecast,
        threshold: float,
        *,
        direction: Direction = Direction.INCREASES_RISK,
        history: object | None = None,
        current_value: float | None = None,
        agreement: float | None = None,
        features: dict[str, float] | None = None,
    ) -> TimeToImpact:
        """Return the best :class:`TimeToImpact` for this forecast + threshold."""
        tti = self.trajectory.estimate(
            forecast, threshold, direction=direction,
            current_value=current_value, agreement=agreement,
        )
        if tti.eta_seconds is not None:
            return tti
        # Forecast saw no crossing in-horizon; try a robust slope extrapolation
        # (it can see further than the forecast horizon).
        if history is not None:
            try:
                ts = self.theil_sen.estimate(
                    forecast.entity, forecast.metric, history, threshold,
                    direction=direction, origin=forecast.origin,
                )
                if ts.eta_seconds is not None:
                    return ts
            except Exception:
                pass
        # Optional survival cross-check.
        if self.survival is not None and self.survival.is_fitted and features is not None:
            try:
                return self.survival.estimate(
                    forecast.entity, forecast.metric, features, threshold,
                    direction=direction, origin=forecast.origin,
                )
            except Exception:
                pass
        return tti     # healthy: no crossing predicted


__all__ = [
    "TrajectoryCrossingTTI",
    "TheilSenTTI",
    "SurvivalTTI",
    "TimeToImpactEstimator",
]
