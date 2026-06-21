"""Forecasting ensemble — combine heterogeneous members, cross-verify, pool.

No single forecaster covers all four fault morphologies, so NETRA runs a
*tiered, heterogeneous* set and combines them (research 03 §4, §7). The ensemble:

  * builds the **always-on CPU tier** (EWMA/Holt-trend, Holt-Winters, Theta,
    STL+ETS, online-ARIMA) and, when their backends/history allow, the
    **gradient-boosted** member and the optional **Chronos-Bolt** member;
  * fits every member on the history, dropping any that fail (graceful
    degradation — the classical tier always carries the load);
  * combines the point trajectories by an **inverse-error / median** rule (the
    median is a famously robust combiner) and **pools the quantile bands**;
  * exposes **cross-model agreement** (1 − normalised member spread) as a
    confidence signal — high agreement ⇒ tight, trustworthy band; disagreement ⇒
    a wider band and lower confidence, i.e. "needs human review" rather than
    false certainty.

The combined result is itself a :class:`~netra.contracts.Forecast` (method
``ensemble``) plus, via :meth:`EnsembleForecaster.forecast_with_members`, the
individual member forecasts (so fusion can record per-method provenance and the
UI can show the spread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from netra.contracts import DetectorFamily, EntityRef, Forecast

from .base import build_forecast
from .classical import (
    EwmaForecaster,
    HoltWintersForecaster,
    OnlineArimaForecaster,
    StlEtsForecaster,
    ThetaForecaster,
)
from .foundation import ChronosBoltForecaster
from .ml import GradientBoostedForecaster


@dataclass
class EnsembleResult:
    """Bundle returned by :meth:`EnsembleForecaster.forecast_with_members`.

    Attributes
    ----------
    combined:
        The fused ensemble :class:`Forecast` (point + pooled band).
    members:
        Per-member :class:`Forecast` objects (already fit + forecast).
    agreement:
        Cross-model agreement in [0, 1] (1 = members coincide). Feeds the fusion
        layer's calibrated confidence.
    weights:
        ``{method: weight}`` used to combine the points (inverse backtest error).
    """

    combined: Forecast
    members: list[Forecast] = field(default_factory=list)
    agreement: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)


class EnsembleForecaster:
    """Heterogeneous forecasting ensemble with agreement-as-confidence.

    Parameters
    ----------
    entity, metric:
        What is being forecast.
    seasonality:
        Seasonal period in samples (passed to the seasonal members; 0 disables).
    enable_gbm:
        Include the gradient-boosted member when history is long enough
        (default True; it self-skips on a too-short series or no backend).
    enable_foundation:
        Include the optional Chronos-Bolt member when available (default False —
        the CPU-only demo leaves it off; turn on if local weights are bundled).
    backtest_window:
        Number of trailing points held out to score each member's recent MASE for
        inverse-error weighting (0 disables weighting → equal weights).
    """

    def __init__(
        self,
        entity: EntityRef,
        metric: str,
        *,
        seasonality: int = 0,
        enable_gbm: bool = True,
        enable_foundation: bool = False,
        quantile_lower: float = 0.1,
        quantile_upper: float = 0.9,
        backtest_window: int = 12,
    ) -> None:
        self.entity = entity
        self.metric = metric
        self.seasonality = int(seasonality)
        self.enable_gbm = bool(enable_gbm)
        self.enable_foundation = bool(enable_foundation)
        self.quantile_lower = float(quantile_lower)
        self.quantile_upper = float(quantile_upper)
        self.backtest_window = int(backtest_window)

    # -- member construction ------------------------------------------------

    def _build_members(self, history_len: int) -> list:
        """Instantiate the member forecasters appropriate for this history."""
        kw = dict(quantile_lower=self.quantile_lower,
                  quantile_upper=self.quantile_upper)
        members: list = [
            EwmaForecaster(self.entity, self.metric, **kw),
            ThetaForecaster(self.entity, self.metric, **kw),
            OnlineArimaForecaster(self.entity, self.metric, **kw),
        ]
        if self.seasonality >= 2 and history_len >= 2 * self.seasonality:
            members.append(
                HoltWintersForecaster(self.entity, self.metric,
                                      seasonality=self.seasonality, gamma=0.1, **kw)
            )
            members.append(
                StlEtsForecaster(self.entity, self.metric,
                                 period=self.seasonality, **kw)
            )
        else:
            # still add a non-seasonal Holt-Winters (acts as damped Holt) for diversity
            members.append(HoltWintersForecaster(self.entity, self.metric, **kw))
        if self.enable_gbm:
            gbm = GradientBoostedForecaster(self.entity, self.metric, **kw)
            if history_len >= gbm.min_history:
                members.append(gbm)
        if self.enable_foundation:
            ch = ChronosBoltForecaster(self.entity, self.metric, **kw)
            if ch.backend_importable():
                members.append(ch)
        return members

    # -- public API ---------------------------------------------------------

    def forecast(self, history: object, steps: int,
                 step_seconds: float = 60.0,
                 origin: datetime | None = None) -> Forecast:
        """Convenience: return only the combined ensemble :class:`Forecast`."""
        return self.forecast_with_members(history, steps, step_seconds, origin).combined

    def forecast_with_members(
        self, history: object, steps: int,
        step_seconds: float = 60.0, origin: datetime | None = None,
    ) -> EnsembleResult:
        """Fit every member, combine, and return the full :class:`EnsembleResult`."""
        series = np.asarray(list(history), dtype=float).ravel()
        series = series[np.isfinite(series)]
        if series.size == 0:
            raise ValueError("ensemble received an empty/all-NaN history")
        members = self._build_members(series.size)

        member_fcs: list[Forecast] = []
        member_points: list[np.ndarray] = []
        member_lowers: list[np.ndarray] = []
        member_uppers: list[np.ndarray] = []
        weights: list[float] = []
        methods: list[str] = []

        for m in members:
            try:
                mase = self._backtest_member(m, series, steps, step_seconds)
                m.fit(series)
                fc = m.forecast(steps, step_seconds=step_seconds, origin=origin)
            except Exception:
                continue
            pts = np.array([p.predicted for p in fc.points], dtype=float)
            if pts.size != steps or not np.all(np.isfinite(pts)):
                continue
            lo = np.array([p.lower if p.lower is not None else p.predicted
                           for p in fc.points], dtype=float)
            hi = np.array([p.upper if p.upper is not None else p.predicted
                           for p in fc.points], dtype=float)
            fc.backtest_mase = mase
            member_fcs.append(fc)
            member_points.append(pts)
            member_lowers.append(lo)
            member_uppers.append(hi)
            methods.append(fc.method)
            # inverse-error weight (smaller MASE -> larger weight); +eps for safety
            weights.append(1.0 / (mase + 1e-3) if mase is not None else 1.0)

        if not member_points:
            raise RuntimeError("ensemble: every member failed to forecast")

        P = np.vstack(member_points)            # (k, steps)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(weights), 1.0 / len(weights))

        # Robust point combination: blend the weighted mean with the median.
        weighted_mean = (w[:, None] * P).sum(axis=0)
        median = np.median(P, axis=0)
        combined_point = 0.5 * weighted_mean + 0.5 * median

        # Pool the band: take the envelope of member bands AND add the
        # cross-member spread so disagreement widens the combined interval.
        lo_stack = np.vstack(member_lowers)
        hi_stack = np.vstack(member_uppers)
        member_spread = P.std(axis=0)
        pooled_lower = np.minimum(lo_stack.min(axis=0), combined_point - member_spread)
        pooled_upper = np.maximum(hi_stack.max(axis=0), combined_point + member_spread)

        agreement = self._agreement(P, combined_point)

        combined = build_forecast(
            entity=self.entity, metric=self.metric,
            point=combined_point, lower=pooled_lower, upper=pooled_upper,
            step_seconds=step_seconds, method="ensemble",
            family=DetectorFamily.FORECAST, origin=origin,
            quantile_lower=self.quantile_lower, quantile_upper=self.quantile_upper,
            backtest_mase=float(np.average([f.backtest_mase for f in member_fcs
                                            if f.backtest_mase is not None] or [np.nan])),
        )
        return EnsembleResult(
            combined=combined,
            members=member_fcs,
            agreement=agreement,
            weights={meth: float(wi) for meth, wi in zip(methods, w)},
        )

    # -- internals ----------------------------------------------------------

    def _backtest_member(self, member, series: np.ndarray, steps: int,
                         step_seconds: float) -> float | None:
        """Rolling-origin MASE of one member on a held-out tail (no leakage).

        Fits on ``series[:-h]`` and scores the 1-step error over the last ``h``
        points against the seasonal/naive denominator. Returns ``None`` when the
        history is too short to back-test (then the member gets equal weight).
        """
        h = self.backtest_window
        if h <= 0 or series.size < member.min_history + h + 1:
            return None
        train = series[:-h]
        actual = series[-h:]
        try:
            fresh = member.__class__(self.entity, self.metric,
                                     quantile_lower=self.quantile_lower,
                                     quantile_upper=self.quantile_upper)
            fresh.fit(train)
            fc = fresh.forecast(h, step_seconds=step_seconds)
            pred = np.array([p.predicted for p in fc.points], dtype=float)
        except Exception:
            return None
        if pred.size != h or not np.all(np.isfinite(pred)):
            return None
        mae = np.mean(np.abs(actual - pred))
        # naive (random-walk) scale on the training series
        denom = np.mean(np.abs(np.diff(train))) if train.size > 1 else 1.0
        denom = denom if denom > 1e-9 else 1.0
        return float(mae / denom)

    @staticmethod
    def _agreement(P: np.ndarray, combined: np.ndarray) -> float:
        """Cross-model agreement in [0,1] = 1 − normalised member spread.

        Spread is the mean per-step std across members, normalised by the level
        of the combined trajectory (so it is scale-free). 1.0 means the members
        coincide; values drop as they diverge.
        """
        if P.shape[0] < 2:
            return 1.0
        spread = float(np.mean(P.std(axis=0)))
        scale = float(np.mean(np.abs(combined))) + 1e-9
        rel = spread / scale
        return float(np.clip(1.0 - rel, 0.0, 1.0))


__all__ = ["EnsembleForecaster", "EnsembleResult"]
