"""Common forecasting abstractions — the ``Forecaster`` ABC + helpers.

Every forecaster in NETRA — classical (EWMA/Holt-Winters/Theta/STL+ETS), the
online state-space/ARIMA members, the gradient-boosted lag regressor, and the
optional Chronos-Bolt foundation wrapper — implements the same tiny interface so
the ensemble can swap, weight and cross-verify them uniformly:

    forecaster.fit(history) -> self
    forecaster.forecast(steps, step_seconds, origin) -> Forecast

``Forecast`` is the canonical contract (point + quantile band + producing
``method``/``family``) from :mod:`netra.contracts`. The win condition of the
project — *lead time* — is read off the produced trajectory band by
:mod:`netra.analytics.forecasting.tti`, so every member must emit honest
``lower``/``upper`` quantile bounds (widening when it is less certain) rather
than a bare point.

This module imports **only** numpy + the contracts, so it always loads on the
CPU/offline tier; heavier members import their backends lazily behind
``try/except`` (see :mod:`~.ml` and :mod:`~.foundation`).
"""

from __future__ import annotations

import abc
from datetime import UTC, datetime

import numpy as np

from netra.contracts import (
    DetectorFamily,
    EntityRef,
    Forecast,
    QuantilePoint,
)

# ---------------------------------------------------------------------------
# Small helpers shared by every forecaster
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Timezone-aware UTC now (contracts use aware datetimes)."""
    return datetime.now(tz=UTC)


def to_series(history: object) -> np.ndarray:
    """Coerce an input history into a 1-D float ``np.ndarray`` of finite values.

    Accepts a list/tuple/ndarray of numbers, or anything ``np.asarray`` can turn
    into floats (e.g. a pandas Series). NaN/inf are dropped so a single bad
    SNMP poll cannot poison a classical fit. Raises ``ValueError`` on an empty
    series so callers fail loudly rather than forecasting noise.
    """
    arr = np.asarray(list(history) if not isinstance(history, np.ndarray) else history,
                     dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("forecaster received an empty/all-NaN history")
    return arr


def build_forecast(
    *,
    entity: EntityRef,
    metric: str,
    point: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    step_seconds: float,
    method: str,
    family: DetectorFamily = DetectorFamily.FORECAST,
    origin: datetime | None = None,
    quantile_lower: float = 0.1,
    quantile_upper: float = 0.9,
    backtest_mase: float | None = None,
) -> Forecast:
    """Assemble a :class:`~netra.contracts.Forecast` from raw point/band arrays.

    Centralises the per-step :class:`QuantilePoint` construction (and enforces a
    monotone, non-crossing band: ``lower <= point <= upper``) so every member
    produces a contract-valid forecast the same way.
    """
    point = np.asarray(point, dtype=float).ravel()
    n = point.size
    if n == 0:
        raise ValueError("build_forecast received an empty point trajectory")
    origin = origin or _utcnow()
    step_seconds = float(step_seconds)
    if step_seconds <= 0:
        raise ValueError("step_seconds must be > 0")

    lo = None if lower is None else np.asarray(lower, dtype=float).ravel()
    hi = None if upper is None else np.asarray(upper, dtype=float).ravel()

    points: list[QuantilePoint] = []
    for i in range(n):
        p = float(point[i])
        lv = None if lo is None else float(min(lo[i], p))   # never cross the point
        uv = None if hi is None else float(max(hi[i], p))
        points.append(
            QuantilePoint(
                horizon_seconds=step_seconds * (i + 1),
                predicted=p,
                lower=lv,
                upper=uv,
            )
        )

    return Forecast(
        entity=entity,
        metric=metric,
        origin=origin,
        horizon_seconds=step_seconds * n,
        points=points,
        method=method,
        family=family,
        quantile_lower=quantile_lower,
        quantile_upper=quantile_upper,
        backtest_mase=backtest_mase,
    )


def residual_std(history: np.ndarray, fitted: np.ndarray | None = None) -> float:
    """Robust estimate of one-step noise std for symmetric quantile bands.

    Uses the MAD of first differences (×1.4826 for normal-consistency) so a few
    outliers/level-shifts in the history don't inflate the band; falls back to a
    tiny positive floor relative to the level so bands are never degenerate.
    """
    h = np.asarray(history, dtype=float).ravel()
    if fitted is not None and len(fitted) == len(h):
        resid = h - np.asarray(fitted, dtype=float).ravel()
    else:
        resid = np.diff(h) if h.size > 1 else np.array([0.0])
    if resid.size == 0:
        resid = np.array([0.0])
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(resid)) if resid.size > 1 else 0.0
    level = float(np.median(np.abs(h))) if h.size else 1.0
    floor = max(1e-9, 1e-3 * (level if level > 0 else 1.0))
    return float(max(sigma, floor))


# Z multipliers for a couple of common symmetric quantile levels (Gaussian).
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.975: 2.2414, 0.99: 2.3263}


def z_for_quantile(q_upper: float) -> float:
    """Gaussian z-score for an upper quantile (e.g. 0.9 -> 1.6449)."""
    return _Z.get(round(float(q_upper), 3), 1.6449)


def growing_band(
    point: np.ndarray,
    sigma: float,
    *,
    q_upper: float = 0.9,
    grow: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric quantile band that widens with the horizon.

    Uncertainty about a trajectory grows the further out you predict; we model
    that as ``sigma * sqrt(grow * step)`` (random-walk-like spread). ``grow``
    lets smoother models (state-space) claim tighter long-horizon bands than a
    naive extrapolation. Returns ``(lower, upper)`` arrays the same length as
    ``point``.
    """
    point = np.asarray(point, dtype=float).ravel()
    z = z_for_quantile(q_upper)
    steps = np.arange(1, point.size + 1, dtype=float)
    width = z * sigma * np.sqrt(np.maximum(grow, 1e-9) * steps)
    return point - width, point + width


class Forecaster(abc.ABC):
    """Abstract base class for all NETRA trajectory forecasters.

    Subclasses implement :meth:`_fit` and :meth:`_predict`; the base wires in the
    entity/metric bookkeeping and the ``Forecast`` assembly so members stay
    small. Contract: ``forecast`` must return a fully-populated
    :class:`~netra.contracts.Forecast` with a per-step quantile band, even if the
    member is a point model (in which case it adds a horizon-growing Gaussian
    band via :func:`growing_band`).

    Parameters
    ----------
    entity, metric:
        Identify what is being forecast (carried straight onto the ``Forecast``).
    quantile_lower, quantile_upper:
        Quantile levels of the emitted band (default p10/p90).
    """

    #: Stable model id recorded on every Forecast (override per subclass).
    method: str = "forecaster"
    #: Method family for fusion/agreement bookkeeping.
    family: DetectorFamily = DetectorFamily.FORECAST
    #: Minimum history length the member needs to fit meaningfully.
    min_history: int = 3

    def __init__(
        self,
        entity: EntityRef,
        metric: str,
        *,
        quantile_lower: float = 0.1,
        quantile_upper: float = 0.9,
    ) -> None:
        self.entity = entity
        self.metric = metric
        self.quantile_lower = float(quantile_lower)
        self.quantile_upper = float(quantile_upper)
        self._history: np.ndarray | None = None
        self._fitted: bool = False
        self._backtest_mase: float | None = None

    # -- public API ---------------------------------------------------------

    def fit(self, history: object) -> Forecaster:
        """Fit the member on a 1-D history (most recent value last)."""
        series = to_series(history)
        self._history = series
        self._fit(series)
        self._fitted = True
        return self

    def forecast(
        self,
        steps: int,
        step_seconds: float = 60.0,
        origin: datetime | None = None,
    ) -> Forecast:
        """Produce a ``steps``-ahead :class:`Forecast` (each step ``step_seconds``)."""
        if not self._fitted or self._history is None:
            raise RuntimeError(f"{self.method}: call fit() before forecast()")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        point, lower, upper = self._predict(steps)
        return build_forecast(
            entity=self.entity,
            metric=self.metric,
            point=point,
            lower=lower,
            upper=upper,
            step_seconds=step_seconds,
            method=self.method,
            family=self.family,
            origin=origin,
            quantile_lower=self.quantile_lower,
            quantile_upper=self.quantile_upper,
            backtest_mase=self._backtest_mase,
        )

    # -- to implement -------------------------------------------------------

    @abc.abstractmethod
    def _fit(self, series: np.ndarray) -> None:
        """Fit member-specific state on a clean 1-D float series."""

    @abc.abstractmethod
    def _predict(self, steps: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``(point, lower, upper)`` arrays of length ``steps``.

        ``lower``/``upper`` may be ``None`` (the base will not synthesise a band
        in that case) but members are strongly encouraged to provide one.
        """

    # -- convenience for subclasses ----------------------------------------

    def _symmetric_band(
        self, point: np.ndarray, sigma: float, grow: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Horizon-growing Gaussian band at the configured upper quantile."""
        return growing_band(point, sigma, q_upper=self.quantile_upper, grow=grow)


__all__ = [
    "Forecaster",
    "build_forecast",
    "to_series",
    "residual_std",
    "growing_band",
    "z_for_quantile",
]
