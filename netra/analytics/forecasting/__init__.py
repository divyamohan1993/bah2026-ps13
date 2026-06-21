"""netra.analytics.forecasting — trajectory forecasting + time-to-impact (WS3).

Tiered forecasters (always-on CPU classical/state-space/GBDT + feature-flagged
deep/foundation members) producing ``netra.contracts.Forecast``, and the
time-to-impact estimators (forecast-trajectory threshold-crossing + Theil-Sen
extrapolation + Cox/lifelines survival) producing ``netra.contracts.TimeToImpact``
— the engine's answer to Q1 ("what fails next AND WHEN").

Public surface
--------------
``Forecaster``                 common ABC (``fit`` -> ``forecast`` -> ``Forecast``)
``EwmaForecaster`` ...          classical CPU members (see ``classical``)
``GradientBoostedForecaster``   global ML member (LightGBM/HistGBR/RF, ``ml``)
``ChronosBoltForecaster``       optional foundation member (``foundation``)
``EnsembleForecaster``          heterogeneous ensemble + agreement (``ensemble``)
``TimeToImpactEstimator`` ...   time-to-impact estimators (``tti``)

Everything is import-light except the optional members, which guard their heavy
backends behind ``try/except`` so the module loads with only the core tier.
"""

from __future__ import annotations

from .base import Forecaster
from .classical import (
    EwmaForecaster,
    HoltWintersForecaster,
    OnlineArimaForecaster,
    StlEtsForecaster,
    ThetaForecaster,
)
from .ensemble import EnsembleForecaster, EnsembleResult
from .foundation import ChronosBoltForecaster
from .ml import GradientBoostedForecaster
from .tti import (
    SurvivalTTI,
    TheilSenTTI,
    TimeToImpactEstimator,
    TrajectoryCrossingTTI,
)

__all__ = [
    "Forecaster",
    "EwmaForecaster",
    "HoltWintersForecaster",
    "ThetaForecaster",
    "StlEtsForecaster",
    "OnlineArimaForecaster",
    "GradientBoostedForecaster",
    "ChronosBoltForecaster",
    "EnsembleForecaster",
    "EnsembleResult",
    "TrajectoryCrossingTTI",
    "TheilSenTTI",
    "SurvivalTTI",
    "TimeToImpactEstimator",
]
