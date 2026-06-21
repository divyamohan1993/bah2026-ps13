"""netra.analytics.anomaly — tiered detector bank + EVT thresholds (WS3).

The non-forecasting detectors (#19-#60: statistical/streaming, ML unsupervised,
deep, change-point/drift, matrix-profile) each producing a
``netra.contracts.AnomalyScore`` with a [0,1] ``normalized_score`` for fusion,
plus EVT/SPOT/DSPOT (#68) adaptive, risk-controlled thresholds that replace
hand-set ones.

Public surface
--------------
``Detector``                       common ABC (``fit`` / ``update`` -> AnomalyScore)
``RobustZDetector`` ...            statistical members (``statistical``)
``HalfSpaceTreesDetector`` ...     ML/streaming members + forecast-residual (``ml``)
``PageHinkleyDetector`` ...        change-point/drift members (``changepoint``)
``MatrixProfileDiscordDetector``   matrix-profile discord (``matrixprofile``)
``AutoEncoderDetector`` ...        optional deep + PCA-recon (``deep``)
``POT`` / ``SPOT`` / ``DSPOT``     EVT thresholding (``evt``)
``DetectorBank``                   tiered ensemble orchestrator (``detectors``)

All members import their heavy/optional backends lazily and degrade to a
surrogate, so the module loads and scores on the light tier alone.
"""

from __future__ import annotations

from .base import Detector, RollingNormalizer
from .changepoint import (
    AdwinDetector,
    KswinDetector,
    PageHinkleyDetector,
    RupturesChangePointDetector,
)
from .deep import AutoEncoderDetector, PcaReconstructionDetector
from .detectors import DetectorBank, build_detector_bank
from .evt import DSPOT, POT, SPOT
from .matrixprofile import MatrixProfileDiscordDetector
from .ml import (
    ForecastResidualDetector,
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    LofDetector,
)
from .statistical import (
    CopodDetector,
    EcodDetector,
    EwmaControlChart,
    HbosDetector,
    RobustZDetector,
)

__all__ = [
    # base
    "Detector",
    "RollingNormalizer",
    # statistical
    "RobustZDetector",
    "EwmaControlChart",
    "HbosDetector",
    "CopodDetector",
    "EcodDetector",
    # ml / streaming
    "HalfSpaceTreesDetector",
    "IsolationForestDetector",
    "LofDetector",
    "ForecastResidualDetector",
    # change-point
    "PageHinkleyDetector",
    "AdwinDetector",
    "KswinDetector",
    "RupturesChangePointDetector",
    # matrix profile
    "MatrixProfileDiscordDetector",
    # deep / pca
    "AutoEncoderDetector",
    "PcaReconstructionDetector",
    # EVT
    "POT",
    "SPOT",
    "DSPOT",
    # bank
    "DetectorBank",
    "build_detector_bank",
]
