"""netra.analytics.fusion — score fusion + calibration -> FusedRisk (WS3).

Combines the many detectors (score-normalisation + weighted-agreement across
*independent* families + optional stacking) into one calibrated
``netra.contracts.FusedRisk`` (recording ``MethodWeight`` provenance for every
contributing method) and attaches the ``TimeToImpact``. Calibration (Platt/
isotonic) is trained on the labelled ``ScenarioLabel`` fault scenarios so the
copilot's confidence is honest.

Public surface
--------------
``RiskFuser``                  weighted-agreement fusion -> ``FusedRisk`` (``fuse``)
``ProbabilityCalibrator``      Platt/isotonic calibration (``calibrate``)
``list_methods`` / ...         the deployed-method census (``registry``)
``minmax`` / ``zscore`` / ...  score normalisers (``normalize``)
``POT`` / ``SPOT`` / ``DSPOT`` EVT adaptive thresholds (``evt``)

Remember: ``FusedRisk.risk_score>0`` MUST carry ``contributing_methods`` —
``RiskFuser`` enforces this by construction.
"""

from __future__ import annotations

from .calibrate import ProbabilityCalibrator
from .evt import DSPOT, POT, SPOT, ScoreStreamThresholder
from .fuse import RiskFuser
from .normalize import OnlineScoreNormalizer, minmax, rank, unify, zscore
from .registry import (
    FAMILIES,
    MethodInfo,
    count_by_family,
    list_methods,
    method_count,
    method_names,
)

__all__ = [
    # fusion
    "RiskFuser",
    # calibration
    "ProbabilityCalibrator",
    # registry
    "MethodInfo",
    "FAMILIES",
    "list_methods",
    "count_by_family",
    "method_count",
    "method_names",
    # normalisation
    "minmax",
    "zscore",
    "unify",
    "rank",
    "OnlineScoreNormalizer",
    # EVT thresholds
    "POT",
    "SPOT",
    "DSPOT",
    "ScoreStreamThresholder",
]
