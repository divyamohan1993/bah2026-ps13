"""Tiered detector bank — assemble & run the whole anomaly ensemble (#19-#60).

Convenience orchestration over the individual detector families so callers (the
fusion layer, the streaming engine, tests) can spin up the *deployed subset*
(research 04 §12) in one call and score a stream through all of them at once.

Tiers (independent inductive biases so they cover each other's blind spots):

  * **Tier-1 streaming** — robust-z, EWMA control chart, Half-Space Trees,
    Page-Hinkley, ADWIN, forecast-residual.
  * **Tier-2 batch ML** — Isolation Forest, COPOD, ECOD, HBOS, LOF, PCA-recon,
    matrix-profile discord.
  * **Tier-3 change-point / deep** — KSWIN, ruptures PELT; optional torch AE.

:class:`DetectorBank` returns, per sample, the list of every detector's
:class:`~netra.contracts.AnomalyScore`. The fusion layer combines those into a
:class:`~netra.contracts.FusedRisk`. Detectors whose backends are missing degrade
to surrogates inside themselves, so the bank always produces a full score set on
the light tier.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from netra.contracts import AnomalyScore, DetectorFamily, EntityRef

from .base import Detector
from .changepoint import (
    AdwinDetector,
    KswinDetector,
    PageHinkleyDetector,
    RupturesChangePointDetector,
)
from .deep import AutoEncoderDetector, PcaReconstructionDetector
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


def build_detector_bank(
    entity: EntityRef,
    metric: str,
    *,
    tier1: bool = True,
    tier2: bool = True,
    tier3: bool = True,
    enable_deep: bool = False,
    include_forecast_residual: bool = False,
) -> list[Detector]:
    """Instantiate the deployed detector subset for one entity+metric.

    Parameters
    ----------
    tier1, tier2, tier3:
        Toggle whole tiers (all on by default).
    enable_deep:
        Include the optional torch autoencoder (skipped unless torch present).
    include_forecast_residual:
        Include the predict-then-flag detector. Off by default because it needs
        (actual, predicted) pairs rather than bare values — enable it when you
        will drive it via :meth:`ForecastResidualDetector.update_residual`.
    """
    bank: list[Detector] = []
    if tier1:
        bank += [
            RobustZDetector(entity, metric),
            EwmaControlChart(entity, metric),
            HalfSpaceTreesDetector(entity, metric),
            PageHinkleyDetector(entity, metric),
            AdwinDetector(entity, metric),
        ]
        if include_forecast_residual:
            bank.append(ForecastResidualDetector(entity, metric))
    if tier2:
        bank += [
            IsolationForestDetector(entity, metric),
            CopodDetector(entity, metric),
            EcodDetector(entity, metric),
            HbosDetector(entity, metric),
            LofDetector(entity, metric),
            PcaReconstructionDetector(entity, metric),
            MatrixProfileDiscordDetector(entity, metric),
        ]
    if tier3:
        bank += [
            KswinDetector(entity, metric),
            RupturesChangePointDetector(entity, metric),
        ]
        if enable_deep:
            ae = AutoEncoderDetector(entity, metric)
            if ae.is_available():
                bank.append(ae)
    return bank


class DetectorBank:
    """Run a tiered detector ensemble over a stream and collect AnomalyScores.

    Wraps :func:`build_detector_bank`. Use :meth:`warmup` to fit the batch members
    on a benign reference window, then :meth:`update` per live sample (returns the
    list of every detector's :class:`AnomalyScore` for that instant), or
    :meth:`score_series` to push a whole array through and get the per-step lists.
    """

    def __init__(self, entity: EntityRef, metric: str, **kwargs) -> None:
        self.entity = entity
        self.metric = metric
        self.detectors: list[Detector] = build_detector_bank(entity, metric, **kwargs)

    @property
    def methods(self) -> list[str]:
        """Ids of the detectors in the bank (post-construction)."""
        return [d.method for d in self.detectors]

    @property
    def families(self) -> set[DetectorFamily]:
        """Distinct detector families represented in the bank."""
        return {d.family for d in self.detectors}

    def warmup(self, reference: object) -> DetectorBank:
        """Fit/prime every detector on a benign reference series."""
        for d in self.detectors:
            try:
                d.fit(reference)
            except Exception:
                pass
        return self

    def update(self, value: object,
               timestamp: datetime | None = None) -> list[AnomalyScore]:
        """Score one sample through every detector; return their AnomalyScores."""
        out: list[AnomalyScore] = []
        for d in self.detectors:
            try:
                out.append(d.update(value, timestamp=timestamp))
            except Exception:
                continue
        return out

    def score_series(self, series: object,
                     timestamps: list[datetime] | None = None,
                     ) -> list[list[AnomalyScore]]:
        """Push a whole 1-D series through the bank; per-step score lists."""
        arr = np.asarray(list(series), dtype=float).ravel()
        results: list[list[AnomalyScore]] = []
        for i, v in enumerate(arr):
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else None
            results.append(self.update(float(v), timestamp=ts))
        return results


__all__ = ["DetectorBank", "build_detector_bank"]
