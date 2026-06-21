"""Common anomaly-detection abstractions — the ``Detector`` ABC + helpers.

Every detector in NETRA — robust-z/MAD, EWMA control chart, Half-Space Trees,
Isolation Forest, HBOS/COPOD/ECOD, LOF, change-point (PELT/BinSeg, ADWIN/Page-
Hinkley), forecast-residual, matrix-profile discord, and the optional deep AE —
implements the same small streaming-friendly interface so the fusion layer can
treat them uniformly and weight cross-family agreement:

    det.fit(reference_series)          # optional warm-up on benign data
    det.update(value | featuredict)    # fold one sample, get its AnomalyScore
    det.score_batch(series)            # convenience: score a whole array

Each call yields a :class:`~netra.contracts.AnomalyScore` carrying the raw
method-specific ``score`` **and** a ``normalized_score`` in [0,1] (the comparable
value fusion consumes) plus the detector's own ``is_anomaly`` decision — made
against an adaptive threshold (often EVT/SPOT, see :mod:`~.evt`) rather than a
hand-set one. ``normalized_score`` is produced by a rolling reference normaliser
(:class:`RollingNormalizer`) so detectors of wildly different score scales become
directly comparable.

This module imports only numpy + the contracts, so it always loads on the
light/offline tier; ML/deep members import their backends lazily and guarded.
"""

from __future__ import annotations

import abc
from collections import deque
from datetime import UTC, datetime

import numpy as np

from netra.contracts import AnomalyScore, DetectorFamily, EntityRef


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class RollingNormalizer:
    """Map an arbitrary raw detector score to [0,1] over a rolling reference.

    Detectors emit incomparable raw scores; fusion needs a common scale. This
    keeps a bounded window of recent raw scores and converts a new score to a
    rolling percentile-rank (robust, distribution-free) blended toward a robust
    z-based squashing, both in [0,1]. Rank handles heavy tails; the z-term keeps
    it responsive before the window fills. Higher = more anomalous.
    """

    def __init__(self, window: int = 200, higher_is_anomalous: bool = True) -> None:
        self.window = int(window)
        self.higher = bool(higher_is_anomalous)
        self._buf: deque[float] = deque(maxlen=self.window)

    def normalize(self, raw: float, *, learn: bool = True) -> float:
        x = float(raw)
        if not self.higher:
            x = -x
        buf = self._buf
        if len(buf) >= 8:
            arr = np.fromiter(buf, dtype=float)
            # percentile rank of x within the reference window
            rank = float(np.mean(arr <= x))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            if mad > 1e-12:
                z = (x - med) / mad
                zsquash = 1.0 / (1.0 + np.exp(-(z - 2.5)))   # ~0 below 2.5σ, ->1 above
            else:
                zsquash = 1.0 if x > med else 0.0
            norm = float(np.clip(0.5 * rank + 0.5 * zsquash, 0.0, 1.0))
        else:
            norm = 0.0
        if learn:
            buf.append(x)
        return norm


class Detector(abc.ABC):
    """Abstract base class for all NETRA anomaly detectors.

    Subclasses implement :meth:`_score_one` (raw score for one sample) and may
    override :meth:`_fit`. The base wires in entity/metric bookkeeping, the
    rolling normaliser, and the :class:`AnomalyScore` assembly. ``higher_is_anomalous``
    tells the normaliser the polarity of the raw score.

    The default decision rule flags an anomaly when ``normalized_score`` exceeds
    ``norm_threshold`` (0.8 by default), which corresponds to a high rolling
    percentile — an adaptive, per-stream cutoff. Detectors with a native /
    EVT-derived threshold override :meth:`_decide`.
    """

    method: str = "detector"
    family: DetectorFamily = DetectorFamily.STATISTICAL
    higher_is_anomalous: bool = True

    def __init__(
        self,
        entity: EntityRef,
        metric: str,
        *,
        norm_window: int = 200,
        norm_threshold: float = 0.8,
    ) -> None:
        self.entity = entity
        self.metric = metric
        self.norm_threshold = float(norm_threshold)
        self._normalizer = RollingNormalizer(
            window=norm_window, higher_is_anomalous=self.higher_is_anomalous
        )
        self._fitted = False

    # -- public API ---------------------------------------------------------

    def fit(self, reference: object) -> Detector:
        """Warm up on a benign reference series (optional for most detectors)."""
        series = np.asarray(list(reference), dtype=float).ravel()
        series = series[np.isfinite(series)]
        self._fit(series)
        # prime the normaliser with reference raw scores so early live scores
        # are already comparable
        for v in series:
            try:
                raw = self._score_one(float(v))
                self._normalizer.normalize(raw, learn=True)
            except Exception:
                pass
        self._fitted = True
        return self

    def update(self, value: object, timestamp: datetime | None = None) -> AnomalyScore:
        """Fold one sample and return its :class:`AnomalyScore`."""
        raw = self._score_one(value)
        norm = self._normalizer.normalize(raw, learn=True)
        is_anom, thr = self._decide(raw, norm)
        return AnomalyScore(
            entity=self.entity,
            metric=self.metric,
            timestamp=timestamp or _utcnow(),
            method=self.method,
            family=self.family,
            score=float(raw),
            normalized_score=float(norm),
            is_anomaly=bool(is_anom),
            threshold=thr,
        )

    def score_batch(self, series: object,
                    timestamps: list[datetime] | None = None) -> list[AnomalyScore]:
        """Score a whole 1-D series, returning one :class:`AnomalyScore` per point."""
        arr = np.asarray(list(series), dtype=float).ravel()
        out: list[AnomalyScore] = []
        for i, v in enumerate(arr):
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else None
            out.append(self.update(float(v), timestamp=ts))
        return out

    # -- to implement / override -------------------------------------------

    def _fit(self, series: np.ndarray) -> None:  # noqa: B027 — optional hook, default no-op by design
        """Optional warm-up; default no-op (streaming detectors learn online)."""

    @abc.abstractmethod
    def _score_one(self, value: object) -> float:
        """Return the raw (method-specific) anomaly score for one sample."""

    def _decide(self, raw: float, norm: float) -> tuple[bool, float | None]:
        """Default decision: flag when the normalised score clears the cutoff.

        Returns ``(is_anomaly, threshold)``. The threshold reported is the
        *normalised* cutoff (adaptive by construction); detectors that compute a
        raw EVT/SPOT threshold override this and report that instead.
        """
        return norm >= self.norm_threshold, self.norm_threshold


__all__ = ["Detector", "RollingNormalizer"]
