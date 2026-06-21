"""Optional foundation-model forecaster (Chronos-Bolt), gracefully skipped.

Open-weight, zero-shot time-series foundation models give strong forecasts with
no training — ideal for newly-provisioned interfaces/tunnels that lack history,
and as a diverse ensemble member (research 03 §3). Amazon **Chronos-Bolt** is the
air-gap pick: Apache-2.0, CPU-runnable, and loadable from a **local directory**
(``HF_HUB_OFFLINE=1``) so no runtime network is needed.

This wrapper is deliberately *optional-heavy*: the ``chronos`` package and its
torch backend are imported lazily under ``try/except``. If they (or the local
weights) are absent — the default on the CPU-only demo box — :meth:`is_available`
returns ``False`` and the ensemble simply omits this member. It never blocks the
light tier.

Air-gap usage::

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    f = ChronosBoltForecaster(entity, metric, model_path="/opt/models/chronos-bolt-base")
    if f.is_available():
        fc = f.fit(history).forecast(steps=12, step_seconds=60)
"""

from __future__ import annotations

import os

import numpy as np

from netra.contracts import DetectorFamily

from .base import Forecaster, residual_std


class ChronosBoltForecaster(Forecaster):
    """Zero-shot Chronos-Bolt foundation forecaster (optional, CPU, offline).

    Parameters
    ----------
    model_path:
        Local directory or HF id of the Chronos-Bolt weights. Defaults to the
        ``NETRA_CHRONOS_PATH`` env var, else ``"amazon/chronos-bolt-base"`` (which
        only resolves offline if pre-bundled). Keep weights local for air-gap.
    device:
        ``"cpu"`` (default) or ``"cuda"`` on a GPU appliance.

    Notes
    -----
    The class imports nothing heavy at construction; :meth:`is_available` probes
    the optional backend so callers can feature-flag cleanly. Quantile bounds use
    Chronos's native quantile output when present, else a residual band.
    """

    method = "chronos_bolt"
    family = DetectorFamily.FORECAST
    min_history = 8

    def __init__(self, *args, model_path: str | None = None,
                 device: str = "cpu", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_path = (
            model_path
            or os.environ.get("NETRA_CHRONOS_PATH")
            or "amazon/chronos-bolt-base"
        )
        self.device = device
        self._pipeline = None
        self._sigma = 1.0

    @staticmethod
    def backend_importable() -> bool:
        """True if the ``chronos`` package imports (does not load weights)."""
        try:
            import chronos  # noqa: F401

            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        """True if the backend imports *and* the weights load (lazy, cached).

        Loads the pipeline on first call; returns ``False`` (never raises) if the
        package or local weights are missing, so the ensemble can skip silently.
        """
        if self._pipeline is not None:
            return True
        try:
            from chronos import BaseChronosPipeline

            # honour offline mode for air-gap safety
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_path, device_map=self.device
            )
            return True
        except Exception:
            self._pipeline = None
            return False

    def _fit(self, series: np.ndarray) -> None:
        # Zero-shot: nothing to train. We only stash residual scale for the band
        # and verify the backend is loadable.
        self._sigma = residual_std(series)
        if not self.is_available():
            raise RuntimeError(
                "ChronosBoltForecaster unavailable (chronos/torch or local "
                "weights absent) — ensemble should skip this member"
            )

    def _predict(self, steps: int):
        import torch  # only reachable when is_available() succeeded

        assert self._history is not None and self._pipeline is not None
        ctx = torch.tensor(self._history, dtype=torch.float32)
        ql, qu = self.quantile_lower, self.quantile_upper
        try:
            quantiles, mean = self._pipeline.predict_quantiles(
                context=ctx, prediction_length=steps,
                quantile_levels=[ql, 0.5, qu],
            )
            q = quantiles[0].cpu().numpy()         # (steps, 3)
            lower, point, upper = q[:, 0], q[:, 1], q[:, 2]
        except Exception:
            fc = self._pipeline.predict(context=ctx, prediction_length=steps)
            arr = np.asarray(fc[0].cpu().numpy() if hasattr(fc[0], "cpu") else fc[0],
                             dtype=float)
            point = np.median(arr, axis=0) if arr.ndim > 1 else arr
            lower, upper = self._symmetric_band(point, self._sigma, grow=1.0)
        point = np.asarray(point, dtype=float).ravel()[:steps]
        lower = np.asarray(lower, dtype=float).ravel()[:steps]
        upper = np.asarray(upper, dtype=float).ravel()[:steps]
        return point, np.minimum(lower, point), np.maximum(upper, point)


__all__ = ["ChronosBoltForecaster"]
