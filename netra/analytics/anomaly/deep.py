"""Optional deep anomaly detector — torch autoencoder (#50-#52), gracefully skipped.

Deep reconstruction models lead on multivariate correlated telemetry and subtle
precursors (research 04 §7). This module provides a light **LSTM-free dense
autoencoder** over lag-embedded windows: trained on benign telemetry, a large
reconstruction error flags an anomaly. It is the *optional-heavy* tier — ``torch``
is imported lazily under ``try/except`` and, when absent (the CPU-only demo
default), :meth:`is_available` returns ``False`` and the ensemble omits the
member. A PCA-reconstruction fallback (sklearn, light tier) is offered as
:class:`PcaReconstructionDetector` so a reconstruction-style detector is always
present even without torch.

No weights are downloaded; the AE trains locally on the supplied benign window,
so it is fully air-gap-safe.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from netra.contracts import DetectorFamily, EntityRef

from .base import Detector


class PcaReconstructionDetector(Detector):
    """PCA reconstruction-error detector (#33) — light-tier deep-style member.

    Lag-embeds the stream, fits a top-``k`` PCA on a benign window, and scores the
    reconstruction residual of each new window (a point that doesn't lie in the
    'normal' subspace reconstructs poorly). Deterministic and CPU-cheap; the
    violated-component loadings double as a 'why' signal. Always available (uses
    sklearn from the light tier); used as the torch-free reconstruction member.
    """

    method = "pca_recon"
    family = DetectorFamily.ML_UNSUPERVISED
    higher_is_anomalous = True

    def __init__(self, *args, lags: int = 8, n_components: int = 3,
                 window: int = 200, refit_every: int = 25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lags = int(lags)
        self.n_components = int(n_components)
        self.window = int(window)
        self.refit_every = int(refit_every)
        self._buf: deque[float] = deque(maxlen=self.window)
        self._pca = None
        self._mean = None
        self._since = 0
        self._fallback = False

    def _embed(self, arr: np.ndarray) -> np.ndarray:
        n = arr.size - self.lags + 1
        if n <= 0:
            return np.empty((0, self.lags))
        return np.stack([arr[i:i + self.lags] for i in range(n)])

    def _refit(self) -> None:
        arr = np.fromiter(self._buf, dtype=float)
        X = self._embed(arr)
        if X.shape[0] < self.lags + 2:
            return
        try:
            from sklearn.decomposition import PCA

            k = max(1, min(self.n_components, self.lags - 1, X.shape[0] - 1))
            self._mean = X.mean(axis=0)
            self._pca = PCA(n_components=k).fit(X - self._mean)
            self._fallback = False
        except Exception:
            self._pca = None
            self._fallback = True

    def _fit(self, series: np.ndarray) -> None:
        for v in series[-self.window:]:
            self._buf.append(float(v))
        self._refit()

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._buf.append(x)
        self._since += 1
        if self._pca is None or self._since >= self.refit_every:
            self._refit()
            self._since = 0
        arr = np.fromiter(self._buf, dtype=float)
        if self._pca is not None and not self._fallback and arr.size >= self.lags:
            w = arr[-self.lags:].reshape(1, -1) - self._mean
            try:
                recon = self._pca.inverse_transform(self._pca.transform(w))
                return float(np.linalg.norm(w - recon))
            except Exception:
                self._fallback = True
        # robust-z surrogate
        if arr.size >= 5:
            med = np.median(arr)
            mad = np.median(np.abs(arr - med)) * 1.4826
            return float(abs(x - med) / mad) if mad > 1e-12 else 0.0
        return 0.0


class AutoEncoderDetector(Detector):
    """Optional torch dense autoencoder over lag windows (#50), CPU/offline.

    Trains a small symmetric autoencoder on benign lag-embedded windows during
    :meth:`fit`; the live score is the reconstruction error of each new window.
    ``torch`` is imported lazily; if unavailable the detector transparently
    delegates to :class:`PcaReconstructionDetector` so callers get a working
    reconstruction detector either way (graceful degradation).

    Parameters
    ----------
    lags, hidden, epochs:
        Window length, bottleneck width, and training epochs (kept small for CPU).
    """

    method = "autoencoder"
    family = DetectorFamily.DEEP
    higher_is_anomalous = True

    def __init__(self, *args, lags: int = 12, hidden: int = 4, epochs: int = 60,
                 lr: float = 0.01, seed: int = 1337, **kwargs) -> None:
        self._init_args = (args, kwargs)
        super().__init__(*args, **kwargs)
        self.lags = int(lags)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self._model = None
        self._mean = 0.0
        self._std = 1.0
        self._res_scale = 1.0
        self._delegate: PcaReconstructionDetector | None = None
        self._buf: deque[float] = deque(maxlen=400)

    @staticmethod
    def backend_importable() -> bool:
        try:
            import torch  # noqa: F401

            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self.backend_importable()

    def _embed(self, arr: np.ndarray) -> np.ndarray:
        n = arr.size - self.lags + 1
        if n <= 0:
            return np.empty((0, self.lags))
        return np.stack([arr[i:i + self.lags] for i in range(n)])

    def _build_delegate(self) -> None:
        args, kwargs = self._init_args
        self._delegate = PcaReconstructionDetector(
            self.entity, self.metric, lags=self.lags,
        )

    def _fit(self, series: np.ndarray) -> None:
        for v in series:
            self._buf.append(float(v))
        if not self.is_available():
            self._build_delegate()
            self._delegate._fit(series)
            return
        try:
            import torch
            import torch.nn as nn

            torch.manual_seed(self.seed)
            X = self._embed(series.astype(float))
            if X.shape[0] < self.lags + 2:
                self._build_delegate()
                self._delegate._fit(series)
                return
            self._mean = float(X.mean())
            self._std = float(X.std()) or 1.0
            Xn = (X - self._mean) / self._std
            Xt = torch.tensor(Xn, dtype=torch.float32)

            model = nn.Sequential(
                nn.Linear(self.lags, max(self.hidden * 2, self.hidden + 2)), nn.ReLU(),
                nn.Linear(max(self.hidden * 2, self.hidden + 2), self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, max(self.hidden * 2, self.hidden + 2)), nn.ReLU(),
                nn.Linear(max(self.hidden * 2, self.hidden + 2), self.lags),
            )
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                out = model(Xt)
                loss = loss_fn(out, Xt)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                resid = (model(Xt) - Xt).pow(2).mean(dim=1).sqrt().cpu().numpy()
            self._res_scale = float(np.median(resid)) or 1.0
            self._model = model
        except Exception:
            self._model = None
            self._build_delegate()
            self._delegate._fit(series)

    def _score_one(self, value: object) -> float:
        x = float(value)
        self._buf.append(x)
        if self._model is None:
            if self._delegate is None:
                self._build_delegate()
            return self._delegate._score_one(value)
        try:
            import torch

            arr = np.fromiter(self._buf, dtype=float)
            if arr.size < self.lags:
                return 0.0
            w = (arr[-self.lags:] - self._mean) / self._std
            wt = torch.tensor(w.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                err = float((self._model(wt) - wt).pow(2).mean().sqrt().item())
            return err / (self._res_scale + 1e-9)
        except Exception:
            if self._delegate is None:
                self._build_delegate()
            return self._delegate._score_one(value)


__all__ = ["AutoEncoderDetector", "PcaReconstructionDetector"]
