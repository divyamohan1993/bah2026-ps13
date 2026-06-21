"""Score normalisation for fusion (#67) — make detectors comparable.

Detectors emit incomparable raw scores (an Isolation-Forest path length, a COPOD
tail probability, a Page-Hinkley pulse). Before they can be combined, fusion maps
each onto a common [0,1] scale where higher = more anomalous (research 04 §10).
This module provides the normalisers:

  * :func:`minmax` / :func:`zscore` / :func:`rank` — batch normalisers over a
    reference array (the PyOD ``standardizer`` family).
  * :func:`unify` — convert a z-score to a probability via the Gaussian CDF
    (PyOD's "unification"), giving a calibrated-ish [0,1].
  * :class:`OnlineScoreNormalizer` — streaming per-method normaliser keeping a
    bounded reference window; what fusion uses live so a detector's score becomes
    comparable as the stream evolves.

The :class:`~netra.contracts.AnomalyScore.normalized_score` field is already a
[0,1] value each detector produces, so fusion can use it directly; these helpers
are for (re)normalising raw scores or pooling across a reference when needed.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def minmax(scores: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Min-max scale ``scores`` into [0,1] using ``reference`` (or itself)."""
    s = np.asarray(scores, dtype=float)
    ref = s if reference is None else np.asarray(reference, dtype=float)
    lo, hi = float(np.min(ref)), float(np.max(ref))
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def zscore(scores: np.ndarray, reference: np.ndarray | None = None,
           robust: bool = True) -> np.ndarray:
    """Standardise ``scores`` against ``reference`` (robust median/MAD by default)."""
    s = np.asarray(scores, dtype=float)
    ref = s if reference is None else np.asarray(reference, dtype=float)
    if robust:
        center = float(np.median(ref))
        scale = float(np.median(np.abs(ref - center))) * 1.4826
    else:
        center = float(np.mean(ref))
        scale = float(np.std(ref))
    if scale < 1e-12:
        return np.zeros_like(s)
    return (s - center) / scale


def unify(scores: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """PyOD-style 'unification': z-score then Gaussian CDF → probability in [0,1]."""
    from scipy.stats import norm

    z = zscore(scores, reference, robust=True)
    return np.clip(norm.cdf(z), 0.0, 1.0)


def rank(scores: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Percentile-rank of each score within ``reference`` (distribution-free)."""
    s = np.asarray(scores, dtype=float)
    ref = s if reference is None else np.asarray(reference, dtype=float)
    if ref.size == 0:
        return np.zeros_like(s)
    ref_sorted = np.sort(ref)
    idx = np.searchsorted(ref_sorted, s, side="right")
    return np.clip(idx / ref_sorted.size, 0.0, 1.0)


class OnlineScoreNormalizer:
    """Streaming [0,1] normaliser for one method's raw score.

    Maintains a bounded reference window of recent raw scores and converts a new
    score to a robust, distribution-free value: the mean of its rolling
    percentile-rank and a Gaussian-CDF "unification" of its robust z-score. Higher
    = more anomalous. Used by the fusion layer to bring heterogeneous detector
    scores onto one scale live.
    """

    def __init__(self, window: int = 300) -> None:
        self.window = int(window)
        self._buf: deque[float] = deque(maxlen=self.window)

    def normalize(self, raw: float, *, learn: bool = True) -> float:
        from scipy.stats import norm

        x = float(raw)
        if len(self._buf) >= 8:
            ref = np.fromiter(self._buf, dtype=float)
            r = float(np.mean(ref <= x))
            center = float(np.median(ref))
            scale = float(np.median(np.abs(ref - center))) * 1.4826
            u = float(norm.cdf((x - center) / scale)) if scale > 1e-12 else (1.0 if x > center else 0.0)
            out = float(np.clip(0.5 * r + 0.5 * u, 0.0, 1.0))
        else:
            out = 0.0
        if learn:
            self._buf.append(x)
        return out


__all__ = ["minmax", "zscore", "unify", "rank", "OnlineScoreNormalizer"]
