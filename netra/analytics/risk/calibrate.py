"""Score calibration — Platt scaling (logistic) + isotonic option (research 07 A2.2).

A raw model "confidence" of 0.8 is meaningless unless it means "happens 80% of the
time". This module maps raw risk scores → calibrated probabilities so the copilot's
stated confidence is honest (which also supports the grounded / no-hallucination
criterion).

  * **Platt scaling** — fit a sigmoid ``1/(1+exp(-(a·s+b)))`` to (score, label)
    pairs. Best for sigmoid-shaped distortion, few parameters, robust on the small
    labelled fault-injection sets we have. **Default.**
  * **Isotonic regression** — corrects any monotonic distortion; with ≥1000
    calibration points it is ≥ Platt, but overfits on small sets — offered as an
    upgrade.

Both are backed by scikit-learn. When sklearn is unavailable, a pure-NumPy
Platt fit (gradient descent on log-loss) and a NumPy isotonic (pool-adjacent-
violators) keep the module working offline with zero heavy deps. Reliability
metrics (Brier score, ECE) are provided as calibration-quality evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # sklearn is a core dep; guard so the module imports on a bare env.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore[assignment]
    IsotonicRegression = None  # type: ignore[assignment]
    _HAVE_SKLEARN = False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


@dataclass
class _PlattParams:
    a: float
    b: float


class RiskCalibrator:
    """Calibrate raw risk scores → probabilities (Platt default, isotonic option).

    Usage::

        cal = RiskCalibrator(method="platt").fit(scores, labels)
        p = cal.transform(0.42)          # one score → calibrated probability
        ps = cal.transform([0.1, 0.9])   # batch

    Unfitted, ``transform`` is the identity (clipped to [0,1]) so the pipeline runs
    before any labelled data exists — graceful degradation.
    """

    def __init__(self, method: str = "platt") -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method
        self._fitted = False
        self._platt: _PlattParams | None = None
        self._sk_model = None
        self._iso_x: np.ndarray | None = None
        self._iso_y: np.ndarray | None = None

    # -- fit ----------------------------------------------------------------
    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> RiskCalibrator:
        """Fit the calibrator on raw scores and binary outcome labels (0/1)."""
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if s.size != y.size:
            raise ValueError("scores and labels must be the same length")
        if s.size == 0:
            return self
        # need both classes present to fit a meaningful calibrator.
        if len(np.unique(y)) < 2:
            self._fitted = False
            return self

        if self.method == "platt":
            self._fit_platt(s, y)
        else:
            self._fit_isotonic(s, y)
        self._fitted = True
        return self

    def _fit_platt(self, s: np.ndarray, y: np.ndarray) -> None:
        if _HAVE_SKLEARN:
            self._sk_model = LogisticRegression(C=1e6, solver="lbfgs")
            self._sk_model.fit(s.reshape(-1, 1), y.astype(int))
        else:  # pragma: no cover - exercised only without sklearn
            self._platt = _platt_fit_numpy(s, y)

    def _fit_isotonic(self, s: np.ndarray, y: np.ndarray) -> None:
        if _HAVE_SKLEARN:
            self._sk_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._sk_model.fit(s, y)
        else:  # pragma: no cover
            self._iso_x, self._iso_y = _isotonic_fit_numpy(s, y)

    # -- transform ----------------------------------------------------------
    def transform(self, scores: float | Sequence[float]) -> float | list[float]:
        """Map raw score(s) → calibrated probability(ies) in [0,1]."""
        scalar = np.isscalar(scores)
        s = np.asarray([scores] if scalar else scores, dtype=float).reshape(-1)
        if not self._fitted:
            out = np.clip(s, 0.0, 1.0)
        elif self.method == "platt":
            out = self._transform_platt(s)
        else:
            out = self._transform_isotonic(s)
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar else [float(v) for v in out]

    def _transform_platt(self, s: np.ndarray) -> np.ndarray:
        if self._sk_model is not None:
            return self._sk_model.predict_proba(s.reshape(-1, 1))[:, 1]
        p = self._platt
        assert p is not None
        return _sigmoid(p.a * s + p.b)

    def _transform_isotonic(self, s: np.ndarray) -> np.ndarray:
        if self._sk_model is not None:
            return self._sk_model.predict(s)
        assert self._iso_x is not None and self._iso_y is not None
        return np.interp(s, self._iso_x, self._iso_y)

    @property
    def fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Pure-NumPy fallbacks (used only when sklearn is unavailable).
# ---------------------------------------------------------------------------
def _platt_fit_numpy(
    s: np.ndarray, y: np.ndarray, *, lr: float = 0.1, iters: int = 2000
) -> _PlattParams:  # pragma: no cover
    a, b = 1.0, 0.0
    n = len(s)
    for _ in range(iters):
        p = _sigmoid(a * s + b)
        grad_a = float(np.dot(p - y, s) / n)
        grad_b = float(np.sum(p - y) / n)
        a -= lr * grad_a
        b -= lr * grad_b
    return _PlattParams(a=a, b=b)


def _isotonic_fit_numpy(
    s: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    order = np.argsort(s)
    xs = s[order]
    ys = y[order].astype(float)
    # pool-adjacent-violators
    weights = np.ones_like(ys)
    i = 0
    blocks = [[ys[k], weights[k], xs[k]] for k in range(len(ys))]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (
                    blocks[i][1] + blocks[i + 1][1]
                )
                blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2]]
                del blocks[i + 1]
                merged = True
            else:
                i += 1
    out_x = np.array([blk[2] for blk in blocks])
    out_y = np.array([blk[0] for blk in blocks])
    return out_x, out_y


# ---------------------------------------------------------------------------
# Calibration-quality metrics (reliability evidence).
# ---------------------------------------------------------------------------
def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between predicted probabilities and outcomes (lower=better)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> float:
    """Expected Calibration Error: |confidence − accuracy| averaged over bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for k in range(n_bins):
        lo, hi = bins[k], bins[k + 1]
        mask = (p > lo) & (p <= hi) if k > 0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / n) * abs(conf - acc)
    return float(ece)


def reliability_diagram(
    probs: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin (mean_confidence, empirical_accuracy, count) for a reliability plot."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for k in range(n_bins):
        lo, hi = bins[k], bins[k + 1]
        mask = (p > lo) & (p <= hi) if k > 0 else (p >= lo) & (p <= hi)
        cnt = int(np.sum(mask))
        out.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_confidence": float(np.mean(p[mask])) if cnt else 0.0,
                "empirical_accuracy": float(np.mean(y[mask])) if cnt else 0.0,
                "count": cnt,
            }
        )
    return out


__all__ = [
    "RiskCalibrator",
    "brier_score",
    "expected_calibration_error",
    "reliability_diagram",
]
