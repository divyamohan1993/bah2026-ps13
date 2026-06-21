"""Probability calibration for fused risk — Platt + isotonic (#69).

A raw fused anomaly score is *monotone* in risk but is **not** a probability: a
score of 0.8 does not mean "80% chance this is a real fault". Calibration learns
the map ``score -> P(fault | score)`` from the labelled fault scenarios
(:class:`~netra.contracts.ScenarioLabel`) so the copilot's stated confidence is
*honest* — reliability-diagram validated rather than asserted (research 04 §13).

Two standard post-hoc calibrators, the same pair scikit-learn's
``CalibratedClassifierCV`` offers:

  * **Platt scaling** — fit a 1-D logistic regression ``sigmoid(a*score + b)``.
    Parametric, data-thrifty, ideal when the score-vs-label relationship is a
    smooth S-curve. The workhorse for the small labelled set we have.
  * **Isotonic regression** — fit a free-form non-decreasing step function. More
    flexible (can model any monotone miscalibration) but needs more labels to
    avoid overfitting; we fall back to Platt when data is scarce.

Both have a **pure-NumPy fallback** so the layer calibrates even in the most
stripped air-gapped bundle (no scikit-learn): a Newton-fit logistic for Platt and
the Pool-Adjacent-Violators algorithm (PAVA) for isotonic. ``sklearn`` is used
when present (better-tested, identical interface), guarded behind ``try/except``.

Typical use::

    cal = ProbabilityCalibrator(method="platt").fit(scores, labels)
    p = cal.transform(0.83)          # -> calibrated P(fault) in [0,1]

The fusion layer (:mod:`~netra.analytics.fusion.fuse`) holds an (optionally
pre-fit) calibrator and uses it to turn the raw weighted-agreement score into the
``risk_score`` / ``calibrated_confidence`` it writes onto a ``FusedRisk``.
"""

from __future__ import annotations

import numpy as np

try:  # optional, better-tested backend
    from sklearn.isotonic import IsotonicRegression as _SkIsotonic
    from sklearn.linear_model import LogisticRegression as _SkLogistic

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only in a no-sklearn bundle
    _SkIsotonic = None
    _SkLogistic = None
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# pure-numpy fallbacks (so calibration works with zero optional deps)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _platt_fit_numpy(scores: np.ndarray, labels: np.ndarray,
                     iters: int = 100, l2: float = 1e-6) -> tuple[float, float]:
    """Newton-Raphson logistic fit ``sigmoid(a*x + b)`` → ``(a, b)``.

    Minimises the (lightly L2-regularised) cross-entropy. Closed-form Newton
    steps on the 2×2 Hessian converge in a handful of iterations on the small
    labelled set; the L2 ridge keeps the Hessian invertible under perfect
    separation (which is common when the score already separates the classes).
    """
    x = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    a, b = 1.0, 0.0
    X = np.column_stack([x, np.ones_like(x)])     # design matrix (n, 2)
    ridge = l2 * np.eye(2)
    for _ in range(iters):
        p = _sigmoid(a * x + b)
        w = np.clip(p * (1.0 - p), 1e-9, None)    # IRLS weights
        grad = X.T @ (p - y) + l2 * np.array([a, b])
        hess = (X * w[:, None]).T @ X + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:             # pragma: no cover
            break
        a -= float(step[0])
        b -= float(step[1])
        if np.linalg.norm(step) < 1e-9:
            break
    return float(a), float(b)


def _isotonic_fit_numpy(scores: np.ndarray, labels: np.ndarray,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators isotonic fit → ``(x_thresholds, y_values)``.

    Returns the breakpoints of a non-decreasing step function fit to
    ``(score, label)`` by PAVA (the textbook O(n) isotonic regression). The
    fitted function is later evaluated by interpolation in :meth:`transform`.
    """
    x = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order].astype(float)
    # PAVA: maintain blocks of (sum, weight); merge while monotonicity violated
    vals: list[float] = []
    weights: list[float] = []
    xs_block: list[float] = []
    for xi, yi in zip(xs, ys, strict=False):
        vals.append(yi)
        weights.append(1.0)
        xs_block.append(xi)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = weights[-2] + weights[-1]
            v = (vals[-2] * weights[-2] + vals[-1] * weights[-1]) / w
            vals[-2] = v
            weights[-2] = w
            xs_block[-2] = xs_block[-1]   # right edge of the merged block
            vals.pop()
            weights.pop()
            xs_block.pop()
    # expand block means back over their x-extent (use right edges as knots)
    knots_x = np.asarray(xs_block, dtype=float)
    knots_y = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    return knots_x, knots_y


# ---------------------------------------------------------------------------
# public calibrator
# ---------------------------------------------------------------------------


class ProbabilityCalibrator:
    """Map a raw monotone score to a calibrated probability in [0,1].

    Parameters
    ----------
    method:
        ``"platt"`` (logistic / sigmoid) or ``"isotonic"`` (free-form monotone).
        Unknown values fall back to ``"platt"``.
    min_isotonic:
        Minimum number of labelled points required to fit isotonic; below this it
        silently downgrades to Platt (isotonic overfits on tiny samples).

    Notes
    -----
    The calibrator is *identity* (``transform`` returns its input, clipped to
    [0,1]) until :meth:`fit` succeeds, so an un-trained fusion layer still emits a
    sensible score rather than erroring — it just isn't probability-calibrated yet.
    """

    def __init__(self, method: str = "platt", *, min_isotonic: int = 10) -> None:
        m = str(method).lower()
        self.method = m if m in ("platt", "isotonic") else "platt"
        self.min_isotonic = int(min_isotonic)
        self._fitted = False
        # platt params / sklearn estimator / isotonic knots
        self._a = 1.0
        self._b = 0.0
        self._sk = None
        self._iso_x: np.ndarray | None = None
        self._iso_y: np.ndarray | None = None
        self._effective_method = self.method

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, scores: object, labels: object) -> ProbabilityCalibrator:
        """Fit the calibrator on labelled ``(score, label)`` pairs.

        ``labels`` are 0/1 (e.g. 1 inside a :class:`ScenarioLabel` fault window,
        0 on benign baseline). Degenerate inputs (one class only, <2 points) leave
        the calibrator as identity — a calibrator can't learn a map from a single
        class, and silently passing through is safer than fabricating one.
        """
        x = np.asarray(list(scores), dtype=float).ravel()
        y = np.asarray(list(labels), dtype=float).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2 or np.unique(y).size < 2:
            self._fitted = False
            return self

        method = self.method
        if method == "isotonic" and x.size < self.min_isotonic:
            method = "platt"                 # not enough data for a free-form fit
        self._effective_method = method

        if method == "isotonic":
            if _HAS_SKLEARN:
                iso = _SkIsotonic(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(x, y)
                self._sk = iso
            else:
                self._iso_x, self._iso_y = _isotonic_fit_numpy(x, y)
        else:  # platt
            if _HAS_SKLEARN:
                # LogisticRegression on the single score feature == Platt scaling.
                lr = _SkLogistic(C=1e6, solver="lbfgs")
                lr.fit(x.reshape(-1, 1), y.astype(int))
                self._sk = lr
            else:
                self._a, self._b = _platt_fit_numpy(x, y)
        self._fitted = True
        return self

    def transform(self, score: float | object) -> float | np.ndarray:
        """Map a score (or array of scores) to a calibrated probability in [0,1]."""
        scalar = np.isscalar(score)
        x = np.asarray([score] if scalar else list(score), dtype=float).ravel()
        if not self._fitted:
            out = np.clip(x, 0.0, 1.0)        # identity until trained
            return float(out[0]) if scalar else out

        if self._effective_method == "isotonic":
            if self._sk is not None:
                out = np.clip(self._sk.predict(x), 0.0, 1.0)
            else:
                assert self._iso_x is not None and self._iso_y is not None
                out = np.interp(x, self._iso_x, self._iso_y,
                                left=float(self._iso_y[0]),
                                right=float(self._iso_y[-1]))
                out = np.clip(out, 0.0, 1.0)
        else:  # platt
            if self._sk is not None:
                out = np.clip(self._sk.predict_proba(x.reshape(-1, 1))[:, 1],
                              0.0, 1.0)
            else:
                out = np.clip(_sigmoid(self._a * x + self._b), 0.0, 1.0)
        return float(out[0]) if scalar else out

    # convenience -----------------------------------------------------------

    def __call__(self, score: float | object) -> float | np.ndarray:
        return self.transform(score)


__all__ = ["ProbabilityCalibrator"]
