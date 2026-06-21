"""EVT/POT adaptive thresholding for fusion — SPOT / DSPOT (#68).

The fusion layer applies Extreme-Value-Theory thresholds to *score streams* (each
detector's normalised score and the fused ensemble score) so even the combined
risk gets a principled, self-calibrating cutoff rather than a hand-set one — one
risk knob ``q`` controls the false-positive rate fleet-wide (research 04 §11).

The implementation lives in :mod:`netra.analytics.anomaly.evt` (POT/SPOT/DSPOT,
GPD tail fit, pure NumPy/SciPy). It is re-exported here so fusion code can import
it from its own subpackage, and a small :class:`ScoreStreamThresholder` adapter is
added to run DSPOT over a fused-score stream and emit an adaptive decision.
"""

from __future__ import annotations

import numpy as np

from netra.analytics.anomaly.evt import DSPOT, POT, SPOT


class ScoreStreamThresholder:
    """Adaptive EVT threshold over a (fused) anomaly-score stream.

    Wraps :class:`~netra.analytics.anomaly.evt.DSPOT` to threshold a stream of
    fused risk/anomaly scores: warm up on benign scores, then :meth:`update` each
    new score to learn whether it is an extreme (anomalous) value under the
    adaptive GPD tail. Because the fused score is bounded in [0,1], DSPOT's
    drift-tracking keeps the cutoff sensible even as the baseline score level
    shifts.

    Parameters
    ----------
    q:
        Target tail probability (false-alarm risk) for the EVT threshold.
    depth:
        Drift moving-average window for DSPOT.
    """

    def __init__(self, q: float = 1e-3, depth: int = 20,
                 init_quantile: float = 0.9) -> None:
        self.q = float(q)
        self._dspot = DSPOT(q=q, init_quantile=init_quantile, depth=depth)
        self._initialized = False

    def warmup(self, scores: object) -> ScoreStreamThresholder:
        """Seed the EVT tail from a window of benign fused scores."""
        arr = np.asarray(list(scores), dtype=float).ravel()
        self._dspot.initialize(arr)
        self._initialized = True
        return self

    def update(self, score: float) -> tuple[bool, float]:
        """Process one fused score → ``(is_extreme, raw_threshold)``."""
        if not self._initialized:
            # permissive until warmed
            self._dspot.initialize([float(score)])
            self._initialized = True
            return False, float("inf")
        is_anom = self._dspot.step(float(score))
        return bool(is_anom), self._dspot.current_raw_threshold()

    @property
    def threshold(self) -> float:
        """Current EVT threshold on the raw score scale."""
        return self._dspot.current_raw_threshold()


__all__ = ["POT", "SPOT", "DSPOT", "ScoreStreamThresholder"]
