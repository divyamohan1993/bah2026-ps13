"""Cross-encoder reranker — bge-reranker-v2-m3 (optional) with identity fallback.

A cross-encoder jointly encodes ``(query, chunk)`` and is the single biggest
grounding boost in the pipeline (Anthropic: reranking pushed retrieval-failure
reduction from 49% -> 67%, research 06 §4). The heavy path is
``sentence-transformers`` ``CrossEncoder`` with **BAAI/bge-reranker-v2-m3** (or
``ms-marco-MiniLM-L-6-v2`` as a faster English option); both are import-guarded
and loaded offline only if present.

When no reranker model is available — the CPU-only default — :class:`Reranker`
degrades to an **identity reranker** that preserves the fused first-stage order
(it simply trims to ``top_k``). Retrieval still works and stays grounded; the
reranker is a quality upgrade, never a hard dependency.
"""

from __future__ import annotations

import os
from typing import Sequence

from .store import Chunk

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class Reranker:
    """Re-score (query, chunk) pairs with a cross-encoder, or pass through.

    Parameters
    ----------
    prefer_model:
        Attempt to load the heavy cross-encoder when True; otherwise identity.
    model_name:
        HF cross-encoder id (present locally; offline).
    """

    def __init__(
        self,
        *,
        prefer_model: bool = False,
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ) -> None:
        self.model_name = model_name
        self._model = None
        self.backend = "identity"
        if prefer_model:
            self._try_load(model_name)

    def _try_load(self, model_name: str) -> None:
        try:  # optional-heavy
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(model_name)  # offline
            self.backend = "cross-encoder"
        except Exception:
            self._model = None
            self.backend = "identity"

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` candidates re-ranked for ``query``.

        With a cross-encoder, scores are the model's relevance logits. In the
        identity fallback, scores are a gently-decaying rank proxy so callers can
        still compare/threshold, while the *order* of the fused first stage is
        preserved.
        """
        cands = list(candidates)
        if not cands:
            return []

        if self.backend == "cross-encoder" and self._model is not None:
            try:
                pairs = [[query, c.text] for c in cands]
                scores = self._model.predict(pairs)
                ranked = sorted(
                    zip(cands, (float(s) for s in scores)),
                    key=lambda t: t[1],
                    reverse=True,
                )
                return ranked[:top_k]
            except Exception:
                pass  # degrade to identity on any inference error

        # Identity: preserve incoming order, attach a decaying pseudo-score.
        n = len(cands)
        return [(c, 1.0 - (i / max(n, 1))) for i, c in enumerate(cands)][:top_k]


__all__ = ["Reranker"]
