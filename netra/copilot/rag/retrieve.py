"""Hybrid retrieval — dense + BM25 sparse fused with Reciprocal Rank Fusion.

Pure-dense retrieval underweights the short literal tokens NOC operators search
on (interface names, ASNs, prefixes, syslog mnemonics); BM25 nails those exact
matches; **hybrid** ensures a doc is found whether the operator used a perfect
keyword *or* a paraphrase (research 06 §3). The two ranked lists are combined
with **Reciprocal Rank Fusion** (``RRF(d)=Σ 1/(k+rank)``, ``k=60``), which is
score-agnostic so cosine and BM25 scales never have to be reconciled.

BM25 backends (best-available, all offline):
  * ``bm25s`` (optional, light, very fast) ->
  * ``rank_bm25`` (optional) ->
  * a small **pure-Python Okapi BM25** implemented here (always works, zero deps).

The dense half uses :class:`~netra.copilot.rag.embed.Embedder` +
:class:`~netra.copilot.rag.store` and the optional cross-encoder
:class:`~netra.copilot.rag.rerank.Reranker` finishes the top list. With no heavy
deps at all (numpy + scikit-learn only) the whole hybrid+rerank path still runs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .embed import Embedder
from .rerank import Reranker
from .store import Chunk, make_vector_store

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokeniser that keeps NOC literals (dots/colons/slashes) intact."""
    return _TOKEN_RE.findall(text.lower())


class _PurePythonBM25:
    """Minimal Okapi BM25 over an in-memory corpus (the always-available sparse).

    Implements the standard BM25 with ``k1=1.5``, ``b=0.75``. Deterministic and
    dependency-free so the sparse half of hybrid retrieval never needs an
    optional package.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n: int = 0

    def fit(self, docs: Sequence[str]) -> _PurePythonBM25:
        self._docs_tokens = [_tokenize(d) for d in docs]
        self._doc_freqs = [Counter(toks) for toks in self._docs_tokens]
        self._n = len(self._docs_tokens)
        self._df = Counter()
        for toks in self._docs_tokens:
            for term in set(toks):
                self._df[term] += 1
        # BM25 idf with the standard +0.5 smoothing (kept non-negative).
        self._idf = {
            term: max(
                0.0, math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            )
            for term, df in self._df.items()
        }
        total_len = sum(len(t) for t in self._docs_tokens)
        self._avgdl = (total_len / self._n) if self._n else 0.0
        return self

    def scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        out = [0.0] * self._n
        for i, freqs in enumerate(self._doc_freqs):
            dl = len(self._docs_tokens[i])
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            s = 0.0
            for term in q_terms:
                if term not in freqs:
                    continue
                tf = freqs[term]
                idf = self._idf.get(term, 0.0)
                s += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
            out[i] = s
        return out


class HybridRetriever:
    """Dense + BM25 hybrid retriever with RRF fusion and optional reranking.

    Build once via :meth:`index` (or the :mod:`~netra.copilot.rag.ingest` helper),
    then call :meth:`retrieve`. The retriever owns its embedder, dense store,
    sparse index and reranker so the orchestrator just asks for chunks.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder or Embedder(prefer_model=False)
        self.reranker = reranker or Reranker(prefer_model=False)
        self.rrf_k = rrf_k
        self._chunks: list[Chunk] = []
        self._store = None
        self._bm25 = None
        self._bm25_backend = "pure_python"

    # -- indexing ---------------------------------------------------------------
    def index(self, chunks: Sequence[Chunk]) -> HybridRetriever:
        """Embed + index ``chunks`` for both dense and sparse retrieval."""
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]

        # Dense: fit the (TF-IDF fallback) embedder on the corpus, then encode.
        self.embedder.fit(texts)
        if self._chunks:
            vecs = self.embedder.encode(texts)
            self._store = make_vector_store(dim=vecs.shape[1])
            self._store.add(self._chunks, vecs)
        else:
            self._store = make_vector_store(dim=None)

        # Sparse: best-available BM25 over the same texts.
        self._init_bm25(texts)
        return self

    def _init_bm25(self, texts: Sequence[str]) -> None:
        """Initialise the best-available BM25 backend (bm25s/rank_bm25/pure)."""
        # rank_bm25 is the most portable optional; bm25s is faster but heavier to
        # set up. We try rank_bm25 first, then fall back to the pure-python BM25.
        try:  # optional
            from rank_bm25 import BM25Okapi  # type: ignore

            tokenized = [_tokenize(t) for t in texts] or [[""]]
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_backend = "rank_bm25"
            return
        except Exception:
            pass
        self._bm25 = _PurePythonBM25().fit(list(texts) or [""])
        self._bm25_backend = "pure_python"

    # -- retrieval --------------------------------------------------------------
    def _dense_ranking(self, query: str, top_n: int) -> list[int]:
        """Indices of chunks ranked by dense cosine similarity."""
        if self._store is None or not self._chunks:
            return []
        qv = self.embedder.encode([query])
        if qv.shape[0] == 0:
            return []
        results = self._store.search(qv[0], top_k=top_n)
        # Map chunk identity back to index position.
        id_to_pos = {id(c): i for i, c in enumerate(self._chunks)}
        return [id_to_pos[id(c)] for c, _ in results if id(c) in id_to_pos]

    def _sparse_ranking(self, query: str, top_n: int) -> list[int]:
        """Indices of chunks ranked by BM25 score (descending)."""
        if self._bm25 is None or not self._chunks:
            return []
        if self._bm25_backend == "rank_bm25":
            scores = list(self._bm25.get_scores(_tokenize(query)))
        else:
            scores = self._bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Drop strictly-zero scores so RRF doesn't reward non-matches.
        return [i for i in order if scores[i] > 0][:top_n]

    @staticmethod
    def _rrf(rankings: Sequence[Sequence[int]], k: int) -> list[int]:
        """Reciprocal Rank Fusion of several ranked index lists -> fused order."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return [idx for idx, _ in sorted(fused.items(), key=lambda t: t[1], reverse=True)]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_pool: int = 50,
        metadata_filter: dict | None = None,
    ) -> list[Chunk]:
        """Return the top ``top_k`` chunks for ``query`` (hybrid + RRF + rerank).

        Parameters
        ----------
        top_k:
            Final number of chunks to return (small k => better grounding).
        candidate_pool:
            How many candidates each stage considers before fusion/rerank.
        metadata_filter:
            Optional ``{key: value}`` filter applied to chunk metadata *before*
            ranking (e.g. ``{"site": "hub1"}``) — the in-process analogue of a
            Qdrant payload filter.
        """
        if not self._chunks:
            return []

        # Optional pre-filter on metadata (acts like a Qdrant payload filter).
        if metadata_filter:
            allowed = {
                i
                for i, c in enumerate(self._chunks)
                if all(c.metadata.get(k) == v for k, v in metadata_filter.items())
            }
        else:
            allowed = set(range(len(self._chunks)))

        dense = [i for i in self._dense_ranking(query, candidate_pool) if i in allowed]
        sparse = [i for i in self._sparse_ranking(query, candidate_pool) if i in allowed]

        fused = self._rrf([dense, sparse], self.rrf_k)
        # If both stages returned nothing (e.g. degenerate query), fall back to
        # the allowed set in original order so we never silently lose grounding.
        if not fused:
            fused = [i for i in range(len(self._chunks)) if i in allowed]

        fused_chunks = [self._chunks[i] for i in fused[:candidate_pool]]
        reranked = self.reranker.rerank(query, fused_chunks, top_k=top_k)
        return [c for c, _ in reranked]

    @property
    def bm25_backend(self) -> str:
        return self._bm25_backend

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["HybridRetriever", "_PurePythonBM25"]
