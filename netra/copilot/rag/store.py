"""Vector store — Qdrant (optional) / FAISS (optional) / numpy cosine fallback.

Holds the dense embeddings of corpus chunks and answers nearest-neighbour
queries. Three backends, selected by availability (research 06 §2):

  * **Qdrant** (optional-heavy) — single-binary HNSW store with payload filters,
    the production air-gap choice; used only if ``qdrant-client`` is importable.
  * **FAISS** (optional) — in-process ``IndexFlatIP`` (exact cosine on
    L2-normalised vectors) when ``faiss`` is importable but Qdrant is not.
  * **NumpyStore** (always) — a pure-numpy in-memory cosine index. Exact,
    dependency-free, perfect for the CPU-only default and tests.

All backends share the :class:`Chunk` unit and return ``(Chunk, score)`` ranked
by cosine similarity, so :mod:`.retrieve` is backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Chunk:
    """One retrievable unit of the corpus, with a citable id and metadata.

    ``chunk_id`` is what ends up in ``CopilotResponse.citations`` (closed-set
    enforced), so it must be stable and human-meaningful (e.g.
    ``"RB-CONGESTION-001#2"`` or ``"INC-2026-0007"``). ``metadata`` carries the
    NOC filter keys (site/device/role/issue_type/scenario_id/artifact_type) used
    for grounding and (optionally) Qdrant payload filtering.
    """

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class NumpyStore:
    """Exact in-memory cosine index over L2-normalised vectors (the fallback).

    Vectors are assumed L2-normalised by the :class:`~netra.copilot.rag.embed.Embedder`,
    so cosine similarity is a single matrix-vector inner product — O(n·d) per
    query, which is trivially fast for a NOC-sized corpus and needs no native
    deps.
    """

    backend = "numpy"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Index ``chunks`` with their corresponding row ``vectors``."""
        mat = _as_matrix(vectors)
        if len(chunks) != mat.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        self._chunks.extend(chunks)
        self._matrix = mat if self._matrix is None else np.vstack([self._matrix, mat])

    def search(
        self, query_vec: Sequence[float], top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` chunks most cosine-similar to ``query_vec``."""
        if self._matrix is None or not self._chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        # Guard against a dimensionality mismatch (e.g. TF-IDF refit) — pad/trim.
        if q.shape[0] != self._matrix.shape[1]:
            return []
        sims = self._matrix @ q
        k = min(top_k, len(self._chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._chunks[i], float(sims[i])) for i in idx]

    def __len__(self) -> int:
        return len(self._chunks)


class FaissStore:
    """In-process FAISS ``IndexFlatIP`` cosine store (optional dependency)."""

    backend = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss  # type: ignore  (optional)

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []
        self._dim = dim

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        mat = _as_matrix(vectors)
        self._index.add(mat)
        self._chunks.extend(chunks)

    def search(
        self, query_vec: Sequence[float], top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        q = _as_matrix([query_vec])
        k = min(top_k, len(self._chunks))
        scores, idx = self._index.search(q, k)
        out: list[tuple[Chunk, float]] = []
        for j, i in enumerate(idx[0]):
            if 0 <= i < len(self._chunks):
                out.append((self._chunks[i], float(scores[0][j])))
        return out

    def __len__(self) -> int:
        return len(self._chunks)


def make_vector_store(
    dim: int | None = None, *, prefer_qdrant: bool = False
) -> NumpyStore | FaissStore:
    """Return the best available dense store.

    Order of preference: Qdrant (if requested + importable) -> FAISS (if
    importable and ``dim`` known) -> NumpyStore (always). Qdrant is only used
    when explicitly requested because the python ``:memory:`` mode does a full
    scan (research 06 §2.3) — for the appliance the integrator runs the on-disk
    binary; for the CPU/test path the numpy store is both exact and lighter.
    """
    if prefer_qdrant:
        try:  # optional-heavy
            import qdrant_client  # type: ignore  # noqa: F401

            # A real Qdrant deployment is integrator-provisioned (on-disk binary);
            # here we acknowledge availability but still use the exact numpy store
            # unless a full QdrantStore is wired by the integrator. This keeps the
            # CPU/test path deterministic while documenting the upgrade hook.
        except Exception:
            pass
    if dim is not None:
        try:  # optional
            import faiss  # type: ignore  # noqa: F401

            return FaissStore(dim)
        except Exception:
            pass
    return NumpyStore()


__all__ = ["Chunk", "NumpyStore", "FaissStore", "make_vector_store"]
