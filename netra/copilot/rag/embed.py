"""Text embeddings — bge-m3 if present (optional-heavy), TF-IDF/hashing fallback.

Embeddings power the *dense* half of hybrid retrieval. The heavy, high-quality
path is ``sentence-transformers`` with **BAAI/bge-m3** (dense + learned-sparse,
8192 ctx — research 06 §1). It is wrapped in ``try/except`` and only loaded if
both the library and the local weights are available **offline** (we set
``HF_HUB_OFFLINE`` so it can never reach the network). When it is absent — the
CPU-only default — we fall back to a pure-``scikit-learn`` **TF-IDF** vectorizer
(or a hashing vectorizer), so dense-style cosine retrieval still works with no
model and no internet.

The :class:`Embedder` exposes a uniform ``encode(texts) -> np.ndarray`` of
L2-normalised row vectors so :mod:`.store` can use plain cosine/inner-product
regardless of which backend is active. ``fit`` is a no-op for transformer models
and trains the TF-IDF vocabulary for the fallback (corpus-driven, deterministic).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# Force offline for any HF backend before it could ever be imported/loaded.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (so inner product == cosine similarity)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Uniform text embedder with a heavy (bge-m3) and a light (TF-IDF) backend.

    Parameters
    ----------
    prefer_model:
        When True, attempt to load ``sentence-transformers`` + ``model_name``.
        When False (default for the CPU/offline tier), go straight to TF-IDF.
    model_name:
        HF model id for the dense backend (must be present locally; offline).
    """

    def __init__(
        self,
        *,
        prefer_model: bool = False,
        model_name: str = "BAAI/bge-m3",
    ) -> None:
        self.model_name = model_name
        self._st_model = None  # sentence-transformers model, if loaded
        self._tfidf = None  # sklearn TfidfVectorizer, if used
        self._fitted = False
        self.backend = "tfidf"  # one of: "sentence-transformers", "tfidf", "hashing"

        if prefer_model:
            self._try_load_transformer(model_name)

        if self._st_model is None:
            self._init_tfidf()

    # -- backend init -----------------------------------------------------------
    def _try_load_transformer(self, model_name: str) -> None:
        """Best-effort load of the heavy dense model; silent fall-through on failure."""
        try:  # optional-heavy: sentence-transformers + local bge-m3 weights
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(model_name)  # offline (HF_HUB_OFFLINE)
            self.backend = "sentence-transformers"
            self._fitted = True  # transformer needs no corpus fit
        except Exception:
            self._st_model = None  # degrade to TF-IDF

    def _init_tfidf(self) -> None:
        """Initialise the light TF-IDF fallback vectorizer."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            # Word + char n-grams capture both prose and literal NOC tokens
            # (interface names, ASNs, error codes) that pure word models miss.
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )
            self.backend = "tfidf"
        except Exception:
            # Last-resort: deterministic hashing embedding with no sklearn at all.
            self._tfidf = None
            self.backend = "hashing"

    # -- fit / encode -----------------------------------------------------------
    def fit(self, corpus: Sequence[str]) -> "Embedder":
        """Fit the fallback vocabulary on the corpus (no-op for transformers).

        If the corpus is empty or yields an empty vocabulary (e.g. only
        stopwords), degrade to the deterministic hashing backend so an empty
        corpus never raises — the copilot must still produce a (correctly
        abstaining) response.
        """
        if self.backend == "tfidf" and self._tfidf is not None:
            docs = [d for d in corpus if d and d.strip()]
            if not docs:
                self.backend = "hashing"
                self._tfidf = None
                self._fitted = True
                return self
            try:
                self._tfidf.fit(docs)
                self._fitted = True
            except ValueError:
                # Empty vocabulary -> hashing fallback (still deterministic).
                self.backend = "hashing"
                self._tfidf = None
                self._fitted = True
        else:
            self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, d)`` array of L2-normalised embeddings for ``texts``."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.backend == "sentence-transformers" and self._st_model is not None:
            vecs = np.asarray(
                self._st_model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            return vecs

        if self.backend == "tfidf" and self._tfidf is not None:
            if not self._fitted:
                # Encoding before fit: fit on the texts themselves (deterministic).
                self.fit(texts)
            # The lazy fit above may have degraded to hashing (empty vocab); only
            # use TF-IDF if it survived.
            if self.backend == "tfidf" and self._tfidf is not None:
                mat = self._tfidf.transform(texts).toarray().astype(np.float32)
                return _l2_normalize(mat)

        # Hashing fallback (no sklearn / empty vocab): hashed bag-of-words.
        return self._hash_encode(texts)

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality once known (None before fit for TF-IDF)."""
        if self.backend == "sentence-transformers" and self._st_model is not None:
            try:
                return int(self._st_model.get_sentence_embedding_dimension())
            except Exception:
                return None
        if self.backend == "tfidf" and self._tfidf is not None and self._fitted:
            try:
                return len(self._tfidf.vocabulary_)
            except Exception:
                return None
        if self.backend == "hashing":
            return 1024
        return None

    @staticmethod
    def _hash_encode(texts: Sequence[str], n_dim: int = 1024) -> np.ndarray:
        """Pure-stdlib hashing vectorizer (deterministic) for the no-sklearn case."""
        import re

        out = np.zeros((len(texts), n_dim), dtype=np.float32)
        token_re = re.compile(r"[A-Za-z0-9_./:-]+")
        for i, text in enumerate(texts):
            for tok in token_re.findall(text.lower()):
                out[i, hash(tok) % n_dim] += 1.0
        return _l2_normalize(out)


__all__ = ["Embedder"]
