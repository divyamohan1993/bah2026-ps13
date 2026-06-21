"""netra.copilot.rag — offline hybrid RAG over internal NOC artifacts.

The grounding engine (the 35% "Copilot Effectiveness" lever). Pipeline:
``ingest -> structure-aware chunk -> embed (bge-m3 | TF-IDF) -> dense store
(Qdrant | FAISS | numpy) + BM25 sparse -> RRF fusion -> cross-encoder rerank
(bge-reranker-v2-m3 | identity) -> top-k chunks`` fused with **GraphRAG-lite**
deterministic topology/blast-radius facts.

Every heavy component (sentence-transformers/bge-m3, Qdrant, FAISS, bm25s,
cross-encoder) is import-guarded with a working light fallback, so the whole
hybrid+rerank+graph path runs on numpy + scikit-learn alone, fully offline.
"""

from __future__ import annotations

from .embed import Embedder
from .graphrag import (
    TopologyGraph,
    affected_scope_from_blast_radius,
    graph_facts,
)
from .ingest import (
    DEFAULT_CORPUS_DIR,
    build_retriever,
    document_ids,
    load_corpus_chunks,
)
from .rerank import Reranker
from .retrieve import HybridRetriever
from .store import Chunk, FaissStore, NumpyStore, make_vector_store

__all__ = [
    "Embedder",
    "Reranker",
    "HybridRetriever",
    "Chunk",
    "NumpyStore",
    "FaissStore",
    "make_vector_store",
    "load_corpus_chunks",
    "build_retriever",
    "document_ids",
    "DEFAULT_CORPUS_DIR",
    "TopologyGraph",
    "graph_facts",
    "affected_scope_from_blast_radius",
]
