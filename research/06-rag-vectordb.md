# Phase 4/5 — Air-Gapped RAG over Internal NOC Artifacts (Deep Research)

**Problem Statement 13: Air-Gapped Predictive Copilot for Secure MPLS Operations**
**Domain:** Local, fully offline Retrieval-Augmented Generation over INTERNAL artifacts only (topology maps, runbooks, past incidents), with strong grounding.

> **Why this matters for scoring.** "Copilot Effectiveness" is **35%** of the evaluation, judged on whether explanations are *"correct, operator-relevant, and grounded in local retrieval without hallucination."* RAG quality is the single biggest lever on that 35%. "Security & Offline Compliance" (20%) additionally requires *"verifiably zero outbound dependency during runtime."* Every component below is chosen to be **100% offline, free/open-source, low-footprint, and low-latency**, with grounding/anti-hallucination as the primary design goal.

---

## 0. TL;DR — Recommended Stack (all offline, all OSS)

| Layer | Primary recommendation | Tiny / CPU fallback | License |
|---|---|---|---|
| **Embedding model** | **BAAI/bge-m3** (dense+sparse+ColBERT in one model, 1024-dim, 8192 ctx) | **bge-small-en-v1.5** (384-dim) or **all-MiniLM-L6-v2** (22M, 384-dim, ~14k sent/s CPU) | MIT / Apache-2.0 |
| **Vector DB / index** | **Qdrant** (self-hosted OSS, native dense+sparse hybrid, payload filtering, HNSW) | **FAISS** (`IndexHNSWFlat`, embeddable, MIT) | Apache-2.0 / MIT |
| **Sparse / keyword** | Qdrant **sparse vectors** (BM25/BM42) or **bm25s** (pure-python, very fast) | `rank_bm25` | MIT |
| **Fusion** | **Reciprocal Rank Fusion (RRF)** (score-agnostic; Qdrant native) | RRF in-code | — |
| **Reranker** | **BAAI/bge-reranker-v2-m3** (cross-encoder, multilingual) | **ms-marco-MiniLM-L-6-v2** (fast, English) | Apache-2.0 |
| **Chunking** | Structure-aware + **Anthropic Contextual Retrieval** (chunk prefixes written by the *local* LLM) | recursive 400–512 tok | OSS |
| **Graph / topology** | **Live topology graph (NetworkX/Neo4j) + text RAG** ("hybrid GraphRAG-lite"): graph for blast-radius, vectors for prose | NetworkX in-process | OSS |
| **RAG framework** | **Thin custom pipeline** (or **Haystack 2.x** if a framework is wanted) | — | Apache-2.0 |
| **Offline eval** | **RAGAS** + **DeepEval** with a **local Ollama judge**; retrieval metrics hit-rate/MRR/nDCG | — | Apache-2.0 / Apache-2.0 |

**The end-to-end pipeline (one line):**
`ingest → structure-aware chunk → LLM-written contextual prefix → embed (bge-m3 dense+sparse) → Qdrant (HNSW + sparse) → hybrid retrieve top-150 → RRF → bge-reranker-v2-m3 → top-5..20 → grounded generation with enforced citations → offline RAGAS/DeepEval gate.`

This mirrors Anthropic's published "Contextual Retrieval + hybrid + reranking" result of a **67% reduction in retrieval failures** (5.7% → 1.9%) — the strongest published, fully-reproducible-offline grounding recipe ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)).

---

## 1. Local Embedding Models (offline)

### 1.1 Comparison table

All models below are open-weight and run fully offline (download once, bundle into the air-gap). MTEB v1 (English) retrieval-oriented scores; note **MTEB v2 (2026) scores are not directly comparable to v1** ([Modal](https://modal.com/blog/mteb-leaderboard-article), [Awesome Agents](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)).

| Model | Params | Dim | Max tokens | MTEB (Eng v1, ~) | Speed / footprint | License | Notes |
|---|---|---|---|---|---|---|---|
| **bge-m3** ⭐ | 567M | 1024 (+sparse +ColBERT) | **8192** | ~63 (multi-mode) | ~1.2 GB, GPU preferred | MIT | **Dense + sparse + multi-vector in ONE model** — ideal for offline hybrid. 100+ langs. ([HF](https://huggingface.co/BAAI/bge-m3), [arXiv](https://arxiv.org/html/2402.03216v3)) |
| **mxbai-embed-large** | 335M | 1024 | 512 | 64.68 | 670 MB | Apache-2.0 | Strong English; short context. Needs task prefix. ([Morph](https://www.morphllm.com/ollama-embedding-models)) |
| **nomic-embed-text-v1.5** ⭐ | 137M | 768 (**Matryoshka** 768/512/256/128/64) | 8192 | 62.28 | **274 MB, runs on laptop CPU** | Apache-2.0 | Truncate to 256-d for only 0.24–1.24 MTEB drop. Great CPU option w/ long ctx. ([HF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), [Zilliz](https://zilliz.com/ai-models/nomic-embed-text-v1.5)) |
| **bge-large-en-v1.5** | 335M | 1024 | 512 | ~64.2 | ~1.3 GB | MIT | English SOTA-class bi-encoder; short ctx. |
| **bge-base-en-v1.5** | 109M | 768 | 512 | ~63.5 | ~440 MB | MIT | Balanced accuracy/latency (~82 ms). ([BentoML](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)) |
| **bge-small-en-v1.5** ⭐(fallback) | 33M | 384 | 512 | ~62.2 | ~130 MB, CPU-friendly | MIT | Best accuracy-per-MB small option. |
| **e5-large-v2** | 335M | 1024 | 512 | ~62.3 | ~1.3 GB | MIT | Requires `query:`/`passage:` prefixes. |
| **e5-base-v2** | 110M | 768 | 512 | ~61.5 | ~440 MB | MIT | ~83.5% top-5 in head-to-head. ([Supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)) |
| **multilingual-e5-large** | 560M | 1024 | 512 | ~61 | ~2.2 GB | MIT | If non-English NOC docs appear. |
| **gte-large** | 335M | 1024 | 512 | ~63.1 | ~1.3 GB | MIT | Competitive English; no prefix needed. |
| **snowflake-arctic-embed-l** / **v2** | 335M / 568M | 1024 | 512 / 8192 | ~55.6 BEIR (v2) | ~1.2 GB | Apache-2.0 | Strong retrieval-tuned; v2 adds long ctx + multilingual. ([Morph](https://www.morphllm.com/ollama-embedding-models)) |
| **jina-embeddings-v3** | 570M | 1024 (Matryoshka) | 8192 | ~64–65 | ~2.2 GB | CC-BY-NC (⚠ non-commercial) | Strong long-ctx but **license is non-commercial — avoid for a deployable product.** |
| **all-MiniLM-L6-v2** ⭐(tiny) | 22M | 384 | 256 | ~56 | **46 MB, ~14k sent/s CPU** | Apache-2.0 | Tiny/fast prototyping & pure-CPU fallback; ~5–8% lower recall. ([BentoML](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models), [Ailog](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)) |

> ⚠ **License watch:** `jina-embeddings-v3` weights are **CC-BY-NC** (non-commercial). For a deployable/judged hackathon product prefer **MIT/Apache-2.0** models (bge-*, nomic, e5, gte, arctic, MiniLM). Note also: a Hacker-News thread argues *"don't use all-MiniLM-L6-v2 for new datasets"* for production retrieval — use it only as the tiny fallback ([HN](https://news.ycombinator.com/item?id=46081800)).

### 1.2 Matryoshka embeddings (latency/footprint lever)

`nomic-embed-text-v1.5`, `jina-v3`, and `EmbeddingGemma` are trained with **Matryoshka Representation Learning (MRL)**: a single forward pass produces a vector you can **truncate** (take first *N* dims) to 512/256/128/64 with minimal quality loss (nomic: **0.24–1.24 MTEB drop** for 512/256) ([Zilliz](https://zilliz.com/ai-models/nomic-embed-text-v1.5), [Nomic](https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2)). In an air-gapped NOC this lets you store **256-d** vectors → ~3× smaller index + faster ANN, then re-rank with the cross-encoder to recover precision. Excellent fit for a resource-constrained on-prem box.

### 1.3 Recommendation

- **Primary: `BAAI/bge-m3`.** Decisive reason for *this* project: it emits **dense + learned-sparse + ColBERT multi-vectors from one model**, so hybrid retrieval (Section 3) needs no second model in the air-gap, and 8192-token context handles long runbooks/configs. MIT-licensed ([HF](https://huggingface.co/BAAI/bge-m3)).
- **CPU fallback: `bge-small-en-v1.5` (384-d)** for accuracy-per-MB, or **`all-MiniLM-L6-v2`** if you need ~14k sentences/sec on a pure-CPU NOC appliance.
- **If GPU is tight but long context matters: `nomic-embed-text-v1.5` @256-d (Matryoshka)** — 274 MB, laptop-CPU-capable, 8192 ctx.
- Always store the **embedding model + version + dimensionality** in chunk metadata so the index is reproducible and re-indexing is deterministic inside the air-gap.

---

## 2. Vector Database / Index (offline, embeddable)

### 2.1 Comparison table

| Engine | Footprint / deploy | Dense+sparse **hybrid** | Metadata filter | Persistence | Index | Air-gap ease | License |
|---|---|---|---|---|---|---|---|
| **Qdrant** ⭐ | Single Rust binary **or** embedded python `:memory:`/on-disk; no external deps | **Yes — native sparse vectors + RRF/`Query API`** | **Strong** (rich JSON payload filters) | Yes (mmap/on-disk) | HNSW (+quantization) | **Excellent** — one static binary, no cloud calls | Apache-2.0 |
| **FAISS** ⭐(embed) | **Library, in-process**, no server | No (DIY) | No (DIY) | `write_index()`/`read_index()` | Flat, **HNSWFlat**, IVF, IVFPQ, OPQ | **Excellent** — pip/conda, MIT | **MIT** |
| **LanceDB** ⭐(embed) | **Embedded, in-process**, reads/writes disk (Lance columnar), zero-copy | Yes (vector + FTS/BM25) | Yes | Yes (Lance files) | IVF-PQ, HNSW, disk-based | **Excellent** — no server, edge-first | Apache-2.0 |
| **Chroma** | Embedded in-process (Rust core, 2025 rewrite) | Metadata + full-text | Yes | Yes (DuckDB/sqlite-style) | HNSW | **Excellent** — great for prototypes | Apache-2.0 |
| **Milvus-Lite** | Embedded (pip) variant of Milvus | Limited native hybrid | Yes | Yes | HNSW, IVF, DiskANN | Good (Lite); full Milvus is heavy | Apache-2.0 |
| **Weaviate** | Server (Docker/K8s) | **Native hybrid** (vector+keyword+filter) | Excellent | Yes | HNSW | OK self-hosted; heavier | BSD-3 |
| **pgvector (+ pgvectorscale)** | Postgres extension (server) | Combine w/ PG full-text | Native SQL | Yes (Postgres) | IVFFlat, HNSW, DiskANN | OK **if Postgres already in stack** | PostgreSQL/Apache |
| **Vespa** | Server (heavy, JVM) | Yes (text+tensor) | Yes | Yes | HNSW + more | Heavier ops; powerful | Apache-2.0 |
| **hnswlib** | **Library, in-process** | No (DIY) | No (DIY) | Yes (`save_index`) | HNSW only | **Excellent** — minimal | Apache-2.0 |
| **ScaNN** | Library (Google) | No | No | DIY | Anisotropic quantization | OK (TF dep) | Apache-2.0 |

Sources: [Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases), [Medium top-5 OSS](https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3), [Qdrant storage docs](https://qdrant.tech/documentation/manage-data/storage/), [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [LanceDB via Firecrawl].

### 2.2 ANN nature & keeping latency low (HNSW tuning)

All recommended engines use **HNSW** (Hierarchical Navigable Small World) graphs — approximate nearest neighbor with **~O(log n)** search instead of O(n) brute force. Three knobs control the recall↔latency tradeoff ([Milvus](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall), [Qdrant course](https://qdrant.tech/course/essentials/day-2/what-is-hnsw/), [OSC](https://opensourceconnections.com/blog/2025/02/27/vector-search-navigating-recall-and-performance/)):

- **`M`** (edges/node): higher = better recall, larger index, slower build. **Default `M=16`.**
- **`ef_construction`** (build beam width): higher = better graph, slower build. **Default `200`** (doubling can ~4× build time).
- **`ef_search`** (query beam width): the **main runtime dial** — tunable *after* build, no rebuild. e.g. `ef=100`→~85% recall @~1 ms; `ef=500`→~98% recall @~5 ms.

**NOC-appliance recipe:** start `M=16, ef_construction=200`; measure recall/latency on a labeled validation set (Section 8); raise `ef_search` until recall is acceptable. For a corpus of internal NOC docs (thousands–low-millions of chunks) this yields **sub-10 ms** dense retrieval on a single node — leaving budget for sparse search + reranking while still meeting an interactive copilot SLA. FAISS `IndexHNSWFlat` costs ~2× memory vs `IndexIVFFlat` but is simpler and very fast; for >1M vectors under tight RAM use `IVFPQ` ([FAISS HNSW](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWFlat.html), [Markaicode](https://markaicode.com/tutorial/faiss-tutorial-production-setup-guide/)).

### 2.3 Recommendation

- **Primary: Qdrant (self-hosted OSS, single binary).** It is the best *air-gap* fit that **also** gives **native hybrid (dense + sparse vectors), RRF fusion, and rich payload filtering** out of the box — and filtering on `{site, device, vendor, role(CE/PE/P), timestamp, artifact_type}` is essential for NOC retrieval. Apache-2.0, zero licensing cost at any scale self-hosted, no outbound calls ([Qdrant LICENSE](https://github.com/qdrant/qdrant/blob/master/LICENSE), [Cohorte tutorial](https://cohorte.co/blog/a-developers-friendly-guide-to-qdrant-vector-database)).
  - ⚠ Qdrant's **python-only `:memory:` local mode** does a *full scan* (no HNSW) and is for tests only — for the real appliance run the **Qdrant binary in on-disk mode** ([discussion #2540](https://github.com/orgs/qdrant/discussions/2540)).
- **Embeddable fallback (no service at all): FAISS `IndexHNSWFlat` + `bm25s`.** When the deliverable must be a single python process with **zero** background service, FAISS (MIT) for dense + `bm25s` for sparse + in-code RRF reproduces the same pipeline. `LanceDB` is the modern in-process alternative (vector + built-in BM25 FTS + filters + disk persistence) and is excellent if you want hybrid *without* hand-rolling BM25.
- Avoid **Weaviate/Vespa/full-Milvus** here purely on **ops weight** in an air-gap; avoid **pgvector** unless Postgres is already mandated by the telemetry stack.

---

## 3. Hybrid & Sparse Retrieval (critical for NOC jargon)

**Why hybrid beats pure-dense — and why it specifically matters here.** Dense vectors capture *meaning/paraphrase* but **underweight short literal tokens** — exactly the tokens NOC operators search on: interface names (`GigabitEthernet0/0/1`, `Te0/1/0.100`), **ASNs** (`AS64512`), IPs/prefixes (`10.20.30.0/24`), BGP/OSPF **error codes / notifications**, MPLS labels, VRF names, rekey/SA identifiers, syslog mnemonics (`%BGP-5-ADJCHANGE`). *"Dense retrieval regularly returns answers that sound plausible but are wrong because it misses short literal tokens like IPs, error codes, and config flags"* ([Medium/Hybrid](https://ashutoshkumars1ngh.medium.com/hybrid-search-done-right-fixing-rag-retrieval-failures-using-bm25-hnsw-reciprocal-rank-fusion-a73596652d22)). **BM25/sparse nails exact-match** rare technical terms; dense handles "why is my tunnel flapping after the maintenance window." Hybrid ensures you don't miss a doc because the operator used a perfect keyword *or* a synonym ([Weaviate](https://weaviate.io/blog/hybrid-search-explained), [MongoDB](https://www.mongodb.com/resources/products/capabilities/hybrid-search)).

**Reciprocal Rank Fusion (RRF)** is the recommended combiner because it is **score-agnostic** — it uses only each list's *rank*, sidestepping the incompatible-scale problem of fusing cosine vs BM25 scores: `RRF(d) = Σ 1/(k + rank_i(d))`, typically `k=60`. On the WANDS benchmark, RRF (NDCG 0.7068) beat both BM25-alone (0.6983) and pure-KNN (0.6953), with tuned hybrid reaching 0.7497 ([Medium/RRF](https://ashutoshkumars1ngh.medium.com/hybrid-search-done-right-fixing-rag-retrieval-failures-using-bm25-hnsw-reciprocal-rank-fusion-a73596652d22), [CEUR](https://ceur-ws.org/Vol-4173/T3-7.pdf)).

**Sparse implementation options (offline):**
- **Qdrant sparse vectors** — store dense + sparse on the same point; server fuses via the Query API/RRF. Sparse can be classic **BM25**, Qdrant's **BM42**, or learned **SPLADE** (DistilBERT MLM head → ~200 non-zero terms, outperforms BM25) via **FastEmbed**, all runnable offline ([Qdrant sparse](https://qdrant.tech/articles/sparse-vectors/), [SPLADE/FastEmbed](https://qdrant.tech/documentation/fastembed/fastembed-splade/), [BM42](https://qdrant.tech/articles/bm42/)).
- **bm25s** — **pure-python BM25**, Scipy-sparse, "orders of magnitude faster" than popular libs via eager scoring; perfect when you don't run a server ([bm25s](https://bm25s.github.io/), [PyPI](https://pypi.org/project/bm25s)).
- **rank_bm25** — simplest, most-cited; fine for small corpora.
- **OpenSearch** — only if an ES/OpenSearch cluster already exists (heavier for an air-gap).

> **Free win with bge-m3:** because bge-m3 already outputs **learned-sparse weights alongside dense**, you get hybrid with **one** model and no separate SPLADE deployment — store both vector types in Qdrant and fuse with RRF.

---

## 4. Reranking (offline) — the biggest single grounding boost

A **cross-encoder reranker** re-scores the top-K retrieved chunks by jointly encoding (query, chunk), which is far more precise than the bi-encoder first stage. In Anthropic's pipeline, **adding reranking pushed retrieval-failure reduction from 49% → 67%** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)). For grounding, this is the highest-ROI add-on.

| Reranker | Type | Offline | Quality | Speed | License | Use when |
|---|---|---|---|---|---|---|
| **bge-reranker-v2-m3** ⭐ | Cross-encoder | Yes | High, multilingual, strong BEIR | ~12 s / 1000 cands (≈10.6× slower than MiniLM) | Apache-2.0 | **Default deep reranker** on small candidate sets ([arXiv 2409.07691](https://arxiv.org/html/2409.07691v1), [BSWEN](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)) |
| **ms-marco-MiniLM-L-6/-12-v2** | Cross-encoder | Yes | Good (English) | **Fast/small** | Apache-2.0 | Latency-critical; shallow reranker scoring many candidates |
| **mxbai-rerank-v2** | Cross-encoder | Yes | Strong | Medium | Apache-2.0 | Alt to bge-reranker |
| **ColBERTv2 / PLAID** | Late-interaction | Yes | Near cross-encoder, **reusable index** | Fast query; **heavy storage** | OSS | When you want late-interaction as a *retriever-reranker* hybrid |
| **Cohere Rerank** | API | **❌ cloud — DISQUALIFIED** | — | — | — | **Never** (violates air-gap) |

**ColBERT/PLAID note:** late interaction (one vector per token) approaches cross-encoder accuracy at bi-encoder query speed, but storage explodes (~200×768 floats/doc; ~6 TB for 10M docs vs 30 GB bi-encoder) — **ColBERTv2 centroid-residual quantization cuts this ~10×**, and **PLAID** prunes candidates so centroids alone recover the top-k. For a *modest* internal NOC corpus this is viable, but a cross-encoder reranker over top-100 is simpler and lighter ([Weaviate late-interaction](https://weaviate.io/blog/late-interaction-overview), [PLAID arXiv](https://arxiv.org/pdf/2205.09707), [emergentmind](https://www.emergentmind.com/topics/colbertv2-retriever)).

**Recommendation — two-stage rerank:**
1. Hybrid retrieve **top ~150** (Anthropic uses 150) → optional fast **ms-marco-MiniLM-L-6-v2** shallow pass →
2. **bge-reranker-v2-m3** deep rerank → keep **top 5–20** for the prompt.

Rerank materially improves grounding whenever the first-stage list is noisy or queries mix jargon + intent (the NOC norm). The cost is one extra small model in the bundle and a few hundred ms — well worth it for the 35% grounding score.

---

## 5. Chunking & Ingestion (heterogeneous NOC artifacts)

NOC artifacts are *not* uniform prose. Use **structure-aware, per-type** chunkers, not one global splitter:

| Artifact | Recommended chunking | Rationale |
|---|---|---|
| **Markdown runbooks** | **Header/section-aware** (`MarkdownHeaderTextSplitter`) then recursive 400–512 tok, **keep tables/code intact** | Preserve procedure boundaries & step numbering ([Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)) |
| **Device configs** (IOS/Junos) | **Code/structure-aware** split on stanza boundaries (`interface`, `router bgp`, `policy-map`); don't split mid-stanza | Each stanza is a semantic unit; exact tokens must survive for BM25 |
| **Incident tickets** | **One ticket = parent**; chunk by field (symptom / RCA / resolution / timeline) | Field-level retrieval + parent-doc return for full context |
| **Topology / JSON** | **Element-based**, one node/link/site per record; emit BOTH a text serialization (for vectors) AND the structured edge (for the graph, Section 6) | JSON shouldn't be blindly char-split |
| **Syslog excerpts** | Group by **device+time window+event**; keep mnemonic codes verbatim | Bursts are the signal; codes drive exact-match |

**Default splitter:** `RecursiveCharacterTextSplitter` at **400–512 tokens, ~10–15% overlap** — reported **85–90% recall** with low overhead; semantic chunking adds up to ~9% recall at higher cost ([Firecrawl chunking](https://www.firecrawl.dev/blog/best-chunking-strategies-rag), [Agenta](https://agenta.ai/blog/the-ultimate-guide-for-chunking-strategies), [Matheus](https://matheusjerico.medium.com/chunking-strategies-for-rag-fixed-recursive-semantic-language-based-and-context-aware-4ab476aea7d1)). For code/config prefer recursive language-aware over semantic.

**Mandatory metadata on every chunk** (powers Qdrant payload filters + citations): `artifact_type, site, device, vendor, role(CE/PE/P), interface, vrf/vpn, severity, timestamp, source_path, line_range, embed_model+dim`.

**Parent-Document Retrieval:** index *small* chunks for precise matching but return the *parent* section/ticket to the LLM for complete context ([Databricks](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)).

### 5.1 Anthropic Contextual Retrieval — computed by the LOCAL LLM (high-impact, fully offline)

Before embedding, prepend a **50–100 token, chunk-specific context** generated by your **own air-gapped LLM** (Mistral-7B / Llama-3-8B / Phi-3), then embed *and* BM25-index the contextualized chunk ([Anthropic](https://www.anthropic.com/news/contextual-retrieval), [Claude cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)):

> Prompt (run locally per chunk): *"Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval."*

For NOC, bias the prompt to inject the **document identity** (e.g. *"From the BGP-flap runbook for the Mumbai hub PE; this step covers clearing a stuck adjacency on the MPLS-facing interface."*). Published gains:

- Contextual **embeddings** alone: **−35%** retrieval failure (5.7% → 3.7%)
- **+ Contextual BM25 (hybrid):** **−49%** (5.7% → 2.9%)
- **+ reranking:** **−67%** (5.7% → 1.9%)

This is a **one-time, offline, batch** preprocessing step (no per-query cost; Anthropic's $1.02/M-token figure is only relevant to their *cloud* Haiku — **your local LLM makes it free**). It is arguably the single highest-leverage technique for the grounding score. Libraries: **LlamaIndex** has a Contextual Retrieval cookbook; easy to hand-roll with **LangChain**/Haystack ([LlamaIndex](https://developers.llamaindex.ai/python/examples/cookbooks/contextual_retrieval/), [DataCamp](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic)).

**Heavy-parse fallback:** for PDFs/scanned diagrams use **`unstructured`** (offline) to extract elements/tables before chunking.

---

## 6. GraphRAG / Structured Retrieval (topology is a graph!)

The topology is **inherently a graph** (sites → CE/PE/P devices → interfaces → links → tunnels → VPNs), and the copilot must reason about **device relationships and blast-radius** ("if PE-2 flaps, which sites/VPNs/tunnels are affected?"). Pure text RAG cannot do multi-hop dependency reasoning; pure graph cannot read runbook prose. **Combine them.**

**Options considered:**
- **Microsoft GraphRAG** — LLM extracts entities/relations into a graph + hierarchical *community summaries*; great for "global" questions over large unstructured corpora, but it's a **heavy LLM-driven ETL** that *rediscovers* structure you already have ([Towards AI](https://pub.towardsai.net/exploring-and-comparing-graph-based-rag-approaches-microsoft-graphrag-vs-neo4j-langchain-3837cd3dddef)).
- **LlamaIndex `KnowledgeGraphIndex` / PropertyGraph** — fastest prototype (GraphRAG in <50 LoC), community detection + summaries ([LlamaIndex GraphRAG v2](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/)).
- **Neo4j (or Memgraph) + native vector** — mature graph DB with Cypher traversal **and** vector search in one store; most powerful, but adds a DB to operate ([Fastio](https://fast.io/resources/best-knowledge-graph-tools-rag/)).

**Pragmatic recommendation — "GraphRAG-lite" / dual-store, topology-first:**
1. **Build the graph from the source of truth, not from an LLM.** You already have the simulated topology (Phase 1) + live telemetry (Phase 2) — load nodes/edges directly into **NetworkX (in-process, zero-ops)** or **Neo4j** if you want Cypher + persistence. Far more accurate than LLM entity extraction, and fully offline.
2. **Blast-radius as a graph query, not an LLM guess.** Compute affected scope deterministically: descendants/reachable set, shortest paths, betweenness/articulation points to flag single-points-of-failure. *"Blast radius of a router failure on network flows traversing it"* and *"dependency graphs predict incident blast radius and which downstream systems are affected"* ([USPTO failure-impact](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11269711), [Grafana service deps](https://grafana.com/docs/grafana-cloud/knowledge-graph/use-cases/explore-dependencies/)).
3. **Fuse graph facts into the RAG context.** On a query/alert about device X: (a) run the graph query → structured "affected sites/VPNs/tunnels + path" facts; (b) run hybrid text RAG → relevant runbook/incident chunks; (c) put **both** into the grounded prompt. The graph answers *"what/where/how-bad"* (affected scope, time-to-impact propagation), the text RAG answers *"why/what-to-do"* (root-cause hypotheses, remediation steps). This directly serves the three required questions **Q1 (what fails next/when), Q2 (which signals), Q3 (what action)** and the structured response fields (issue, confidence, root-cause, affected scope, ETA).

> Net: **don't pay GraphRAG's LLM-ETL tax to rebuild a graph you already own.** Use the authoritative topology graph for relationship/blast-radius reasoning and reserve vector RAG for prose. Optionally store node/edge embeddings in Qdrant so "find devices similar to the failing one" works too.

---

## 7. RAG Framework (leanest, most controllable for air-gap)

| Framework | Overhead | Control / observability | Air-gap fit | Verdict |
|---|---|---|---|---|
| **Thin custom pipeline** ⭐ | Minimal | **Total** | Best | **Recommended** — fewest deps to vendor into the air-gap, no hidden network calls, full control of grounding/citations |
| **Haystack 2.x** ⭐(if framework wanted) | **~5.9 ms** | **Explicit DAG, inputs/outputs validated**, easy ground-truth eval | Excellent | Best *framework* for search-heavy, observable RAG ([Medium showdown](https://mayur-ds.medium.com/langchain-vs-haystack-vs-llamaindex-rag-showdown-2025-28c222d34b0a)) |
| **LlamaIndex** | ~6 ms | Great ingestion/index abstractions | Good | Best for fast ingestion + ready GraphRAG/Contextual cookbooks |
| **LangChain** | **~10 ms** | Flexible but heavy, many transitive deps | OK | Fast prototyping; **most deps to audit for an air-gap** |

Overhead/token figures from [RAG showdown 2025](https://mayur-ds.medium.com/langchain-vs-haystack-vs-llamaindex-rag-showdown-2025-28c222d34b0a) and [iCert](https://www.icertglobal.com/community/haystack-vs-langchain-vs-llamaindex-for-production-rag-2026).

**Recommendation:** For a *judged, security-reviewed, air-gapped* build where **"verifiably zero outbound dependency"** is 20% of the score, prefer a **thin custom pipeline** (sentence-transformers + Qdrant client + bm25s/SPLADE + reranker + your local LLM) — it minimizes the dependency surface you must vendor and audit, and gives you direct control over **citation enforcement**. If you want a framework's scaffolding/observability, use **Haystack 2.x** (explicit, validated DAG, lowest overhead, strong retrieval focus). Use **LlamaIndex** specifically for its ingestion + Contextual-Retrieval/GraphRAG cookbooks if you don't want to hand-roll those. Pin every wheel and mirror to a local PyPI for reproducible offline installs.

---

## 8. Grounding & Evaluation (offline) — proving low hallucination for the 35%

**Goal:** demonstrably show the copilot is *grounded in local retrieval without hallucination*. Two metric families, **both runnable fully offline**.

### 8.1 Retrieval-quality metrics (no LLM judge needed)
Build a small **golden set**: queries ↔ known-relevant chunk IDs (hand-label real NOC questions + synthesize from runbooks/incidents). Compute ([Future AGI](https://medium.com/@future_agi/how-to-evaluate-rag-systems-the-complete-technical-guide-bea586a01c69), [ThinkingLoop](https://medium.com/@ThinkingLoop/top-10-retrieval-metrics-for-tuning-your-rag-14c61d5957f4)):
- **Hit-Rate@k** — % queries with ≥1 relevant doc in top-k (target **≥0.90 @ k=10**).
- **MRR** — rank of first relevant doc (higher = answer found faster).
- **nDCG@10** — rank-weighted relevance (target **>0.8**).
- **Recall@k / Precision@k**, **Context Precision / Context Recall**.

These objectively tune embedding choice, hybrid weights, `ef_search`, and rerank depth — **no LLM required**, so they're deterministic and air-gap-trivial.

### 8.2 Generation grounding / anti-hallucination (LOCAL LLM judge)
Run **RAGAS** and/or **DeepEval** with a **local Ollama judge model** (e.g. Llama-3-8B / Qwen-2.5 / Mistral-7B) and **local embeddings** — *"RAGAS evaluation can be conducted using an Ollama-hosted local LLM judge, enabling offline evaluation without commercial APIs"* ([RAGAS guide](https://dkaarthick.medium.com/ragas-for-rag-in-llms-a-comprehensive-guide-to-evaluation-metrics-3aca142d6e38), [Local RAG+RAGAS tutorial](https://medium.com/@QuarkAndCode/local-rag-tutorial-langchain-ollama-chromadb-with-ragas-481c1c346624)). Core metrics:
- **Faithfulness** — are answer claims entailed by the retrieved context? (the primary hallucination signal for RAG — *"the source of truth is the retrieval_context your retriever fetched"* ([DeepEval](https://deepeval.com/docs/metrics-hallucination))).
- **Answer Relevancy** — does the answer address the query?
- **Context Precision / Context Recall** — was the retrieved context on-point and sufficient?
- **TruLens** alternative: the **RAG Triad** (context relevance, groundedness, answer relevance) for continuous monitoring ([Atlan](https://atlan.com/know/llm-evaluation-frameworks-compared/)).
- Optional offline classifier: **Vectara HHEM-2.1-Open** (open-source hallucination detector) to corroborate faithfulness ([Atlan](https://atlan.com/know/llm-evaluation-frameworks-compared/)).

All three frameworks are **reference-free** (no labeled answers needed) and LLM-as-judge based, so they run entirely in the air-gap ([Atlan](https://atlan.com/know/llm-evaluation-frameworks-compared/)). **DeepEval** is Pytest-style with pass/fail thresholds — wire it into the build so a regression that drops faithfulness **fails the pipeline** (great for the documentation/rigor story).

### 8.3 Citation enforcement (hard grounding guarantee)
Beyond metrics, *force* grounding at generation time:
- Make the LLM cite the **chunk IDs / source_path+line_range** it used; **reject/regenerate** any sentence lacking a citation.
- Constrain answers to the structured schema (predicted issue, confidence, root-cause hypothesis, affected scope, recommended actions) with **every field traceable** to a retrieved chunk or a graph fact.
- If retrieval confidence is low (e.g. top rerank score below threshold or low Hit-Rate), have the copilot **say "insufficient local evidence"** rather than hallucinate — this is exactly the behavior judges reward under "grounded without hallucination."

### 8.4 What to put in the demo/report (to win the 35%)
A table showing, on the golden set: **Hit-Rate@10, MRR, nDCG@10** for *dense-only vs hybrid vs hybrid+rerank vs +contextual-retrieval*, plus **RAGAS Faithfulness/Answer-Relevancy/Context-Precision** for the final config — demonstrating the **monotonic grounding improvement** mirroring Anthropic's 5.7%→1.9% failure curve, all computed offline. Cross-check faithfulness against the live fault-injection scenarios (Phase 6) so copilot explanations are validated against ground-truth labels.

---

## Appendix A — Concrete library/version checklist (vendor into the air-gap)
- **Embeddings:** `sentence-transformers` + `FlagEmbedding` (bge-m3); fallback `bge-small-en-v1.5` / `all-MiniLM-L6-v2`. Optionally `fastembed` for SPLADE.
- **Vector DB:** Qdrant server binary + `qdrant-client` (on-disk mode); OR `faiss-cpu`/`faiss-gpu` + `bm25s` for the embeddable variant; `lancedb` as embedded hybrid alternative.
- **Sparse:** Qdrant sparse vectors / `bm25s` / `rank_bm25`.
- **Reranker:** `bge-reranker-v2-m3` (+ `ms-marco-MiniLM-L-6-v2` fast tier) via sentence-transformers CrossEncoder / FlagEmbedding.
- **Chunking/parse:** `langchain-text-splitters` (Recursive/MarkdownHeader), `unstructured` for PDFs/tables; custom stanza splitter for configs.
- **Graph:** `networkx` (in-process) or Neo4j + driver for blast-radius/Cypher.
- **LLM runtime:** Ollama / llama.cpp serving quantized Mistral-7B / Llama-3-8B / Phi-3 (per idea.md) — same model does **contextual-prefix generation, answer generation, and the eval judge**.
- **Eval:** `ragas`, `deepeval`, optional `trulens`, plus a small custom Hit-Rate/MRR/nDCG harness.
- **Air-gap hygiene:** pin all wheels, mirror to a local PyPI, pre-download all HF weights, set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, and block egress at runtime (proves the 20% offline-compliance score).

## Appendix B — Recommended retrieval pipeline (exact flow)
```
[Ingest]  runbooks(md) | configs | tickets | topology(json) | syslog
   │  per-type structure-aware chunking (400–512 tok, 10–15% overlap;
   │  keep tables/code/stanzas intact); attach metadata
   ▼
[Contextual prefix]  local LLM writes 50–100-tok context per chunk  (offline, batch)
   ▼
[Embed]  bge-m3 → dense(1024) + learned-sparse        ┐
[Index]  Qdrant: HNSW(M=16, efc=200) + sparse index;  ├─ also emit topology
         payload = {site,device,vendor,role,ts,...}    ┘    edges → graph store
   ▼
[Query/Alert]
   ├─ Hybrid retrieve top ~150  (dense ⊕ sparse)
   ├─ RRF fuse (k=60)                          ── score-agnostic
   ├─ (optional) ms-marco-MiniLM shallow rerank
   ├─ bge-reranker-v2-m3 deep rerank → top 5–20
   └─ Graph query (blast-radius / affected scope / path) on the failing device
   ▼
[Generate]  local LLM, grounded prompt = reranked chunks + graph facts;
            enforce per-field citations (chunk_id / source:line) ;
            abstain if evidence insufficient
   ▼
[Eval gate]  offline RAGAS+DeepEval (faithfulness/relevancy/precision)
             + Hit-Rate/MRR/nDCG on golden set  → fail build on regression
```

---

### Key sources
- Anthropic, *Introducing Contextual Retrieval* — https://www.anthropic.com/news/contextual-retrieval
- BAAI/bge-m3 model card — https://huggingface.co/BAAI/bge-m3 ; M3 paper — https://arxiv.org/html/2402.03216v3
- Nomic Embed v1.5 (Matryoshka) — https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 ; https://zilliz.com/ai-models/nomic-embed-text-v1.5
- Ollama embedding benchmark table — https://www.morphllm.com/ollama-embedding-models
- Open-source embedding guides — https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models ; https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/
- Vector DB comparisons — https://www.firecrawl.dev/blog/best-vector-databases ; https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3
- Qdrant: storage — https://qdrant.tech/documentation/manage-data/storage/ ; sparse vectors — https://qdrant.tech/articles/sparse-vectors/ ; SPLADE/FastEmbed — https://qdrant.tech/documentation/fastembed/fastembed-splade/ ; BM42 — https://qdrant.tech/articles/bm42/
- FAISS indexes — https://github.com/facebookresearch/faiss/wiki/Faiss-indexes ; HNSWFlat — https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWFlat.html
- HNSW tuning — https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall ; https://opensourceconnections.com/blog/2025/02/27/vector-search-navigating-recall-and-performance/
- Hybrid + RRF — https://ashutoshkumars1ngh.medium.com/hybrid-search-done-right-fixing-rag-retrieval-failures-using-bm25-hnsw-reciprocal-rank-fusion-a73596652d22 ; https://weaviate.io/blog/hybrid-search-explained
- bm25s — https://bm25s.github.io/
- Reranker benchmark — https://arxiv.org/html/2409.07691v1 ; best rerankers — https://docs.bswen.com/blog/2026-02-25-best-reranker-models/
- ColBERT/PLAID — https://weaviate.io/blog/late-interaction-overview ; https://arxiv.org/pdf/2205.09707
- Chunking — https://www.firecrawl.dev/blog/best-chunking-strategies-rag ; https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
- GraphRAG — https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/ ; https://pub.towardsai.net/exploring-and-comparing-graph-based-rag-approaches-microsoft-graphrag-vs-neo4j-langchain-3837cd3dddef ; blast-radius — https://grafana.com/docs/grafana-cloud/knowledge-graph/use-cases/explore-dependencies/
- RAG frameworks — https://mayur-ds.medium.com/langchain-vs-haystack-vs-llamaindex-rag-showdown-2025-28c222d34b0a
- Eval — https://atlan.com/know/llm-evaluation-frameworks-compared/ ; https://medium.com/@QuarkAndCode/local-rag-tutorial-langchain-ollama-chromadb-with-ragas-481c1c346624 ; retrieval metrics — https://medium.com/@future_agi/how-to-evaluate-rag-systems-the-complete-technical-guide-bea586a01c69 ; DeepEval hallucination — https://deepeval.com/docs/metrics-hallucination
