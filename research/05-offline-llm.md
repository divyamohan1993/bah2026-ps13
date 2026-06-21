# Phase 4 — Offline, Air-Gapped, Quantized LLM for the NOC Copilot

Research report for **Problem Statement 13 — Air-Gapped Predictive Copilot for Secure MPLS Operations.**

**Scope:** model selection, quantization, air-gapped serving runtimes (with telemetry/phone-home analysis), schema-valid structured output, anti-hallucination/grounding, copilot prompt/agent design, and hardware sizing/latency.

**Design constraints that drive every recommendation below:**
- **100% offline / air-gapped** — no outbound network access at runtime; the entire model + runtime + RAG index must be bundled and run with verifiable zero egress (evaluation: 20% "Security & Offline Compliance").
- **Free / open-weight** — license must permit offline redistribution and bundling.
- **Grounded & schema-valid output** — answers must be correct, operator-relevant, grounded only in local retrieval, with no hallucination, and emitted in a fixed structured schema (evaluation: 35% "Copilot Effectiveness").

---

## 0. TL;DR / Executive Recommendation

| Decision | Recommendation | Why |
|---|---|---|
| **Primary model** | **Qwen2.5-7B-Instruct** (GGUF **Q5_K_M** on GPU, **Q4_K_M** on CPU) | Best reasoning + structured-output among 7-9B class; **Apache-2.0** (cleanest for air-gap bundling); 128K context for RAG; strong tool/function calling. |
| **CPU-only fallback** | **Qwen2.5-3B-Instruct** (Q4_K_M) or **Llama-3.2-3B-Instruct** (Q4_K_M); **Phi-3.5-mini** (MIT) as alt | Runs at usable speed on CPU (6-10 tok/s); Qwen2.5-3B is **Apache-2.0**, Phi-3.5-mini is **MIT**. |
| **Higher-quality option (if ≥12 GB VRAM)** | **Mistral-Nemo-12B-Instruct** (Q4_K_M, Apache-2.0) or **Gemma-2-9B-it** (Q5_K_M) | Better quality than 8B class when VRAM allows; Nemo is Apache-2.0 + 128K context. |
| **Serving runtime** | **llama.cpp `llama-server`** (primary). Ollama acceptable for dev ergonomics **only if its update-check is firewalled/disabled**. | `llama-server` has **no auto-updater and no telemetry**, OpenAI-compatible API, native GBNF + JSON-schema constrained decoding. Strongest verifiable air-gap story. |
| **Structured output** | **JSON-Schema → GBNF grammar-constrained decoding** in `llama-server` (`response_format: json_schema` / `grammar`), wrapped with **Pydantic** validation (via Instructor or a thin client). | *Guarantees* schema-valid JSON at the sampler level — not "best effort" prompting. |
| **Grounding / anti-hallucination** | "Answer only from context" + mandatory **citations to chunk IDs** + **abstain-when-insufficient**, verified offline with **Vectara HHEM-2.1-Open** (DeBERTa-v3 NLI, <600 MB, CPU) for faithfulness scoring and a local RAGAS-style harness. | Gives a *measurable, fully offline* hallucination rate for the 35% copilot score. |
| **Quant levels** | GPU: **Q5_K_M** (quality-first) or **Q4_K_M** (speed). CPU: **Q4_K_M**. Avoid ≤Q3 for math/reasoning. | Q5_K_M ≈ FP16 quality; Q4_K_M is the size/speed sweet spot. |

---

## 1. Model Selection

### 1.1 Candidate landscape (small/open-weight instruct models runnable fully offline)

All candidates are downloadable as weights (safetensors and/or pre-quantized GGUF) from Hugging Face and can be **snapshot-downloaded once, then bundled** into the air-gapped image. None require a license server or online activation.

| Model | Params | License | Context | Notable strengths | Air-gap notes |
|---|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 7.6B | **Apache-2.0** (7B, 3B, 1.5B, 0.5B, 14B, 32B) | 128K (YaRN) | Top reasoning/coding/structured-output in class; strong tool calling | GGUF on HF (`Qwen/Qwen2.5-7B-Instruct-GGUF` + community); fully bundlable |
| **Qwen2.5-14B-Instruct** | 14.7B | **Apache-2.0** | 128K | Noticeably stronger reasoning; needs ~10-12 GB VRAM at Q4-Q5 | Use if a single 16-24 GB GPU is available |
| **Qwen2.5-Coder-7B-Instruct** | 7.6B | **Apache-2.0** | 128K | Best at emitting/processing configs, CLI, structured text (useful for runbook/CLI-heavy NOC answers) | Good secondary "config specialist" |
| **Llama-3.1-8B-Instruct** | 8B | **Llama 3.1 Community License** (custom) | 128K | Strong general instruct + agentic/tool fine-tuning; huge ecosystem | License has redistribution/attribution + 700M-MAU clause (see §1.4) |
| **Llama-3.2-3B-Instruct** | 3.2B | Llama 3.2 Community License | 128K | Good CPU-class model, tool-calling tuned | Same custom-license caveats |
| **Phi-3.5-mini-instruct** | 3.8B | **MIT** | 128K | Excellent reasoning-per-byte; MIT = simplest license | Great CPU fallback; MIT ideal for redistribution |
| **Phi-3-mini-4k/128k-instruct** | 3.8B | **MIT** | 4K / 128K | Microsoft ships official GGUF | Named in the PS suggested tools |
| **Gemma-2-9B-it** | 9B | **Gemma Terms of Use** (custom) | 8K | Very strong quality for 9B | 8K context limits RAG; custom license w/ prohibited-use policy |
| **Mistral-7B-Instruct** | 7.3B | **Apache-2.0** | 32K | Named in PS; fast, permissive | Older; outclassed by Qwen2.5-7B on reasoning |
| **Mistral-Nemo-Instruct-2407** | 12B | **Apache-2.0** | 128K | Best Apache-2.0 quality if you have ~12 GB VRAM; 128K context | Strong upgrade path; Apache-2.0 |

### 1.2 Quality comparison (why Qwen2.5-7B is the primary pick)

Per the Qwen2.5 technical report and Qwen blog, **Qwen2.5-7B-Instruct** scores **MMLU ≈ 74.2** (up from 70.3 on Qwen2-7B), **MATH ≈ 49.8-75.5** (reported variously by config), **HumanEval ≈ 57.9-84.8**, and the Qwen team states it *"significantly outperforms Gemma2-9B-IT and Llama3.1-8B-Instruct across all tasks except IFEval"* despite having fewer parameters. ([qwenlm.github.io/blog/qwen2.5-llm](https://qwenlm.github.io/blog/qwen2.5-llm/), [arxiv.org/pdf/2412.15115](https://arxiv.org/pdf/2412.15115))

Independent commentary corroborates: *"Qwen 2.5 7B beats Llama 3.1 8B on multilingual and Asian languages and matches it on English reasoning,"* and *"Mistral Nemo 12B is recommended when you have ~12 GB VRAM and want better-than-8B quality."* ([glukhov.org](https://www.glukhov.org/llm-performance/benchmarks/mistral-small-gemma2-qwen2-5-mistral-nemo/))

The one consistent gap is **IFEval** (strict instruction-following), where **Llama-3.1-8B** is competitive or slightly ahead. We mitigate this for the copilot by **constraining output with a grammar** (§4) rather than relying on the model to follow formatting instructions perfectly — which neutralizes Qwen's IFEval gap for our specific structured-output need.

### 1.3 Structured-output / function-calling ability

- **Qwen2.5** ships with first-class tool-calling support and integrates with vLLM's built-in tool calling (`--enable-auto-tool-choice --tool-call-parser hermes`). It is widely used with Outlines/XGrammar for constrained JSON. ([qwenlm.github.io/blog/qwen2.5](https://qwenlm.github.io/blog/qwen2.5/))
- **Llama-3.1-8B-Instruct** is explicitly *"fine-tuned on tool calling for agentic use cases"* with documented tool-use schemas/chat templates. ([huggingface.co/blog/llama31](https://huggingface.co/blog/llama31))
- For our design, **the runtime's grammar constraint (§4) is the source of truth for structure**, so model-native function-calling is a convenience, not a dependency. This makes the choice robust across any of the candidates.

### 1.4 License analysis (critical for air-gapped *bundling/redistribution*)

| Model | License | Can you bundle weights into an offline appliance/image? | Caveats |
|---|---|---|---|
| Qwen2.5 (7B/14B/3B/Coder) | **Apache-2.0** | **Yes, freely** | None material for hackathon/enterprise; outputs unrestricted. ([huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE)) |
| Phi-3 / Phi-3.5 / Phi-4 | **MIT** | **Yes, freely** | Most permissive; redistribute/modify without restriction. ([huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)) |
| Mistral-7B / Mistral-Nemo-12B | **Apache-2.0** | **Yes, freely** | Clean. |
| Llama 3.1 / 3.2 | **Llama Community License** (custom) | **Yes, with conditions** | Must include attribution ("Built with Llama"), ship the license, follow Acceptable Use Policy; commercial license required only if >700M MAU. Redistribution allowed under these terms. ([llama.com/llama3_1/license](https://www.llama.com/llama3_1/license/)) |
| Gemma 2 | **Gemma Terms of Use** (custom) | **Yes, with conditions** | Custom terms + Prohibited Use Policy, flow-down obligations to downstream users, unilateral termination rights. ([ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms)) |

**Recommendation:** For an air-gapped, *redistributable* government/enterprise appliance, prefer **Apache-2.0 (Qwen2.5, Mistral-Nemo) or MIT (Phi-3.5)** to avoid custom flow-down/usage clauses and termination risk. Llama 3.1/3.2 is usable but adds attribution + AUP compliance overhead; Gemma's 8K context and custom terms make it the least attractive here. ([techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions](https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/))

### 1.5 Final model recommendation

- **Primary:** **Qwen2.5-7B-Instruct** — best balance of reasoning, structured output, 128K context for RAG, and an unambiguous **Apache-2.0** license. Run **Q5_K_M** on a GPU, **Q4_K_M** on CPU.
- **CPU-only fallback:** **Qwen2.5-3B-Instruct (Q4_K_M)** (Apache-2.0) — keeps the entire stack on one permissive license. **Phi-3.5-mini-instruct (Q4_K_M, MIT)** is an equally good, slightly stronger-reasoning alternative and is explicitly named-compatible with the PS.
- **Quality upgrade (≥12 GB VRAM):** **Mistral-Nemo-12B-Instruct (Q4_K_M, Apache-2.0, 128K)** or **Qwen2.5-14B-Instruct (Q4_K_M)**.

---

## 2. Quantization

### 2.1 Format primer

| Format | Engine | Target HW | Notes |
|---|---|---|---|
| **GGUF + k-quants** (Q2_K…Q8_0, `_S/_M/_L`) | **llama.cpp / Ollama / LM Studio** | **CPU, GPU, Apple, hybrid CPU+GPU offload** | Most portable. `_M` variants keep sensitive layers (attn, some FFN) at higher precision — better quality per byte. **Best for air-gapped CPU-or-modest-GPU.** |
| **AWQ** (Activation-aware Weight Quant, 4-bit) | vLLM / TGI / AutoAWQ | **GPU only** | Calibrated 4-bit; strong quality at high GPU throughput. |
| **GPTQ** (4-bit, calibrated) | vLLM / TGI / ExLlama | **GPU only** | Mature; AWQ usually ≥ GPTQ quality at same bits. |
| **bitsandbytes** (NF4/INT8, on-the-fly) | Transformers / vLLM | **GPU** | Easy load-time quant; slower than AWQ/GPTQ for serving; good for quick tests. |
| **EXL2** (variable bpw) | ExLlamaV2 | **GPU (NVIDIA)** | Very fast, flexible bits-per-weight; excellent single-GPU latency, less portable than GGUF. |

### 2.2 Quality vs size vs speed — measured numbers (Llama-3.1-8B-Instruct)

From a unified llama.cpp quantization study on Llama-3.1-8B-Instruct (WikiText-2 perplexity, FP16 baseline 7.32; aggregate of GSM8K/HellaSwag/IFEval/MMLU). ([arxiv.org/html/2601.14277v1](https://arxiv.org/html/2601.14277v1))

| Quant | Perplexity (↓) | Avg benchmark (↑) | Size (MiB) | Size reduction | CPU token-gen (tok/s) |
|---|---|---|---|---|---|
| **FP16** | 7.32 | 69.47 | 15,317 | — | 2.83 |
| Q3_K_S | 8.96 | 65.49 | 3,487 | 77.2% | 9.91 |
| Q3_K_M | 7.96 | — | — | — | — |
| **Q4_K_S** | 7.62 | 69.17 | 4,468 | 70.8% | 4.65 |
| **Q5_0** | 7.43 | **69.92** | 5,332 | 65.2% | 6.66 |
| **Q6_K** | 7.35 | — | — | — | — |
| **Q8_0** | 7.33 | 69.41 | 8,138 | 46.9% | 5.03 |

**Key takeaways from the study:**
- **Q5_0 / Q5_K_M slightly *match or beat* FP16** on aggregate downstream tasks while cutting size ~65% — i.e., 5-bit k-quants are effectively lossless for our purposes.
- **Q4_K_S/Q4_K_M** lose only ~0.3 ppl and ~0.3 avg-points vs FP16 — the size/speed sweet spot (~70-75% smaller).
- **Q3 is risky for reasoning:** GSM8K drops ~9 points at Q3_K_S (77.6% → 68.3%). **Avoid ≤Q3 for the NOC copilot** because root-cause reasoning and time-to-impact arithmetic matter.
- *"Quantization format matters, not just nominal bit-width"* — a well-designed Q4_K_M beats a naive legacy 5-bit format. ([arxiv.org/html/2601.14277v1](https://arxiv.org/html/2601.14277v1), [github.com/ggml-org/llama.cpp discussions/2094](https://github.com/ggml-org/llama.cpp/discussions/2094))

General-purpose guidance from practitioners aligns: **Q4_K_M is the workhorse (~4.8 bits/weight), Q5_K_M when you can afford it, Q8_0 indistinguishable from FP16 but rarely worth the size/speed cost.** ([runaihome.com/blog/quantization-q4-q5-q6-q8-quality-loss-2026](https://runaihome.com/blog/quantization-q4-q5-q6-q8-quality-loss-2026/), [tonisagrista.com/blog/2026/quantization](https://tonisagrista.com/blog/2026/quantization/))

### 2.3 Approximate RAM/VRAM footprint (weights only; add KV cache + overhead)

GGUF k-quant on-disk size ≈ in-memory weight size. Add **~0.5-2 GB** for KV cache at moderate context (more at 32K-128K) and runtime overhead. Numbers below are rounded weight footprints; engineer for **disk size + ~20-30% headroom**.

| Model size | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | FP16 | Min. practical RAM/VRAM (Q4_K_M, ~8K ctx) |
|---|---|---|---|---|---|---|
| **3B** (Qwen2.5-3B, Phi-3.5, Llama-3.2-3B) | ~2.0 GB | ~2.3 GB | ~2.6 GB | ~3.3 GB | ~6.5 GB | **~4 GB** (runs on CPU laptop) |
| **7B** (Qwen2.5-7B, Mistral-7B) | ~4.4 GB | ~5.3 GB | ~6.0 GB | ~8.1 GB | ~15.3 GB | **~6 GB** (fits 8 GB GPU / 16 GB RAM) |
| **8B** (Llama-3.1-8B) | ~4.9 GB | ~5.7 GB | ~6.6 GB | ~8.5 GB | ~16 GB | **~6-7 GB** |
| **9B** (Gemma-2-9B) | ~5.8 GB | ~6.6 GB | ~7.6 GB | ~9.8 GB | ~18 GB | **~8 GB** |
| **12B** (Mistral-Nemo) | ~7.5 GB | ~8.7 GB | ~10 GB | ~13 GB | ~24 GB | **~10-12 GB** |
| **14B** (Qwen2.5-14B) | ~9 GB | ~10.5 GB | ~12 GB | ~16 GB | ~30 GB | **~12 GB** |

(Footprints consistent with TheBloke/Bartowski GGUF release sizes and the 8B study above; e.g. 8B Q4_K_S = 4.47 GB measured.) Rule of thumb for fit: **8 GB VRAM → 7-8B comfortably; 12 GB → 8-9B with long context or 12-14B at Q4; 24 GB → 14B at Q8 or 30B at Q4.** ([huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally))

### 2.4 Quant recommendation

- **GPU (RTX 3060/4090/T4):** **Q5_K_M** (quality-first; ≈FP16) for the 7-8B primary. Drop to **Q4_K_M** only if you need maximum throughput or longer context in limited VRAM. Use **AWQ 4-bit** instead of GGUF only if you commit to **vLLM** for batched multi-operator throughput.
- **CPU-only:** **Q4_K_M** for 7B, or **Q4_K_M** on a **3B** fallback for snappy interactivity. Never go below Q4 for the reasoning-heavy copilot.

---

## 3. Serving Runtimes (Air-Gapped) — with Telemetry / Phone-Home Analysis

This is the **20% "Security & Offline Compliance"** lever. The goal: a runtime that (a) serves a **pre-pulled** model from local disk, (b) exposes an **OpenAI-compatible API + structured outputs**, and (c) makes **no outbound calls at runtime** that we cannot fully disable and *verify*.

### 3.1 Comparison

| Runtime | OpenAI API | Structured output | CPU support | GPU throughput | Telemetry / phone-home | Air-gap verdict |
|---|---|---|---|---|---|---|
| **llama.cpp `llama-server`** | **Yes** (`/v1/chat/completions`) | **Native** GBNF + `response_format: json_schema`/`grammar` (JSON-Schema→GBNF) | **Excellent** (CPU-first project) | Good (CUDA/Vulkan/Metal); single-stream | **None.** No analytics, **no auto-updater** (you build/download the binary yourself). | **Best.** Smallest, most auditable; nothing to disable. ([github.com/ggml-org/llama.cpp/.../server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) |
| **Ollama** | **Yes** (`/v1/...`) | JSON mode + `format` (JSON-Schema) | Good | Good | **No prompt/usage telemetry by default, BUT it checks for updates on launch** on macOS/Windows and has an auto-update path; this is an outbound call you must block. | **Good with hardening** — must firewall/disable the update check to be verifiably air-gapped (see §3.3). ([docs.ollama.com/faq](https://docs.ollama.com/faq), [github.com/ollama/ollama/issues/6024](https://github.com/ollama/ollama/issues/6024)) |
| **vLLM** | **Yes** (reference impl) | **Yes** — guided decoding via **Outlines/XGrammar** (`guided_json`, `guided_grammar`) | Limited/secondary | **Best** (paged-attention, batching, multi-operator) | **Usage stats collection** (anonymous) — **disable with `VLLM_NO_USAGE_STATS=1` / `DO_NOT_TRACK=1`**; also tries HF Hub unless `HF_HUB_OFFLINE=1` + local path. | **Good with config** — set offline env vars; serve from local path. Heaviest dependency footprint. ([github.com/vllm-project/vllm/issues/9255](https://github.com/vllm-project/vllm/issues/9255), [docs.forjinn.com/.../air-gapped-llm-support](https://docs.forjinn.com/components-guide/air-gapped-llm-support)) |
| **TGI** (HF Text-Generation-Inference) | Messages API | Guided/JSON-schema support | Limited | Very good | Pulls from HF Hub by default; supports offline with pre-downloaded weights + `HF_HUB_OFFLINE=1`. | OK; air-gap documented but heavier than llama.cpp. ([docs.forjinn.com/.../air-gapped-llm-support](https://docs.forjinn.com/components-guide/air-gapped-llm-support)) |
| **LM Studio** | Yes (local server) | JSON schema | Good | Good | **GUI app**; offline-capable but closed-source desktop app — auditing egress is harder; update checks. | Dev/demo only; **not** ideal for a verifiable headless appliance. |
| **MLC-LLM** | Yes-ish | Limited | Yes (compiled) | Good (TVM, broad HW) | No server telemetry; compiles model to target. | Niche; good for exotic HW, more build complexity. |
| **LocalAI** | **Yes** (drop-in OpenAI) | JSON/grammar (wraps llama.cpp/others) | Yes | Good | Self-hosted; no mandatory phone-home. | Good OpenAI-compatible wrapper if you want a multi-backend gateway. ([markaicode.com/best/air-gapped-ai-stack](https://markaicode.com/best/air-gapped-ai-stack/)) |

Corroboration: *"llama.cpp and Ollama run entirely on local hardware with no background telemetry… ideal for air-gapped deployments… vLLM requires network configuration for maximum data isolation."* ([quantizelab.dev/.../llama-cpp-vs-ollama-vs-vllm](https://www.quantizelab.dev/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide), [insiderllm.com/guides/running-ai-offline-complete-guide](https://insiderllm.com/guides/running-ai-offline-complete-guide/))

### 3.2 Recommendation

**Use `llama.cpp` `llama-server` as the production air-gapped runtime.**
Rationale for the 20% offline-compliance score:
1. **No telemetry and no auto-updater** — the binary you compile/copy is inert with respect to the network; there is literally nothing to "disable." This is the strongest *verifiable* claim.
2. **Native structured output** — JSON-Schema is converted to a **GBNF grammar** and enforced at the sampler (`response_format` / `grammar`), satisfying §4 with no extra library. ([github.com/ggml-org/llama.cpp/.../server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md))
3. **CPU + modest-GPU first** — matches our hardware target and the CPU fallback.
4. **OpenAI-compatible** — the RAG/orchestration layer talks to `http://127.0.0.1:8080/v1` exactly like any OpenAI client, so swapping models/runtimes later is trivial.

**For multi-operator/high-throughput NOC** (many concurrent queries), add **vLLM with AWQ** as an optional high-throughput backend, hardened per §3.4. **Ollama** is fine for *development ergonomics* (`ollama pull`/`ollama run`) but should be either (a) replaced by `llama-server` in the appliance, or (b) hardened by blocking its update endpoint (§3.3).

### 3.3 Hardening Ollama for verifiable air-gap (if used)

- Ollama **does not send prompt/usage telemetry** and *"runs locally… We don't see your prompts or data when you run locally."* ([docs.ollama.com/faq](https://docs.ollama.com/faq))
- **However**, Ollama **checks for / downloads updates on launch** (macOS/Windows) — an open issue requests disabling auto-update, confirming the behavior exists. ([github.com/ollama/ollama/issues/6024](https://github.com/ollama/ollama/issues/6024))
- Hardening steps for the appliance:
  - Run the **Linux server build** (no desktop auto-updater) and **bind to localhost**: `OLLAMA_HOST=127.0.0.1:11434`.
  - Set `OLLAMA_NO_CLOUD=1` (disable cloud features) and pre-pull all models so no `ollama pull` is needed at runtime; set `OLLAMA_MODELS` to the bundled blob dir and `OLLAMA_NOPRUNE=1` to keep blobs. ([modelpiper.com/blog/ollama-environment-variables](https://modelpiper.com/blog/ollama-environment-variables), [pkg.go.dev/github.com/ollama/ollama/envconfig](https://pkg.go.dev/github.com/ollama/ollama/envconfig))
  - **Enforce no-egress at the OS/network layer** (the real guarantee): default-deny outbound firewall, no DNS, no default route (§3.4). This makes any residual update check fail harmlessly and is the verifiable control regardless of app behavior.

### 3.4 Verifiable no-egress (the actual compliance evidence)

Don't rely on app config alone — **prove it at the network layer** and capture evidence:
- **Default-deny outbound firewall** (`iptables -P OUTPUT DROP` / nftables), allow only `lo`; no nameservers in `/etc/resolv.conf`; remove the default route. Run inside a **network namespace / container with `--network=none`** (or a bridge with no NAT).
- For the demo: bind the runtime to **127.0.0.1** and run the whole stack (`llama-server` + vector DB + RAG API + UI) on loopback.
- **Audit & record** during a full copilot run: `ss -tunap` (no foreign ESTAB sockets), `sudo tcpdump -ni any 'not (host 127.0.0.1)'` (zero packets), optional `strace -f -e trace=connect` to show no `connect()` to non-loopback. ([ai-ollama.github.io/privacy-offline.html](https://ai-ollama.github.io/privacy-offline.html))
- Set offline env vars belt-and-suspenders: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `GRADIO_ANALYTICS_ENABLED=False` (if Gradio UI). ([github.com/vllm-project/vllm/issues/9255](https://github.com/vllm-project/vllm/issues/9255))

This packet-capture + socket-table evidence is exactly what "verifiably zero outbound dependency during runtime" asks for.

---

## 4. Structured / Grounded Output (schema-valid, guaranteed)

The copilot must always emit a fixed structure: **predicted issue, confidence score, root-cause hypothesis, affected scope, recommended actions** (plus time-to-impact and citations).

### 4.1 Approaches, ranked

1. **Grammar-constrained decoding (RECOMMENDED).** llama.cpp converts a **JSON Schema → GBNF grammar** and masks the sampler so only schema-valid tokens are generated. Output **cannot** be malformed JSON. Exposed via `llama-server` `response_format: {"type":"json_schema","json_schema":{...}}` or a raw `grammar`. ([deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output](https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output), [github.com/ggml-org/llama.cpp/blob/master/grammars/README.md](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md))
2. **Guided decoding in vLLM** via **Outlines / XGrammar** (`guided_json`, `guided_grammar`) — same guarantee, GPU-batched; XGrammar gives up to ~5× TPOT speedup under load. ([arxiv.org/pdf/2403.06988](https://arxiv.org/pdf/2403.06988))
3. **Outlines (library)** with a local GGUF (`outlines.models.llamacpp()` + `outlines.generate.json(model, PydanticModel)`) — guaranteed-valid JSON, fully offline, no network. ([python.useinstructor.com/integrations/llama-cpp-python](https://python.useinstructor.com/integrations/llama-cpp-python/))
4. **Instructor + Pydantic** on top of the OpenAI-compatible endpoint — adds **type-safe validation, retries, and field descriptions** as model hints; pairs perfectly with (1)/(2). ([python.useinstructor.com/blog/2024/03/07/...](https://python.useinstructor.com/blog/2024/03/07/open-source-local-structured-output-pydantic-json-openai/))
5. **lm-format-enforcer** — token-filter alternative; works but (1)/(2) are first-class in our chosen runtimes.
6. **Plain "return JSON" prompting / JSON mode** — **not sufficient alone**; use only as a fallback. (Note: `llama-server` historically *fails open* if schema-grammar parsing fails, so validate downstream. ([github.com/ggml-org/llama.cpp/issues/19051](https://github.com/ggml-org/llama.cpp/issues/19051)))

**Chosen approach (defense in depth):** **GBNF/JSON-Schema grammar in `llama-server` (guarantee at sampler) → Pydantic validation in the client (Instructor) → one constrained retry on validation failure.** This makes invalid output essentially impossible and any residual edge case caught before it reaches the operator.

### 4.2 Example JSON Schema (copilot response)

```json
{
  "type": "object",
  "properties": {
    "predicted_issue": {
      "type": "string",
      "enum": ["interface_congestion", "latency_drift", "bgp_route_flap",
               "ospf_convergence_stress", "tunnel_degradation", "mpls_underlay_failure",
               "policy_drift", "path_asymmetry", "none"]
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
    "time_to_impact_minutes": { "type": ["integer", "null"], "minimum": 0 },
    "root_cause_hypothesis": { "type": "string", "minLength": 1, "maxLength": 600 },
    "contributing_signals": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "signal": { "type": "string" },
          "observation": { "type": "string" },
          "shap_contribution": { "type": "number" }
        },
        "required": ["signal", "observation"]
      }
    },
    "affected_scope": {
      "type": "object",
      "properties": {
        "sites": { "type": "array", "items": { "type": "string" } },
        "devices": { "type": "array", "items": { "type": "string" } },
        "services_or_vpns": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["sites", "devices"]
    },
    "recommended_actions": {
      "type": "array", "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "step": { "type": "string" },
          "runbook_ref": { "type": "string" },
          "urgency": { "type": "string", "enum": ["immediate", "soon", "monitor"] }
        },
        "required": ["step", "urgency"]
      }
    },
    "citations": {
      "type": "array", "minItems": 1,
      "items": { "type": "string", "description": "IDs of retrieved chunks/telemetry windows actually used" }
    },
    "insufficient_context": { "type": "boolean" }
  },
  "required": ["predicted_issue", "confidence", "root_cause_hypothesis",
               "affected_scope", "recommended_actions", "citations", "insufficient_context"],
  "additionalProperties": false
}
```

`required` + `additionalProperties: false` + `enum`s do real work: they force the model to commit to a known failure class, always cite, and always declare whether context was sufficient.

### 4.3 GBNF snippet (excerpt — `llama-server` auto-generates this from the schema; shown for transparency)

```gbnf
root        ::= "{" ws "\"predicted_issue\":" ws issue "," ws
                     "\"confidence\":" ws number "," ws
                     "\"root_cause_hypothesis\":" ws string "," ws
                     "\"recommended_actions\":" ws actions "," ws
                     "\"citations\":" ws citations "," ws
                     "\"insufficient_context\":" ws boolean ws "}"
issue       ::= "\"interface_congestion\"" | "\"latency_drift\"" | "\"bgp_route_flap\""
              | "\"ospf_convergence_stress\"" | "\"tunnel_degradation\""
              | "\"mpls_underlay_failure\"" | "\"policy_drift\"" | "\"path_asymmetry\"" | "\"none\""
actions     ::= "[" ws action (", " ws action)* ws "]"
action      ::= "{" ws "\"step\":" ws string "," ws "\"urgency\":" ws urgency ws "}"
urgency     ::= "\"immediate\"" | "\"soon\"" | "\"monitor\""
citations   ::= "[" ws string (", " ws string)* ws "]"
boolean     ::= "true" | "false"
number      ::= "-"? [0-9]+ ("." [0-9]+)?
string      ::= "\"" ([^"\\] | "\\" .)* "\""
ws          ::= [ \t\n]*
```

### 4.4 Outlines (offline, fully local) equivalent

```python
from pydantic import BaseModel, Field, conlist
from typing import Literal, Optional
import outlines

class Action(BaseModel):
    step: str
    runbook_ref: Optional[str] = None
    urgency: Literal["immediate", "soon", "monitor"]

class CopilotResponse(BaseModel):
    predicted_issue: Literal["interface_congestion","latency_drift","bgp_route_flap",
        "ospf_convergence_stress","tunnel_degradation","mpls_underlay_failure",
        "policy_drift","path_asymmetry","none"]
    confidence: float = Field(ge=0, le=1)
    time_to_impact_minutes: Optional[int] = None
    root_cause_hypothesis: str
    recommended_actions: conlist(Action, min_length=1)
    citations: conlist(str, min_length=1)
    insufficient_context: bool

model = outlines.models.llamacpp("/models/qwen2.5-7b-instruct-q5_k_m.gguf")  # local file, no network
generator = outlines.generate.json(model, CopilotResponse)
result = generator(prompt)   # guaranteed schema-valid CopilotResponse
```

### 4.5 `llama-server` request (OpenAI-compatible) equivalent

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [ {"role":"system","content":"<copilot system prompt §6>"},
                {"role":"user","content":"<analytics + SHAP + retrieved runbooks + operator question>"} ],
  "response_format": { "type": "json_schema", "json_schema": { "schema": { /* §4.2 schema */ } } },
  "temperature": 0.2, "top_p": 0.9
}'
```

(If you hit the historical `json_schema`+`grammar` conflict on `/v1/chat/completions`, pass the equivalent **`grammar`** field on the native `/completion` endpoint instead. ([github.com/ggml-org/llama.cpp/issues/11988](https://github.com/ggml-org/llama.cpp/issues/11988)))

---

## 5. Anti-Hallucination / Grounding Plan (with offline evaluation)

### 5.1 Generation-time techniques (keep answers grounded ONLY in retrieved artifacts + live analytics)

1. **"Answer only from context" contract** in the system prompt, with an explicit **abstain rule**: if the retrieved runbooks/telemetry don't support an answer, set `insufficient_context: true`, lower `confidence`, and recommend "escalate / gather more data" rather than inventing a cause.
2. **Mandatory citations** — `citations` (schema-required, `minItems: 1`) must reference **IDs of chunks/telemetry windows actually provided**. Post-generation, **reject any citation ID not in the supplied context** (closed-set check) and force a constrained retry.
3. **Quote-grounding for root cause** — instruct the model that every claim in `root_cause_hypothesis` and `contributing_signals` must be traceable to a provided SHAP attribution or retrieved text; discourage outside knowledge.
4. **Low temperature** (0.1-0.3) for deterministic, conservative answers.
5. **Confidence calibration** — derive `confidence` primarily from the **analytics engine's predicted probability / anomaly score**, not the LLM's own guess; the LLM *explains* the score rather than fabricating one. Optionally bin/clip via isotonic/temperature scaling computed offline on the validation fault scenarios.
6. **Tool/data scoping** — the only "tools" exposed are local: vector search over internal artifacts and a read-only query into the analytics/telemetry store. No web/search tool exists, so the model *cannot* reach outside even if prompted.

### 5.2 Offline groundedness / faithfulness verification (the measurable part)

Run a **post-generation faithfulness gate**, entirely offline:

- **Vectara HHEM-2.1-Open** (factual-consistency / hallucination model): `microsoft/deberta-v3-base` fine-tuned on NLI + FEVER/VitaminC/PAWS-style factual-consistency data. Given **(premise = retrieved context, hypothesis = copilot claim)** it returns a **0-1 consistency score**; low score ⇒ likely hallucination. Runs on **CPU in ~1.5 s for ~2k tokens, <600 MB RAM**, MIT-style availability on HF — ideal for an air-gapped guardrail. ([huggingface.co/vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model), [vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model))
- **NLI claim-decomposition (alternative/augment):** split the answer into atomic claims; classify each as **entailed / neutral / contradicted** vs context with an offline DeBERTa-v3-NLI model; **faithfulness = fraction entailed**; flag if < threshold (e.g., 0.9). ([123ofai.com/qnalab/.../faithfulness](https://123ofai.com/qnalab/system-design/blocks/faithfulness))
- **Runtime gate:** if HHEM/NLI score < threshold → **retry with a stricter "use only the cited text" instruction**, then **abstain** (`insufficient_context: true`) if still unfaithful. (Same "faithfulness floor at the gateway" pattern recommended in production RAG. ([towardsdatascience.com/rag-hallucinates-i-built-a-self-healing-layer](https://towardsdatascience.com/rag-hallucinates-i-built-a-self-healing-layer-that-fixes-it-in-real-time/)))

### 5.3 Offline evaluation method (how to *measure* hallucination rate for the 35% score)

Build a **local eval harness** over the simulated fault scenarios (these double as ground truth):

- **RAGAS-style metrics computed locally** with a **local LLM judge** (DeepEval/RAGAS can point at a local Ollama/`llama-server` instead of OpenAI), reporting:
  - **Faithfulness / groundedness** (answer supported by retrieved context) — primary anti-hallucination metric.
  - **Answer relevancy** and **context precision/recall** (RAG quality).
  ([deepeval.com/docs/metrics-hallucination](https://deepeval.com/docs/metrics-hallucination), [letsdatascience.com/blog/llm-evaluation-ragas-llm-as-judge](https://letsdatascience.com/blog/llm-evaluation-ragas-llm-as-judge-and-production-evals))
- **HHEM-based hallucination rate**: % of copilot responses scoring below the consistency threshold across all scenario queries — a single, defensible, fully-offline number.
- **Task-grounded accuracy** against the injected-fault labels: does `predicted_issue` match the injected fault class? Is `time_to_impact_minutes` within tolerance of the actual breach time? Are `recommended_actions` the correct runbook?
- **Abstention correctness**: on deliberately under-specified queries, does the copilot correctly set `insufficient_context: true` instead of hallucinating? (Abstention-policy benchmarking is an established RAG eval design. ([researchgate.net/.../Benchmarking_Hallucination_Evaluation_for_RAG_Under_an_Abstention_Policy](https://www.researchgate.net/publication/399331938_Benchmarking_Hallucination_Evaluation_for_RAG_Under_an_Abstention_Policy_A_Controlled_30-Query_Study_with_RAGAS_DeepEval_and_LLM-as-Judge)))

Report these per the four PS validation scenarios (congestion buildup, BGP flap cascade, MPLS underlay/tunnel degradation, controller policy drift) to directly populate "explanation quality / accuracy of recommended remediation."

---

## 6. Prompt / Agent Design

### 6.1 Inputs assembled into the LLM context (by the RAG/orchestration layer, all local)

For each alert/query, the orchestration layer composes a **single grounded prompt** containing:
1. **Analytics predictions** — predicted fault class, probability/confidence, **estimated time-to-impact**, anomaly scores (from the Phase-3 models).
2. **SHAP / feature attributions** — the top contributing signals (e.g., "ifutil_eth3 +0.41", "bgp_flap_rate +0.22", "tunnel_jitter_ms +0.18") so the LLM can explain *why*.
3. **Retrieved internal artifacts** — top-k chunks from the local vector DB: relevant **runbooks**, **topology metadata** (which sites/devices/VPNs), and **past incident records**, each tagged with a **chunk ID** for citation.
4. **Live telemetry window** — the recent metric snapshots relevant to the affected interfaces/tunnels.
5. **The operator's question** (or the standing 3 questions).

### 6.2 System prompt (drop-in)

```
You are the NOC Copilot for an air-gapped SD-WAN-over-MPLS network operations center.
You assist operators by turning the predictive analytics engine's output and retrieved
internal documents into clear, correct, operator-ready guidance.

GROUNDING RULES (strict):
- Use ONLY the information in the provided CONTEXT block: analytics predictions, SHAP
  attributions, retrieved runbooks/topology/past-incidents (each has an ID), and the live
  telemetry window. Do NOT use outside knowledge or invent device names, metrics, or causes.
- Every factual claim must be supported by the CONTEXT. Put the IDs of the chunks/telemetry
  you actually relied on into "citations". If you cannot cite it, do not say it.
- If the CONTEXT is insufficient to answer confidently, set "insufficient_context": true,
  give a low "confidence", and recommend gathering specific additional data or escalating —
  do NOT guess a root cause.
- Use the analytics engine's probability as the basis for "confidence" and its estimate for
  "time_to_impact_minutes". Explain these numbers; do not fabricate new ones.

ANSWER THE OPERATOR'S THREE QUESTIONS, grounded in the context:
  Q1 What is likely to fail next, and when?  -> "predicted_issue", "time_to_impact_minutes",
       "affected_scope".
  Q2 Why is risk elevated, which signals contributed?  -> "root_cause_hypothesis" +
       "contributing_signals" (tie each to the SHAP attribution / telemetry you cite).
  Q3 What corrective action should be taken before SLA/security impact?  ->
       "recommended_actions" with the matching "runbook_ref" and an "urgency".

STYLE: concise, concrete, NOC-operator language (interface/site/VPN/tunnel names, specific
thresholds). No hedging beyond the confidence score. No markdown — output ONLY the JSON object
that conforms to the provided schema.
```

The **user message** carries the structured CONTEXT (clearly delimited sections for ANALYTICS, SHAP, RETRIEVED[id], TELEMETRY, QUESTION). Output is constrained to the §4.2 schema via grammar. A **few-shot example** of one well-grounded response (with real citation IDs and an `insufficient_context: true` example) further stabilizes formatting and the abstain behavior.

### 6.3 Why this design scores on "Copilot Effectiveness"

- **Correct & operator-relevant:** answers are framed around the exact three operational questions and use concrete network entities.
- **Grounded / no hallucination:** schema-forced citations + abstain flag + HHEM faithfulness gate (§5) make ungrounded claims hard to emit and easy to catch.
- **Confidence-scored:** confidence and time-to-impact come from the analytics engine, so they are calibrated and defensible, not LLM guesses.

---

## 7. Hardware Sizing & Latency

### 7.1 Measured tokens/sec (llama.cpp, Q4_K_M, single-stream, warm model)

| Hardware | 7B (tok/s) | 8B (tok/s) | 13B (tok/s) | Notes |
|---|---|---|---|---|
| **RTX 4090 24 GB** | **135** | ~104-150 | 78 | 16K ctx ~104 tok/s on 8B; batch32 → ~2,200 aggregate tok/s for multi-operator. |
| **RTX 3090 24 GB** | 95 | ~90 | 55 | Great value for the appliance. |
| **RTX 4060 Ti 16 GB** | 55 | ~50 | 30 | Comfortable for 7-9B. |
| **RTX 3060 12 GB** | **45** | **42** | 22 | Solid interactive single-operator target. |
| **NVIDIA T4 16 GB** | ~25-40 (est.) | ~25-35 (est.) | ~12-18 | Turing, no FP8; usable, slower than Ampere. |
| **Apple M3 Max 64 GB** | 40 | ~36 | 22 | Unified memory; good for large-context. |
| **CPU only (modern, DDR5)** | **6-10** | ~5-9 | 3-5 | Interactive but slow; prefer a **3B** here (~15-25 tok/s). |

Sources: [mustafa.net/llm-tokens-per-second-benchmarks](https://mustafa.net/llm-tokens-per-second-benchmarks/), [developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/), [singhajit.com/llm-inference-speed-comparison](https://singhajit.com/llm-inference-speed-comparison/). The 8B quantization study measured **CPU token-gen 4.65 tok/s @ Q4_K_S, 6.66 @ Q5_0** for Llama-3.1-8B, consistent with the 6-10 tok/s CPU band. ([arxiv.org/html/2601.14277v1](https://arxiv.org/html/2601.14277v1))

### 7.2 Interactive-latency budget for the NOC

A copilot response is bounded but not tiny (structured JSON ≈ 150-400 output tokens) plus a sizable grounded prompt (analytics + SHAP + a few retrieved chunks ≈ 1-3K input tokens).

- **RTX 3060 @ 7B Q5_K_M (~40 tok/s):** ~0.3-0.5 s prompt processing + ~4-10 s generation ⇒ **~5-10 s end-to-end** — fine for alert triage.
- **RTX 4090 @ 7B (~135 tok/s):** **~1-3 s** end-to-end — snappy.
- **CPU @ 3B Q4_K_M (~15-25 tok/s):** **~6-15 s** — acceptable for a fallback; for CPU prefer the 3B model and shorter `max_tokens`.

### 7.3 Keeping inference fast for interactive use

- **Right-size the model to the box:** GPU → 7-8B Q5_K_M; CPU → 3B Q4_K_M. *"A fast 7B that runs smoothly beats a huge model that crawls."* ([huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally))
- **Cap output:** the schema is compact; set `max_tokens` ~512 and prune `contributing_signals` to top-k.
- **Trim the prompt:** retrieve **top-3 to top-5** chunks, not 20; pre-summarize long runbooks at index time; keep telemetry windows short. This cuts prompt-processing time and *improves* grounding.
- **KV-cache / prefix reuse:** keep the (long, static) system prompt constant so the runtime can reuse the prefix across queries; `llama-server` slot reuse and vLLM prefix caching both help.
- **GPU offload:** offload all layers to GPU when VRAM allows (`-ngl 99` in llama.cpp); use **flash-attention** builds; on CPU set threads = physical cores and use AVX2/AVX-512 builds.
- **Batch for multiple operators:** if many concurrent queries, switch the high-throughput path to **vLLM + AWQ** (paged attention, continuous batching → 1,500-2,200+ aggregate tok/s on a 4090).
- **Speculative decoding (optional):** a tiny draft model (e.g., Qwen2.5-0.5B) can speed up the 7B target on GPU.

---

## 8. Concrete Bundle / Deployment Checklist (air-gap)

1. **On a connected build host**, `snapshot_download` the chosen GGUF (e.g., `Qwen2.5-7B-Instruct-Q5_K_M.gguf`) + the CPU-fallback 3B GGUF + **HHEM-2.1-Open** + a **DeBERTa-v3 NLI** model; copy into the appliance image. (`HF_HUB_OFFLINE=1` thereafter.)
2. **Ship `llama-server`** (statically built) as the runtime; no Ollama auto-updater in the appliance. Start: `llama-server -m qwen2.5-7b-instruct-q5_k_m.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 8080 --grammar-file copilot.gbnf` (or pass `response_format` per request).
3. **RAG stack local** — local embedding model + local vector DB (e.g., FAISS/Chroma/Qdrant) over internal artifacts only; chunk IDs flow into `citations`.
4. **Faithfulness gate** — HHEM/NLI check on every response; retry-then-abstain on low score.
5. **No-egress enforcement** — `--network=none`/netns, default-deny outbound firewall, no DNS/default route; env: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1`.
6. **Evidence capture** — `ss -tunap`, `tcpdump 'not host 127.0.0.1'`, `strace -e connect` during a full demo run; archive as the air-gap compliance artifact.

---

## 9. Sources

- Open-weight model landscape / local-run guides: https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally · https://acecloud.ai/blog/best-open-source-llms/ · https://www.vellum.ai/open-llm-leaderboard · https://whatllm.org/best-open-source-llm
- Qwen2.5 quality, license, tool-calling: https://qwenlm.github.io/blog/qwen2.5-llm/ · https://qwenlm.github.io/blog/qwen2.5/ · https://arxiv.org/pdf/2412.15115 · https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE
- Llama 3.1/3.2 license & capability: https://www.llama.com/llama3_1/license/ · https://huggingface.co/blog/llama31 · https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
- Phi-3 / Gemma licenses: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf · https://ai.google.dev/gemma/terms · https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/
- Cross-model benchmarks: https://www.glukhov.org/llm-performance/benchmarks/mistral-small-gemma2-qwen2-5-mistral-nemo/
- GGUF quantization study (Llama-3.1-8B, ppl/accuracy/size/speed): https://arxiv.org/html/2601.14277v1 · https://github.com/ggml-org/llama.cpp/discussions/2094 · https://runaihome.com/blog/quantization-q4-q5-q6-q8-quality-loss-2026/ · https://tonisagrista.com/blog/2026/quantization/ · https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- Runtimes & air-gap/telemetry: https://www.quantizelab.dev/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide · https://markaicode.com/best/air-gapped-ai-stack/ · https://insiderllm.com/guides/running-ai-offline-complete-guide/ · https://ai-ollama.github.io/privacy-offline.html
- Ollama config/updates: https://docs.ollama.com/faq · https://github.com/ollama/ollama/issues/6024 · https://modelpiper.com/blog/ollama-environment-variables · https://pkg.go.dev/github.com/ollama/ollama/envconfig
- vLLM/TGI offline: https://github.com/vllm-project/vllm/issues/9255 · https://github.com/vllm-project/vllm/issues/23684 · https://docs.forjinn.com/components-guide/air-gapped-llm-support
- Structured output (GBNF/Outlines/Instructor/vLLM guided): https://deepwiki.com/ggml-org/llama.cpp/8.1-grammar-and-structured-output · https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md · https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md · https://github.com/ggml-org/llama.cpp/issues/11988 · https://github.com/ggml-org/llama.cpp/issues/19051 · https://arxiv.org/pdf/2403.06988 · https://python.useinstructor.com/integrations/llama-cpp-python/ · https://python.useinstructor.com/blog/2024/03/07/open-source-local-structured-output-pydantic-json-openai/
- Grounding / anti-hallucination / offline eval: https://huggingface.co/vectara/hallucination_evaluation_model · https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model · https://123ofai.com/qnalab/system-design/blocks/faithfulness · https://deepeval.com/docs/metrics-hallucination · https://letsdatascience.com/blog/llm-evaluation-ragas-llm-as-judge-and-production-evals · https://towardsdatascience.com/rag-hallucinates-i-built-a-self-healing-layer-that-fixes-it-in-real-time/ · https://www.researchgate.net/publication/399331938_Benchmarking_Hallucination_Evaluation_for_RAG_Under_an_Abstention_Policy_A_Controlled_30-Query_Study_with_RAGAS_DeepEval_and_LLM-as-Judge
- Hardware / latency: https://mustafa.net/llm-tokens-per-second-benchmarks/ · https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/ · https://singhajit.com/llm-inference-speed-comparison/

---

*Prepared for PS-13 Phase 4. Emphasis: 100% offline/air-gapped, free/open-weight, verifiable no-egress, grounded & schema-valid output.*
