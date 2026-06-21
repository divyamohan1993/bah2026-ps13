# NETRA — Build Plan & Workstream Ownership

This document assigns **exact, non-overlapping file subtrees** to seven builder
workstreams so they can implement in parallel without colliding. It is derived
from and subordinate to [`../ARCHITECTURE.md`](../ARCHITECTURE.md) and the shared
contracts in [`../netra/contracts/`](../netra/contracts).

## Ground rules (read first)

1. **Each workstream OWNS a distinct subtree and edits ONLY that subtree.** If
   your change requires touching a path owned by another workstream — or any
   shared top-level file — STOP and flag it for the integrator.
2. **Shared top-level files are OFF-LIMITS to builders.** These were created by
   the architecture step and are owned by the integrator:
   `ARCHITECTURE.md`, `README.md`, `pyproject.toml`, `requirements.txt`,
   `requirements-core.txt`, `.gitignore`, and the entire
   `netra/contracts/` package. **Do not edit the contracts.** If you need a new
   field/enum/model, propose it to the integrator (it is a model-grammar change
   for the copilot and must be coordinated).
3. **Extra dependencies:** if your module needs deps beyond the core tier, add a
   `requirements-<module>.txt` *inside your own subtree* (e.g.
   `netra/copilot/requirements-copilot.txt`). The integrator folds approved
   extras into the top-level manifests. Never edit `requirements.txt` directly.
4. **Import the contracts; never redefine them.** Always
   `from netra.contracts import ...`. Producing/consuming the canonical types is
   how your module plugs into the pipeline.
5. **Respect graceful degradation.** Anything heavy (GPU model, LLM, sim, vector
   DB) must be import-guarded and feature-flagged so the CPU-only path still
   runs. Provide a working CPU/no-op fallback.
6. **Every module ships a CPU-only smoke test** that runs against the synthetic
   `TelemetrySource` with no GPU / no internet / no sim.

## Dependency / build order

```
WS1 datagen ─┐                      (synthetic TelemetrySource = everyone's input)
             ├─> WS2 streaming ──> WS3 analytics ──> WS4 correlation/risk/explain ─┐
             │                                                                     ├─> WS6 API+UI
             └────────────────────────────> WS5 copilot+RAG ──────────────────────┘
WS7 security+packaging  (cross-cuts all; finalises docker-compose + air-gap proof)
```
WS1 unblocks everyone (it provides labeled data without a sim). WS2→WS3→WS4 form
the analytics spine. WS5 (copilot) depends on WS4 outputs + the corpus. WS6
surfaces everything. WS7 wraps it for the air-gap.

---

## WS1 — Simulation + synthetic scenario generator

**Owns:** `sim/**`, `netra/datagen/**`
**Consumes (contracts):** `ScenarioId`, `DeviceRole`, `SiteType`, `TelemetryKind`, `MetricName`, `IssueType`
**Produces (contracts):** `TelemetryRecord`, `SyslogEvent`, `RoutingEvent`, `FlowRecord`, `TunnelStat`, `ScenarioLabel`; defines the `TelemetrySource` interface (`SIM` + `SYNTHETIC` + `REPLAY` backends)
**Key libraries:** Containerlab, netlab 26.06, FRRouting, Nokia SR Linux, strongSwan, iperf3/TRex/Scapy, tc/netem, Pumba, ExaBGP/GoBGP (sim side); numpy/pandas/scipy (datagen side — core tier only, must stay CPU-light)
**Deliverable files (concrete):**
- `sim/topology.clab.yml` (+ `sim/netlab/topology.yml`) — the 5-site, ~20-node IaC topology.
- `sim/configs/` — FRR/SR Linux + strongSwan + tc QoS render templates.
- `sim/scenarios/{a_congestion,b_bgp_flap,c_tunnel,d_drift}.py` — seeded fault drivers writing `ScenarioLabel` JSONL.
- `netra/datagen/source.py` — the `TelemetrySource` ABC + a `replay` backend.
- `netra/datagen/synthetic.py` — the high-fidelity 4-scenario generator emitting labeled `TelemetryRecord` streams (the CPU-only default source).
- `netra/datagen/scenarios.py` — per-scenario signal models (diurnal baselines + injected precursors) shared by sim labels and synthetic output.
**Notes:** the synthetic generator is the linchpin of the CPU-only promise — it MUST produce realistic diurnal seasonality + the four precursor signatures with ground-truth `ScenarioLabel`s, deterministically (seeded).

---

## WS2 — Telemetry pipeline + O(1) streaming feature engine

**Owns:** `telemetry/**`, `netra/streaming/**`
**Consumes:** `TelemetryRecord`, `SyslogEvent`, `RoutingEvent`, `FlowRecord`, `TunnelStat`, `EntityRef`, `MetricName`
**Produces:** `FeatureVector`
**Key libraries:** gnmic, Telegraf, pmacct, NATS JetStream, VictoriaMetrics (configs); `river`, `stumpy`, `ddsketch`, `nats-py` (engine — all core tier)
**Deliverable files (concrete):**
- `telemetry/gnmic.yaml`, `telemetry/telegraf.conf` — collector configs (→ NATS + VictoriaMetrics).
- `telemetry/nats-streams.sh`, `telemetry/victoriametrics.yaml` — bus + TSDB provisioning.
- `netra/streaming/features.py` — the O(1) feature operators (Welford/EWMA/DDSketch/Page-Hinkley/HST/stumpi) producing `FeatureVector`.
- `netra/streaming/engine.py` — NATS consumer loop: subscribe `telemetry.>`, update per-entity online state, publish `FeatureVector`.
- `netra/streaming/sources.py` — adapter that lets the engine read from a `TelemetrySource` directly (CPU-only path, no NATS needed).
**Notes:** keep every feature O(1)/amortised-O(1); features scaled to [0,1] before HST. Must run against the synthetic source with no bus for the smoke test.

---

## WS3 — Predictive ensemble (forecasting + anomaly + fusion)

**Owns:** `netra/analytics/forecasting/**`, `netra/analytics/anomaly/**`, `netra/analytics/fusion/**`
**Consumes:** `FeatureVector`, `TelemetryRecord`, `EntityRef`, `MetricName`, `DetectorFamily`, `ScenarioLabel` (for training/backtest)
**Produces:** `Forecast`, `AnomalyScore`, `FusedRisk`, `TimeToImpact`
**Key libraries:** core — `statsforecast`-style via `statsmodels`, `pyod`, `ruptures`, `pymannkendall`, `lifelines`, `scikit-learn`, `stumpy`; optional-heavy — `lightgbm`/`xgboost`/`mlforecast`, `neuralforecast`, `deepod`, `chronos-forecasting`, `mapie`, `scikit-survival`
**Deliverable files (concrete):**
- `netra/analytics/forecasting/ensemble.py` — tiered forecasters → `Forecast` (CPU tier always-on; deep/foundation feature-flagged).
- `netra/analytics/forecasting/timeimpact.py` — trajectory threshold-crossing + Theil-Sen + Cox/RSF survival → `TimeToImpact`.
- `netra/analytics/anomaly/detectors.py` — the tiered detector bank (#19–#60) → `AnomalyScore`.
- `netra/analytics/anomaly/evt.py` — DSPOT/SPOT/POT (#68) adaptive thresholds.
- `netra/analytics/fusion/fuse.py` — score-normalisation + weighted-agreement + stacker (#67) → `FusedRisk` (records `MethodWeight` provenance).
- `netra/analytics/fusion/calibrate.py` — Platt/isotonic calibration trained on `ScenarioLabel`s.
**Notes:** `FusedRisk.risk_score>0` MUST carry `contributing_methods` (the contract enforces it). Feature-flag deep/foundation members; the CPU ensemble alone must produce usable risk + lead time.

---

## WS4 — Correlation / RCA / risk prioritisation / explainability

**Owns:** `netra/analytics/correlation/**`, `netra/analytics/risk/**`, `netra/analytics/explain/**`
**Consumes:** `FusedRisk`, `TimeToImpact`, `AnomalyScore`, `RoutingEvent`, `FlowRecord`, `EntityRef`, topology (from WS1)
**Produces:** `Incident` (with `BlastRadius`, `ContributingSignal[]`, `Playbook` reference), `ContributingSignal`
**Key libraries:** `networkx` (graph/BFS/centrality/WCC-SCC), `statsmodels` (Granger), `scikit-learn` (Platt calibration), `shap` (TreeSHAP) — all core tier
**Deliverable files (concrete):**
- `netra/analytics/correlation/graph.py` — the `networkx` digital twin (build from WS1 topology + routing).
- `netra/analytics/correlation/correlate.py` — temporal+topological grouping (WCC/SCC), alarm compression, root-cause ranking (`centrality × earliest_onset × Granger`).
- `netra/analytics/correlation/blast_radius.py` — BFS reachability ∩ NetFlow → `BlastRadius`.
- `netra/analytics/risk/prioritize.py` — product-form risk + severity bucketing + flap suppression → `Incident`.
- `netra/analytics/explain/shap_explain.py` — TreeSHAP/ECOD attributions → `ContributingSignal`.
**Notes:** blast radius and affected scope are computed deterministically here — the copilot must NOT recompute or guess them. Granger is a *ranking hint*, cross-checked by graph correlation + SHAP.

---

## WS5 — Copilot (LLM + RAG + grounding)

**Owns:** `netra/copilot/**`, `corpus/**`
**Consumes:** `Incident`, `FusedRisk`, `TimeToImpact`, `ContributingSignal`, `BlastRadius`, `Playbook`, `CopilotRequest`
**Produces:** `CopilotResponse` (grammar-constrained from the LLM **or** the deterministic template fallback — same schema)
**Key libraries (optional-heavy, all import-guarded):** `llama-cpp-python` / `llama-server`, `sentence-transformers` + `FlagEmbedding` (bge-m3, bge-reranker-v2-m3), `qdrant-client`, `bm25s`, `transformers` (HHEM-2.1 / DeBERTa-v3 NLI), `instructor`
**Deliverable files (concrete):**
- `netra/copilot/llm.py` — llama-server OpenAI-compatible client with GBNF/JSON-schema constraint **and** the deterministic template fallback (`used_fallback=True`) — both return `CopilotResponse`.
- `netra/copilot/grammar.py` — GBNF generated from the `CopilotResponse` schema (must match `netra.contracts.copilot`).
- `netra/copilot/rag.py` — ingest + structure-aware chunk + contextual prefix + bge-m3 embed + Qdrant hybrid + RRF + bge-reranker; with a no-Qdrant in-process keyword fallback.
- `netra/copilot/grounding.py` — HHEM/NLI faithfulness gate + closed-set citation check + abstain logic.
- `netra/copilot/orchestrate.py` — assembles the grounded prompt (analytics + SHAP + retrieved chunks + graph facts) → `CopilotResponse`.
- `corpus/runbooks/*.md`, `corpus/incidents/*.json`, `corpus/topology/*.json`, `corpus/playbooks/*.json` — sample internal artifacts (CACAO-style playbooks) keyed for citation.
**Notes:** the template fallback is a **hard requirement** (graceful degradation) — when no model is present the copilot still answers Q1/Q2/Q3 in the same schema. Confidence comes from the analytics objects, never invented.

---

## WS6 — Operator UI + API

**Owns:** `netra/api/**`, `ui/**`, `grafana/**`
**Consumes:** `Incident`, `FusedRisk`, `TimeToImpact`, `ContributingSignal`, `CopilotRequest`/`CopilotResponse`, `FeatureVector`
**Produces:** HTTP/JSON + WebSocket surfaces; no new contracts (serialises existing ones)
**Key libraries:** `fastapi`, `uvicorn`, `httpx` (core); Cytoscape.js / vis-network + a lightweight frontend (vendored, no CDN); Grafana (provisioned-as-code)
**Deliverable files (concrete):**
- `netra/api/app.py` — FastAPI app: `/incidents`, `/risk`, `/forecast/{entity}`, `/copilot` (POST `CopilotRequest` → `CopilotResponse`), `/ws` stream.
- `netra/api/routes_copilot.py`, `netra/api/routes_analytics.py` — route modules.
- `ui/index.html` + `ui/app.js` + `ui/styles.css` — single-page console: Cytoscape topology (root-cause node + blast-radius shaded), risk timeline, 3-answer incident card, chat box.
- `ui/vendor/` — vendored JS/CSS/fonts (no external CDN — air-gap requirement).
- `grafana/provisioning/{datasources,dashboards}/*.yaml` + `grafana/dashboards/*.json` — telemetry/alert wall + the air-gap blocked-attempt counter panel.
**Notes:** every asset served from localhost; no Google Fonts, no CDNs. The API must return identical shapes whether the copilot used the LLM or the template fallback.

---

## WS7 — Security + packaging (air-gap enforcement, verification, bundling)

**Owns:** `security/**`, `tests/airgap/**`, `scripts/**`; **finalises** `docker-compose.yml` (stub note below — integrator-coordinated)
**Consumes:** all built images + the wheelhouse
**Produces:** air-gap enforcement configs, the conformance test, the offline bundle + SBOM
**Key libraries:** nftables, Falco, Docker (`internal: true` / `--network none`), firejail/bubblewrap + seccomp; `pytest` (conformance); `cyclonedx-bom`/`syft`, `cosign`, `docker save`, `pip --no-index --require-hashes`
**Deliverable files (concrete):**
- `security/nftables.conf` — default-DROP egress + log/counter.
- `security/falco-rules.yaml` — CRITICAL outbound-connection rule.
- `security/seccomp-llm.json` + `security/docker-network.md` — LLM sandbox + internal-bridge network config.
- `tests/airgap/test_airgap_conformance.py` — pytest suite trying TCP/UDP/DNS/HTTPS egress; passes only if all blocked.
- `scripts/build_bundle.sh` — `docker save | gzip` + SHA-256 + cosign sign.
- `scripts/build_wheelhouse.sh` — `pip download` transitive closure for offline `--no-index --require-hashes` install.
- `scripts/gen_sbom.sh` — CycloneDX SBOM for images + wheels.
- `scripts/install.sh` — verified offline install: checksum → cosign verify → load → `pip --no-index` → compose up → run conformance test on first boot (abort on any failure).
**docker-compose.yml stub:** the integrator finalises the service mesh per [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §9.1 on the `airgap_net` `internal: true` bridge; WS7 supplies the network/security stanzas and the per-service hardening (cap-drop, no-new-privileges, seccomp).

---

## Contract change protocol

If any workstream finds the shared contracts insufficient:
1. Do **not** edit `netra/contracts/**`.
2. Write up the needed change (new field/model/enum value + why) and hand it to
   the integrator.
3. The integrator evaluates the ripple (especially the copilot GBNF grammar,
   which is generated from `CopilotResponse`), makes the change once, and
   notifies all workstreams.

This keeps the interface stable and the parallel build collision-free.
