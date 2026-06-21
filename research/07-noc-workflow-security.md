# Phase 5/6 Research — Integrated NOC Workflow Automation, Event Correlation, Alert Prioritization, Operator UI, and Air-Gap Security Compliance + Verification

**Problem Statement 13 — Air-Gapped Predictive Copilot for Secure MPLS Operations**
**Domain:** Phase 5 (Copilot Integration & Decision Support) + Phase 6 (Scenario Validation), with rigorous treatment of the **20% "Security & Offline Compliance"** evaluation dimension.

This document is the deep-research deliverable for two intertwined concerns:

- **Part A — NOC Workflow & Decision Support:** how the copilot turns raw predictive-model outputs + telemetry into *one ranked incident with a probable root cause, a blast-radius estimate, a confidence score, a recommended playbook, and an operator-ready summary* — answering the three operational questions (Q1 what/when, Q2 why, Q3 what-to-do).
- **Part B — Air-Gap Security & Compliance:** how to **guarantee** and, critically, **verifiably prove** zero outbound dependency at runtime, plus the offline security controls (secrets, audit, RBAC, integrity, reproducible/hermetic packaging) that a regulated/government NOC requires.

> **Judging note (do not skip):** Evaluation explicitly says "Air-gap integrity — **verifiably** zero outbound dependency during runtime; offline security controls implemented." The word *verifiably* means we must ship a **conformance test + continuous egress monitor** that a judge can run/observe live, not just claim isolation. Part B Section 7 is the highest-leverage section in this doc.

---

## PART A — NOC WORKFLOW & DECISION SUPPORT

### A1. Graph-Based Event Correlation & Root-Cause Analysis (RCA)

#### A1.1 The core idea: a topology graph + a time axis = correlate "where" and "when"

Modern AIOps does **not** treat alarms as independent rows. It maintains a **service/network dependency graph** (often learned from traffic patterns rather than hand-configured) so that when an alarm fires on one node, the correlation engine already knows which downstream nodes will cascade. Correlation is performed along **two axes simultaneously**:

- **Temporal correlation** — events inside a sliding time window (e.g. last *T* minutes) are candidates to belong to the same incident. Production systems *adaptively widen/narrow T* until a target "event density" is met. [AWS Neptune RCA blog](https://aws.amazon.com/blogs/database/beyond-correlation-finding-root-causes-using-a-network-digital-twin-graph-and-agentic-ai/)
- **Topological (spatial) correlation** — events on nodes that are adjacent/reachable in the dependency graph are candidates to belong to the same incident; the **propagation sequence within the topology** is the spatial relation. [Frontiers — alarm reduction & root-cause via association mining](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1211739/full) · [Selector AIOps 2025](https://www.selector.ai/learning-center/aiops-in-2025-4-components-and-4-key-capabilities/)

The canonical worked example from the literature: a storage bottleneck causes cascading failures; topology-based correlation traces dependencies from web servers → app servers → database, and **surfaces the storage issue as the single root cause rather than emitting a separate alert per downstream symptom.** [HEAL — AIOps event correlation](https://healsoftware.ai/blog/how-aiops-event-correlation-transforms-incident-response.html)

#### A1.2 The RCA pipeline we will implement (mapped to AWS DOCOMO production design)

The AWS "network digital twin + agentic AI" blog is the most concrete public reference and maps almost 1:1 onto our MPLS/SD-WAN problem. Their pipeline (DOCOMO reported **15 s failure isolation** in transport + RAN). We adopt the same stages: [AWS Neptune RCA blog](https://aws.amazon.com/blogs/database/beyond-correlation-finding-root-causes-using-a-network-digital-twin-graph-and-agentic-ai/)

1. **Trigger / pattern detection.** Watch alarm volume, **z-score deviation from baseline**, severity clustering, and cascading-failure patterns. A precursor signal from the predictive engine (Phase 3) is itself a trigger.
2. **Temporal gather.** Pull last *T* minutes of alarms/KPIs; adapt *T* to hit a target event density.
3. **Dependency-graph (failure subgraph) build.** From the affected node IDs, **expand up to N hops** of dependency neighbors in the graph DB. Iterate until the subgraph is "enough" to contain the incident.
4. **Decompose & isolate fault domains.** Run **Weakly-Connected-Components (WCC)** and **Strongly-Connected-Components (SCC)** to split the subgraph into independent fault domains, so two unrelated incidents don't merge.
5. **Rank candidate root causes.** Compute **centrality** to rank nodes by structural importance; use **label propagation** to cluster related nodes; return the **top-K (default 10)** `⟨node_id, score⟩` pairs.
6. **Correlate anomalies + forecast drift.** For each candidate, compare actual vs forecast KPI: flag when residual `r = |actual − forecast| > σ × band_factor` (i.e. a forecast-band breach). This is exactly our predictive engine's job and ties Phase 3 → Phase 5.
7. **Match against known incidents.** Vector-similarity search of the incoming alarm *sequence/pattern* against an incident knowledge base (this is our RAG over past incidents).
8. **Compile a single structured record:** failing nodes, alarm pattern, neighboring/affected nodes, incident description, recommended actions. This record is what we hand to the LLM and to alert prioritization.

#### A1.3 Causal RCA (beyond correlation) — choosing the right technique

Correlation tells you *what co-occurred*; causal inference distinguishes *which thing caused which*. Options, with our recommendation for a hackathon timeline:

- **PC algorithm (constraint-based structure learning).** Builds a causal DAG from conditional-independence tests. **Known limitation: it ignores the temporal order of time series and the rich temporal information** — multiple sources call this out. Use only as a baseline. [arXiv 2402.01140 (RUN / Neural Granger)](https://arxiv.org/abs/2402.01140) · [ACM — Evaluation of Causal Inference Techniques for AIOps](https://dl.acm.org/doi/fullHtml/10.1145/3430984.3431027)
- **Granger causality (temporal).** "Does the past of X improve prediction of Y?" Granger has been validated specifically on *log/telemetry data from benchmark microservice systems* for inferring inter-component dependency impact. Good fit for our SNMP/latency/jitter/BGP time-series. [ACM — Evaluation of Causal Inference Techniques for AIOps](https://dl.acm.org/doi/fullHtml/10.1145/3430984.3431027)
- **Neural Granger causal discovery (RUN).** State-of-the-art: enhances an encoder with contextual info from the time series and uses a forecasting model to do neural Granger discovery + contrastive learning. Best accuracy but heaviest to implement. [arXiv 2402.01140](https://arxiv.org/abs/2402.01140)
- **Structural Causal Models (SCM) / Bayesian networks + counterfactual.** Counterfactual reasoning ("would Y have happened if X hadn't?") is what *separates correlation from causation* and improves RCA accuracy. SCMs, Granger, and Bayesian nets are the three model families recommended for embedding RCA in ops pipelines. [ResearchGate — Causal Inference AI Models for RCA in DevOps](https://www.researchgate.net/publication/392499895_Applying_Causal_Inference_AI_Models_to_Root_Cause_Analysis_RCA_in_DevOps) · [IJRTI — Applying Causal Inference to RCA](https://ijrti.org/papers/IJRTI2505203.pdf)
- **GNN-based RCA.** Recent (2025/2026) work combines an **anomaly-correlation graph + graph-attention network + LightGBM**, learning causal correlations from historical alarms to build the graph, then attending over it. Also **AST-GNN (Attention Spatio-Temporal GNN)** jointly learns topology + temporal patterns and emits per-KPI forecasts *with confidence bands* — directly reusable for both prediction and correlation. [ScienceDirect — RCA via anomaly-correlation graph + GNN](https://www.sciencedirect.com/science/article/abs/pii/S0952197626007323)

**Recommended stack for this hackathon (pragmatic, defensible):**
- **Topology layer:** a `networkx.DiGraph` of the simulated CE/PE/P + tunnels + sites (built from the Containerlab topology + routing tables). This is the digital twin.
- **Correlation:** temporal sliding window + topological grouping (WCC/SCC over the failure subgraph) — pure `networkx`, no GPU, fully offline.
- **Causal ranking:** **pairwise Granger causality** over candidate KPIs (cheap, temporal-aware, defensible) to order root-cause hypotheses; optionally a small **PyTorch-Geometric (PyG) GNN** as the "advanced" component if time permits.
- **Known-incident match:** vector similarity in the local RAG store (the same store the LLM uses).

#### A1.4 Blast-radius via graph reachability (this powers prioritization in A2)

"Blast radius" = the set of sites/services/SLAs reachable from the failing node under the failure-propagation relation. This is a **graph traversal / reachability** problem of complexity **O(V+E)**: [Endor Labs — measuring blast radius](https://www.endorlabs.com/learn/vulnerability-blast-radius-how-to-measure-and-reduce-impact) · [Medium — graph-theory reachability analysis](https://medium.com/@a.talsma/data-driven-vulnerability-management-graph-theory-based-reachability-analysis-1-2-61f2fe185339)

- To find **all impacted downstream services**, compute reachable nodes from the failure node. **BFS is preferred over DFS** because it is iterative, easy to explain to operators, and naturally yields **"distance from failure"** (hop count), which doubles as an urgency/propagation-time proxy. [Endor Labs](https://www.endorlabs.com/learn/vulnerability-blast-radius-how-to-measure-and-reduce-impact)
- In `networkx`: `nx.descendants(G, failed_node)` for the affected set, or `nx.single_source_shortest_path_length(G, failed_node)` to get hop-distance per affected node. For "what's upstream that could cause this," reverse the edges and BFS (reverse-BFS to extract the precursor subgraph). [Medium — reverse-BFS blast radius](https://medium.com/@a.talsma/data-driven-vulnerability-management-graph-theory-based-reachability-analysis-1-2-61f2fe185339)
- For a router/link, blast radius is "the network flows traversing it" — i.e. intersect reachability with the NetFlow/IPFIX flow set to get **# affected flows / SLAs / sites**, which is the real impact number operators care about. [USPTO 11637861 — reachability-graph safe remediation](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11637861)

#### A1.5 Alarm compression / dedup / single-incident grouping — the algorithm

To collapse correlated precursor signals into **one** incident:

1. **Dedup** identical alarms (same node+type) within the window (hash key = `node_id|alarm_type|severity_bucket`).
2. **Topological grouping** — union-find / connected-components over the failure subgraph so all symptoms on reachable nodes join one incident.
3. **Root-cause selection** — within each group, pick the node maximizing `centrality × earliest_onset × causal_score` (earliest onset + highest causal/centrality = most likely cause; downstream/later = symptom).
4. **Compression ratio** is a metric to report: (# raw alarms ÷ # incidents). Association-mining approaches in carrier networks are explicitly designed to *reduce alarms* and infer the root cause. [Frontiers — alarm reduction & root-cause inference](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1211739/full)
5. A **hybrid knowledge-aware aggregation** (combining learned correlation with operator knowledge/runbooks) is the current SOTA for large-scale alert aggregation — relevant because we *have* runbooks in RAG. [arXiv 2403.06485 — Knowledge-aware Alert Aggregation](https://arxiv.org/pdf/2403.06485)

---

### A2. Confidence-Scored Alert Prioritization

#### A2.1 The risk score (the formula we will ship)

We adopt a **Risk-Based Alerting (RBA)** model — assign each *incident* (post-correlation, not each raw alarm) a calculated risk score from multiple weighted factors, so analysts spend time on the few incidents most likely to cause real harm. [Deepwatch — Risk-Based Alerting](https://www.deepwatch.com/glossary/risk-based-alerting-rba/) · [Safe Security — Risk Prioritization Framework 2026](https://safe.security/resources/blog/the-modern-risk-prioritization-framework-for-2026/)

Recommended composite (all factors normalized 0–1, then combined):

```
Risk = w1 · AnomalyConfidence          # calibrated P(failure) from the predictive model
     × w2 · TimeToImpactUrgency        # = 1 / (1 + minutes_to_impact); sooner ⇒ higher
     × w3 · BlastRadius                 # normalized #affected_sites / #SLAs / #flows (A1.4)
     × w4 · AssetCriticality            # business/role weight: DC-PE > hub-PE > branch-CE, etc.
```

Use a **product** (geometric) form rather than a sum so that a near-zero factor (e.g. zero blast radius) correctly suppresses the score — this avoids "high anomaly score but affects nothing" false-urgency. Asset/service criticality tied to business services + data classification, and *shared* blast radius from dependencies, are the standard RBA inputs. [JupiterOne — prioritize by asset criticality](https://www.jupiterone.com/blog/prioritizing-exploitable-vulnerabilities-to-protect-your-business-critical-assets) · [Medium (Silas Potter) — a formula for consistent alert severity scoring](https://medium.com/@silaspotter17/rethinking-alert-severity-a-formula-for-consistent-scoring-abbcb60e42ac)

#### A2.2 Calibration — turning model scores into *trustworthy* probabilities

A raw model "confidence" of 0.8 is meaningless unless it means "happens 80% of the time." Calibrate so the copilot's stated confidence is honest (this directly supports the "no hallucination / grounded" 35% criterion too):

- **Platt scaling** (fit a sigmoid to scores) — best when the score distortion is **sigmoid-shaped**; few parameters, works with small calibration sets. [FastML — Platt vs isotonic](https://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/) · [Train in Data — guide to Platt scaling](https://www.blog.trainindata.com/complete-guide-to-platt-scaling/)
- **Isotonic regression** — corrects **any monotonic** distortion; **with ≥1000 calibration points it is always as good as or better than Platt**, but overfits on small sets. [Niculescu-Mizil & Caruana, ICML'05 — Predicting Good Probabilities](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) · [Abzu — calibration part II](https://www.abzu.ai/data-science/calibration-introduction-part-2/)
- **Recommendation:** since our labeled fault-injection dataset is modest, use **Platt scaling** for the demo (robust on small data) and mention isotonic as the upgrade once we have ≥1k labeled incidents. Report a **reliability diagram + Brier score / ECE** as evidence of calibration quality.

#### A2.3 Suppression of flapping / noise (reduce alert fatigue)

Alert fatigue = teams desensitized by a barrage of non-actionable alerts; the fixes are **dedup, suppression, and correlation/consolidation**. [Palo Alto — reduce alert fatigue](https://www.paloaltonetworks.com/cyberpedia/how-to-reduce-security-alert-fatigue) · [ACM Computing Surveys — Alert Fatigue in SOCs](https://dl.acm.org/doi/10.1145/3723158) · [Torq — alert management 2026](https://torq.io/blog/cybersecurity-alert-management-2026/)

Concrete suppressors to implement:

- **Flap detection (penalty/decay, à la BGP route-flap damping).** Maintain a per-entity *flap penalty* that increments on each state change and **exponentially decays**; suppress notifications while penalty > suppress-threshold, re-enable when it decays below reuse-threshold. This is exactly BGP/MPLS flap-dampening logic reused for alerting. [Juniper — BGP session/route flaps](https://www.juniper.net/documentation/us/en/software/junos/bgp/topics/topic-map/bgp-session-flaps.html) · [USPTO 8289856 — alarm threshold for BGP flapping](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8289856)
- **Dedup + consolidation** of identical/duplicate alarms before scoring. [Conifers — false-positive suppression](https://www.conifers.ai/glossary/false-positive-suppression)
- **Correlation-as-suppression:** because A1 collapses N symptoms into 1 incident, downstream symptom alarms are auto-suppressed (they become evidence, not alerts).
- **Threshold calibration via historical data + iterative testing** to avoid both over-suppression and noise. [CyberDefenders — alert triage guide](https://cyberdefenders.org/blog/alert-triage-process/)

#### A2.4 Output: a ranked triage queue

Sort incidents by calibrated Risk, bucket into **urgency classes** (e.g. P1 imminent-SLA-breach / P2 elevated / P3 watch) for the "urgency classification" required by Objective 4, and attach to each: the top contributing signals (for Q2 "why"), time-to-impact (Q1 "when"), and the recommended playbook (Q3 "what-to-do", from A3).

---

### A3. Automated Playbook Suggestion & Action Sequencing

#### A3.1 Represent runbooks as machine-readable artifacts

Two complementary representations:

- **CACAO Security Playbooks v2.0 (OASIS standard).** A standardized **JSON** schema + taxonomy for course-of-action workflows: machine-readable, shareable across orgs/tools, and able to **encapsulate OpenC2 / Sigma / Kestrel commands** at the command level. Both CACAO v2 and OpenC2 are OASIS standards (OpenC2 = standardized command-and-control language for cyber-defense actions). This is the rigorous, standards-aligned way to model "predicted issue → ordered remediation steps." [OASIS — CACAO Security Playbooks v2.0](https://www.oasis-open.org/standard/cacao-security-playbooks-v2-0/) · [CACAO v2.0 spec](https://docs.oasis-open.org/cacao/security-playbooks/v2.0/security-playbooks-v2.0.html) · [SecurityIntelligence — intro to CACAO](https://securityintelligence.com/posts/introducing-collaborative-automated-course-of-action-operations-cacao-an-emerging-cybersecurity-standard-to-quickly-define-and-share-playbooks/)
- **Simpler pragmatic form for the demo:** YAML/JSON runbook docs (id, trigger-signature, ordered steps, each step = {description, command-template, safety-class, expected-result, rollback}) stored in the **RAG corpus** so the LLM can retrieve and the orchestrator can execute. The LLM maps the predicted-issue signature → the matching runbook; the orchestrator sequences the steps.

#### A3.2 "Suggest, human-approve" vs auto-remediation (critical safety posture)

For a regulated/government NOC the default **must** be **suggest → require explicit operator approval → then execute** (a.k.a. *human-in-the-loop*). SOAR/event-driven systems support fully automated response *with no or limited human intervention*, but we **gate** that behind approval for anything that changes device state. Reserve full auto-remediation only for *read-only diagnostics* (e.g. collecting `show` outputs). [GitHub SOARCA — CACAO-based open-source orchestrator](https://github.com/COSSAS/SOARCA) · [StackStorm — event-driven auto-remediation](https://github.com/networktocode/awesome-network-automation/blob/master/README.md)

#### A3.3 The execution tooling (all offline-capable)

- **Network device actions (the "hands"):** **Netmiko** (Paramiko-based SSH for network gear), **NAPALM** (vendor-abstraction over SSH/NETCONF/REST: get/merge/replace config, with rollback), **Nornir** (pure-Python, multithreaded, inventory-driven orchestration — by NAPALM's author), and **Ansible** (agentless, declarative). All run locally with no internet. [PacketCoders — tooling landscape](https://www.packetcoders.io/network-automation-101-tooling-landscape/) · [APNIC — Paramiko/Netmiko/NAPALM/Ansible/Nornir](https://blog.apnic.net/2023/02/13/automation-tools-paramiko-netmiko-napalm-ansible-nornir-or/) · [awesome-network-automation](https://github.com/networktocode/awesome-network-automation)
- **Workflow orchestration (the "brain" that sequences steps + approvals):**
  - **StackStorm** — event-driven automation/auto-remediation platform; self-hosted, offline-capable. [awesome-network-automation](https://github.com/networktocode/awesome-network-automation/blob/master/README.md)
  - **Rundeck** — job scheduler + runbook/Ansible-playbook automation with built-in **ACL/RBAC and approval gates**; self-hosted. [awesome-network-automation](https://github.com/networktocode/awesome-network-automation/blob/master/README.md)
  - **Shuffle / n8n** — open-source SOAR/workflow engines, self-hostable via Docker (offline once images are vendored). **SOARCA** — open-source CACAO-native orchestrator (executes CACAO JSON playbooks directly). [GitHub — SOARCA](https://github.com/COSSAS/SOARCA)
- **Recommendation for the demo:** keep it light — model playbooks as **CACAO-style JSON in RAG**, let a thin **FastAPI** "playbook engine" sequence steps and present an **approve/reject** control to the operator, and execute approved steps via **Nornir/Netmiko**. This is fully offline, demoable, and standards-flavored without the operational weight of a full SOAR install. (Mention StackStorm/Rundeck/SOARCA as the production upgrade path.)

#### A3.4 Mapping the 4 validation scenarios → playbooks (see consolidated table in A6).

---

### A4. Operator UI / Dashboards (fully offline, demo-ready)

#### A4.1 Recommended stack (pragmatic + offline)

| Layer | Recommendation | Why / offline notes |
|---|---|---|
| **Metrics dashboards + alerting** | **Grafana** (self-hosted) over **Prometheus** (+ Loki for syslog/logs) | Industry-standard NOC visualization; native alerting. Installs/runs air-gapped: download plugin zips and unzip into the plugins dir, set `GF_PLUGINS_PREINSTALL_SYNC`, allow unsigned plugins via `allow_loading_unsigned_plugins`, provision dashboards/datasources as code (YAML), enable anonymous/local auth. [Grafana plugin-management docs](https://grafana.com/docs/grafana/latest/administration/plugin-management/) · [Air-gapped monitoring stack guide](https://mylinux.work/guides/air-gapped-monitoring-stack/) · [Grafana community — install plugin in disconnected env](https://community.grafana.com/t/install-plug-in-disconnected-network/132273) |
| **Copilot console (custom app)** | **FastAPI (backend) + React or Svelte (frontend)**; for *fastest* demo, **Streamlit** | FastAPI gives WebSocket streaming for the chat + a clean API the playbook engine reuses. Streamlit is the quickest Python-only path (no JS build) but has **no alerting** and weaker layout control — fine for a hackathon demo. [Red Hat — monitor infra with Streamlit](https://www.redhat.com/en/blog/streamlit-monitor-infrastructure) · [Medium — Streamlit vs Grafana for dashboards](https://medium.com/@lasyachowdary1703/day-39-building-a-real-time-dashboard-with-streamlit-or-grafana-9fef8232b77f) |
| **Topology graph viz** | **Cytoscape.js** (preferred) or **vis-network**; **D3** if fully custom | Cytoscape.js is purpose-built for network/graph rendering, supports layouts + interaction (click node → drill into incident), ships as a self-contained JS bundle (vendor it; no CDN). vis-network is simpler for quick force-directed topology. |
| **Live risk timeline** | Grafana time-series panel **or** a Plotly/uPlot chart embedded in the custom app | Shows Risk score trending up *before* impact — the visual proof of "prediction with lead time." |
| **The 3 answers + chat** | Custom FastAPI/React panel: Q1 what/when, Q2 why (top signals), Q3 what-to-do (playbook), + a chat box hitting the local LLM | This is the copilot surface; streams tokens from the offline model. |

**Pragmatic recommendation:** **Grafana for the telemetry/alerting wall** + a **single custom FastAPI+React (or Streamlit) copilot page** that embeds a **Cytoscape.js topology**, a **risk timeline**, the **3-answer incident card**, and a **chat box**. Everything served from localhost; **all JS/CSS/fonts vendored locally** (no Google Fonts, no CDNs) — this is also an air-gap requirement (Part B).

#### A4.2 What the topology view must show
Color/size nodes by risk; highlight the **root-cause node** and shade the **blast-radius subgraph** (from A1.4) so the operator instantly sees "this PE is the cause; these 4 branch sites are at risk." Clicking a node opens its incident card. Topology-aware visualization of cascading dependencies is exactly what differentiates AIOps consoles from flat alert lists. [InsightFinder — dependency graph for reliability](https://insightfinder.com/blog/announcing-insightfinders-dependency-graph-a-new-way-to-ensure-service-reliability/)

---

### A5. Incident Summary Generation (operator-ready, grounded, no hallucination)

**Approach: templated skeleton + LLM fill, strictly grounded in the correlated evidence (RAG), with inline citations.** The MDPI "agentic network-traffic incident-report" work is the closest blueprint: a graph-based multi-agent system that uses **RAG so the report is based on retrieved context rather than free-form text completion** (which is what causes hallucination/fabrication), producing a **structured report**: classification verdict + confidence + threat level, attack analysis, key contributing features (e.g. SHAP), **reasoning with inline citations to retrieved evidence**, and mitigation recommendations grounded in retrieved intel. [MDPI — LLM agentic incident-report for XAI network defense](https://www.mdpi.com/2224-2708/15/2/32)

Design rules (these defend the 35% "grounded, no hallucination" score):
- **RAG-ground every claim.** Retrieve from local artifacts only (topology, runbooks, past incidents, the correlated evidence record from A1.2). RAG reduces hallucination ~30–70% across domains. [Sharvari Raut — stop RAG hallucinations](https://sharur7.medium.com/how-to-stop-llm-hallucinations-in-retrieval-augmented-generation-rag-5ef2894f9cd6)
- **Lightweight model + reduced-hallucination prompting.** A 2025 paper specifically does *incident-response planning with a lightweight LLM and reduced hallucination* — supports our quantized-7B/8B choice (Mistral-7B / Llama-3-8B / Phi-3). [arXiv 2508.05188 — Incident Response with a lightweight LLM, reduced hallucination](https://arxiv.org/pdf/2508.05188)
- **Structured/templated output** with fixed fields: *Predicted issue · Confidence (calibrated) · Probable root cause · Affected sites/services/SLAs · Time-to-impact · Recommended actions · Evidence citations · Urgency class.* Fixed slots keep the model on-rails and make it auditable.
- **Faithfulness/groundedness checks.** Score the summary's faithfulness to retrieved evidence (the literature flags that security LLM systems *lack rigorous faithfulness evaluation*); a simple grounding check = every named entity/number in the summary must appear in the evidence record, else flag. [MDPI](https://www.mdpi.com/2224-2708/15/2/32)
- **Templated fallback.** If retrieval confidence is low, emit the deterministic template *without* LLM prose (numbers + citations only) so the operator never gets a confident-sounding fabrication.

---

### A6. Scenario → Detection-Signal → Playbook Mapping (Phase 6 validation set)

For each of the four required scenarios: the **precursor signals** the predictive/correlation engine watches, the **correlation/RCA behavior**, and a concrete **recommended remediation playbook** (suggest → approve → execute). Signal choices are grounded in the BGP/MPLS/OSPF behavior cited below.

| # | Scenario | Precursor detection signals (Q1/Q2) | Correlation / RCA behavior | Recommended playbook (Q3, suggest→approve→execute) |
|---|---|---|---|---|
| **1** | **Progressive congestion on a hub-spoke link** | Rising interface utilization slope toward saturation; queue-drop/output-discard counter creep; latency/jitter drift; throughput plateau — **forecast-band breach** `r=|actual−forecast|>σ·band` flags *before* threshold. | Single root node = the hub-spoke interface; blast radius = spoke sites + apps whose flows traverse it (NetFlow intersect). Time-to-impact from utilization trend extrapolation. | 1) Collect `show interface`/queue stats (read-only, auto-ok). 2) Identify top-talker flows (NetFlow). 3) **Propose**: raise QoS priority for business-critical class / shift traffic to alternate SD-WAN path / rate-limit bulk class. 4) On approval, push via Nornir/NAPALM; verify utilization drops; else rollback. |
| **2** | **BGP route flap → downstream path-reroute cascade** | BGP adjacency up/down churn; rapid best-path changes A→B→A; **recursive-routing / next-hop instability**; **flap penalty** rising; OSPF SPF recompute storms. Precursor = flap penalty trend + adjacency timer jitter. [Cisco — flapping BGP / recursive routing](https://www.cisco.com/c/en/us/support/docs/ip/border-gateway-protocol-bgp/19167-bgp-rec-routing.html) · [IP With Ease — BGP flapping](https://ipwithease.com/what-is-bgp-flapping/) | Group all reroute symptoms under the flapping peer/prefix (earliest onset + highest centrality = root). OSPF recompute alarms are *symptoms*. Suppress with flap-damping logic (A2.3). [Threatshare — BGP route-flapping impact](https://threatshare.ai/networking/bgp-route-flapping/) | 1) Collect BGP/OSPF neighbor + flap stats (auto-ok). 2) Identify flapping prefix/peer + cause (link vs config). 3) **Propose**: enable/adjust **BGP flap damping**, pin next-hop / add static fallback, or shut the unstable peer pending fix. 4) On approval, apply via NAPALM; confirm convergence stable; rollback if worse. |
| **3** | **Intermittent MPLS underlay failure → tunnel degradation** | Tunnel packet-loss progression; jitter trend; LDP/RSVP-TE label-path churn / continuous tunnel re-routing; **IPSec rekey anomalies**; BFD flaps on the underlay. Precursor = loss/jitter slope + rekey-interval anomaly. [USPTO 9667559 — MPLS/GMPLS tunnel flap dampening](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9667559) | Root = the MPLS underlay segment/LSP; affected = all overlay tunnels + sites riding that LSP. Distinguish underlay flap from overlay symptom via topology layering (underlay vs overlay graph). | 1) Collect tunnel/LSP/BFD stats + rekey logs (auto-ok). 2) Localize faulty LSP/underlay hop. 3) **Propose**: reroute LSP onto healthy TE path / fail tunnel over to backup transport / trigger controlled IPSec rekey. 4) On approval, apply + verify loss/jitter recover; rollback path if needed. |
| **4** | **Controller misconfiguration → policy drift** | Config-version/intent diff vs last-known-good; sudden policy/route-map/QoS-class changes from the SD-WAN controller; correlated multi-site behavior change with **no** corresponding physical fault. Precursor = config-drift event + downstream metric shift. | Root = the controller change event (earliest, fans out to many sites simultaneously → high centrality, broad blast radius, no underlay fault) — the "many sites at once, no hardware alarm" signature *is* the discriminator. | 1) Diff running vs intended/golden config across affected sites (read-only). 2) Attribute drift to the specific controller push/commit. 3) **Propose**: revert to last-known-good policy / roll back the controller change / re-push golden config. 4) On approval, execute via controller API or NAPALM `replace`; verify metrics normalize; keep audit record of who approved. |

> Each playbook ends with **verification + rollback**, and every state-changing step requires **operator approval** (A3.2) and is **audit-logged** (Part B, B8).

---

## PART B — AIR-GAP SECURITY & COMPLIANCE (the 20% — rigorous)

> Goal: the LLM + RAG + analytics **literally cannot** reach the internet at runtime, and we can **prove it on demand** to a judge. We use **defense-in-depth**: multiple independent layers, each of which alone blocks egress, plus an always-on **monitor** and a runnable **conformance test**.

### B6. Enforcing Zero Egress — concrete Linux/container controls

Layer these (any one suffices; together they're belt-and-suspenders):

#### B6.1 Layer 1 — Container has *no* network path to the outside

- **Hardest isolation — `--network none`:** the container gets *no* networking stack at all; it cannot reach any external network or other containers. Use this for components that need zero network (e.g. a batch analytics job). [Docker networking overview](https://docs.docker.com/engine/network/) · [Docker forums — internal network w/o external access](https://forums.docker.com/t/internal-network-between-containers-without-external-network-access/41751)
  ```bash
  docker run --network none my-analytics:pinned
  ```
- **Internal-only bridge (components must talk to each other but not the internet):** a user-defined network with `internal: true` **intentionally blocks internet access** while allowing inter-container traffic. This is the right mode for the LLM + RAG + API + UI mesh. [Docker docs](https://docs.docker.com/engine/network/) · [Netdata — container has no internet](https://www.netdata.cloud/guides/docker/docker-container-cannot-connect-to-internet/)
  ```yaml
  # docker-compose.yml
  networks:
    airgap_net:
      driver: bridge
      internal: true        # <-- no route to the host's external interface / internet
  services:
    llm:      { image: llm:pinned,    networks: [airgap_net] }
    rag:      { image: rag:pinned,    networks: [airgap_net] }
    api:      { image: copilot:pinned, networks: [airgap_net] }
    grafana:  { image: grafana:pinned, networks: [airgap_net] }
  ```
- **No `--add-host`/public DNS; disable name resolution to the internet** (see B6.4).

#### B6.2 Layer 2 — Host firewall default-DROP egress (nftables)

Even if a container somehow had a route, the host kernel drops outbound. **Default policy DROP on OUTPUT**, allow only loopback + the internal docker subnet + (optionally) the LAN telemetry sources. nftables example:

```bash
# /etc/nftables.conf  — default-deny egress, allow only loopback + internal bridge
table inet airgap {
  chain output {
    type filter hook output priority 0; policy drop;   # DEFAULT DROP all egress
    oif "lo" accept                                     # loopback ok
    ip daddr 172.18.0.0/16 accept                       # docker internal bridge subnet
    ip daddr 10.0.0.0/8 accept                          # (optional) LAN telemetry sources only
    ct state established,related accept                 # replies to allowed inbound
    log prefix "AIRGAP-EGRESS-DROP " counter            # log + count every blocked attempt
  }
}
```
Equivalent iptables: `iptables -P OUTPUT DROP` then `-A OUTPUT -o lo -j ACCEPT`, allow the docker bridge, and `-A OUTPUT -j LOG --log-prefix "AIRGAP-EGRESS-DROP "`. Note Docker manages its own iptables chains; ensure nothing flushes/reorders them or traffic could leak — the host policy DROP is the backstop. [Netdata — Docker iptables/firewall behavior](https://www.netdata.cloud/guides/docker/docker-container-cannot-connect-to-internet/) · [OneUptime — Docker can't reach internet](https://oneuptime.com/blog/post/2026-02-08-how-to-troubleshoot-docker-container-cannot-reach-internet/view)

> **Pro move (verifiability):** instead of silently dropping, **route all egress to a blackhole / sinkhole and alert on every attempt**. `ip route add blackhole 0.0.0.0/0` on an isolated test netns, or point default route at an unrouted address; combine with the LOG rule above so each attempt is *counted* and visible in the demo.

#### B6.3 Layer 3 — Kubernetes (if used) default-deny egress NetworkPolicy

Kubernetes default posture *allows all*; you must impose **default-deny egress** and allow only intra-cluster. Requires a CNI that enforces policy (Calico/Cilium/Weave). [Kubernetes Recipes — default deny all](https://kubernetes.recipes/recipes/networking/networkpolicy-deny-all/) · [Medium — enforcing NetworkPolicies for egress control](https://medium.com/@bavicnative/network-security-in-kubernetes-implementing-enforcing-network-policies-c3d9d7d14a3b)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: default-deny-egress, namespace: copilot }
spec:
  podSelector: {}
  policyTypes: [Egress]            # deny ALL egress by default
  # (no egress rules = nothing allowed out)
```
- **Calico** uniquely supports **explicit Deny** rules (vanilla NetworkPolicy is allow-list only) — useful to *prove* a deny is in force. [Tigera/Calico — explicit deny](https://www.tigera.io/blog/deep-dive/what-you-cant-do-with-kubernetes-network-policies-unless-you-use-calico-the-ability-to-explicitly-deny-policies/)
- **Cilium** adds a **DNS-proxy** to observe/filter all DNS egress for selected pods. [Cilium docs — ingress & network policy](https://docs.cilium.io/en/latest/network/servicemesh/ingress-and-network-policy/) · [CNCF — safely managing Cilium policies](https://www.cncf.io/blog/2025/11/06/safely-managing-cilium-network-policies-in-kubernetes-testing-and-simulation-techniques/)
- **Mind DNS:** after default-deny, pods can't resolve DNS — in an air-gap that's *desired* (we want no external DNS), but allow only intra-cluster `kube-dns` UDP/53 if internal service discovery is needed. [Medium — DNS under default-deny](https://medium.com/@bavicnative/network-security-in-kubernetes-implementing-enforcing-network-policies-c3d9d7d14a3b)

#### B6.4 Layer 4 — DNS sinkhole / disable external resolution

No external DNS = no domain the LLM/RAG could resolve even if a request escaped. Point `/etc/resolv.conf` at a local-only resolver (or `0.0.0.0`), and/or run a sinkhole that answers all external names with `0.0.0.0`. Cilium DNS-proxy (above) can additionally log/deny DNS egress. [Cilium DNS-proxy](https://docs.cilium.io/en/latest/network/servicemesh/ingress-and-network-policy/)

#### B6.5 Layer 5 — Process sandboxing (the LLM process itself)

Wrap the inference process so it has its **own empty network namespace** and a **seccomp** filter that forbids/limits socket syscalls:

- **firejail** — uses Linux namespaces + seccomp-bpf + capabilities (+ AppArmor if present); `firejail --net=none ./serve_llm` gives the process a private network view with no connectivity. [GitHub — netblue30/firejail](https://github.com/netblue30/firejail) · [ArchWiki — Firejail](https://wiki.archlinux.org/title/Firejail)
- **bubblewrap** — lower-level, scriptable, minimal deps; `bwrap --unshare-net ...` runs the process in a new network namespace with no interfaces. Preferred when you want fine-grained, auditable isolation. [LinuxTechi — Firejail vs Bubblewrap](https://www.linuxtechi.com/sandbox-linux-apps-firejail-bubblewrap/) · [IT'S FOSS — sandbox with Firejail/Bubblewrap](https://itsfoss.gitlab.io/blog/how-to-sandbox-linux-apps-with-firejail-and-bubblewrap/)
- **seccomp** — kernel syscall allow-list; on a disallowed syscall the kernel **kills the process**. Drop `socket`/`connect` for any component that must never touch the network. [LinuxTechi](https://www.linuxtechi.com/sandbox-linux-apps-firejail-bubblewrap/)
- **AppArmor / SELinux** profiles + Docker `--cap-drop NET_RAW NET_ADMIN`, `--security-opt no-new-privileges`, and a seccomp profile denying network syscalls.

#### B6.6 Layer 6 — choose telemetry-free, offline-by-design software

- **Ollama** has **no telemetry and is fully offline-safe**; **LM Studio** likewise. **llama.cpp** needs no GPU/internet and runs on an air-gapped workstation. **vLLM** works offline if you **pull all images/weights before disconnecting**. Models packaged as **GGUF** (single file: weights+tokenizer+metadata). [Red Hat — llama.cpp vs vLLM](https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine) · [InsiderLLM — air-gapped local LLM guide](https://insiderllm.com/guides/running-ai-offline-complete-guide/) · [Sesame Disk — local inference engines 2026](https://sesamedisk.com/local-inference-engines-2026-comparison/)
- **Recommendation:** **Ollama or llama.cpp** serving a **quantized Mistral-7B / Llama-3-8B / Phi-3 GGUF**, with `OLLAMA_*`/no-update flags and the network layers above. Local vector DB (Chroma/FAISS/Qdrant) for RAG — all on the internal bridge.

---

### B7. Verifying / Proving Air-Gap (the part judges reward) — continuous monitor + conformance test

Two deliverables: **(a)** an **always-on egress monitor** that shows zero successful external connections (and counts/logs every *attempt*), and **(b)** a **one-command conformance test** that actively tries to escape and asserts every attempt fails.

#### B7.1 Always-on egress monitoring (passive proof)

- **nftables/iptables LOG + counters** (from B6.2): every blocked egress increments a counter and logs `AIRGAP-EGRESS-DROP`. Surface the counter in Grafana — a flat line at "0 successful egress, N blocked attempts" is the live proof.
- **conntrack:** `conntrack -L` / `conntrack -E` shows the live connection table — demonstrate there are **no ESTABLISHED flows to non-RFC1918 addresses**. `ss -tunp` / `ss -tuln` likewise enumerates current sockets (the standard air-gap spot-check). [InsiderLLM — verify with `ss -tuln`](https://insiderllm.com/guides/running-ai-offline-complete-guide/)
- **Falco runtime rule — alert on ANY outbound connection** from our pods/containers. Falco watches syscalls in real time and fires on unexpected outbound connects; the rule keys on `evt.type=connect`, IPv4/IPv6 socket (`fd.typechar in (4,6)`), excludes loopback, and matches the `outbound` macro / non-allowed destinations. Working rule (adapted from the community network-monitoring rule and Falco defaults): [Falco — default rules](https://falco.org/docs/reference/rules/default-rules/) · [rkatz — Falco outbound monitoring rule](https://www.rkatz.xyz/post/2021-04-16-falco-network-monitoring/) · [Sysdig — tuning Falco network rules](https://www.sysdig.com/blog/day-2-falco-container-security-tuning-the-rules)
  ```yaml
  - macro: outbound_corp
    condition: >
      (((evt.type = connect and evt.dir=<) or
        (evt.type in (sendto,sendmsg) and evt.dir=< and fd.l4proto != tcp
         and fd.connected=false and fd.name_changed=true)) and
       (fd.typechar = 4 or fd.typechar = 6) and
       (fd.ip != "0.0.0.0" and fd.net != "127.0.0.0/8") and
       (evt.rawres >= 0 or evt.res = EINPROGRESS))
  - rule: airgap outbound connection attempt
    desc: Any container in the air-gapped copilot attempted to connect outward
    condition: outbound_corp and container
    output: "AIRGAP VIOLATION outbound attempt (container=%container.name proc=%proc.name dstip=%fd.sip dstport=%fd.sport proto=%fd.l4proto)"
    priority: CRITICAL
  ```
  > Tighten by excluding the internal-bridge CIDR so only *external* attempts fire. Test the rule fires using a known-good trigger (Falco Test Suite / Atomic Red Team for containers). [Sysdig — Falco tuning](https://www.sysdig.com/blog/day-2-falco-container-security-tuning-the-rules)
- **Span/IDS (optional, strongest):** mirror the host/segment to **Zeek** or **Suricata** and assert zero external conversations — carrier-grade evidence if a judge wants packet-level proof.

#### B7.2 Active conformance test (one command, asserts NO egress succeeds)

Adapt egress-testing tools by **inverting their success criteria**: the test *tries every common exfil path* and **passes only if all attempts fail/time out**. Tools to model on: **egression** (tries cleartext FTP, SCP over multiple ports, and **DNS-query exfil** of a sensitive file; reports which levels were *not* blocked) and **Egresser** (client/server outbound-firewall rule tester). [GitHub — danielmiessler/egression](https://github.com/danielmiessler/egression) · [GitHub — cyberisltd/Egresser](https://github.com/cyberisltd/Egresser) · [HackerTarget — egress firewall test](https://hackertarget.com/egress-firewall-test/)

A self-contained **pytest** conformance test (runs inside each container; ships in the bundle):

```python
# test_airgap_conformance.py  —  PASSES only if every egress path is blocked.
import socket, subprocess, pytest

EXTERNAL_IPS   = ["1.1.1.1", "8.8.8.8", "9.9.9.9"]          # well-known public IPs
EXTERNAL_PORTS = [53, 80, 443, 22, 21, 123]                 # DNS/HTTP/HTTPS/SSH/FTP/NTP
TIMEOUT = 3

@pytest.mark.parametrize("ip", EXTERNAL_IPS)
@pytest.mark.parametrize("port", EXTERNAL_PORTS)
def test_tcp_egress_blocked(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(TIMEOUT)
    try:
        s.connect((ip, port))
        pytest.fail(f"AIR-GAP BREACH: reached {ip}:{port}")   # any success = FAIL
    except (socket.timeout, OSError):
        pass                                                  # blocked/timed out = PASS
    finally:
        s.close()

def test_dns_resolution_blocked():
    # external name resolution must fail (no exfil-via-DNS, no name leakage)
    with pytest.raises(Exception):
        socket.gethostbyname("example.com")

def test_https_fetch_blocked():
    # curl with a hard timeout must NOT succeed (exit!=0 expected)
    r = subprocess.run(["curl","-sS","--max-time","3","https://example.com"],
                       capture_output=True)
    assert r.returncode != 0, "AIR-GAP BREACH: HTTPS egress succeeded"

def test_udp_dns_egress_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(TIMEOUT)
    try:
        s.sendto(b"\x00", ("8.8.8.8", 53))
        s.recvfrom(512)                                       # a reply = breach
        pytest.fail("AIR-GAP BREACH: UDP/53 egress got a reply")
    except (socket.timeout, OSError):
        pass
    finally:
        s.close()
```
Run as part of demo/CI: `pytest -q test_airgap_conformance.py` → all green = **verifiable** zero egress. Reference egress-tester behavior (FTP/SCP/DNS, multi-port) shows what paths to cover. [danielmiessler/egression](https://github.com/danielmiessler/egression)

> **Demo script for judges:** (1) show `nftables` policy = drop + the live blocked-attempt counter, (2) run the pytest conformance suite live → all pass, (3) show the Falco CRITICAL rule armed and that it would fire (trigger once with the rule's external-CIDR excluded to demonstrate detection works), (4) show `conntrack -L` with no external ESTABLISHED flows. That is a complete, reproducible "verifiable zero outbound" demonstration.

#### B7.3 Supply-chain: vendor everything, hash-pin, SBOM

So that *build/install time* also has no outbound dependency and is tamper-evident:

- **Python wheels offline:** resolve the **full transitive closure** on a connected build host, mirror wheels + PEP 503 metadata for the target platform (**bandersnatch** for full PyPI mirror, or **minirepo / pypi-disconnected** for a selective subset), copy to the air-gapped host, and install with **`pip install --no-index --find-links=/path/to/wheels -r requirements.txt`** so pip never touches the network. [python.org — full offline PyPI via devpi+bandersnatch](https://mail.python.org/archives/list/devpi-dev@python.org/thread/7P46MDSD7XVNSTIZONZAXRCQEK2D6D3D/) · [GitHub — scohmer/pypi-disconnected](https://github.com/scohmer/pypi-disconnected) · [discuss.python.org — host PyPI on air-gapped server](https://discuss.python.org/t/how-do-i-locally-host-a-pypi-repository-on-an-air-gapped-server/60704) · [minirepo](https://pypi.org/project/minirepo/)
- **Hash-pin** with a fully pinned `requirements.txt` + **`pip install --require-hashes`** so any altered wheel is rejected.
- **SBOM** (CycloneDX/SPDX via `syft`/`cyclonedx-py`) for every image and the wheel set — required evidence for regulated environments and pairs with SLSA/attestations (B8).

---

### B8. Offline Security Controls (secrets, audit, RBAC, integrity, reproducibility)

| Control area | Recommendation (offline) | Source |
|---|---|---|
| **Secrets management** | **HashiCorp Vault self-hosted** with **Transit-engine auto-unseal** (designed for air-gapped, avoids cloud KMS lock-in); centralized control plane, strict policies, **complete audit trail of every operation**, encryption-as-a-service at rest + in transit; root CA generated/stored **offline (air-gapped machine)**, intermediate CA per env. Lightweight alt for the demo: **SOPS + age** to encrypt secrets-at-rest in the repo. | [HashiCorp Vault — product](https://www.hashicorp.com/en/products/vault) · [HashiCorp — HA auto-unseal in air-gapped Vault](https://support.hashicorp.com/hc/en-us/articles/42233123177491-HA-Auto-Unseal-with-dual-transit-clusters-in-air-gapped-vault-environments-with-secure-token-injection) · [OneUptime — Vault auto-unseal](https://oneuptime.com/blog/post/2026-01-27-vault-auto-unseal/view) |
| **Audit logging** | Append-only audit log of every operator action, every approved/executed playbook step (who/what/when), and every Falco alert; ship to local Loki/Elasticsearch. Vault's own audit device covers secret access. | [HashiCorp Vault](https://www.hashicorp.com/en/products/vault) |
| **RBAC + approval gates** | Operator/approver roles; state-changing playbook steps require explicit approval (A3.2). **Rundeck** provides built-in ACL/RBAC + approval workflow if a heavier engine is wanted. | [awesome-network-automation (Rundeck)](https://github.com/networktocode/awesome-network-automation/blob/master/README.md) |
| **Integrity verification** | **cosign --offline verify** with the **Rekor inclusion proof bundled** alongside the signature (sign with `--tlog-upload=false`, or mirror **Rekor + Fulcio** internally for full transparency). Verify all images on load. Plus plain **SHA-256 checksums** for the bundle + model files. | [OpenSSF — Sigstore for image signing](https://openssf.org/blog/2024/02/16/scaling-up-supply-chain-security-implementing-sigstore-for-seamless-container-image-signing/) · [GitGuardian — Sigstore/Cosign II](https://blog.gitguardian.com/supply-chain-security-sigstore-and-cosign-part-ii/) · [IntelligenceX — Sigstore/Cosign/SLSA](https://blog.intelligencex.org/secure-supply-chain-with-sigstore-cosign-slsa-framework) |
| **Reproducible / hermetic builds + provenance** | Pinned base images by **digest** (not tag), hermetic inputs (vendored wheels, no network at build), **SLSA build provenance** via in-toto/ITE-6 attestations, optional SLSA L3 if built in a hardened/ephemeral builder. | [AquilaX — beyond SBOMs: Sigstore/SLSA/provenance](https://aquilax.ai/blog/supply-chain-artifact-signing-slsa) · [arXiv 2503.20079 — ARGO-SLSA](https://arxiv.org/pdf/2503.20079) |
| **Data-at-rest (regulated)** | LUKS/dm-crypt full-disk encryption for the host; encrypted volumes for telemetry/RAG stores; Vault-managed keys; data **never leaves the air-gap boundary** (matches the dataset constraint in the problem statement). | [NIST SP 800-53 SC-7](https://csf.tools/reference/nist-sp-800-53/r5/sc/sc-7/) |

**Standards mapping (high level):**
- **NIST SP 800-53 r5 — SC-7 Boundary Protection** and enhancement **SC-7(21) Isolation of System Components**: isolate components via **physically/logically separate subnetworks** to limit unauthorized information flows and reduce breach susceptibility — exactly the air-gap posture. **SC-7(13)** isolates security tools/mechanisms; **SC-3** security-function isolation. [CSF Tools — SC-7](https://csf.tools/reference/nist-sp-800-53/r5/sc/sc-7/) · [GRC Academy — SC-7(21)](https://grcacademy.io/nist-800-53/controls/sc-7-21/) · [CSF Tools — SC-7(13)](https://csf.tools/reference/nist-sp-800-53/r4/sc/sc-7/sc-7-13/) · [Tenable — SC-3](https://www.tenable.com/audits/references/800-53/SC-3)
- Map our controls to **SC-7** (egress deny + boundary), **AU** (audit), **AC** (RBAC/least-privilege), **SI/SR** (integrity, supply-chain), and reference **ISO 27001** Annex A (access control, cryptography, ops security) at a high level for the documentation deliverable.

---

### B9. Packaging for Air-Gap Delivery (deterministic, hash-verified offline bundle)

Ship the whole product as a single offline bundle a NOC loads with **no internet**:

1. **Save all images** to a tarball: `docker save` exports images with **all layers/metadata/tags**; both Docker- and **OCI-format** archives auto-detect on load. Bundle + compress (`gzip`/`xz`):
   ```bash
   docker save llm:pinned rag:pinned copilot:pinned grafana:pinned prometheus:pinned \
     | gzip > copilot-airgap-images.tar.gz
   sha256sum copilot-airgap-images.tar.gz > copilot-airgap-images.tar.gz.sha256
   ```
   [OneUptime — docker save/load offline transfer](https://oneuptime.com/blog/post/2026-01-16-docker-export-import-images/view) · [DevOpsSchool — docker save](https://www.devopsschool.com/blog/docker-save/) · [RepoFlow — preparing Docker images for air-gap](https://docs.repoflow.io/Self-Hosting/air-gapped-preparation)
2. **Include** the `docker-compose.yml` (internal-only network), `nftables.conf`, Falco rules, the pytest conformance test, the GGUF model file(s), the vendored wheel dir, SBOMs, cosign signatures + Rekor bundles, and an `install.sh`. K3s/RKE2 air-gap docs are the canonical pattern if delivering on Kubernetes (`images` dir + tar import). [K3s — air-gap install](https://docs.k3s.io/installation/airgap) · [RKE2 — air-gap install](https://docs.rke2.io/install/airgap)
3. **On the air-gapped host (deterministic, verified):**
   ```bash
   sha256sum -c copilot-airgap-images.tar.gz.sha256          # integrity gate
   tar -tzf copilot-airgap-images.tar.gz                     # inspect before load
   gunzip -c copilot-airgap-images.tar.gz | docker load      # load images, no registry
   cosign verify --offline --key cosign.pub llm:pinned       # signature/provenance gate
   pip install --no-index --find-links=./wheels --require-hashes -r requirements.txt
   docker compose up -d                                       # internal-only network
   pytest -q test_airgap_conformance.py                      # PROVE zero egress on first boot
   ```
   Tip: verify tarball integrity *before* load and inspect `manifest.json` `RepoTags` to confirm names/tags survive. [w3tutorials — preserve image names on save/load](https://www.w3tutorials.net/blog/docker-save-load-lose-original-image-repository-name-tag/)
4. **Determinism:** pin images by **digest**, pin wheels by hash, checksum the whole bundle, and have `install.sh` **abort** on any checksum/signature/conformance failure. The conformance test running at first boot is what makes the delivery *verifiably* air-gapped from minute one.

---

## TL;DR — Concrete Recommendations for the Team

- **Correlation/RCA:** `networkx` digital-twin graph; temporal-window + topological grouping (WCC/SCC over a failure subgraph); rank root cause by `centrality × earliest-onset × Granger-causal-score`; blast radius via **BFS reachability** (`nx.descendants` / shortest-path-length) intersected with NetFlow. (Model on the AWS Neptune/DOCOMO pipeline.)
- **Prioritization:** product-form Risk = `AnomalyConf × TimeToImpactUrgency × BlastRadius × AssetCriticality`; **Platt-scale** the model confidence (report reliability diagram + Brier); suppress flapping with **BGP-style penalty/decay**.
- **Playbooks:** **CACAO-style JSON** runbooks in RAG; **suggest → operator-approve → execute** via **Nornir/Netmiko/NAPALM**; never auto-change device state. Each of the 4 scenarios mapped to signals + a verify/rollback playbook (table in A6).
- **UI:** **Grafana** (offline, provisioned-as-code) for the telemetry wall + a **FastAPI+React (or Streamlit)** copilot page with a **Cytoscape.js** topology (root-cause node + blast-radius highlighted), a **risk timeline**, the **3-answer card**, and a **chat box** to the local LLM. All assets vendored, no CDNs.
- **Summaries:** templated structured fields, RAG-grounded with inline citations, faithfulness check, deterministic fallback on low retrieval confidence.
- **Air-gap enforcement (defense-in-depth):** Docker `internal: true` bridge (or `--network none`) + host **nftables default-DROP egress (+log/counter)** + (K8s) **default-deny egress NetworkPolicy** + **DNS sinkhole** + **firejail/bwrap `--unshare-net` + seccomp** on the LLM + telemetry-free software (Ollama/llama.cpp).
- **Air-gap VERIFICATION (the 20%):** always-on **nftables blocked-attempt counter + Falco CRITICAL outbound rule + `conntrack`** monitor, *plus* a runnable **pytest conformance suite** that actively tries TCP/UDP/DNS/HTTPS egress and **passes only if all fail** — demo it live to the judges.
- **Supply chain / packaging:** vendored hash-pinned wheels (`pip --no-index --require-hashes`), SBOMs, **cosign --offline** verify (bundled Rekor proof), **`docker save | gzip`** image tarball with SHA-256, `install.sh` that aborts on any integrity/conformance failure and runs the air-gap test on first boot.

---

### Source URLs (consolidated)

**Correlation / RCA / causal / graph:**
- AWS — Network digital twin graph + agentic AI RCA: https://aws.amazon.com/blogs/database/beyond-correlation-finding-root-causes-using-a-network-digital-twin-graph-and-agentic-ai/
- ScienceDirect — RCA via anomaly-correlation graph + GNN: https://www.sciencedirect.com/science/article/abs/pii/S0952197626007323
- Frontiers — alarm reduction & root-cause inference (assoc. mining): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1211739/full
- HEAL — AIOps event correlation: https://healsoftware.ai/blog/how-aiops-event-correlation-transforms-incident-response.html
- Selector — AIOps 2025 components: https://www.selector.ai/learning-center/aiops-in-2025-4-components-and-4-key-capabilities/
- arXiv 2402.01140 — RUN / Neural Granger causal discovery: https://arxiv.org/abs/2402.01140
- ACM — Evaluation of Causal Inference Techniques for AIOps: https://dl.acm.org/doi/fullHtml/10.1145/3430984.3431027
- ResearchGate — Causal Inference AI for RCA in DevOps: https://www.researchgate.net/publication/392499895_Applying_Causal_Inference_AI_Models_to_Root_Cause_Analysis_RCA_in_DevOps
- IJRTI — Applying Causal Inference to RCA: https://ijrti.org/papers/IJRTI2505203.pdf
- arXiv 2403.06485 — Knowledge-aware Alert Aggregation: https://arxiv.org/pdf/2403.06485
- Endor Labs — measuring/reducing blast radius: https://www.endorlabs.com/learn/vulnerability-blast-radius-how-to-measure-and-reduce-impact
- Medium — graph-theory reachability analysis: https://medium.com/@a.talsma/data-driven-vulnerability-management-graph-theory-based-reachability-analysis-1-2-61f2fe185339
- USPTO 11637861 — reachability-graph safe remediation: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11637861
- InsightFinder — dependency graph for reliability: https://insightfinder.com/blog/announcing-insightfinders-dependency-graph-a-new-way-to-ensure-service-reliability/

**Prioritization / calibration / alert fatigue:**
- Deepwatch — Risk-Based Alerting: https://www.deepwatch.com/glossary/risk-based-alerting-rba/
- Safe Security — Risk Prioritization Framework 2026: https://safe.security/resources/blog/the-modern-risk-prioritization-framework-for-2026/
- JupiterOne — prioritize by asset criticality: https://www.jupiterone.com/blog/prioritizing-exploitable-vulnerabilities-to-protect-your-business-critical-assets
- Medium (Silas Potter) — alert severity scoring formula: https://medium.com/@silaspotter17/rethinking-alert-severity-a-formula-for-consistent-scoring-abbcb60e42ac
- Niculescu-Mizil & Caruana ICML'05 — Predicting Good Probabilities: https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- FastML — Platt vs isotonic calibration: https://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/
- Train in Data — guide to Platt scaling: https://www.blog.trainindata.com/complete-guide-to-platt-scaling/
- Abzu — calibration part II (Platt/isotonic/beta): https://www.abzu.ai/data-science/calibration-introduction-part-2/
- Palo Alto — reduce alert fatigue: https://www.paloaltonetworks.com/cyberpedia/how-to-reduce-security-alert-fatigue
- ACM Computing Surveys — Alert Fatigue in SOCs: https://dl.acm.org/doi/10.1145/3723158
- Torq — alert management 2026: https://torq.io/blog/cybersecurity-alert-management-2026/
- Conifers — false-positive suppression: https://www.conifers.ai/glossary/false-positive-suppression
- CyberDefenders — alert triage process: https://cyberdefenders.org/blog/alert-triage-process/

**Playbooks / automation / network tooling:**
- OASIS — CACAO Security Playbooks v2.0: https://www.oasis-open.org/standard/cacao-security-playbooks-v2-0/
- CACAO v2.0 spec: https://docs.oasis-open.org/cacao/security-playbooks/v2.0/security-playbooks-v2.0.html
- SecurityIntelligence — intro to CACAO: https://securityintelligence.com/posts/introducing-collaborative-automated-course-of-action-operations-cacao-an-emerging-cybersecurity-standard-to-quickly-define-and-share-playbooks/
- GitHub — SOARCA (CACAO orchestrator): https://github.com/COSSAS/SOARCA
- PacketCoders — network automation tooling landscape: https://www.packetcoders.io/network-automation-101-tooling-landscape/
- APNIC — Paramiko/Netmiko/NAPALM/Ansible/Nornir: https://blog.apnic.net/2023/02/13/automation-tools-paramiko-netmiko-napalm-ansible-nornir-or/
- awesome-network-automation (StackStorm/Rundeck/Nornir): https://github.com/networktocode/awesome-network-automation/blob/master/README.md

**UI / dashboards / summaries:**
- Grafana — plugin management (air-gap): https://grafana.com/docs/grafana/latest/administration/plugin-management/
- Grafana community — install plugin in disconnected env: https://community.grafana.com/t/install-plug-in-disconnected-network/132273
- Air-gapped monitoring stack guide: https://mylinux.work/guides/air-gapped-monitoring-stack/
- Red Hat — monitor infra with Streamlit: https://www.redhat.com/en/blog/streamlit-monitor-infrastructure
- Medium — Streamlit vs Grafana for dashboards: https://medium.com/@lasyachowdary1703/day-39-building-a-real-time-dashboard-with-streamlit-or-grafana-9fef8232b77f
- MDPI — LLM agentic incident-report (XAI network defense): https://www.mdpi.com/2224-2708/15/2/32
- arXiv 2508.05188 — Incident Response with lightweight LLM, reduced hallucination: https://arxiv.org/pdf/2508.05188
- Sharvari Raut — stop RAG hallucinations: https://sharur7.medium.com/how-to-stop-llm-hallucinations-in-retrieval-augmented-generation-rag-5ef2894f9cd6

**MPLS / BGP / OSPF signal references:**
- Cisco — flapping BGP / recursive routing: https://www.cisco.com/c/en/us/support/docs/ip/border-gateway-protocol-bgp/19167-bgp-rec-routing.html
- IP With Ease — BGP flapping: https://ipwithease.com/what-is-bgp-flapping/
- Juniper — BGP session/route flaps: https://www.juniper.net/documentation/us/en/software/junos/bgp/topics/topic-map/bgp-session-flaps.html
- Threatshare — BGP route-flapping impact: https://threatshare.ai/networking/bgp-route-flapping/
- USPTO 8289856 — alarm threshold for BGP flapping: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8289856
- USPTO 9667559 — MPLS/GMPLS tunnel flap dampening: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9667559

**Air-gap enforcement / verification / supply-chain / packaging:**
- Docker networking overview (none/internal): https://docs.docker.com/engine/network/
- Docker forums — internal network without external access: https://forums.docker.com/t/internal-network-between-containers-without-external-network-access/41751
- Netdata — Docker container has no internet (iptables behavior): https://www.netdata.cloud/guides/docker/docker-container-cannot-connect-to-internet/
- OneUptime — Docker can't reach internet (firewall): https://oneuptime.com/blog/post/2026-02-08-how-to-troubleshoot-docker-container-cannot-reach-internet/view
- Kubernetes Recipes — NetworkPolicy default-deny all: https://kubernetes.recipes/recipes/networking/networkpolicy-deny-all/
- Medium — enforcing NetworkPolicies / egress control: https://medium.com/@bavicnative/network-security-in-kubernetes-implementing-enforcing-network-policies-c3d9d7d14a3b
- Tigera/Calico — explicit deny policies: https://www.tigera.io/blog/deep-dive/what-you-cant-do-with-kubernetes-network-policies-unless-you-use-calico-the-ability-to-explicitly-deny-policies/
- Cilium docs — ingress & network policy / DNS-proxy: https://docs.cilium.io/en/latest/network/servicemesh/ingress-and-network-policy/
- CNCF — safely managing Cilium network policies: https://www.cncf.io/blog/2025/11/06/safely-managing-cilium-network-policies-in-kubernetes-testing-and-simulation-techniques/
- GitHub — netblue30/firejail: https://github.com/netblue30/firejail
- ArchWiki — Firejail: https://wiki.archlinux.org/title/Firejail
- LinuxTechi — sandbox with Firejail/Bubblewrap: https://www.linuxtechi.com/sandbox-linux-apps-firejail-bubblewrap/
- IT'S FOSS — sandbox Linux apps with Firejail/Bubblewrap: https://itsfoss.gitlab.io/blog/how-to-sandbox-linux-apps-with-firejail-and-bubblewrap/
- Falco — default rules (outbound detection): https://falco.org/docs/reference/rules/default-rules/
- rkatz — Falco outbound monitoring rule (full YAML): https://www.rkatz.xyz/post/2021-04-16-falco-network-monitoring/
- Sysdig — tuning Falco network rules: https://www.sysdig.com/blog/day-2-falco-container-security-tuning-the-rules
- GitHub — danielmiessler/egression (egress tester): https://github.com/danielmiessler/egression
- GitHub — cyberisltd/Egresser (outbound firewall tester): https://github.com/cyberisltd/Egresser
- HackerTarget — egress firewall test: https://hackertarget.com/egress-firewall-test/
- Red Hat — llama.cpp vs vLLM (offline inference): https://developers.redhat.com/articles/2026/06/15/llamacpp-vs-vllm-choosing-right-local-llm-inference-engine
- InsiderLLM — air-gapped local LLM guide (`ss -tuln`, no-telemetry): https://insiderllm.com/guides/running-ai-offline-complete-guide/
- Sesame Disk — local inference engines 2026 comparison: https://sesamedisk.com/local-inference-engines-2026-comparison/
- python.org — offline PyPI mirror (devpi + bandersnatch): https://mail.python.org/archives/list/devpi-dev@python.org/thread/7P46MDSD7XVNSTIZONZAXRCQEK2D6D3D/
- GitHub — scohmer/pypi-disconnected: https://github.com/scohmer/pypi-disconnected
- discuss.python.org — host PyPI on air-gapped server: https://discuss.python.org/t/how-do-i-locally-host-a-pypi-repository-on-an-air-gapped-server/60704
- minirepo (selective PyPI mirror): https://pypi.org/project/minirepo/
- OpenSSF — Sigstore for image signing: https://openssf.org/blog/2024/02/16/scaling-up-supply-chain-security-implementing-sigstore-for-seamless-container-image-signing/
- GitGuardian — Sigstore & Cosign (Part II): https://blog.gitguardian.com/supply-chain-security-sigstore-and-cosign-part-ii/
- IntelligenceX — Sigstore/Cosign/SLSA: https://blog.intelligencex.org/secure-supply-chain-with-sigstore-cosign-slsa-framework
- AquilaX — beyond SBOMs: Sigstore/SLSA/provenance: https://aquilax.ai/blog/supply-chain-artifact-signing-slsa
- arXiv 2503.20079 — ARGO-SLSA: https://arxiv.org/pdf/2503.20079
- OneUptime — docker save/load offline transfer: https://oneuptime.com/blog/post/2026-01-16-docker-export-import-images/view
- DevOpsSchool — docker save: https://www.devopsschool.com/blog/docker-save/
- RepoFlow — preparing Docker images for air-gap: https://docs.repoflow.io/Self-Hosting/air-gapped-preparation
- w3tutorials — preserve image names on save/load: https://www.w3tutorials.net/blog/docker-save-load-lose-original-image-repository-name-tag/
- K3s — air-gap install: https://docs.k3s.io/installation/airgap
- RKE2 — air-gap install: https://docs.rke2.io/install/airgap

**Secrets / standards:**
- HashiCorp Vault — product: https://www.hashicorp.com/en/products/vault
- HashiCorp — HA auto-unseal in air-gapped Vault: https://support.hashicorp.com/hc/en-us/articles/42233123177491-HA-Auto-Unseal-with-dual-transit-clusters-in-air-gapped-vault-environments-with-secure-token-injection
- OneUptime — Vault auto-unseal: https://oneuptime.com/blog/post/2026-01-27-vault-auto-unseal/view
- CSF Tools — NIST SP 800-53 SC-7 Boundary Protection: https://csf.tools/reference/nist-sp-800-53/r5/sc/sc-7/
- GRC Academy — SC-7(21) Isolation of System Components: https://grcacademy.io/nist-800-53/controls/sc-7-21/
- CSF Tools — SC-7(13) Isolation of Security Tools: https://csf.tools/reference/nist-sp-800-53/r4/sc/sc-7/sc-7-13/
- Tenable — 800-53 SC-3 Security Function Isolation: https://www.tenable.com/audits/references/800-53/SC-3
