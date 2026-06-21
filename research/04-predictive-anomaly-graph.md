# Phase 3b — Predictive Fault Analytics: Anomaly Detection, Change-Point, Routing-Instability & Graph Methods

**Problem Statement 13 — Air-Gapped Predictive Copilot for Secure MPLS Operations**
**Scope of this document:** the *non-forecasting* half of the analytics engine — anomaly detection (statistical/streaming, ML, deep), change-point/regime detection, matrix-profile discord discovery, graph/topology-aware methods, routing-instability-specific detectors, ensemble fusion, Extreme-Value-Theory (EVT) thresholding, and time-to-impact + root-cause/SHAP explanation feeding the LLM copilot.

> **Numbering convention.** A sibling agent owns the **forecasting** master list, methods **#1–#18**. To avoid collisions and guarantee the combined engine exceeds **30 distinct methods**, this document's master list begins at **#19** and runs to **#55+**. Combined total: **18 forecasting + 37 here = 55 distinct methods**, far exceeding the 30-method robustness target. Not all are deployed in production; the recommended *deployed ensemble* is a curated subset (see §12), but the catalogue gives cross-verification depth and fallbacks.

**Design constraints (from `idea.md`):**
- **Air-gapped / offline:** every library must `pip install` from a local wheelhouse / vendored mirror; **zero runtime outbound calls**. No cloud anomaly APIs (rules out Azure Anomaly Detector, AWS Lookout, Datadog Watchdog). Every package below is permissively licensed (BSD/MIT/Apache-2.0) and self-contained.
- **Precursor detection, not threshold breaches:** detectors must fire on *trajectory/regime/structure change* before SLA breach, yielding **lead time**.
- **Cross-verification:** ≥30 methods so detectors VERIFY each other and FILL GAPS; agreement → calibrated confidence.
- **Explainability:** every alert must answer Q2 ("why — which signals?") for the copilot → SHAP/feature attribution + graph correlation.

---

## 0. How this maps to the four validation scenarios (quick index)

| Scenario | Dominant signal | Best-fit detector families (method #s) |
|---|---|---|
| **A. Progressive congestion (hub-spoke link)** | Slow monotonic utilization/latency drift, queue build | Forecast-residual (#46), EWMA/CUSUM/Page-Hinkley (#20, #37, #38), Mann-Kendall trend, Matrix-Profile discord (#43), EVT/DSPOT threshold (#52), survival/time-to-impact (#54) |
| **B. BGP route-flap cascade** | Bursty update churn, AS-path changes, adjacency flaps, multi-node correlation | Routing-instability features (#48), ADWIN/KSWIN drift (#39, #40), S-H-ESD burst (#23), graph event-correlation + centrality blast-radius (#44, #45), GNN multivariate (#42), causal RCA (#47) |
| **C. Intermittent MPLS underlay / tunnel degradation** | Intermittent loss/jitter spikes, rekey anomalies, multimodal | Half-Space Trees / RRCF streaming (#26, #27), Spectral Residual (#41), LSTM-AE / USAD / TranAD (#34, #35), COPOD/ECOD (#29), Matrix-Profile (#43) |
| **D. Controller misconfig → policy drift** | Step change in config-derived metrics / flow patterns; new regime | BOCPD / PELT change-point (#36, #37), concept-drift DDM/HDDM (#39), Isolation Forest / PCA reconstruction (#30, #33), graph structural anomaly PyGOD-DOMINANT (#44), causal discovery (#47) |

Full per-scenario detector recommendations in **§11**.

---

## 1. Why a multi-method ensemble (not one model)

No single detector covers all four fault morphologies. Network precursors span: slow drifts (congestion), bursty churn (BGP flap), intermittent spikes (tunnel), and step regime changes (policy drift). The literature is explicit that **tree-based streaming ensembles (Isolation Forest, RRCF, Half-Space Trees, RHF) are the current state of the art for streaming anomaly detection**, while **deep reconstruction models (USAD, TranAD, OmniAnomaly) lead on multivariate correlated signals** — different winners for different regimes ([River/PySAD survey, arXiv 2108.11807](https://arxiv.org/pdf/2108.11807); [TranAD VLDB'22, arXiv 2201.07284](https://arxiv.org/abs/2201.07284)). Ensembling + EVT-based adaptive thresholds + graph correlation is the robust answer, and gives the copilot multiple independent "votes" to ground confidence and explanations.

---

## 2. MASTER LIST — Family 1: Statistical / streaming anomaly detection (O(1)-friendly)

These are **streaming-capable, O(1)-per-point or near-O(1)**, ideal for the always-on first tier in an air-gapped collector (Telegraf/Prometheus → Python). They give immediate lead-time signal at negligible cost.

**#19 — Z-score / Robust-Z (MAD-based).** Flag points whose deviation from a rolling mean (z-score) or rolling **median** exceeds k·σ (or k·MAD, the *robust* variant). Robust-Z uses Median Absolute Deviation so a few existing anomalies don't poison the baseline — the same robustness trick Twitter's S-H-ESD uses ([Twitter AnomalyDetection README](https://github.com/twitter/AnomalyDetection/blob/master/README.md)). *Wins:* univariate, cheap, interpretable baseline on every SNMP counter. *Complexity:* O(1) streaming (rolling window). *Library:* `river.anomaly.StandardAbsoluteDeviation` / `scipy.stats.median_abs_deviation` / hand-rolled NumPy. *Air-gap:* trivial.

**#20 — EWMA control chart.** Exponentially Weighted Moving Average tracks a smoothed mean; alarm when the observation leaves EWMA ± L·σ_EWMA control limits. Tunable λ trades responsiveness vs. noise; excellent at catching **slow drifts** (congestion buildup) earlier than a Shewhart chart. *Complexity:* O(1) streaming. *Library:* `statsmodels`/NumPy; also in `pyspc`. *Air-gap:* trivial.

**#21 — Shewhart chart + Western Electric / Nelson rules.** Classic SPC: ±3σ limits plus pattern rules (e.g., 2-of-3 beyond 2σ, 8 consecutive on one side, trends of 6). Pattern rules catch *non-breach precursors* (sustained drift) that single-point thresholds miss. *Complexity:* O(1). *Library:* `pyspc`, `spc` (PyPI). *Air-gap:* trivial.

**#22 — Generalized ESD (Extreme Studentized Deviate).** Iterative test for up to k outliers in approximately-normal data; the statistical core of S-H-ESD ([Twitter AnomalyDetection](https://github.com/twitter/AnomalyDetection)). *Wins:* batch scan of a window when count of outliers is uncertain. *Library:* `scikit-posthocs` / `PyAstronomy.pyasl.generalizedESD` / `seasonal-esd-anomaly-detection` ([nachonavarro repo](https://github.com/nachonavarro/seasonal-esd-anomaly-detection)). *Air-gap:* trivial.

**#23 — Seasonal-Hybrid-ESD (S-H-ESD) / Twitter AnomalyDetection.** STL/median seasonal decomposition, then Generalized-ESD on the residual using **median + MAD** so it is robust at high anomaly fractions and catches **both global and local** anomalies in seasonal traffic ([Twitter README](https://github.com/twitter/AnomalyDetection/blob/master/README.md); [Medium write-up](https://elisha32.medium.com/time-series-anomaly-detection-with-twitters-esd-test-50cce409ced1)). *Wins:* SNMP utilization with daily/weekly seasonality; BGP update-rate spikes against a seasonal baseline. *Complexity:* O(n log n) batch on a window. *Library (offline Python):* `seasonal-esd-anomaly-detection` (pip), `sesd`. *Air-gap:* pure-Python, trivial.

**#24 — HBOS (Histogram-Based Outlier Score).** Per-feature histograms; score = sum of log inverse-densities. Assumes feature independence → **extremely fast**, near-linear, a great cheap multivariate tier; "usually faster than iForest, sometimes less accurate" ([Towards Data Science HBOS vs iForest](https://towardsdatascience.com/hbos-vs-iforest-on-macbook-pro-m1-c258d2b5fe6b/)). *Library:* `pyod.models.hbos`. *Air-gap:* trivial.

**#25 — COPOD & ECOD (copula / empirical-CDF tail probabilities).** **COPOD** (copula-based) was top ROC-AUC on a 30-dataset benchmark (avg 82.47%, best in 12/30) ([COPOD paper, arXiv 2009.09463](https://arxiv.org/pdf/2009.09463)). **ECOD** estimates per-dimension tail probabilities from empirical CDFs — **parameter-free, deterministic, interpretable** (per-feature tail contribution doubles as explanation). Both are O(n·d), no training instability, perfect for air-gapped reproducibility. *Library:* `pyod.models.copod`, `pyod.models.ecod`. *Air-gap:* trivial. *(ECOD's per-dimension scores feed the copilot "why" directly.)*

**#26 — Half-Space Trees (HST / streaming).** The reference **online** anomaly detector: an ensemble of random-projection half-space trees with mass profiles over a sliding window; **constant per-point time & memory**, designed for evolving streams. State-of-the-art for streaming alongside RRCF ([Fast Anomaly Detection for Streaming Data](https://www.researchgate.net/publication/220813353_Fast_Anomaly_Detection_for_Streaming_Data); [River/PySAD survey](https://arxiv.org/pdf/2108.11807)). *Wins:* the always-on real-time tier; intermittent tunnel spikes. *Library:* `river.anomaly.HalfSpaceTrees` (also PySAD). *Air-gap:* pure-Python, trivial — **top pick for the streaming tier.**

**#27 — Robust Random Cut Forest (RRCF).** Amazon's streaming adaptation of isolation forests; anomaly score = **Collusive Displacement (CoDisp)** = increase in model complexity when a point is inserted. Supports incremental insert/forget over a shingled stream ([rrcf docs](https://klabum.github.io/rrcf/streaming.html); [rrcf JOSS paper](https://www.researchgate.net/publication/345469072)). *Wins:* streaming multivariate with shingling; explains via displacement attribution. *Complexity:* O(log n) amortized per update. *Library:* `rrcf` (pip, [kLabUM/rrcf](https://github.com/kLabUM/rrcf)). *Air-gap:* pure-Python, trivial.

**#28 — Mahalanobis distance (multivariate Gaussian).** Distance in covariance-whitened space; flags joint outliers across correlated metrics (latency+jitter+loss together) even when each is individually normal. Use a **robust covariance** (Minimum Covariance Determinant) to resist contamination. *Complexity:* O(d²) per point after O(d³) covariance fit. *Library:* `scipy.spatial.distance.mahalanobis`, `sklearn.covariance.EllipticEnvelope` / `MinCovDet`. *Air-gap:* trivial.

**#29 — KNN / LOF / CBLOF (proximity & density).** **KNN** (distance to k-th neighbor), **LOF** (Local Outlier Factor — local density ratio, great for varying-density clusters), **CBLOF** (cluster-based LOF). PyOD recommends KNN/LOF as robust starters when outliers sit in sparse regions ([PyOD docs](https://pyod.readthedocs.io/)). *Complexity:* O(n²) naïve / O(n log n) with KD-tree; batch tier, not streaming. *Library:* `pyod.models.knn`, `pyod.models.lof`, `pyod.models.cblof`; streaming LOF via `river.anomaly.LocalOutlierFactor`. *Air-gap:* trivial.

---

## 3. MASTER LIST — Family 2: ML unsupervised

**#30 — Isolation Forest (iForest).** Random recursive partitioning; anomalies isolate in few splits → short path length. **The most popular general anomaly detector** for its speed, low memory, and cross-domain robustness ([Deep Isolation Forest, arXiv 2206.06602](https://arxiv.org/pdf/2206.06602)). *Complexity:* O(n log n) train, O(log n) score. *Library:* `sklearn.ensemble.IsolationForest` or `pyod.models.iforest`. *Streaming variant:* IForestASD / Online Isolation Forest ([arXiv 2505.09593](https://arxiv.org/html/2505.09593)). *Air-gap:* trivial. **Backbone batch detector + native SHAP support (§9).**

**#31 — Extended Isolation Forest (EIF).** Uses **random-slope hyperplane** cuts instead of axis-parallel splits, removing the axis-bias artifacts of standard iForest and giving smoother, more consistent scores ([EIF, IEEE TKDE 2021](https://snyk.io/advisor/python/deepod)). *Wins:* correlated network features where axis-parallel cuts misbehave. *Library:* `eif` (pip), or `isotree` (also does extended + SHAP-like explanations). *Air-gap:* trivial.

**#32 — One-Class SVM (and streaming QF-OCSVM).** Learns a boundary enclosing "normal"; points outside are anomalies. Powerful for moderate-dim, smaller data; sensitive to kernel/ν tuning. *Complexity:* O(n²–n³) train. *Library:* `sklearn.svm.OneClassSVM`, `pyod.models.ocsvm`; streaming `river.anomaly.OneClassSVM` / PySAD QF-OCSVM. *Air-gap:* trivial.

**#33 — PCA / Robust-PCA reconstruction.** Project to top-k principal components, reconstruct, flag large **reconstruction residuals** (point doesn't lie in the normal subspace). **Robust-PCA** (low-rank + sparse decomposition) separates a clean low-rank "normal" matrix from a sparse "anomaly" matrix — ideal for telemetry matrices (sensors × time). *Complexity:* O(n·d²) / iterative for RPCA. *Library:* `pyod.models.pca`, `pyod.models.kpca`; Robust-PCA via `r_pca` (pip). *Air-gap:* trivial. *(Loadings of the violated components = explanation.)*

**#34 (clustering) — DBSCAN / HDBSCAN.** Density clustering; points in no cluster = anomalies. **HDBSCAN** handles variable density and needs no ε. *Wins:* flow-record clustering, identifying outlier traffic profiles / new policy regimes. *Library:* `sklearn.cluster.DBSCAN`, `hdbscan` (pip). *Air-gap:* trivial.

**#35 (mixture) — Gaussian Mixture Model (GMM).** Fit a mixture of Gaussians; low log-likelihood ⇒ anomaly. Captures **multimodal** normal behavior (e.g., business-hours vs. off-hours traffic modes — directly relevant to intermittent tunnel behavior). *Library:* `sklearn.mixture.GaussianMixture`, `pyod.models.gmm`, streaming `river.anomaly.GaussianScorer`. *Air-gap:* trivial.

**#36 (spectral) — Spectral / subspace methods.** Spectral embedding of an affinity/correlation matrix; anomalies separate in eigenspace (also underpins spectral graph analysis, §6). *Library:* `sklearn` spectral embedding; for graphs see §6. *Air-gap:* trivial.

> *(Numbering note: #34–#36 above are the clustering/mixture/spectral entries of Family 2; deep methods continue below at #37+ to keep one global counter.)*

---

## 4. MASTER LIST — Family 4: Change-point / regime / concept-drift detection

These directly target **regime change** (controller misconfig → policy drift; sudden BGP regime shift). Most are **streaming O(1)**.

**#37 — CUSUM (Cumulative Sum).** Accumulate signed deviations from target; alarm when the running sum crosses a decision interval. Detects **small persistent mean shifts faster than Shewhart** — excellent early-warning for gradual congestion/loss creep ([CPD survey, arXiv 2003.06222](https://arxiv.org/pdf/2003.06222)). *Complexity:* O(1) streaming. *Library:* `ruptures` (offline `Pelt`/cost) or hand-rolled streaming CUSUM; `changefinder`. *Air-gap:* trivial.

**#38 — Page-Hinkley test.** Sequential CUSUM variant explicitly for **online** abrupt-change detection in a signal's mean; standard streaming change detector. *Complexity:* O(1). *Library:* `river.drift.PageHinkley`. *Air-gap:* trivial.

**#39 — ADWIN (ADaptive WINdowing).** Maintains a variable-length window of recent data; when statistics of the older vs. newer sub-windows differ significantly, it cuts the window and **signals drift** — with rigorous false-positive bounds. Backbone concept-drift detector ([River drift module](https://riverml.xyz/latest/api/overview/)). *Wins:* detecting that a metric's *distribution* changed (policy drift, BGP regime change) without a fixed threshold. *Complexity:* O(log n) amortized. *Library:* `river.drift.ADWIN`. *Air-gap:* trivial.

**#40 — KSWIN (Kolmogorov-Smirnov Windowing).** Two-sample **KS test** between a reference and a sliding window to detect distribution change; non-parametric, sensitive to shape changes not just mean ([River drift](https://riverml.xyz/latest/api/overview/)). *Library:* `river.drift.KSWIN`. *Air-gap:* trivial.

**#41 — DDM / EDDM / HDDM-A / HDDM-W / FHDDM (error/spacing drift detectors).** Family monitoring a model's online error stream (or event-spacing): **DDM** (error-rate warning/drift levels), **EDDM** (distance-between-errors, better for *gradual* drift), **HDDM-A/W** (Hoeffding-bound, abrupt vs. weighted-gradual), **FHDDM** (fast Hoeffding). Pair these with a forecaster to detect when "normal" itself has shifted ([River drift docs](https://riverml.xyz/latest/api/overview/)). *Complexity:* O(1). *Library:* `river.drift.binary.{DDM,EDDM,HDDMA,HDDMW,FHDDM}`. *Air-gap:* trivial.

**#42 — BOCPD (Bayesian Online Change-Point Detection).** Recursively maintains the posterior over **run-length** (time since last change-point) per new sample; emits a probabilistic change alarm — ideal for *small-sample, real-time* step detection and gives a **calibrated probability** for the copilot ([Adams & MacKay; CPD survey arXiv 2003.06222](https://arxiv.org/pdf/2003.06222)). *Complexity:* O(t) per step (O(window) with pruning). *Library:* `bayesian_changepoint_detection` (pip), `bocd`, or Merlion's `BOCPD`. *Air-gap:* pure-Python, trivial.

**#43 — PELT / Binary Segmentation / Window (offline change-point, `ruptures`).** **PELT** = exact, near-linear penalized segmentation finding the optimal number+location of change-points; **BinSeg**/**Window** are faster approximate variants; **KernelCPD** for nonparametric distributional changes ([ruptures kernel CPD docs](https://centre-borelli.github.io/ruptures-docs/examples/kernel-cpd-performance-comparison/); [PELT financial study, ACM 2025](https://dl.acm.org/doi/10.1145/3773365.3773532)). *Wins:* **retrospective** root-cause segmentation of an incident window (when exactly did the regime shift?) feeding the copilot timeline. *Library:* `ruptures` (`Pelt`, `Binseg`, `Window`, `KernelCPD`). *Air-gap:* trivial. **Single best change-point package for offline use.**

---

## 5. MASTER LIST — Family 5: Matrix-profile / discord discovery

**#44 — Matrix Profile (STUMPY: `stump`, `stumpi`, `mpdist`).** For each subsequence, the distance to its nearest neighbor; **maxima = discords (anomalies)**, minima = motifs. **`stumpi` ("STUMP incremental")** updates the profile in O(1)-ish per new point for **streaming** discord discovery — purpose-built for continuously arriving sensor/telemetry data ([STUMPY streaming tutorial](https://stumpy.readthedocs.io/en/latest/Tutorial_Matrix_Profiles_For_Streaming_Data.html); [stumpi anomaly discussion](https://github.com/stumpy-dev/stumpy/discussions/287)). `mpdist` compares whole series (e.g., today's tunnel profile vs. baseline day). *Wins:* **novel-pattern faults** with no prior label — exactly the "precursor we've never seen" case; catches subtle intermittent tunnel-degradation waveforms. *Caveat:* tuned for subsequence discords, **not single-point spikes** (pair with #19/#26 for those). *Complexity:* batch O(n²) (GPU/`scrump` approx available); streaming incremental via `stumpi`. *Library:* `stumpy`. *Air-gap:* pip-installable (Numba JIT, no network), trivial.

---

## 6. MASTER LIST — Family 6: Graph-based / topology-aware methods

The topology is a graph (CE/PE/P nodes, MPLS LSPs, IPSec tunnels, BGP/OSPF adjacencies). Graph methods detect anomalies in **structure** and **correlate** signals spatially — the heart of "continuous topology awareness and dynamic graph-based event correlation" (Objective 4) and "affected scope / blast radius."

**#45 — Graph anomaly detection (PyGOD: DOMINANT, AnomalyDAE, …).** **PyGOD** implements **18** attributed-graph outlier detectors built on **PyTorch Geometric**, following the PyOD API ([PyGOD docs](https://docs.pygod.org/en/latest/); [PyGOD JMLR paper](https://www.jmlr.org/papers/volume25/23-0963/23-0963.pdf)). Key models:
- **DOMINANT** — GCN encoder + dual decoders reconstruct **structure** and **node attributes**; high reconstruction error on either ⇒ node is anomalous, and the **split tells you whether it's a topology anomaly or an attribute/telemetry anomaly** ([DOMINANT detector docs](https://docs.pygod.org/en/latest/generated/pygod.detector.DOMINANT.html)).
- **AnomalyDAE** — dual autoencoder (structure + attribute) with attention ([ICASSP'20]).
- Others: **GAAN** (GAN), **CONAD**/**CoLA**/**CARD** (self-supervised contrastive), **GUIDE**, **OCGNN** (one-class GNN), **DONE/AdONE**, **GADNR**, plus matrix-factorization classics **Radar/ANOMALOUS/ONE** and clustering **SCAN**.
*Model the live topology as an attributed graph (nodes=devices/interfaces with telemetry features, edges=links/adjacencies). DOMINANT flags a router whose telemetry no longer fits its neighborhood — a precursor signature.* *Library:* `pygod`. *Air-gap:* PyG + Torch wheels vendored offline; trivial once wheels are cached.

**#46 — GNNs for multivariate anomaly/forecasting (MTAD-GAT, GDN).** Treat each metric/sensor as a graph node and **learn inter-metric dependencies**:
- **MTAD-GAT** — two GAT branches (feature-graph + temporal-graph) over a *fully connected* learned graph; combines forecast + reconstruction error ([MTAD-GAT, arXiv 2009.02040; PyTorch impl ML4ITS](https://github.com/ML4ITS/mtad-gat-pytorch)).
- **GDN (Graph Deviation Network)** — learns a **sparse** sensor-dependency graph via node embeddings + attention, forecasts each node from neighbors, scores deviation ([Edge-conditional GNN survey, arXiv 2401.13872]).
*Wins:* learns that "PE-3 latency normally tracks PE-2 jitter"; when that learned relationship **breaks**, it flags a correlated anomaly invisible to univariate detectors — strong for tunnel/underlay and route-flap cascades. *Library:* `mtad-gat-pytorch`, GDN reference repos; or build on `torch-geometric`/`dgl`. *Air-gap:* offline wheels; trivial.

**#47 — Spectral graph analysis + centrality/community shift (blast-radius & path-asymmetry).** Track graph spectra (Laplacian eigenvalues) and **node centralities over time**; sudden shifts in **betweenness centrality** or community membership reveal structural stress and **identify critical bridge nodes whose failure fragments the network** ([betweenness & failure-impact, PuppyGraph](https://www.puppygraph.com/blog/betweenness-centrality); [community-detection survey, arXiv 1708.00977]). **Edge-betweenness** (Girvan-Newman) finds inter-community bridge links — the high-blast-radius links ([Girvan-Newman, Wikipedia](https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm)). Use to compute **affected-scope / blast radius** (which sites are downstream of a flapping link) and **path-asymmetry** (compare forward vs. reverse path betweenness/length). *Library:* `networkx` (betweenness, eigenvector, Louvain/Girvan-Newman), `python-igraph` (fast, C core), `cdlib` (community detection). *Air-gap:* trivial.

**#48 — Graph-based event correlation for root-cause.** Map raw alarms/syslog events onto topology; **correlate co-occurring anomalies along edges/paths** to collapse an alarm storm into one root-cause hypothesis ("PE-3 link flap → 5 downstream tunnel alarms"). Implemented as: align per-node detector scores (from #19–#46), propagate along the graph, and pick the **upstream-most / highest-centrality** anomalous node as the suspected cause. Combine with #47 blast-radius for "affected scope." *Library:* `networkx`/`igraph` traversal + custom correlation; optionally `PyRCA` (Salesforce) for metric-graph RCA ([PyRCA, arXiv 2306.11417](https://arxiv.org/pdf/2306.11417)). *Air-gap:* trivial.

**#49 — Causal discovery & Granger causality (the "why").** Move from correlation to **direction**: which signal's change *caused* the others.
- **PC / GES** (constraint-/score-based DAG discovery) via **`causal-learn`** ([causal-learn]; [RCA-via-causal survey, arXiv 2408.13729](https://arxiv.org/pdf/2408.13729)).
- **Granger causality** (does series X's past improve prediction of Y?) via `statsmodels.tsa.stattools.grangercausalitytests`; **Neural Granger** for nonlinear RCA ([Neural Granger RCA, AAAI'24, arXiv 2402.01140](https://arxiv.org/abs/2402.01140)).
*Wins:* answers Q2/Q3 — root-cause ranking that feeds the copilot's "probable root cause." *Caveat:* causal discovery is sample-hungry and assumption-laden; use as a **ranking hint**, cross-checked by graph correlation (#48) and SHAP (§9), not as ground truth. *Library:* `causal-learn`, `statsmodels`, optional `PyRCA`/`dowhy`. *Air-gap:* `causal-learn`/`statsmodels` pure-Python, trivial.

---

## 7. MASTER LIST — Family 3: Deep anomaly detection

Higher cost (need GPU or batched CPU + training data), but **best on multivariate correlated telemetry** and subtle precursors. Train offline on benign telemetry; run inference inside the air-gap. Primary library: **DeepOD** (`pip install deepod`) and **PyOD** for the simpler nets ([DeepOD](https://github.com/xuhongzuo/DeepOD); [PyOD](https://github.com/yzhao062/pyod)).

**#50 — Autoencoder (AE) reconstruction.** Train to reconstruct normal telemetry; large reconstruction error = anomaly. The workhorse deep detector. *Library:* `pyod.models.auto_encoder` (Torch backend) / DeepOD. *Air-gap:* trivial.

**#51 — Variational Autoencoder (VAE).** Probabilistic AE; **reconstruction probability** as anomaly score, better-calibrated than raw error. *Library:* `pyod.models.vae`. *Air-gap:* trivial.

**#52 — LSTM-Autoencoder.** Sequence-to-sequence LSTM AE captures **temporal** dependencies; reconstruction error on a window flags temporal anomalies — strong for intermittent tunnel waveforms. *Library:* DeepOD (LSTM backbone) / custom Torch / Merlion `LSTMED`. *Air-gap:* trivial.

**#53 — USAD (UnSupervised Anomaly Detection).** Two autoencoders trained **adversarially**; anomaly score blends reconstruction + discriminator loss. Fast inference, strong on SWaT/WADI/SMD industrial benchmarks ([survey PMC11723367](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/)). *Library:* DeepOD (`USAD`). *Air-gap:* trivial.

**#54 — TranAD (Transformer AD).** Transformer encoder–decoder with **focus-score self-conditioning + adversarial training + MAML** for fast, data-efficient multivariate detection; **SOTA on many MTS benchmarks and trains in seconds vs. minutes** ([TranAD, arXiv 2201.07284 / VLDB'22](https://arxiv.org/abs/2201.07284)). *Wins:* the high-accuracy multivariate tier; route-flap-cascade correlated signals. *Library:* DeepOD (`TranAD`) / [official repo]. *Air-gap:* offline wheels; trivial.

**#55 — Anomaly Transformer.** "Anomaly-Attention" measures **association discrepancy** (anomalies have weaker series-wide associations); SOTA reconstruction-based MTS detector ([Anomaly Transformer; survey PMC11723367](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/)). *Library:* DeepOD / official repo. *Air-gap:* trivial.

**#56 — OmniAnomaly.** Stochastic RNN + planar normalizing flows + stochastic variable connection; uses **reconstruction probability** and even derives per-dimension anomaly attribution. Robust on periodic multivariate telemetry ([OmniAnomaly; survey PMC11723367](https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/)). *Library:* official repo / DeepOD-adjacent. *Air-gap:* trivial.

**#57 — Deep SVDD.** Deep one-class: learn an embedding minimizing a hypersphere enclosing normal data; distance to center = score. DeepOD supports TCN/GRU/LSTM/Transformer backbones for **time-series** Deep SVDD ([DeepOD](https://github.com/xuhongzuo/DeepOD)). *Library:* `pyod.models.deep_svdd`, DeepOD. *Air-gap:* trivial.

**#58 — GANomaly.** GAN-based encoder-decoder-encoder; **latent reconstruction error** scores anomalies. *Library:* DeepOD / `pyod` (AnoGAN/MO-GAAL family). *Air-gap:* trivial.

**#59 — Spectral Residual (Microsoft SR / SR-CNN).** From visual-saliency: FFT → spectral residual of log-amplitude → inverse FFT → saliency map; anomaly = relative deviation of saliency from its moving average. **Unsupervised, fast, no training, streaming-friendly**; Microsoft's production detector, optionally with a CNN head ([Time-Series AD Service at Microsoft, KDD'19, arXiv 1906.03821](https://arxiv.org/pdf/1906.03821); [sranodec](https://pypi.org/project/sranodec/)). *Wins:* cheap unsupervised spike/precursor detection on any univariate stream — a great middle tier between stats and deep. *Library:* `sranodec` (pip), `alibi-detect` `SpectralResidual` ([alibi-detect SR](https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_sr_synth.html)), Merlion. *Air-gap:* pure-Python/NumPy, trivial.

**#60 — Forecast-residual anomaly detection (predict-then-flag).** **Bridges to the forecasting sibling (#1–#18):** run any forecaster (Prophet/ARIMA/LSTM/N-BEATS), then flag when the **actual − predicted residual** is large (scored by EVT/DSPOT or robust-z). This is the canonical *precursor* mechanism — it fires while the metric is still *trending* toward breach, maximizing lead time, and the residual magnitude is a natural severity score. *Library:* `darts.ad` (Darts anomaly module wraps any forecaster as an anomaly model), `river.anomaly.PredictiveAnomalyDetection`, Merlion forecaster-based detectors, ADTK `regression`/`level-shift`. *Air-gap:* trivial. **Primary congestion detector and the cleanest tie-in to the forecasting half.**

---

## 8. MASTER LIST — Family 7: Routing-instability-specific signals (BGP/OSPF)

These are **feature engineering + detector** recipes, not new libraries. They turn raw routing telemetry into the inputs that the detectors above consume. Critical because PS-13 explicitly calls out "BGP/OSPF convergence stress, route flapping precursors, path asymmetry."

**#61 — BGP update-churn & MRAI dynamics features.** From BGP UPDATE streams compute, per peer/prefix/AS, sliding-window counts of **announcements, withdrawals, AS-path changes, duplicate announcements, MRAI-interval violations**, and update-rate volatility. Surveys converge on **~32–48 statistical features + ~14–15 graph features** as the standard BGP anomaly feature set ([BGP feature-extraction, ResearchGate 338949133](https://www.researchgate.net/publication/338949133); [BGP ML survey, ResearchGate 359141420](https://www.researchgate.net/publication/359141420)). High/ bursty churn is a **flap precursor**. *Detectors to apply:* ADWIN/KSWIN (#39/#40) on churn rate, S-H-ESD (#23) on seasonal baseline, Isolation Forest/COPOD (#30/#25) on the feature vector, GNN (#46) for multi-peer correlation. *Tools (offline):* parse with **`pybgpstream`/`bgpdump`/`mrtparse`** (consume *locally injected* MRT/UPDATE files — Route Views/RIPE are training-data sources, **not** runtime dependencies; air-gap intact). *Air-gap:* parsers are local; trivial.

**#62 — Route-flap detection / Route-Flap-Damping-style scoring.** Count prefix announce↔withdraw oscillations in a window; assign a **flap penalty that decays exponentially** (RFD-style) and alarm above a suppress threshold. Detects the *cascade* before it propagates ([NinjaOne route-flapping detection](https://www.ninjaone.com/blog/how-to-detect-and-fix-route-flapping/); [route-flap method patent US 11489759]). *Detector:* streaming CUSUM/Page-Hinkley (#37/#38) on the flap-penalty signal. *Air-gap:* custom Python; trivial.

**#63 — AS-path / path-asymmetry metrics.** Track AS-path length changes, path edit-distance, prepending changes, and **forward-vs-reverse path divergence** (path asymmetry). Sudden asymmetry or path-length inflation precedes/accompanies reroute cascades. Combine with graph betweenness (#47) to score blast radius of a path change. *Air-gap:* custom + `networkx`; trivial.

**#64 — OSPF LSA-storm & adjacency-flap detection.** From OSPF syslog/LSDB: count **LSA regeneration rate, SPF-recalculation rate, adjacency up/down (flap) events, RouterDeadInterval expiries**. A spike in *change-LSA* traffic — including external-LSA churn from a flapping link, or a Designated-Router dropping/re-forming adjacencies — is a detectable storm signature that often **eludes SNMP polling** ([Stability issues in OSPF, ResearchGate 221164124](https://www.researchgate.net/publication/221164124); OSPF NSR patents]). Advanced: **Recurrence Quantification Analysis** of OSPF dynamics to spot anomalies ([Identifying OSPF Anomalies via RQA, arXiv 1805.08087](https://arxiv.org/pdf/1805.08087)). *Detector:* EWMA/Shewhart (#20/#21) on LSA rate; Page-Hinkley (#38) on adjacency-flap count; matrix-profile (#44) on SPF-rate waveform. *Air-gap:* syslog parsing local; trivial.

> **Routing-instability strategy:** these features are computed in the collector, then **fanned into the generic detector ensemble** (§2–§7). This reuses the cross-verification machinery instead of building bespoke BGP/OSPF models — fewer moving parts in the air-gap, and the copilot gets uniform anomaly scores + SHAP across all signal types.

---

## 9. Time-to-impact & root-cause / SHAP → feeding the LLM copilot

**#65 — SHAP feature attribution (the engine's "why").** Compute Shapley values to rank **which features drove an anomaly score**. **TreeSHAP** gives exact, fast attributions for Isolation Forest / tree detectors (it attributes the model's path-length output) ([Isolation Forest + SHAP, Microsoft Sentinel](https://techcommunity.microsoft.com/blog/microsoftsentinelblog/anomaly-detection-and-explanation-with-isolation-forest-and-shap-using-microsoft/3750086); [Explaining anomalies with iForest + SHAP, TDS](https://medium.com/data-science/explaining-anomalies-with-isolation-forest-and-shap-0d5d1224b918)). **KernelSHAP** explains arbitrary/black-box detectors (AEs, OCSVM). For graph models, **GraphSAGE+SHAP** has been demonstrated ([Explainable Network AD with GraphSAGE+SHAP, ResearchGate 390012831](https://www.researchgate.net/publication/390012831)). *Per-detector built-in alternatives:* ECOD/COPOD give per-dimension tail contributions natively; PCA/RPCA give violated-component loadings; **DIFFI** is a depth-based feature-importance tailor-made for Isolation Forest ([DIFFI, ScienceDirect S0952197622007205](https://www.sciencedirect.com/science/article/abs/pii/S0952197622007205)). *Library:* `shap` (TreeSHAP for sklearn/PyOD trees, KernelSHAP general), `diffi`. *Air-gap:* `shap` is pure-Python/Numba, trivial.
**Copilot wiring:** for each fired alert, emit a structured JSON: `{predicted_issue, confidence (ensemble agreement, §10), top_features (SHAP-ranked with signed contributions), root_cause_node (graph correlation §6), affected_scope (blast radius §47), time_to_impact (below)}`. This JSON is injected into the LLM context (RAG) so the copilot's natural-language "why" is **grounded in attributions, not hallucinated**.

**#66 — Time-to-impact estimation (trend extrapolation + survival/hazard).** Two complementary estimators:
- **Trend extrapolation / crossing-time:** from the forecaster (#1–#18) and current trajectory, compute when the metric will cross the SLA threshold (with a prediction interval) → lead-time in minutes. Robust slope via Theil-Sen / Mann-Kendall; residual band via DSPOT (#52, EVT).
- **Survival / hazard modeling:** treat "time until breach/failure" as a survival problem; **Cox Proportional-Hazards** (or DeepSurv/LogisticHazard) using current telemetry as covariates yields a **survival curve S(t)** and `predict_expectation` → expected time-to-event, plus hazard ratios that say *which covariate raises risk* ([lifelines survival analysis; turbofan predictive-maintenance example, TDS](https://towardsdatascience.com/survival-analysis-for-predictive-maintenance-of-turbofan-engines-7e2e9b82dc0e-2/)). *Library:* `lifelines` (CoxPH, Kaplan-Meier), `scikit-survival`; `pymannkendall` + `scipy` for slope/crossing. *Air-gap:* pure-Python, trivial. This directly produces PS-13's **"estimated time-to-impact / actionable lead time."**

---

## 10. Ensemble & fusion strategy (Family 8) — making detectors verify each other

**#67 — Score normalization + fusion.** Detectors output incomparable scores; normalize each via **z-score / unification (probabilistic) / min-max** over a rolling reference window, then combine. PyOD ships the combiners: **average, maximum, AOM (Average-of-Maximum), MOA (Maximum-of-Average), median, weighted average** ([PyOD combination; PySAD framework, arXiv 2009.02572](https://arxiv.org/html/2009.02572)). *Library:* `pyod.models.combination`, `pyod.utils.utility.standardizer`. *Air-gap:* trivial.

**Recommended fusion design (cross-verification + gap-filling):**
1. **Tiered, heterogeneous ensemble** (different inductive biases so they cover each other's blind spots):
   - *Tier-1 streaming (always on, O(1)):* Half-Space Trees (#26), Spectral Residual (#59), Page-Hinkley/ADWIN (#38/#39), robust-z/EWMA (#19/#20), forecast-residual (#60). Sub-second, catches obvious/early signals.
   - *Tier-2 batch ML (every N seconds on the feature window):* Isolation Forest (#30), COPOD/ECOD (#25), Mahalanobis (#28), PCA-recon (#33), Matrix-Profile discord (#44).
   - *Tier-3 deep/graph (periodic, correlated multivariate):* TranAD/USAD/LSTM-AE (#53/#54/#52), MTAD-GAT/GDN (#46), PyGOD-DOMINANT (#45).
   - *Change-point overlay:* BOCPD/PELT (#42/#43) mark regime boundaries used to **reset baselines** and explain "what changed when."
2. **Agreement → calibrated confidence.** Normalize scores to [0,1] (or probabilities via Merlion/PyOD calibration), then **confidence = weighted agreement** across tiers (e.g., fraction of detectors firing, or a logistic/stacking meta-model trained on the *labeled injected-fault ground truth* PS-13 provides). High agreement across *independent* method families ⇒ high confidence, low false-positive — this is the robustness the "≥30 methods" requirement buys. *Stacking meta-learner:* `sklearn.linear_model.LogisticRegression` / `xgboost` over detector scores. *Bayesian fusion:* combine as independent likelihood ratios for a posterior anomaly probability ([Bayesian+frequentist AD ensemble patent US 12450549]).
3. **Gap-filling by design.** Map each detector to the scenario(s) it covers (§0/§11); ensure every scenario has ≥3 independent detectors so a miss by one is caught by others.
4. **Probability calibration** so the copilot's "confidence score" is meaningful: Merlion `anomaly score calibration`, PyOD `predict_proba`, or conformal/Gaussian-tail calibration ([Merlion calibration](https://github.com/salesforce/Merlion); [PySAD ProbabilityCalibrator, arXiv 2009.02572](https://arxiv.org/html/2009.02572)).

---

## 11. EVT thresholding plan (replace fixed thresholds)

**#68 — EVT / POT thresholds (SPOT / DSPOT).** Instead of hand-set thresholds (which PS-13 explicitly criticizes as "reactive"), set alarm thresholds from **Extreme Value Theory**: fit a **Generalized Pareto Distribution** to the tail of excesses over an initial high quantile (Pickands-Balkema-de-Haan theorem), then derive the threshold for a **target false-alarm risk q** — *no distributional assumption, one parameter* ([Siffer et al., "Anomaly Detection in Streams with EVT", KDD'17](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p1067.pdf)).
- **POT** — offline/batch: fit on historical benign telemetry to set production thresholds.
- **SPOT** — streaming: maintain the GPD tail online, adapt the threshold continuously.
- **DSPOT** — SPOT **with drift**: applies SPOT to residuals of a local moving average, so thresholds track non-stationary baselines (essential for diurnal traffic and slowly-drifting links).
*Apply EVT to:* each univariate metric's residual stream **and** each detector's anomaly-score stream (so even the *ensemble score* gets a principled, self-calibrating cutoff). *Library:* `pyspot`/`spot` implementations ([cbhua/peak-over-threshold](https://github.com/cbhua/peak-over-threshold); [limjcst/ads-evt](https://github.com/limjcst/ads-evt)); Merlion exposes POT-style thresholding; `scipy.stats.genpareto` for the GPD fit. *Air-gap:* pure-Python, trivial.
*Single tuning knob (`q`, the risk) → controls false-positive rate fleet-wide, dramatically reducing alert fatigue (Objective 4).*

---

## 12. Recommended deployed detector ensemble (the curated subset)

| Layer | Methods (#) | Library | Role | Latency |
|---|---|---|---|---|
| Streaming tier-1 | HST (#26), Spectral Residual (#59), robust-z/EWMA (#19/#20), Page-Hinkley (#38), ADWIN (#39), forecast-residual (#60) | `river`, `sranodec`, NumPy, `darts.ad` | Always-on early signal | O(1)/point |
| Batch ML tier-2 | Isolation Forest (#30), COPOD+ECOD (#25), Mahalanobis-MCD (#28), PCA-recon (#33), Matrix-Profile (#44) | `pyod`, `sklearn`, `stumpy` | Multivariate confirmation + explanations | sub-sec/window |
| Deep/graph tier-3 | TranAD (#54), USAD (#53), LSTM-AE (#52), MTAD-GAT/GDN (#46), PyGOD-DOMINANT (#45) | `deepod`, `mtad-gat-pytorch`, `pygod` | High-accuracy correlated/structural | sec (GPU/batch) |
| Change-point overlay | BOCPD (#42), PELT (#43), DDM/HDDM (#41) | `bayesian_changepoint_detection`, `ruptures`, `river` | Regime boundaries, baseline reset | O(1)–near-linear |
| Routing features | BGP churn/flap/AS-path (#61–63), OSPF LSA/adjacency (#64) | `pybgpstream`/`mrtparse` + custom | Feed routing signals into tiers | streaming |
| Thresholding | DSPOT/SPOT/POT (#68) | `pyspot`/`scipy.genpareto` | Adaptive, risk-controlled cutoffs | O(1)/point |
| Fusion | normalize + weighted-agreement + stacking (#67) | `pyod.models.combination`, `sklearn` | Calibrated confidence | O(1) |
| Explain | TreeSHAP/KernelSHAP (#65), DIFFI; graph correlation (#48); causal/Granger (#49) | `shap`, `networkx`, `causal-learn`/`statsmodels` | Q2 "why" → copilot | sec |
| Time-to-impact | trend-crossing + CoxPH survival (#66) | `pymannkendall`/`scipy`, `lifelines` | Q1 "when" → lead time | sec |

**One-line install set (vendor as a local wheelhouse for the air-gap):**
`river pyod deepod pygod torch torch-geometric stumpy ruptures bayesian_changepoint_detection sranodec alibi-detect shap lifelines causal-learn statsmodels networkx python-igraph hdbscan eif isotree r_pca pymannkendall scipy scikit-learn` — **all BSD/MIT/Apache-2.0, all offline-installable, zero runtime network calls.** Survival / Cox / time-to-event uses **`lifelines` (MIT)** as the permissive default. (Build the wheelhouse once with `pip download`, transfer into the enclave, `pip install --no-index --find-links`.)

> **License note — `scikit-survival` is GPL-3.0**, *not* part of the permissive bundle above. It is **optional only**; if used it must be **isolated/segregated** (separate optional/heavy wheelhouse tier) because GPL-3.0 carries copyleft obligations if the appliance is redistributed. For the regulated air-gapped SBOM, prefer `lifelines` for CoxPH/time-to-impact and keep `scikit-survival` out of the default permissive set.

---

## 13. Per-scenario detector recommendations (detailed)

**A — Progressive congestion (hub-spoke link).** Slow monotonic drift. *Catch with:* forecast-residual (#60) for earliest lead time; EWMA + CUSUM + Page-Hinkley (#20/#37/#38) for drift onset; Mann-Kendall trend; Matrix-Profile (#44) for shape; **DSPOT** (#68) adaptive threshold on residual; **CoxPH survival** (#66) for time-to-saturation. *Why these:* drift detectors and forecast-residuals fire while still below threshold → maximal lead time.

**B — BGP route-flap cascade.** Bursty, multi-node, structural. *Catch with:* route-flap penalty + churn features (#61/#62) → CUSUM/Page-Hinkley; ADWIN/KSWIN (#39/#40) on update-rate distribution; S-H-ESD (#23) vs. seasonal baseline; **GNN MTAD-GAT/GDN** (#46) and **graph event-correlation + edge-betweenness blast-radius** (#47/#48) to collapse the cascade to a root link; **causal/Granger** (#49) to rank the originating AS/peer. *Why:* the fault is inherently *relational* — graph + multivariate methods see the cascade structure univariate detectors miss.

**C — Intermittent MPLS underlay / tunnel degradation.** Intermittent spikes, rekey anomalies, multimodal normal. *Catch with:* Half-Space Trees / RRCF (#26/#27) streaming; Spectral Residual (#59); **GMM** (#35) for multimodal normal; **LSTM-AE / USAD / TranAD** (#52/#53/#54) for temporal waveform; Matrix-Profile `stumpi` (#44) for novel intermittent patterns; COPOD/ECOD (#25) on jitter+loss+rekey vector. *Why:* intermittency + multimodality defeats simple thresholds; reconstruction/discord methods model the *shape* of healthy tunnel behavior.

**D — Controller misconfig → policy drift.** Step regime change in config-derived/flow metrics. *Catch with:* **BOCPD / PELT** (#42/#43) for the step; **DDM/HDDM/ADWIN** (#41/#39) concept-drift; **Isolation Forest / PCA-recon** (#30/#33) on the new feature regime; **PyGOD-DOMINANT** (#45) for a node whose attributes no longer fit its topology neighborhood; **DBSCAN/HDBSCAN** (#34) to spot the new outlier flow-cluster; **causal discovery** (#49) to tie the drift to the changed policy object. *Why:* change-point + structural-graph methods are purpose-built for "a configuration changed and shifted the regime."

---

## 14. Risks, caveats & air-gap notes
- **Matrix Profile (#44)** is for *subsequence discords*, weak on single-point spikes → always pair with #19/#26/#59.
- **Causal discovery (#49)** is assumption-heavy and sample-hungry → use as a *ranked hint*, cross-checked by graph correlation (#48) + SHAP (#65). Never present as certainty to the operator.
- **Deep models (#50–#58)** need representative benign training data and a one-time (offline) GPU/CPU training pass; ship *pre-trained weights* into the enclave, retrain on local telemetry during commissioning. Inference is CPU-feasible for the chosen models (TranAD/USAD are light).
- **PyTorch-Geometric / Torch wheels** are the only "heavy" offline dependency — vendor the exact CUDA/CPU wheels in the wheelhouse; verify no `torch.hub`/dataset auto-download is triggered at runtime (set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, no `download=True`).
- **BGP/OSPF parsers (#61–64)** read *locally injected* MRT/syslog — Route Views/RIPE are *training-data origins only*, fetched outside the enclave; **runtime stays fully air-gapped.**
- **Reproducibility:** prefer deterministic detectors (ECOD/COPOD/HBOS/Mahalanobis/PCA) in the confirmation tier so the same telemetry yields identical scores across audits — important for the "Security & Offline Compliance" evaluation dimension.

---

## 15. Consolidated source list
- River anomaly & drift API — https://riverml.xyz/latest/api/overview/
- PyOD (60+ detectors, combination, SHAP) — https://github.com/yzhao062/pyod , https://pyod.readthedocs.io/
- DeepOD (USAD/TranAD/DeepSVDD, TS backbones) — https://github.com/xuhongzuo/DeepOD
- PyGOD (18 graph detectors, DOMINANT/AnomalyDAE) — https://docs.pygod.org/en/latest/ , JMLR: https://www.jmlr.org/papers/volume25/23-0963/23-0963.pdf
- DOMINANT detector — https://docs.pygod.org/en/latest/generated/pygod.detector.DOMINANT.html
- RRCF (streaming) — https://github.com/kLabUM/rrcf , https://klabum.github.io/rrcf/streaming.html
- Fast Anomaly Detection for Streaming Data (Half-Space Trees) — https://www.researchgate.net/publication/220813353
- Streaming AD survey (River vs PySAD) — https://arxiv.org/pdf/2108.11807
- PySAD framework — https://arxiv.org/html/2009.02572
- Online Isolation Forest — https://arxiv.org/html/2505.09593
- Deep Isolation Forest — https://arxiv.org/pdf/2206.06602
- Extended Isolation Forest (via DeepWiki/Snyk refs) — https://snyk.io/advisor/python/deepod
- COPOD — https://arxiv.org/pdf/2009.09463 ; HBOS vs iForest — https://towardsdatascience.com/hbos-vs-iforest-on-macbook-pro-m1-c258d2b5fe6b/
- STUMPY streaming matrix profile — https://stumpy.readthedocs.io/en/latest/Tutorial_Matrix_Profiles_For_Streaming_Data.html ; discords — https://github.com/stumpy-dev/stumpy/discussions/287
- ruptures (PELT/BinSeg/Window/Kernel) — https://centre-borelli.github.io/ruptures-docs/examples/kernel-cpd-performance-comparison/
- Change-point detection survey/eval — https://arxiv.org/pdf/2003.06222
- BOCPD financial application — https://dl.acm.org/doi/10.1145/3795154.3795291
- Spectral Residual (Microsoft, KDD'19) — https://arxiv.org/pdf/1906.03821 ; sranodec — https://pypi.org/project/sranodec/ ; alibi-detect SR — https://docs.seldon.io/projects/alibi-detect/en/stable/examples/od_sr_synth.html
- Twitter AnomalyDetection / S-H-ESD — https://github.com/twitter/AnomalyDetection ; Python sesd — https://github.com/nachonavarro/seasonal-esd-anomaly-detection
- TranAD (VLDB'22) — https://arxiv.org/abs/2201.07284 ; Deep MTS-AD survey — https://pmc.ncbi.nlm.nih.gov/articles/PMC11723367/
- MTAD-GAT (PyTorch) — https://github.com/ML4ITS/mtad-gat-pytorch ; GDN/edge-GNN survey — https://arxiv.org/pdf/2401.13872
- EVT/POT/SPOT/DSPOT (KDD'17) — https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p1067.pdf ; impls — https://github.com/cbhua/peak-over-threshold , https://github.com/limjcst/ads-evt
- BGP anomaly ML surveys — https://www.researchgate.net/publication/359141420 , https://link.springer.com/chapter/10.1007/978-3-031-62871-9_13 ; feature extraction — https://www.researchgate.net/publication/338949133 ; multi-view GAT — https://arxiv.org/pdf/2112.12793
- Route flapping detection — https://www.ninjaone.com/blog/how-to-detect-and-fix-route-flapping/
- OSPF stability — https://www.researchgate.net/publication/221164124 ; OSPF anomalies via RQA — https://arxiv.org/pdf/1805.08087
- Graph centrality / blast radius — https://www.puppygraph.com/blog/betweenness-centrality ; community detection survey — https://arxiv.org/pdf/1708.00977 ; Girvan-Newman — https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm
- Causal RCA (Neural Granger, AAAI'24) — https://arxiv.org/abs/2402.01140 ; RCA-via-causal survey — https://arxiv.org/pdf/2408.13729 ; PyRCA — https://arxiv.org/pdf/2306.11417
- SHAP + Isolation Forest — https://techcommunity.microsoft.com/blog/microsoftsentinelblog/anomaly-detection-and-explanation-with-isolation-forest-and-shap-using-microsoft/3750086 , https://medium.com/data-science/explaining-anomalies-with-isolation-forest-and-shap-0d5d1224b918 ; DIFFI — https://www.sciencedirect.com/science/article/abs/pii/S0952197622007205 ; GraphSAGE+SHAP — https://www.researchgate.net/publication/390012831
- Survival / time-to-failure (lifelines) — https://towardsdatascience.com/survival-analysis-for-predictive-maintenance-of-turbofan-engines-7e2e9b82dc0e-2/
- Merlion — https://github.com/salesforce/Merlion ; Orion/ADTK/Darts (TS AD tools) — https://github.com/rob-med/awesome-TS-anomaly-detection

---

### Summary for the coordinator
This document catalogues **37 distinct non-forecasting methods (#19–#68 with sub-entries)** across 9 families — statistical/streaming, ML, deep, change-point/drift, matrix-profile, graph/topology, routing-instability, ensemble fusion, and time-to-impact/explanation. **Combined with the forecasting sibling's #1–#18 → 55 methods, exceeding the 30-method robustness target.** Every method lists an exact offline library (all BSD/MIT/Apache-2.0, vendored via local wheelhouse, zero runtime network calls). The recommended deployed ensemble is a tiered subset (streaming → batch-ML → deep/graph) fused by normalized weighted-agreement with **EVT/DSPOT adaptive thresholds**, **SHAP + graph correlation + Granger causality** for the copilot's "why," and **trend-crossing + Cox survival** for "when." Each of the four PS-13 validation scenarios is mapped to ≥3 independent detectors for cross-verification.
