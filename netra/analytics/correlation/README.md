# WS4 — Correlation · RCA · Blast-radius · Risk · Explainability

**Builder 4.** Turns raw per-entity detector/forecaster outputs
(`AnomalyScore` / `FusedRisk` + `TimeToImpact`) into operator-grade
**`Incident`s** with a ranked root cause, deterministic blast radius, calibrated
priority, and the grounded *"why"* (Q2). This is Phase 4 of
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and PART A of
[`research/07-noc-workflow-security.md`](../../../research/07-noc-workflow-security.md).

Owned subtree (this builder edits only these): `netra/analytics/correlation/**`,
`netra/analytics/risk/**`, `netra/analytics/explain/**`, plus
`tests/test_correlation.py`.

> **All types come from `netra.contracts`** — this layer never redefines a model
> and never imports Builder-3 modules; it consumes the contract types
> (`AnomalyScore`, `FusedRisk`, `Forecast`, `TimeToImpact`) directly.

---

## The pipeline (data flow)

```
 AnomalyScore[]                         topology spec (sim/synthetic) ── graph.py ──► TopologyGraph
 FusedRisk[] (+ TimeToImpact)                                                            │ (networkx DiGraph;
        │                                                                                │  upstream→downstream)
        ▼                                                                                ▼
 correlate.normalize_events ─► dedup/compress ─► correlate_events ───────────────► IncidentGroup[]
   (uniform CorrelationEvent)   (alarm storm →    (temporal window ∩ topological      (one per fault domain)
                                 deduped set)      proximity = connected components)
        │
        ▼   per group:
   ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
   │  rca.rank_root_causes       centrality × earliest_onset × Granger(causal-learn opt) → root node │
   │  blast_radius.compute       nx.descendants(root) ∩ NetFlow → BlastRadius (sites/SLAs/flows/hops) │
   │  explain.explain_fused_risk SHAP(opt) / deterministic fallback → ContributingSignal[] (Q2)       │
   │  correlate.assemble_incident  ─────────────────────────────────────────────────► Incident       │
   └──────────────────────────────────────────────────────────────────────────────────────────────┘
        │  Incident[]  (predicted_issue, root_cause_*, blast_radius, contributing_signals, compression)
        ▼
 risk.prioritize_incidents     product-form risk → Platt-calibrate → flap-suppress → sort + P1/P2/P3
   (score.py · calibrate.py · suppress.py)                                   → ORDERED Incident queue
        │
        ▼
   operator triage queue  →  API (/incidents)  ·  copilot (grounds Q1/Q2/Q3 on the Incident)
```

One entry point per stage; `correlate_to_incidents(...)` runs steps 1–4 and
`prioritize_incidents(...)` runs the risk stage.

---

## `correlation/`

| File | Responsibility | Key API |
|---|---|---|
| `graph.py` | `networkx` topology **digital twin** from a topology dict/JSON (nodes = devices/sites with role + criticality; edges = links/tunnels/adjacencies). Directed edge `A→B` = "a failure at A affects B". Undirected media expanded both ways; built-in 5-site demo topology. | `TopologyGraph.from_spec/from_json`, `build_demo_graph()`, `default_criticality()`, `map_to_node()` (interface id → device node) |
| `correlate.py` | **Temporal + topological** correlation: dedup/compress an alarm storm, then group co-occurring events (within `window_seconds` **and** ≤`max_topo_distance` hops) into one incident via connected components. Assembles the `Incident`. | `correlate_to_incidents()`, `correlate_events()`, `assemble_incident()`, `dedup_events()` |
| `rca.py` | Root-cause ranking `centrality × earliest_onset × causal_score`. Centrality = betweenness+eigenvector (`networkx`); causal = pairwise **Granger** (`statsmodels`) with optional **PC** (`causal-learn`, try/except fallback). Builds the grounded hypothesis string. | `rank_root_causes()`, `topology_centrality()`, `granger_causal_scores()`, `build_hypothesis()` |
| `blast_radius.py` | **Deterministic** blast radius: `nx.descendants` + `single_source_shortest_path_length` (hop = propagation proxy) ∩ NetFlow → `BlastRadius` (affected sites/devices/services/SLAs/flow-count). | `compute_blast_radius()`, `blast_urgency_factor()` |

## `risk/`

| File | Responsibility | Key API |
|---|---|---|
| `score.py` | Product-form **Risk = AnomalyConfidence × TimeToImpactUrgency × BlastRadius × AssetCriticality** (product so a zero factor suppresses false urgency). Returns an auditable `RiskFactors` breakdown; `geometric_mean_score` rescales to [0,1]. | `compute_risk_factors()`, `geometric_mean_score()`, `time_to_impact_urgency()` |
| `calibrate.py` | **Platt scaling** (logistic, default) + **isotonic** option mapping raw scores → calibrated probabilities (sklearn; pure-NumPy fallback). Reliability evidence: Brier, ECE, reliability diagram. | `RiskCalibrator(method=...)`, `brier_score()`, `expected_calibration_error()`, `reliability_diagram()` |
| `suppress.py` | **BGP-style flap suppression**: per-entity penalty that increments on each re-fire and **decays exponentially** (half-life); hysteresis suppress/reuse thresholds; smooth `demotion_factor` so a flapping entity is demoted (not dropped). | `FlapSuppressor.observe/penalty_of/is_suppressed/demotion_factor` |
| `prioritize.py` | Orchestrates score → calibrate → suppress → **severity bucket (P1/P2/P3)** → sorted queue. | `prioritize_incidents()` (→ `PrioritizedIncident[]`), `triage_queue()` (→ `Incident[]`), `severity_for()` |

## `explain/`

| File | Responsibility | Key API |
|---|---|---|
| `shap_explain.py` | Feature attributions over a `FusedRisk`. **SHAP** (TreeSHAP/KernelSHAP) when `shap` + a model are present (import-guarded); otherwise a **deterministic fallback** (normalized feature-contribution from `MethodWeight` provenance / permutation importance) — reproducible, no model needed. | `attribute_fused_risk()`, `permutation_importance_fallback()`, `shap_available()` |
| `signals.py` | Renders attributions → `ContributingSignal[]` (name, `shap_value`, `direction` up/down, metric-aware `human_explanation`) — the data behind **Q2**. | `explain_fused_risk()`, `attributions_to_signals()` |

---

## Integration contract (what the copilot & API consume)

**In:** `topology spec` (from WS1), `AnomalyScore[]` + `FusedRisk[]` (+
`TimeToImpact` carried on `FusedRisk`) from WS3, optional `FlowRecord[]`.

**Out:** an **ordered** `list[Incident]` (highest calibrated risk first). Each
`Incident` carries:

- `predicted_issue: IssueType`, `severity: Severity` (P1/P2/P3/info)
- `risk: FusedRisk` (`calibrated_confidence` updated to the queue priority)
- `root_cause_entity: EntityRef` + `root_cause_hypothesis: str` (grounded)
- `correlated_entities: EntityRef[]`, `contributing_signals: ContributingSignal[]` (Q2)
- `blast_radius: BlastRadius` (**deterministic — the copilot must NOT recompute or guess it**)
- `alarm_compression_ratio: float` (raw alarms ÷ incidents)

Minimal end-to-end usage:

```python
from netra.analytics.correlation import build_demo_graph, correlate_to_incidents
from netra.analytics.explain import explain_fused_risk
from netra.analytics.risk import prioritize_incidents, FlapSuppressor, RiskCalibrator

g = build_demo_graph()                       # or TopologyGraph.from_json(<sim topology>)
incidents = correlate_to_incidents(g, anomalies=anoms, fused=fused, flows=flows)
for inc in incidents:                        # enrich Q2 with SHAP/fallback attributions
    inc.contributing_signals = explain_fused_risk(inc.risk, entity=inc.root_cause_entity)

cal = RiskCalibrator("platt").fit(hist_scores, hist_labels)   # optional (else identity)
sup = FlapSuppressor()                                        # optional flap damping
queue = prioritize_incidents(incidents, topology=g, calibrator=cal, suppressor=sup)
# queue[0].incident is the top-priority Incident for the operator card.
```

---

## Dependencies & graceful degradation

Core/light tier only: **networkx, numpy, scipy, scikit-learn, statsmodels,
pydantic** (see [`../requirements-correlation.txt`](../requirements-correlation.txt)).
CPU-only, fully offline. Optional/heavy deps are **import-guarded** and the module
**plus its tests pass without them**:

| Optional dep | Used for | Fallback when absent |
|---|---|---|
| `shap` | TreeSHAP/KernelSHAP attributions | deterministic normalized-contribution / permutation importance |
| `causal-learn` | PC causal-discovery RCA refinement | pairwise **Granger** (statsmodels) |
| `statsmodels` | Granger causal score | RCA on centrality × onset only |
| `scikit-learn` | Platt/isotonic calibration | pure-NumPy Platt (gradient descent) / PAV isotonic |

## Tests

`tests/test_correlation.py` — builds the demo topology + synthetic
`AnomalyScore`/`FusedRisk` events and asserts: correlation groups the right
events into one incident (and keeps unrelated ones apart), RCA picks the
plausible central/earliest root, blast radius counts the right downstream set,
prioritisation orders by calibrated risk, flap suppression demotes a flapping
entity, and explain produces `ContributingSignal`s with sane directions. Runs
with light deps only (and the optional-dep fallbacks are exercised — the suite
passes with `shap`/`causal-learn` absent).

```bash
python -m pytest tests/test_correlation.py -q     # 26 passed
```
