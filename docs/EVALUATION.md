# NETRA — Evaluation Map (rubric → concrete evidence)

This maps each scoring dimension to **runnable, inspectable evidence** in the
repo — files, commands, and measured results. Every command runs **fully offline
on a plain CPU box** (core tier only). The architecture rationale lives in
[ARCHITECTURE.md](../ARCHITECTURE.md); the demo walkthrough in
[DEMO.md](DEMO.md).

| Dimension | Weight |
|---|---|
| [Technical Merit](#technical-merit--35) | 35% |
| [Copilot Effectiveness](#copilot-effectiveness--35) | 35% |
| [Security & Offline Compliance](#security--offline-compliance--20) | 20% |
| [Documentation Quality](#documentation-quality--10) | 10% |

**One-command proof:** `make demo` (4/4 detection + lead time), `make test`
(suite green), `make airgap-verify` (zero egress), `make license` (copyleft-free
bundle).

---

## Technical Merit — 35%
*Prediction accuracy + lead time.*

**Evidence**

- **43-method deployed ensemble (50+ catalogued).** The auditable census is
  `netra/analytics/fusion/registry.py` — `method_count()` returns **43**, asserted
  `>= 30` by `tests/test_analytics.py` (the "30+ methods" robustness floor). The
  census is the source of truth behind this table:

  | Family | Deployed methods |
  |---|---|
  | Forecasting (classical / state-space / GBDT / survival) | 11 |
  | Anomaly (statistical / ML / change-point / matrix-profile) | 16 |
  | Streaming O(1) features (Welford/EWMA/DDSketch/Page-Hinkley/HST/stumpi/…) | 10 |
  | Fusion + EVT thresholding (score-norm + weighted-agreement + DSPOT) | 4 |
  | Graph (event-correlation + Granger) | 2 |
  | **Total** | **43** (≥ 30 target) |

  Robustness comes from **independent families agreeing**, not any single model;
  each of the four scenarios is covered by ≥3 detector families
  ([ARCHITECTURE.md §6](../ARCHITECTURE.md)).

- **EVT/SPOT adaptive thresholds, not hand-set.** `netra/analytics/fusion/evt.py`
  (DSPOT/SPOT/POT over residual + ensemble-score streams) → calibrated, not
  tuned-by-hand, firing.

- **Calibrated lead time (not a point guess).** Forecast trajectory + conformal
  band first-crossing, cross-checked by Cox/RSF survival hazard —
  `netra/analytics/forecasting/` → `TimeToImpact` (eta + CI + confidence). Fusion
  + Platt/isotonic calibration in `netra/analytics/fusion/{fuse,calibrate}.py`.

- **O(1) streaming scoring** recomputes the lead time **every sample**
  (`netra/streaming/`), so the warning is the earliest possible.

- **Measured result — 4/4 detected with lead time** (`make demo`, fully offline,
  CPU-only, deterministic at `--seed 1337`):

  | Scenario (`ScenarioId`) | Predicted issue | Detected? | Lead time | Top method |
  |---|---|---|---|---|
  | `A_congestion`         | `interface_congestion` | ✅ | **2.4 min** | EWMA/Page-Hinkley drift |
  | `B_bgp_flap`           | `bgp_route_flap`       | ✅ | **1.5 min** | route-flap churn + change-point |
  | `C_tunnel_degradation` | `tunnel_degradation`   | ✅ | **1.8 min** | Half-Space-Trees / COPOD |
  | `D_policy_drift`       | `policy_drift`         | ✅ | **0.2 min** | BOCPD/PELT step change |

- **Reproducible ground-truth scoring.** The synthetic source emits labeled
  `ScenarioLabel`s (`netra/datagen/`), so precision/recall/lead-time are measured
  against ground truth, deterministically (seeded). The demo's EVAL block reports
  detected?/lead time/peak risk/methods-fired per scenario.

**Run it**

```bash
make demo                                   # 4/4 detection + lead time
PYTHONPATH=. python scripts/demo.py --json /tmp/out.json   # machine-readable summary
make test                                   # includes the ≥30-method registry assertions
```

---

## Copilot Effectiveness — 35%
*Correct, operator-relevant, grounded, no hallucination.*

**Evidence**

- **Structured, schema-locked `CopilotResponse`.** `netra/contracts/copilot.py`
  defines the answer the copilot MUST return — `predicted_issue` (Q1),
  `time_to_impact_minutes` (Q1), `root_cause_hypothesis` + `contributing_signals`
  (Q2), `recommended_actions` (Q3), `citations` (≥1), and the
  `insufficient_context` **abstain** flag.

- **GBNF grammar from the contract.** `netra/copilot/llm/grammar.py` (+
  `grammar.gbnf`) compiles the schema to a GBNF grammar so the LLM **cannot** emit
  malformed JSON or an out-of-vocabulary `predicted_issue`; a Pydantic validation
  + one constrained retry is the belt-and-suspenders client check
  (`netra/copilot/llm/llama_cpp_client.py`).

- **Grounding (the 35% lever).** `netra/copilot/rag/` does hybrid + reranked +
  contextual retrieval over **internal artifacts only** (`corpus/`), with
  mandatory chunk-ID `citations`; `netra/copilot/grounding/` adds a faithfulness
  gate + closed-set citation check + abstain-on-low-evidence. Confidence is
  **sourced from the analytics engine** (`confidence_score`), never invented by
  the LLM.

- **Deterministic template fallback always answers Q1/Q2/Q3.**
  `netra/copilot/llm/template_client.py` fills the **same** `CopilotResponse`
  schema from the analytics/incident objects with `used_fallback=True` /
  `model_id="template-fallback"`. So the copilot is effective **even with no
  model** — this is the path the offline demo + tests exercise (graceful
  degradation, [ARCHITECTURE.md §5.2](../ARCHITECTURE.md)).

- **Deterministic blast radius, not an LLM guess.** Affected scope / blast radius
  are computed in `netra/analytics/correlation/` (NetworkX BFS ∩ NetFlow) and the
  copilot is forbidden to recompute them.

**Run it**

```bash
make demo                 # prints the Q1/Q2/Q3 card + citations + fallback flag per scenario
make test                 # copilot schema/grounding/fallback tests
make up                   # then POST to http://127.0.0.1:8000/api/copilot/query
```

---

## Security & Offline Compliance — 20%
*Verifiably zero outbound dependency.*

**Evidence — enforcement (defense-in-depth; any one layer blocks egress)**

- **Container isolation (layer 1).** `docker-compose.yml` + `security/networks.md`
  put the mesh on `airgap_net`, an **`internal: true`** bridge with no gateway /
  no NAT — no route to the internet by construction. UI/API/Grafana publish to
  **127.0.0.1 only**.
- **Host firewall (layer 2).** `security/nftables.conf` — `chain forward` and
  `chain output` both `policy drop`, allowing only `lo` + intra-lab RFC1918, and
  **log + counter** every blocked attempt with prefix `EGRESS-DROP`. Because
  container egress traverses **FORWARD**, `security/docker-user.sh` installs the
  same lockdown in the **DOCKER-USER** chain (the chain Docker is contractually
  required to honour, consulted before its own rules).
- **LLM sandbox.** `security/seccomp-llm.json` denies `socket`/`connect`; applied
  via the `x-netra-llm-hardening` anchor (`make up-llm` / the security fragment).
- **Offline-by-design env (layer 6).** `security/.env.example` switches off every
  phone-home path (`HF_HUB_OFFLINE`, `DO_NOT_TRACK`, `PIP_NO_INDEX`, Grafana/Qdrant
  telemetry, …); mirrored into the compose hardening anchors.

**Evidence — verification (the part judges reward)**

- **Active conformance test (one command).**
  `tests/airgap/test_airgap_conformance.py` actively tries TCP to
  1.1.1.1/8.8.8.8 on 53/80/443/22/21/123, external DNS, HTTPS fetch, and UDP/53,
  and **passes only if every attempt fails**. Off the appliance it **xfails**
  (lenient) instead of failing; `NETRA_AIRGAP_STRICT=1` enforces true zero egress.
- **Passive monitor.** `security/falco-egress.yaml` fires **CRITICAL** on any
  outbound `connect` from a NETRA container; `scripts/airgap_verify.sh` also shows
  the nftables `EGRESS-DROP` counters and the `conntrack`/`ss` external-flow check
  (expect 0). Grafana surfaces the blocked-attempt counter + "external ESTABLISHED
  must stay 0" panel.

**Evidence — permissive supply chain**

- **Copyleft-free bundle.** `scripts/license_inventory.py` classifies every
  dependency and **flags copyleft (GPL/AGPL/LGPL/MPL)**. The CORE tier is **22/22
  permissive** — `make license` (= `--no-installed --fail-on-copyleft` over
  `requirements-core.txt`) exits 0. The tool deliberately flags `scikit-survival`
  (GPL-3.0) in the *full* tier; the permissive default uses `lifelines` (MIT)
  instead. This gate is enforced in CI (`.github/workflows/ci.yml` → `license-gate`).
- **Hash-verified offline installer.** `scripts/bundle.sh` (`docker save | gzip`
  + wheel closure + SBOM + per-file `.sha256` + a single `MANIFEST.sha256`) and
  `scripts/install.sh` (verifies the manifest → `docker load` → `pip install
  --no-index` → runs the conformance test as a **first-boot gate**, aborting on any
  mismatch). Neither script makes any outbound request.

**Run it**

```bash
make airgap-verify                          # active conformance + passive evidence
NETRA_AIRGAP_STRICT=1 make airgap-verify    # appliance: enforce zero egress
make license                                # permissive core bundle: CLEAN (exit 0)
make bundle && make install-offline         # build + verified offline install
```

---

## Documentation Quality — 10%

**Evidence**

- **[README.md](../README.md)** — problem → solution → offline promise →
  architecture-at-a-glance → real-command quickstart → the 4/4 results table →
  repo layout.
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** — the locked master architecture: all
  6 phases, the dual-source + graceful-degradation design, per-scenario detector →
  lead-time → playbook map (§6), the air-gap model (§8), deployment topology (§9).
- **[docs/DEMO.md](DEMO.md)** — how to run the demo, what each scenario shows
  (Q1/Q2/Q3 + lead time), expected output, and how to view the UI + Grafana.
- **[docs/BUILD_PLAN.md](BUILD_PLAN.md)** — unambiguous workstream ownership map.
- **[research/](../research)** — the 7 deep-research dossiers backing every
  decision.
- **Self-documenting typed contracts** (`netra/contracts/`, import-light Pydantic
  v2) + **per-module READMEs** (`netra/api/`, `netra/copilot/`, `telemetry/`,
  `grafana/`, `security/`, `scripts/`, `tests/`) — design rationale at every level.
- **CI** (`.github/workflows/ci.yml`) keeps lint / tests / the license gate
  visible and green.
