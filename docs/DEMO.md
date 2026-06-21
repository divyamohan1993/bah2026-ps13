# NETRA — Demo Guide

How to run NETRA end-to-end and what each of the four validation scenarios
shows. **Everything here runs fully offline on a plain CPU box** — no GPU, no
internet, no simulator. The demo path uses the synthetic `TelemetrySource`
(labeled ground truth) and the deterministic **template-fallback copilot**, so it
needs only the light **core** tier (`requirements-core.txt`).

---

## 1. Run the demo

```bash
# one-time: create a venv and install the core tier
make setup
#   == python -m venv .venv && . .venv/bin/activate && pip install -r requirements-core.txt

# run all four scenarios (fast profile)
make demo
#   == PYTHONPATH=. python scripts/demo.py --profile fast
```

Useful variants (all pass straight through to `scripts/demo.py`):

```bash
PYTHONPATH=. python scripts/demo.py --scenario A          # one scenario (A/B/C/D, repeatable)
PYTHONPATH=. python scripts/demo.py --duration 900        # telemetry seconds/scenario (default 900)
PYTHONPATH=. python scripts/demo.py --profile full        # the heavy ensemble (slower, higher fidelity)
PYTHONPATH=. python scripts/demo.py --json /tmp/out.json  # also write a machine-readable summary
PYTHONPATH=. python scripts/demo.py --quiet               # only the summary table
scripts/run_demo.sh --scenario B                          # bash wrapper (sets offline env vars)
```

The demo exits **non-zero if any scenario fails to detect** in its precursor
window, so it doubles as a CI smoke check.

---

## 2. What each scenario shows (Q1 / Q2 / Q3 + lead time)

For every scenario the report prints an operator-style card answering the three
NOC questions, plus an **EVAL** block proving the warning fired *before* the
labeled fault (the lead time):

- **Q1 — what fails next & when:** predicted `IssueType`, root-cause entity,
  time-to-impact, affected scope, with analytics-sourced confidence.
- **Q2 — why / which signals:** the root-cause hypothesis + the ranked
  contributing signals (each with a signed SHAP attribution + a human one-liner).
- **Q3 — what action:** the ordered, approval-gated remediation steps (with
  runbook citations) the copilot recommends.
- **EVAL:** detected? + **lead time** (vs the label target) + peak risk +
  predicted-issue correctness + which detector methods fired.

| Scenario (`ScenarioId`) | Precursor signature | Predicted `IssueType` | What it demonstrates |
|---|---|---|---|
| **A — Progressive congestion** (`A_congestion`) | Monotonic ↑ interface utilisation, queue-drop creep, latency/jitter drift before loss | `interface_congestion` | Drift/forecast detectors fire while still **below** threshold → early QoS/path remediation |
| **B — BGP route-flap cascade** (`B_bgp_flap`) | Bursty BGP UPDATE/withdraw churn, adjacency flaps, best-path A→B→A | `bgp_route_flap` | Flap-penalty + change-point + graph correlation rank the originating peer before mass reachability loss |
| **C — Intermittent tunnel degradation** (`C_tunnel_degradation`) | Intermittent tunnel loss/jitter spikes, LSP path churn, IPSec rekey anomalies | `tunnel_degradation` | Multivariate anomaly (Half-Space-Trees/COPOD) + matrix-profile catch intermittent degradation before SLA loss |
| **D — Controller policy drift** (`D_policy_drift`) | Config-version diff, simultaneous multi-site change with **no** physical fault | `policy_drift` | Step-change detectors (BOCPD/PELT) catch the config event the instant it fans out — the earliest possible warning |

Each scenario is covered by **≥3 independent detector families** so a miss by one
is caught by others. Full per-scenario detector → lead-time → playbook mapping:
[ARCHITECTURE.md §6](../ARCHITECTURE.md).

---

## 3. Results table (expected output)

`make demo` finishes with this verdict (fully offline, CPU-only,
template-fallback copilot):

| Scenario (`ScenarioId`) | Predicted issue | Detected? | Lead time | Top method |
|---|---|---|---|---|
| `A_congestion`          | `interface_congestion` | ✅ YES | **2.4 min** | EWMA/Page-Hinkley drift |
| `B_bgp_flap`            | `bgp_route_flap`       | ✅ YES | **1.5 min** | route-flap churn + change-point |
| `C_tunnel_degradation`  | `tunnel_degradation`   | ✅ YES | **1.8 min** | Half-Space-Trees / COPOD |
| `D_policy_drift`        | `policy_drift`         | ✅ YES | **0.2 min** | BOCPD/PELT step change |

**RESULT: 4/4 scenarios detected with lead time.**

The printed report includes the boxed per-scenario cards and the summary table;
the exact ANSI styling is auto-disabled when output is not a TTY (e.g. piped to a
file or CI log). Lead-time values are deterministic for a fixed `--seed` (default
`1337`).

---

## 4. View the UI + Grafana

Bring up the offline stack (internal-only network; ports bound to loopback only):

```bash
make up                 # NATS + VictoriaMetrics + Grafana + netra-app (builds the image)
#   hardened appliance (adds the Falco egress monitor + LLM seccomp):
#   make up-secure       # docker compose -f docker-compose.yml -f security/compose.security.yml up -d
make ps                 # service status
make logs               # tail logs
```

Then open:

- **Operator console (API + UI):** <http://127.0.0.1:8000>
  - serves `ui/index.html` — the Cytoscape topology (root-cause node +
    blast-radius shaded), the **risk timeline** (risk rising *before* impact),
    the 3-answer incident card, and a copilot chat box.
  - health check: <http://127.0.0.1:8000/api/health>
  - API surface (all under `/api`): `/situation`, `/incidents`,
    `/risk/timeline`, `/topology`, `/copilot/query`, `/stream/risk` (SSE).
  - The API runs the in-process **DemoProvider** by default, so the UI is fully
    populated with no other service. Set `NETRA_API_PROVIDER=live` to wire the
    real engines.
- **Grafana NOC wall:** <http://127.0.0.1:3000> (anonymous read-only viewer)
  - auto-provisions the VictoriaMetrics datasource + both dashboards
    (`netra-telemetry`, `netra-risk-overview`) — including the **air-gap proof
    panels** (nftables `EGRESS-DROP` blocked-attempt counter + "external
    ESTABLISHED connections must stay 0").

> The optional LLM copilot (`make up-llm`) and RAG vector DB
> (`--profile vectordb`) upgrade answer quality; when absent the template
> fallback emits the **same** `CopilotResponse` schema, so the UI/API are
> identical either way.

Tear down with `make down` (named volumes are kept).

---

## 5. Prove the air-gap (the security demo)

```bash
make airgap-verify       # scripts/airgap_verify.sh
#   == active pytest egress conformance (tests/airgap) + passive evidence:
#      nftables EGRESS-DROP counters, conntrack/ss external-flow check, Falco rule armed.
```

On a dev box (not actually air-gapped) the egress-attempt tests **xfail** rather
than fail; on the appliance run `NETRA_AIRGAP_STRICT=1 make airgap-verify` to
enforce true zero egress. See [ARCHITECTURE.md §8](../ARCHITECTURE.md) and
[../security/](../security).
