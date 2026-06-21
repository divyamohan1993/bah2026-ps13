# `netra/datagen/` — Synthetic 4-scenario TelemetrySource (Workstream 1)

**The linchpin of NETRA's CPU-only promise.** This package produces a fully
labeled, time-ordered, deterministic stream of the canonical
[`netra.contracts`](../contracts) telemetry records for the 5-site
SD-WAN-over-MPLS reference topology — **with no sim, no Docker, no GPU and no
internet**. The same record types the live Containerlab lab in [`../../sim/`](../../sim)
would emit are produced here, plus ground-truth `ScenarioLabel`s, so the entire
downstream pipeline (streaming → ensemble → fusion/correlation/risk → copilot)
is runnable, testable and byte-for-byte reproducible.

This is the `SYNTHETIC` backend of the **dual-source telemetry abstraction**
(`ARCHITECTURE.md` §5). Everything downstream depends only on the
`TelemetrySource` interface and the contract record types — never on *how* the
records were produced.

---

## Modules

| File | Role |
|---|---|
| `topology.py` | The deterministic 5-site reference topology (DC, HQ/Hub, Branch-1/2/3, MPLS core, RR, controller) — the single source of truth for *which entities exist*. Shared with the sim labels so both sources align on identity. |
| `scenarios.py` | The fidelity core: diurnal baseline math (per-metric, per-site, with seasonality + noise) and the four precursor injectors (trend / variance / regime-shift / churn) with `ScenarioSpec` metadata. |
| `synthetic.py` | `SyntheticGenerator` + `GeneratorConfig` — walks time, emits the five contract record types with baseline + precursor deltas, and computes ground-truth `ScenarioLabel`s. |
| `source.py` | The `TelemetrySource` ABC and its three backends: `SyntheticSource`, `ReplaySource`, and the documented `ContainerlabSource` (`SIM`) stub. |
| `cli.py` | `netra-datagen` CLI: `generate` (labeled dataset → parquet + JSONL) and `stream` (NDJSON, optionally real-time-paced). |

---

## What it generates

For the whole topology, at every `step_s` tick over `duration_s`:

- **`TelemetryRecord`** — interface utilisation, queue depth, discards, errors,
  latency, jitter, loss (SNMP/gNMI-style); BGP/OSPF session rates
  (update/withdraw/flap-penalty/adjacency-flap/LSA/SPF/path-asymmetry);
  controller config-drift score.
- **`TunnelStat`** — IPSec/GRE overlay loss, jitter, latency, **rekey interval**
  (the rekey-anomaly precursor for scenario C), `oper_up`.
- **`RoutingEvent`** — BGP/OSPF adjacency up/down, announce/withdraw, SPF/LSA
  events emitted when churn is high (scenario B).
- **`SyslogEvent`** — `%BGP-5-ADJCHANGE`, `%IKE-4-REKEY_ANOMALY`,
  `%SYS-5-CONFIG_I` (the config-push that starts scenario D).
- **`FlowRecord`** — NetFlow/IPFIX-style flows per PE (top-talkers, QoS class via
  DSCP) for traffic-matrix / blast-radius input.

### The four scenarios (each with a precursor that *precedes* the fault)

| `ScenarioId` | Predicted `IssueType` | Precursor signature (detectable, with lead time) |
|---|---|---|
| `A_congestion` | `interface_congestion` | Monotonic ↑ utilisation slope + growing queue depth + creeping discards/latency/jitter **before** loss starts (trend + variance). Target: a hub-spoke PE interface. |
| `B_bgp_flap` | `bgp_route_flap` | Rising RFD-style flap penalty + bursty UPDATE/withdraw churn + adjacency flaps + path-asymmetry **before** mass reachability loss (churn + change-point). Target: an RR↔PE peering. |
| `C_tunnel_degradation` | `tunnel_degradation` | Rising tunnel loss/jitter trend + **IPSec rekey-interval shrinking erratically** + intermittent micro-bursts **before** SLA loss. Target: a branch overlay tunnel. |
| `D_policy_drift` | `policy_drift` | A **config-change event** (earliest signal) → a sustained step in config-drift score that fans out to **multiple PEs simultaneously**, with QoS/path-asymmetry divergence but **no** loss/error spike (regime shift, multi-entity). Target: the SD-WAN controller. |

Each scenario emits exactly one `ScenarioLabel` with
`precursor_window_start < fault_window_start ≤ fault_window_end`, so an alert
firing in `[precursor_window_start, fault_window_start)` earns lead-time credit
(the Phase-3/Phase-6 scoring contract).

---

## Determinism

The **entire** output is a pure function of `GeneratorConfig`
(`seed`, `start`, `duration_s`, `step_s`, `scenarios`). Each `(entity, metric,
tick)` stream draws from its own seed-derived RNG (`stream_rng`, an FNV-1a fold
of the human-readable stream key into `numpy.random.default_rng`), so the same
config produces byte-for-byte identical records on any machine — the
reproducibility the evaluation rewards.

---

## Usage

### Python (the interface other modules consume)

```python
from netra.datagen import SyntheticSource

src = SyntheticSource(seed=1337, duration_s=3600, step_s=10)

labels = src.labels()                 # list[ScenarioLabel] — ground truth
for rec in src.iter_records():        # time-ordered union of the 5 record types
    handle(rec)                       # TelemetryRecord | RoutingEvent | ...

# real-time (or accelerated) pacing for the live pipeline:
for rec in src.stream(realtime=True, speed=60):   # 1 sim-minute per real second
    publish(rec)
```

Downstream modules (streaming/analytics) typically do:

```python
from netra.datagen import SyntheticSource
from netra.contracts import TelemetryRecord

for rec in SyntheticSource(seed=1337, duration_s=1800).iter_records():
    if isinstance(rec, TelemetryRecord):
        feature_engine.update(rec)     # → FeatureVector
```

### CLI

```bash
# Generate a labeled dataset (parquet + JSONL + labels + manifest):
python -m netra.datagen.cli generate --out ./data --seed 1337 \
    --duration 3600 --step 10

# Only scenarios A and C, no flow records:
python -m netra.datagen.cli generate --out ./dataAC \
    --scenario a --scenario c --no-flows

# Stream NDJSON to stdout (e.g. pipe into the bus publisher):
python -m netra.datagen.cli stream --duration 600 --step 5 | my_publisher

# Real-time paced (1 min/s) for a live demo:
python -m netra.datagen.cli stream --realtime --speed 60

# Just count records/labels (fast sanity check):
python -m netra.datagen.cli stream --count-only --duration 3600
```

(If the package is installed with a console-script entry point the integrator
may also expose it as `netra-datagen generate ...`; `python -m netra.datagen.cli`
always works.)

### Dataset on disk (what `generate` writes)

```
data/
├── telemetry.parquet   # all records, flat columnar (skipped if pyarrow absent)
├── telemetry.jsonl     # all records, NDJSON, with a `_type` discriminator
├── labels.jsonl        # ground-truth ScenarioLabel objects
└── manifest.json       # exact config + record counts (reproducibility/audit)
```

Round-trip a captured run for deterministic regression tests:

```python
import json
from pathlib import Path
from netra.datagen import ReplaySource

rows = [json.loads(l) for l in Path("data/telemetry.jsonl").read_text().splitlines()]
labels = [json.loads(l) for l in Path("data/labels.jsonl").read_text().splitlines()]
replay = ReplaySource.from_records(rows, labels)      # re-emits in time order
```

---

## Dependencies

CPU-light, **core tier only** — no net-new dependencies (see
`requirements-datagen.txt`):

- **Required:** `pydantic` (contracts), `numpy` (RNG + baseline/precursor math).
- **Optional (import-guarded):** `pandas` + `pyarrow` for the parquet export.
  If absent, `generate` still writes `telemetry.jsonl` + `labels.jsonl` +
  `manifest.json` (sufficient for replay and all downstream loaders); only the
  parquet file is skipped, with a warning.

## Notes for the integrator

- **The dataset/stream interface other modules consume:** the
  `TelemetrySource` ABC (`iter_records()` → time-ordered union of the five
  contract record types; `labels()` → `list[ScenarioLabel]`). WS2's
  `netra/streaming/sources.py` adapter is expected to read a `TelemetrySource`
  directly (no NATS needed) for the CPU-only path.
- **Invoking the generator:** construct `SyntheticSource(seed=…, duration_s=…,
  step_s=…)` or `SyntheticSource(config=GeneratorConfig(…))`. The default config
  (`seed=1337`, `start=2026-06-20T08:00Z`, `duration_s=3600`, `step_s=10`,
  all four scenarios) is a sensible demo dataset.
- **Ground truth for Phase 3/6:** `labels()` provides the `ScenarioLabel`s used
  to train the fusion stacker / survival models and to score lead time / TTI.
- The `ContainerlabSource` (`SIM`) raises a clear, actionable error in the
  air-gapped CPU-only container and documents the live NATS/VictoriaMetrics wiring
  a full deployment would use.
