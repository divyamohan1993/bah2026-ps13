# `netra/` — The NETRA Python product

The importable product package. Sub-packages are owned by distinct build
workstreams (see [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md)); they all
depend on the shared, import-light contracts in [`contracts/`](contracts).

| Sub-package | Workstream | Role | Produces (contracts) |
|---|---|---|---|
| [`contracts/`](contracts) | architecture (shared) | **Stable Pydantic v2 interface** — import-light, every module depends on it | all contract types |
| `datagen/` | WS1 | Synthetic 4-scenario `TelemetrySource` (labeled) + the `TelemetrySource` ABC | `TelemetryRecord`, …, `ScenarioLabel` |
| `streaming/` | WS2 | O(1) online feature engine (River/stumpy/ddsketch) | `FeatureVector` |
| `analytics/forecasting/` | WS3 | Forecasters M1–M24 + foundation | `Forecast` |
| `analytics/anomaly/` | WS3 | Detectors #19–#60 + EVT/SPOT | `AnomalyScore` |
| `analytics/fusion/` | WS3 | Score fusion + weighted-agreement + calibration | `FusedRisk`, `TimeToImpact` |
| `analytics/correlation/` | WS4 | Graph event-correlation + blast-radius | (feeds `Incident`) |
| `analytics/risk/` | WS4 | Calibrated prioritisation | `Incident` |
| `analytics/explain/` | WS4 | SHAP attributions | `ContributingSignal` |
| `copilot/` | WS5 | llama.cpp client + template fallback + RAG + grounding | `CopilotResponse` |
| `api/` | WS6 | FastAPI surface (analytics + copilot + incidents) | HTTP/JSON |

**Golden rule:** `from netra.contracts import ...`. Never redefine the contracts;
never edit `contracts/` (propose changes to the integrator). Importing
`netra.contracts` pulls in only pydantic — keep it that way.

**Graceful degradation:** every heavy dependency (GPU model, LLM, vector DB,
sim) is import-guarded and feature-flagged; the CPU-only path (synthetic source +
template-fallback copilot) must always run.
