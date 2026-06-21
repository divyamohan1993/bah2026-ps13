"""NETRA — Network Early-warning, Telemetry & Reasoning Assistant.

An air-gapped, offline predictive copilot for secure SD-WAN-over-MPLS NOC
operations (Hackathon Problem Statement 13).

The top-level ``netra`` package is intentionally import-light: importing it (or
``netra.contracts``) must NOT pull in any heavy/optional dependency (no numpy,
pandas, torch, llama.cpp, qdrant, etc.). Only the shared Pydantic data
contracts live close to the root so that *every* build workstream can depend on
a stable interface without dragging in another team's runtime deps.

Sub-packages (owned by distinct build workstreams — see ``docs/BUILD_PLAN.md``):
    netra.contracts   Shared Pydantic v2 models (THIS is the stable interface).
    netra.datagen     Synthetic 4-scenario telemetry generator (TelemetrySource).
    netra.streaming   O(1) online streaming feature engine (FeatureVector).
    netra.analytics   Forecasting + anomaly + fusion + correlation + risk + explain.
    netra.copilot     Offline LLM + RAG + grounding orchestration (CopilotResponse).
    netra.api         FastAPI surface exposing analytics, copilot and incidents.
"""

__version__ = "0.1.0"
__product__ = "NETRA"
__tagline__ = "Network Early-warning, Telemetry & Reasoning Assistant"
