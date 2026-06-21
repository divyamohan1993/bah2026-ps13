"""netra.pipeline — the end-to-end NETRA orchestration layer (integration).

Wires the seven module workstreams (datagen → streaming → forecasting/anomaly →
fusion → correlation/risk/explain → copilot) into one runnable, offline, CPU-only
pipeline and reconciles their interface gaps. This is the integration layer that
proves the whole system works together.

Public API::

    from netra.pipeline import NetraPipeline, PipelineConfig, SituationReport

    pipe = NetraPipeline()
    report = pipe.run_scenario()                 # all four validation scenarios
    headline = report.headline_incident
    answer = report.copilot_for(headline.incident_id)   # grounded Q1/Q2/Q3
    for ev in report.scenario_evals:             # detected? lead time? method?
        print(ev.scenario, ev.detected, ev.lead_time_minutes, ev.top_method)

Or incrementally (streaming)::

    pipe = NetraPipeline()
    for record in source.iter_records():
        pipe.process(record)
    report = pipe.assemble()

Everything degrades gracefully: no GPU, no LLM, no vector DB, no sim and no
internet are required — the synthetic source + template-fallback copilot keep the
chain runnable on the core dependency tier alone.
"""

from __future__ import annotations

from .orchestrator import NetraPipeline, PipelineConfig
from .report import RiskPoint, ScenarioEval, SituationReport

__all__ = [
    "NetraPipeline",
    "PipelineConfig",
    "SituationReport",
    "ScenarioEval",
    "RiskPoint",
]
