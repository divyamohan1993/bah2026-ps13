"""Integration tests for the end-to-end ``netra.pipeline`` chain.

Exercises the wired pipeline (datagen → streaming → forecasting/anomaly → fusion →
correlation/risk/explain → copilot) on the synthetic source, asserting that:

  * a scenario run produces a contract-valid :class:`SituationReport`;
  * the four validation scenarios are *detected* with positive lead time and the
    correct predicted issue class (the headline technical-merit claim);
  * the copilot answers Q1/Q2/Q3 in a contract-valid :class:`CopilotResponse`
    (template-fallback path — no model);
  * the incremental ``process`` / ``assemble`` path matches the batch path;
  * the pipeline-backed API ``LiveProvider`` serves real pipeline output.

CPU-only, offline, core-tier deps only (no GPU / LLM / sim / internet). Kept fast
by using short scenario windows; the full-fidelity validation lives in the demo
(``scripts/demo.py``).
"""

from __future__ import annotations

import warnings

import pytest

from netra.contracts import (
    CopilotResponse,
    Incident,
    IssueType,
    ScenarioId,
)
from netra.pipeline import NetraPipeline, PipelineConfig, SituationReport

# statsmodels/sklearn emit benign convergence warnings on these short series.
warnings.filterwarnings("ignore")

# Shorter-but-complete windows keep the suite fast while still containing each
# scenario's full precursor+fault window.
_DURATION = 720.0
_STEP = 10.0


@pytest.fixture(scope="module")
def congestion_report() -> SituationReport:
    """A single A-scenario run, reused across the cheap assertions."""
    pipe = NetraPipeline(PipelineConfig(step_seconds=_STEP))
    return pipe.run_scenario(
        ScenarioId.A_CONGESTION, duration_s=_DURATION, step_s=_STEP
    )


def test_report_is_well_formed(congestion_report: SituationReport) -> None:
    r = congestion_report
    assert r.window_start is not None and r.window_end is not None
    assert r.stats["records_processed"] > 0
    assert r.stats["streams_tracked"] > 0
    # at least one incident, all contract-valid
    assert r.incidents, "pipeline should raise an incident for congestion"
    for inc in r.incidents:
        Incident.model_validate(inc.model_dump())
        # FusedRisk invariant: any risk>0 carries provenance.
        if inc.risk.risk_score > 0:
            assert inc.risk.contributing_methods
    # the risk timeline has points for the elevated entity (lead-time proof)
    assert r.risk_history, "expected a fused-risk timeline"


def test_headline_incident_and_copilot(congestion_report: SituationReport) -> None:
    r = congestion_report
    hi = r.headline_incident
    assert hi is not None
    assert hi.predicted_issue == IssueType.INTERFACE_CONGESTION
    # root cause maps to the hub PE (the injected congestion target's device node)
    assert hi.root_cause_entity is not None
    assert "pe-hub" in hi.root_cause_entity.entity_id
    # the copilot answered Q1/Q2/Q3 for the headline, contract-valid + grounded
    cp = r.copilot_for(hi.incident_id)
    assert cp is not None
    CopilotResponse.model_validate(cp.model_dump())
    assert cp.predicted_issue == IssueType.INTERFACE_CONGESTION  # Q1
    assert cp.root_cause_hypothesis  # Q2
    assert len(cp.recommended_actions) >= 1  # Q3
    assert len(cp.citations) >= 1  # grounding
    assert 0.0 <= cp.confidence_score <= 1.0
    assert cp.used_fallback is True  # CPU-only template path


def test_scenario_eval_detects_with_lead_time(congestion_report: SituationReport) -> None:
    r = congestion_report
    ev = r.eval_for(ScenarioId.A_CONGESTION)
    assert ev is not None
    assert ev.detected, "congestion precursor should be detected before the fault"
    assert ev.lead_time_seconds is not None and ev.lead_time_seconds > 0
    assert ev.predicted_issue_correct
    # several independent detector families fired (the cross-verification claim)
    assert len(ev.methods_fired) >= 3


@pytest.mark.parametrize(
    "scenario,expected_issue",
    [
        (ScenarioId.B_BGP_FLAP, IssueType.BGP_ROUTE_FLAP),
        (ScenarioId.C_TUNNEL_DEGRADATION, IssueType.TUNNEL_DEGRADATION),
        (ScenarioId.D_POLICY_DRIFT, IssueType.POLICY_DRIFT),
    ],
)
def test_other_scenarios_detected(scenario: ScenarioId, expected_issue: IssueType) -> None:
    """B / C / D each detect with lead time and the right issue class."""
    pipe = NetraPipeline(PipelineConfig(step_seconds=_STEP))
    report = pipe.run_scenario(scenario, duration_s=_DURATION, step_s=_STEP)
    ev = report.eval_for(scenario)
    assert ev is not None
    assert ev.detected, f"{scenario.value} should be detected before the fault"
    assert ev.lead_time_seconds is not None and ev.lead_time_seconds > 0
    assert ev.predicted_issue_correct, (
        f"{scenario.value}: expected {expected_issue.value}"
    )
    # the headline incident should classify to the expected issue too
    hi = report.headline_incident
    assert hi is not None and hi.predicted_issue == expected_issue


def test_incremental_matches_batch() -> None:
    """The streaming ``process`` + ``assemble`` path yields the same detection."""
    from netra.datagen import SyntheticSource

    src = SyntheticSource(seed=1337, duration_s=_DURATION, step_s=_STEP,
                          scenarios=(ScenarioId.A_CONGESTION,))
    pipe = NetraPipeline(PipelineConfig(step_seconds=_STEP))
    pipe._labels = list(src.labels())
    for rec in src.iter_records():
        pipe.process(rec)
    report = pipe.assemble()
    assert report.headline_incident is not None
    assert report.headline_incident.predicted_issue == IssueType.INTERFACE_CONGESTION
    ev = report.eval_for(ScenarioId.A_CONGESTION)
    assert ev is not None and ev.detected


def test_live_provider_serves_pipeline_output() -> None:
    """The API ``LiveProvider`` sources real pipeline output (not the demo stub)."""
    from netra.api.providers import LiveProvider

    lp = LiveProvider.from_scenario("A", duration_s=_DURATION, step_s=_STEP)
    incs = lp.incidents()
    assert incs and incs[0].predicted_issue == IssueType.INTERFACE_CONGESTION
    for inc in incs:
        Incident.model_validate(inc.model_dump())
    sit = lp.situation()
    assert sit["source"] == "live"
    CopilotResponse.model_validate(sit["copilot"])
    topo = lp.topology()
    assert topo["elements"]["nodes"], "topology should have nodes"
    assert topo["root_cause_devices"], "a root-cause device should be flagged"


def test_bare_live_provider_is_stub() -> None:
    """A bare LiveProvider (no pipeline attached) stays a documented stub."""
    from netra.api.providers import LiveProvider

    lp = LiveProvider()
    with pytest.raises(NotImplementedError):
        lp.incidents()
    with pytest.raises(NotImplementedError):
        lp.situation()
