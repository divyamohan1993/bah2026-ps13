"""Ground-truth scenario labels — supervision + evaluation contract.

The synthetic generator (``netra.datagen``) and the sim fault orchestrator
(``sim/``) both emit a :class:`ScenarioLabel` for every injected fault. This is
the supervised target for training the predictive ensemble and the yardstick for
Phase-6 scoring: it bounds the fault window, the precursor window (the
pre-breach interval during which an early warning earns lead-time credit), and
the expected fault class / time-to-impact. Emitting the label *before* injection
starts (and closing it on cleanup) guarantees the window exactly bounds the
impairment even if a run is interrupted.

Lead time is then ``fault_window_start - first_valid_alert_time`` for any alert
that fires inside ``[precursor_window_start, fault_window_start)``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from .common import NetraModel
from .enums import IssueType, ScenarioId, Severity


class ScenarioLabel(NetraModel):
    """Ground-truth label for one injected fault window (one scenario instance).

    Example (mirrors the JSONL the sim orchestrator writes alongside telemetry)::

        ScenarioLabel(
            label_id="f0007",
            scenario=ScenarioId.A_CONGESTION,
            expected_issue=IssueType.INTERFACE_CONGESTION,
            target_entity_id="hub1:pe-hub1:PE:eth1",
            precursor_window_start="2026-06-20T14:01:00Z",
            fault_window_start="2026-06-20T14:03:00Z",
            fault_window_end="2026-06-20T14:13:00Z",
            expected_lead_time_seconds=120,
            severity=Severity.P1,
            seed=1337,
        )
    """

    label_id: str = Field(..., description="Unique id for this fault instance.")
    scenario: ScenarioId = Field(..., description="Which validation scenario.")
    expected_issue: IssueType = Field(
        ..., description="The fault class the engine SHOULD predict."
    )
    target_entity_id: str = Field(
        ...,
        description="Primary injected-fault entity (the ground-truth root cause).",
        examples=["hub1:pe-hub1:PE:eth1", "rr1:rr1:RR:peer-pe-dc1"],
    )

    precursor_window_start: datetime = Field(
        ...,
        description="Start of the pre-breach window; alerts here earn lead-time "
        "credit (positive 'elevated risk' label region).",
    )
    fault_window_start: datetime = Field(
        ..., description="When the actual fault/breach begins."
    )
    fault_window_end: datetime = Field(
        ..., description="When the fault is cleared / window closes."
    )

    expected_lead_time_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Target lead time the engine should achieve before breach.",
    )
    expected_time_to_impact_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Ground-truth time from precursor onset to breach (for TTI "
        "error scoring).",
    )
    severity: Severity = Field(
        default=Severity.P2, description="Ground-truth severity of the fault."
    )

    target_sites: list[str] = Field(
        default_factory=list, description="Ground-truth affected sites (blast radius)."
    )
    target_services_or_vpns: list[str] = Field(
        default_factory=list, description="Ground-truth affected services/VPNs."
    )
    expected_playbook_id: str | None = Field(
        default=None, description="The correct remediation playbook id (Q3 scoring)."
    )

    injection_tool: str | None = Field(
        default=None,
        description="How the fault was injected.",
        examples=["pumba+tc", "exabgp", "napalm_config_push", "synthetic"],
    )
    params: dict[str, object] = Field(
        default_factory=dict,
        description="Injection parameters (e.g. rate schedule, flap cadence).",
    )
    seed: int | None = Field(
        default=None, description="RNG seed for byte-level reproducibility."
    )

    @model_validator(mode="after")
    def _windows_ordered(self) -> ScenarioLabel:
        if not (
            self.precursor_window_start
            <= self.fault_window_start
            <= self.fault_window_end
        ):
            raise ValueError(
                "ScenarioLabel windows must satisfy precursor_start <= "
                "fault_start <= fault_end"
            )
        return self


__all__ = ["ScenarioLabel"]
