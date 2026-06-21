"""Shared ground-truth label writer for the sim fault drivers (Workstream 1).

Every fault driver in ``sim/faults/`` writes a :class:`ScenarioLabel` JSONL line
**before** it starts injecting (and the window is fixed up front), exactly as the
contract requires (``netra/contracts/scenario.py`` docstring; research/01 §4.3).
This guarantees the labeled window bounds the impairment even if a run is
interrupted, and that the synthetic generator and the sim emit the *same*
``ScenarioLabel`` shape.

Importing ``netra.contracts`` here keeps the sim labels byte-compatible with the
synthetic ones. The sim side stays import-light (only pydantic via the
contracts). These drivers are offline lab tooling and are NOT on the runtime
import path of the Python product.

Usage inside a driver::

    from _labels import open_label, ScenarioClock
    clk = ScenarioClock(precursor_s=120, fault_s=180, hold_s=420)
    label = open_label(
        scenario=ScenarioId.A_CONGESTION,
        expected_issue=IssueType.INTERFACE_CONGESTION,
        target_entity_id="hub:pe-hub:PE:eth3",
        clock=clk, severity=Severity.P1, seed=1337,
        params={"rate_schedule_mbit": [100, 50, 20, 8]},
        out_path="labels/run.jsonl",
    )
    # ... inject the fault across [clk.fault_start, clk.fault_end] ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from netra.contracts import IssueType, ScenarioId, ScenarioLabel, Severity


@dataclass
class ScenarioClock:
    """Absolute precursor/fault/clear timestamps for one injection.

    The precursor window opens ``precursor_s`` before the fault, the fault runs
    for ``hold_s`` seconds. All timestamps are absolute UTC (never relative), the
    determinism rule from research/01 §5.2.
    """

    precursor_s: float
    fault_s: float
    hold_s: float
    t0: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.t0 is None:
            self.t0 = datetime.now(timezone.utc)
        elif self.t0.tzinfo is None:
            self.t0 = self.t0.replace(tzinfo=timezone.utc)

    @property
    def precursor_start(self) -> datetime:
        return self.t0 + timedelta(seconds=self.fault_s - self.precursor_s)

    @property
    def fault_start(self) -> datetime:
        return self.t0 + timedelta(seconds=self.fault_s)

    @property
    def fault_end(self) -> datetime:
        return self.fault_start + timedelta(seconds=self.hold_s)


def build_label(
    *,
    scenario: ScenarioId,
    expected_issue: IssueType,
    target_entity_id: str,
    clock: ScenarioClock,
    severity: Severity = Severity.P2,
    seed: int | None = None,
    injection_tool: str = "synthetic",
    params: dict | None = None,
    label_id: str | None = None,
    target_sites: list[str] | None = None,
    target_vpns: list[str] | None = None,
    expected_playbook_id: str | None = None,
) -> ScenarioLabel:
    """Construct a contract-valid :class:`ScenarioLabel` for one injection."""
    return ScenarioLabel(
        label_id=label_id or f"{scenario.value}-{int(clock.t0.timestamp())}",
        scenario=scenario,
        expected_issue=expected_issue,
        target_entity_id=target_entity_id,
        precursor_window_start=clock.precursor_start,
        fault_window_start=clock.fault_start,
        fault_window_end=clock.fault_end,
        expected_lead_time_seconds=clock.precursor_s,
        expected_time_to_impact_seconds=clock.precursor_s,
        severity=severity,
        target_sites=target_sites or [],
        target_services_or_vpns=target_vpns or [],
        expected_playbook_id=expected_playbook_id,
        injection_tool=injection_tool,
        params=params or {},
        seed=seed,
    )


def open_label(out_path: str | Path, **kwargs) -> ScenarioLabel:
    """Build a label and append it to ``out_path`` (JSONL) *before* injection."""
    label = build_label(**kwargs)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(label.model_dump(mode="json"), default=str))
        fh.write("\n")
    return label


__all__ = ["ScenarioClock", "build_label", "open_label"]
