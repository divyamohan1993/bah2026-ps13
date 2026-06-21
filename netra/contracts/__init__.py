"""NETRA shared data contracts (Pydantic v2) — the stable interface.

This package is the single source of truth for every cross-module data
structure in NETRA. EVERY workstream imports from here so interfaces stay
stable while teams build their subtrees in parallel (see ``docs/BUILD_PLAN.md``).

Design rules for this package (do not break them):
  * **Import-light.** Importing ``netra.contracts`` must pull in *only* pydantic
    (plus stdlib). No numpy/pandas/torch/llama/qdrant — so any module, however
    minimal, can depend on the contracts. ``python -c "import netra.contracts"``
    must succeed with just pydantic installed.
  * **Closed vocabularies live in ``enums``.** The copilot's GBNF grammar is
    generated from these models, so enum/field changes are model-grammar
    changes — coordinate via the integrator.
  * **``extra='forbid'``.** Contracts reject unknown fields; typos surface as
    errors, not silent drift between teams.

Convenience: everything is re-exported at the package root, so::

    from netra.contracts import TelemetryRecord, FusedRisk, CopilotResponse
"""

from __future__ import annotations

from .analytics import (
    AnomalyScore,
    ContributingSignal,
    Forecast,
    FusedRisk,
    MethodWeight,
    QuantilePoint,
    TimeToImpact,
)
from .common import EntityRef, NetraModel
from .copilot import (
    AffectedScope,
    CopilotAction,
    CopilotRequest,
    CopilotResponse,
    CopilotSignal,
)
from .enums import (
    ApprovalState,
    DetectorFamily,
    DeviceRole,
    Direction,
    IssueType,
    MetricName,
    ScenarioId,
    Severity,
    SiteType,
    TelemetryKind,
    TelemetrySourceKind,
    Urgency,
)
from .features import FeatureVector
from .incident import BlastRadius, Incident, Playbook, RecommendedAction
from .scenario import ScenarioLabel
from .telemetry import (
    FlowRecord,
    RoutingEvent,
    SyslogEvent,
    TelemetryRecord,
    TunnelStat,
)

__all__ = [
    # base / common
    "NetraModel",
    "EntityRef",
    # enums
    "DeviceRole",
    "SiteType",
    "TelemetryKind",
    "MetricName",
    "IssueType",
    "Severity",
    "Urgency",
    "Direction",
    "DetectorFamily",
    "ApprovalState",
    "TelemetrySourceKind",
    "ScenarioId",
    # telemetry
    "TelemetryRecord",
    "SyslogEvent",
    "RoutingEvent",
    "FlowRecord",
    "TunnelStat",
    # features
    "FeatureVector",
    # analytics
    "QuantilePoint",
    "Forecast",
    "AnomalyScore",
    "TimeToImpact",
    "ContributingSignal",
    "MethodWeight",
    "FusedRisk",
    # incident / workflow
    "BlastRadius",
    "RecommendedAction",
    "Playbook",
    "Incident",
    # copilot
    "CopilotRequest",
    "CopilotSignal",
    "AffectedScope",
    "CopilotAction",
    "CopilotResponse",
    # scenario ground-truth
    "ScenarioLabel",
]
