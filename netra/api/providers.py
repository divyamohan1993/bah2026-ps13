"""Situation providers — the dependency-injection seam for the operator API.

The API never imports another builder's *engine* module directly. Instead it
depends on the :class:`SituationProvider` interface, which has two
implementations:

  * :class:`DemoProvider` — fabricates realistic, **seeded**, fully
    contract-conformant data so the API (and the UI on top of it) runs
    **standalone** with no analytics/copilot/datagen engine present. This is the
    CPU-only / no-internet / no-sim default that makes the console always
    demoable and testable.
  * :class:`LiveProvider` — a thin stub the integrator wires to the real
    analytics / correlation / risk / copilot engines. Every method raises
    :class:`NotImplementedError` with a precise note on *which* engine call
    should populate it, so wiring is a fill-in-the-blanks exercise.

All return values are ``netra.contracts`` types (or lists of them) — the API
serialises these as-is, so the wire schema *is* the contract.

The DemoProvider intentionally reuses :data:`netra.datagen.topology.REFERENCE_TOPOLOGY`
(import-light: only contracts enums + stdlib) so the topology graph, blast radius
and entity ids are consistent with the rest of NETRA — it is the shared
*topology source of truth*, not an analytics engine.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from netra.contracts import (
    AffectedScope,
    BlastRadius,
    ContributingSignal,
    CopilotAction,
    CopilotRequest,
    CopilotResponse,
    CopilotSignal,
    DetectorFamily,
    Direction,
    EntityRef,
    FusedRisk,
    Incident,
    IssueType,
    MethodWeight,
    Playbook,
    RecommendedAction,
    Severity,
    TimeToImpact,
    Urgency,
)
from netra.contracts.enums import ApprovalState, ScenarioId


def _load_reference_topology():
    """Load the shared reference topology *without* importing the heavy datagen pkg.

    ``netra/datagen/topology.py`` is deliberately import-light (only
    ``netra.contracts`` enums + stdlib), but ``netra/datagen/__init__.py`` eagerly
    imports the numpy-backed synthetic generator. To honour the API's
    "runs standalone on light deps (no numpy/sim/analytics)" guarantee we load the
    ``topology`` submodule directly from its file via importlib, bypassing the
    package ``__init__``. We register it under a private module name so its own
    ``from __future__ import annotations`` / dataclass machinery works.

    If, in some future layout, the file is absent or numpy *is* present, we fall
    back to the normal import path.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    # Fast path: if datagen already imported cleanly elsewhere, reuse it.
    mod = sys.modules.get("netra.datagen.topology")
    if mod is not None:  # pragma: no cover - depends on import order
        return mod.REFERENCE_TOPOLOGY, mod.Topology

    topo_path = Path(__file__).resolve().parents[1] / "datagen" / "topology.py"
    if topo_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "netra._api_reference_topology", topo_path
        )
        if spec and spec.loader:  # pragma: no branch
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            # Don't write a .pyc into the datagen package's __pycache__ (that
            # directory belongs to another workstream) — the bytecode cache path
            # is derived from the source file path, so we suppress it for this load.
            _prev = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(module)
            finally:
                sys.dont_write_bytecode = _prev
            return module.REFERENCE_TOPOLOGY, module.Topology

    # Last-resort fallback: normal import (requires datagen deps to be installed).
    from netra.datagen.topology import (  # type: ignore[import-not-found]
        REFERENCE_TOPOLOGY as _RT,
    )
    from netra.datagen.topology import Topology as _T

    return _RT, _T


REFERENCE_TOPOLOGY, Topology = _load_reference_topology()


# --------------------------------------------------------------------------- #
#  Provider interface                                                         #
# --------------------------------------------------------------------------- #
class SituationProvider(ABC):
    """Read model behind every API route.

    One small interface the routes depend on; swapping the implementation
    (Demo vs Live) is the only change needed to move from a self-contained demo
    to a fully wired deployment. Methods return canonical contract objects.
    """

    @abstractmethod
    def incidents(self) -> list[Incident]:
        """Prioritised triage queue (already correlated + risk-ranked)."""

    @abstractmethod
    def situation(self) -> dict:
        """Combined Q1/Q2/Q3 snapshot for the headline incident + fleet rollup."""

    @abstractmethod
    def risk_timeline(self, entity_id: str | None = None) -> dict:
        """Risk-over-time series (the visual proof of lead time)."""

    @abstractmethod
    def topology(self) -> dict:
        """Nodes/edges + per-node risk for the Cytoscape graph."""

    @abstractmethod
    def copilot(self, request: CopilotRequest) -> CopilotResponse:
        """Answer an operator query / auto-trigger as a CopilotResponse."""

    @abstractmethod
    def risk_tick(self) -> dict:
        """A single live risk update (one frame of the SSE stream)."""


# --------------------------------------------------------------------------- #
#  Demo provider — seeded, realistic, contract-conformant                     #
# --------------------------------------------------------------------------- #
# The headline demo incident: progressive hub-spoke congestion (scenario A). It
# is the most intuitive to show a judge (utilisation visibly climbing toward
# saturation with lead time) and exercises every contract field.
_HUB = REFERENCE_TOPOLOGY.device("pe-hub")
_HUB_UPLINK_ENTITY = Topology.interface_entity_id(_HUB, "eth1")  # hub:pe-hub:PE:eth1


def _entity_ref_from_id(topo: Topology, entity_id: str) -> EntityRef:
    """Build a structured :class:`EntityRef` from a colon-delimited id.

    Mirrors the convention in ``netra.contracts.common.EntityRef`` and looks up
    the device for site_type/role so the UI can group/colour without re-parsing.
    """
    parts = entity_id.split(":")
    site = parts[0] if parts else entity_id
    device = parts[1] if len(parts) > 1 else site
    role_str = parts[2] if len(parts) > 2 else None
    sub = parts[3] if len(parts) > 3 else None
    try:
        dev = topo.device(device)
        role = dev.role
        site_type = dev.site_type
    except KeyError:
        # Fall back to whatever the id encodes (keeps the API robust to ids not
        # in the reference topology, e.g. a live engine entity).
        from netra.contracts import DeviceRole, SiteType

        role = DeviceRole(role_str) if role_str else DeviceRole.PE
        site_type = None
        try:
            site_type = SiteType(site) if site in {s.value for s in SiteType} else None
        except Exception:
            site_type = None
    return EntityRef(
        entity_id=entity_id,
        site=site,
        device=device,
        role=role,
        site_type=site_type,
        sub=sub,
    )


class DemoProvider(SituationProvider):
    """Self-contained, seeded provider that fabricates realistic NETRA state.

    Deterministic given ``seed`` so the demo and the tests are reproducible. All
    timestamps are anchored to construction time (``now``) so the risk timeline
    and ETAs look live without wall-clock flakiness inside a single process.
    """

    def __init__(self, seed: int = 1337, now: datetime | None = None) -> None:
        self.seed = seed
        self.topo = REFERENCE_TOPOLOGY
        self._now = (now or datetime.now(UTC)).replace(microsecond=0)
        self._rng = random.Random(seed)
        # A monotonically advancing clock for the streaming endpoint so each tick
        # looks like a fresh sample.
        self._tick = 0
        # Pre-compute the headline incident + a couple of supporting ones so the
        # whole snapshot is internally consistent across endpoints.
        self._incidents = self._build_incidents()

    # ----- public API ------------------------------------------------------ #
    def incidents(self) -> list[Incident]:
        return list(self._incidents)

    def situation(self) -> dict:
        headline = self._incidents[0]
        copilot = self._answer_for_incident(headline, request_id="situation-auto")
        # Fleet rollup: counts by severity for the header strip.
        by_sev: dict[str, int] = {}
        for inc in self._incidents:
            by_sev[inc.severity.value] = by_sev.get(inc.severity.value, 0) + 1
        return {
            "generated_at": self._now.isoformat(),
            "source": "demo",
            "headline_incident": headline.model_dump(mode="json"),
            "copilot": copilot.model_dump(mode="json"),
            # Convenience Q1/Q2/Q3 projection for the 3-answer card (the UI can
            # also derive this from headline_incident/copilot directly).
            "answers": {
                "q1_what_when": {
                    "predicted_issue": copilot.predicted_issue.value,
                    "time_to_impact_minutes": copilot.time_to_impact_minutes,
                    "confidence": copilot.confidence_score,
                    "affected_scope": copilot.affected_scope.model_dump(mode="json"),
                },
                "q2_why": {
                    "root_cause_hypothesis": copilot.root_cause_hypothesis,
                    "contributing_signals": [
                        s.model_dump(mode="json") for s in copilot.contributing_signals
                    ],
                },
                "q3_action": {
                    "recommended_actions": [
                        a.model_dump(mode="json") for a in copilot.recommended_actions
                    ],
                },
            },
            "fleet": {
                "incident_count": len(self._incidents),
                "by_severity": by_sev,
            },
        }

    def risk_timeline(self, entity_id: str | None = None) -> dict:
        target = entity_id or _HUB_UPLINK_ENTITY
        series = self._risk_series(target)
        # Annotate the breach point (where risk crosses the action threshold) so
        # the UI can draw the "predicted before impact" marker.
        threshold = 0.7
        breach_idx = next(
            (i for i, p in enumerate(series) if p["risk"] >= threshold), None
        )
        return {
            "entity_id": target,
            "generated_at": self._now.isoformat(),
            "threshold": threshold,
            "breach_index": breach_idx,
            "points": series,
        }

    def topology(self) -> dict:
        return self._build_topology_graph()

    def copilot(self, request: CopilotRequest) -> CopilotResponse:
        # Resolve the request to an incident: explicit ref, else the headline.
        inc = None
        if request.incident_ref:
            inc = next(
                (i for i in self._incidents if i.incident_id == request.incident_ref),
                None,
            )
        if inc is None and request.entity_refs:
            inc = next(
                (
                    i
                    for i in self._incidents
                    if i.root_cause_entity
                    and i.root_cause_entity.entity_id in request.entity_refs
                ),
                None,
            )
        if inc is None:
            inc = self._maybe_route_by_keyword(request)
        if inc is None:
            inc = self._incidents[0]
        return self._answer_for_incident(
            inc, request_id=request.request_id, query=request.operator_query
        )

    def risk_tick(self) -> dict:
        """Emit one live frame: per-entity risk + the headline ETA counting down."""
        self._tick += 1
        ts = self._now + timedelta(seconds=10 * self._tick)
        headline = self._incidents[0]
        # Risk wobbles upward with small noise; ETA shrinks as time passes.
        base = min(0.97, 0.74 + 0.01 * self._tick)
        jitter = (self._rng.random() - 0.5) * 0.02
        risk = max(0.0, min(1.0, base + jitter))
        tti = headline.risk.time_to_impact
        eta_min = None
        if tti and tti.eta_seconds is not None:
            eta_min = max(0.0, round((tti.eta_seconds - 10 * self._tick) / 60.0, 1))
        # A few entities with their current risk for the live graph recolour.
        entities = self._live_entity_risks(risk)
        return {
            "type": "risk_tick",
            "tick": self._tick,
            "timestamp": ts.isoformat(),
            "headline_entity": _HUB_UPLINK_ENTITY,
            "headline_risk": round(risk, 4),
            "headline_eta_minutes": eta_min,
            "predicted_issue": headline.predicted_issue.value,
            "entities": entities,
        }

    # ----- builders -------------------------------------------------------- #
    def _build_incidents(self) -> list[Incident]:
        """Three correlated incidents covering scenarios A, C and B."""
        incidents = [
            self._incident_congestion(),
            self._incident_tunnel(),
            self._incident_bgp_flap(),
        ]
        # Stable priority order: P1 before P2 before P3, then by risk score.
        sev_rank = {Severity.P1: 0, Severity.P2: 1, Severity.P3: 2, Severity.INFO: 3}
        incidents.sort(key=lambda i: (sev_rank[i.severity], -i.risk.risk_score))
        return incidents

    # --- scenario A: progressive hub-spoke congestion (the headline) ------- #
    def _incident_congestion(self) -> Incident:
        ent = _entity_ref_from_id(self.topo, _HUB_UPLINK_ENTITY)
        origin = self._now
        # Time-to-impact: ~7 minutes to SLA breach (90% utilisation), tight CI.
        tti = TimeToImpact(
            entity=ent,
            metric="if_util_pct",
            origin=origin,
            threshold=90.0,
            threshold_direction=Direction.INCREASES_RISK,
            eta_seconds=420.0,
            eta_lower_seconds=300.0,
            eta_upper_seconds=600.0,
            confidence=0.86,
            method="trajectory_crossing",
        )
        methods = [
            MethodWeight(
                method="lightgbm_global",
                family=DetectorFamily.FORECAST,
                normalized_score=0.82,
                weight=0.30,
            ),
            MethodWeight(
                method="page_hinkley",
                family=DetectorFamily.CHANGE_POINT,
                normalized_score=0.91,
                weight=0.25,
            ),
            MethodWeight(
                method="ewma_zscore",
                family=DetectorFamily.STATISTICAL,
                normalized_score=0.77,
                weight=0.20,
            ),
            MethodWeight(
                method="matrix_profile_stumpi",
                family=DetectorFamily.MATRIX_PROFILE,
                normalized_score=0.69,
                weight=0.15,
            ),
            MethodWeight(
                method="cox_survival",
                family=DetectorFamily.SURVIVAL,
                normalized_score=0.74,
                weight=0.10,
            ),
        ]
        risk = FusedRisk(
            entity=ent,
            timestamp=origin,
            risk_score=0.88,
            calibrated_confidence=0.84,
            predicted_issue=IssueType.INTERFACE_CONGESTION,
            agreement=0.83,
            contributing_methods=methods,
            time_to_impact=tti,
        )
        signals = [
            ContributingSignal(
                signal="if_util_pct:eth1",
                shap_value=0.41,
                direction=Direction.INCREASES_RISK,
                observation="utilisation rising ~4%/min, now 78%",
                human_explanation=(
                    "Hub-spoke uplink utilisation is trending toward saturation."
                ),
                entity=ent,
            ),
            ContributingSignal(
                signal="if_out_discards:eth1",
                shap_value=0.22,
                direction=Direction.INCREASES_RISK,
                observation="egress queue drops creeping: 0 -> 14/min",
                human_explanation="Output queue drops are starting to appear.",
                entity=ent,
            ),
            ContributingSignal(
                signal="latency_ms",
                shap_value=0.13,
                direction=Direction.INCREASES_RISK,
                observation="RTT drifting 18ms -> 27ms over 6 min",
                human_explanation="Latency is drifting up ahead of loss.",
                entity=ent,
            ),
            ContributingSignal(
                signal="flow_bytes:bulk",
                shap_value=0.09,
                direction=Direction.INCREASES_RISK,
                observation="a bulk backup flow is the top talker (38% of link)",
                human_explanation="A bulk transfer is consuming business capacity.",
                entity=ent,
            ),
        ]
        blast = self._congestion_blast_radius()
        playbook = self._playbook_congestion()
        return Incident(
            incident_id="INC-2026-0001",
            created_at=origin,
            window_start=origin - timedelta(minutes=8),
            window_end=origin,
            predicted_issue=IssueType.INTERFACE_CONGESTION,
            severity=Severity.P1,
            risk=risk,
            root_cause_entity=ent,
            root_cause_hypothesis=(
                "Progressive congestion on the HQ hub uplink (pe-hub eth1): a bulk "
                "backup flow is driving utilisation toward saturation while queue "
                "drops and latency begin to climb. Forecast + survival models agree "
                "the link will breach its 90% SLA in ~7 minutes if untreated."
            ),
            correlated_entities=[
                ent,
                _entity_ref_from_id(self.topo, "hub:ce-hub:CE:tunnel-br1"),
                _entity_ref_from_id(self.topo, "br1:ce-br1:CE:tunnel-hub"),
                _entity_ref_from_id(self.topo, "br2:ce-br2:CE:tunnel-hub"),
            ],
            contributing_signals=signals,
            blast_radius=blast,
            recommended_playbook=playbook,
            alarm_compression_ratio=6.0,
            scenario_label=ScenarioId.A_CONGESTION,
        )

    def _congestion_blast_radius(self) -> BlastRadius:
        return BlastRadius(
            affected_sites=["hub", "br1", "br2", "br3"],
            affected_devices=["pe-hub", "ce-hub", "ce-br1", "ce-br2", "ce-br3"],
            affected_services_or_vpns=["CORP"],
            affected_slas=["gold-voice", "silver-business"],
            affected_flow_count=312,
            hop_distances={
                "hub:pe-hub:PE:eth1": 0,
                "hub:ce-hub:CE": 1,
                "br1:ce-br1:CE": 2,
                "br2:ce-br2:CE": 2,
                "br3:ce-br3:CE": 2,
            },
            normalized_size=0.62,
        )

    def _playbook_congestion(self) -> Playbook:
        return Playbook(
            playbook_id="PB-CONGESTION-001",
            title="Mitigate progressive hub-spoke link congestion",
            issue_type=IssueType.INTERFACE_CONGESTION,
            trigger_signature="if_util_pct slope>0 approaching SLA with queue-drop creep",
            source_ref="runbook:congestion#PB-CONGESTION-001",
            actions=[
                RecommendedAction(
                    step=1,
                    description="Collect interface and queue statistics on pe-hub eth1.",
                    command_or_guidance="show interface eth1 | include rate|drops",
                    requires_approval=False,
                    urgency=Urgency.IMMEDIATE,
                    verification="counters retrieved",
                    runbook_ref="runbook:congestion#collect",
                    approval_state=ApprovalState.AUTO_OK,
                    safety_class="read_only",
                ),
                RecommendedAction(
                    step=2,
                    description="Identify top-talker flows via NetFlow.",
                    command_or_guidance="nfdump -R /flows -s srcip/bytes -n 5",
                    requires_approval=False,
                    urgency=Urgency.IMMEDIATE,
                    verification="top talkers listed",
                    runbook_ref="runbook:congestion#toptalkers",
                    approval_state=ApprovalState.AUTO_OK,
                    safety_class="read_only",
                ),
                RecommendedAction(
                    step=3,
                    description=(
                        "Raise QoS priority for the business class and rate-limit the "
                        "bulk backup flow on pe-hub eth1."
                    ),
                    command_or_guidance="napalm merge_config qos/hub-uplink-priority.cfg",
                    requires_approval=True,
                    urgency=Urgency.SOON,
                    rollback="napalm rollback",
                    verification="utilisation drops below 80% and queue drops stop",
                    runbook_ref="runbook:congestion#qos",
                    approval_state=ApprovalState.PROPOSED,
                    safety_class="config_change",
                ),
                RecommendedAction(
                    step=4,
                    description=(
                        "If still saturated, shift business traffic to the alternate "
                        "SD-WAN path (DC transit)."
                    ),
                    command_or_guidance="napalm merge_config sdwan/prefer-dc-transit.cfg",
                    requires_approval=True,
                    urgency=Urgency.SOON,
                    rollback="napalm rollback",
                    verification="loss/jitter recover on spoke tunnels",
                    runbook_ref="runbook:congestion#reroute",
                    approval_state=ApprovalState.PROPOSED,
                    safety_class="service_impacting",
                ),
            ],
        )

    # --- scenario C: intermittent tunnel / MPLS underlay degradation ------- #
    def _incident_tunnel(self) -> Incident:
        ent = _entity_ref_from_id(self.topo, "br3:ce-br3:CE:tunnel-hub")
        origin = self._now
        tti = TimeToImpact(
            entity=ent,
            metric="tunnel_loss_pct",
            origin=origin,
            threshold=2.0,
            threshold_direction=Direction.INCREASES_RISK,
            eta_seconds=540.0,
            eta_lower_seconds=360.0,
            eta_upper_seconds=900.0,
            confidence=0.71,
            method="theil_sen_extrapolation",
        )
        methods = [
            MethodWeight(
                method="half_space_trees",
                family=DetectorFamily.STATISTICAL,
                normalized_score=0.79,
                weight=0.35,
            ),
            MethodWeight(
                method="spectral_residual",
                family=DetectorFamily.STATISTICAL,
                normalized_score=0.68,
                weight=0.25,
            ),
            MethodWeight(
                method="copod",
                family=DetectorFamily.STATISTICAL,
                normalized_score=0.64,
                weight=0.2,
            ),
            MethodWeight(
                method="matrix_profile_stumpi",
                family=DetectorFamily.MATRIX_PROFILE,
                normalized_score=0.6,
                weight=0.2,
            ),
        ]
        risk = FusedRisk(
            entity=ent,
            timestamp=origin,
            risk_score=0.66,
            calibrated_confidence=0.63,
            predicted_issue=IssueType.TUNNEL_DEGRADATION,
            agreement=0.6,
            contributing_methods=methods,
            time_to_impact=tti,
        )
        signals = [
            ContributingSignal(
                signal="tunnel_jitter_ms",
                shap_value=0.34,
                direction=Direction.INCREASES_RISK,
                observation="jitter spikes intermittently to 40ms",
                human_explanation="Tunnel jitter is spiking intermittently.",
                entity=ent,
            ),
            ContributingSignal(
                signal="tunnel_rekey_interval_s",
                shap_value=0.27,
                direction=Direction.INCREASES_RISK,
                observation="IPSec rekey interval is anomalously short (1100s)",
                human_explanation="IPSec rekey timing is anomalous on this tunnel.",
                entity=ent,
            ),
            ContributingSignal(
                signal="tunnel_loss_pct",
                shap_value=0.18,
                direction=Direction.INCREASES_RISK,
                observation="loss creeping 0.2% -> 0.9%",
                human_explanation="Tunnel loss is slowly rising toward the SLA.",
                entity=ent,
            ),
        ]
        blast = BlastRadius(
            affected_sites=["br3", "core"],
            affected_devices=["ce-br3", "p3", "p4"],
            affected_services_or_vpns=["CORP", "OT"],
            affected_slas=["gold-voice"],
            affected_flow_count=64,
            hop_distances={"br3:ce-br3:CE:tunnel-hub": 0, "core:p3:P": 1, "core:p4:P": 2},
            normalized_size=0.34,
        )
        return Incident(
            incident_id="INC-2026-0002",
            created_at=origin,
            window_start=origin - timedelta(minutes=12),
            window_end=origin,
            predicted_issue=IssueType.TUNNEL_DEGRADATION,
            severity=Severity.P2,
            risk=risk,
            root_cause_entity=ent,
            root_cause_hypothesis=(
                "Intermittent degradation of the Branch-3 hub tunnel: jitter spikes "
                "and an anomalous IPSec rekey interval correlate with loss creep on a "
                "core LSP (p3-p4). Likely an unstable underlay segment rather than the "
                "overlay itself."
            ),
            correlated_entities=[
                ent,
                _entity_ref_from_id(self.topo, "core:p3:P"),
                _entity_ref_from_id(self.topo, "core:p4:P"),
            ],
            contributing_signals=signals,
            blast_radius=blast,
            recommended_playbook=self._playbook_tunnel(),
            alarm_compression_ratio=3.0,
            scenario_label=ScenarioId.C_TUNNEL_DEGRADATION,
        )

    def _playbook_tunnel(self) -> Playbook:
        return Playbook(
            playbook_id="PB-TUNNEL-001",
            title="Localize and reroute a degrading MPLS underlay / tunnel",
            issue_type=IssueType.TUNNEL_DEGRADATION,
            trigger_signature="tunnel loss/jitter slope + IPSec rekey anomaly",
            source_ref="runbook:tunnel#PB-TUNNEL-001",
            actions=[
                RecommendedAction(
                    step=1,
                    description="Collect tunnel/LSP/BFD stats and rekey logs.",
                    command_or_guidance="show tunnel br3-hub stats; show mpls lsp",
                    requires_approval=False,
                    urgency=Urgency.IMMEDIATE,
                    runbook_ref="runbook:tunnel#collect",
                    approval_state=ApprovalState.AUTO_OK,
                    safety_class="read_only",
                ),
                RecommendedAction(
                    step=2,
                    description="Reroute the LSP onto a healthy TE path (avoid p3-p4).",
                    command_or_guidance="napalm merge_config te/reroute-br3-lsp.cfg",
                    requires_approval=True,
                    urgency=Urgency.SOON,
                    rollback="napalm rollback",
                    verification="loss/jitter recover on the br3 tunnel",
                    runbook_ref="runbook:tunnel#reroute",
                    approval_state=ApprovalState.PROPOSED,
                    safety_class="config_change",
                ),
            ],
        )

    # --- scenario B: BGP route-flap cascade (a P3 watch item) -------------- #
    def _incident_bgp_flap(self) -> Incident:
        ent = _entity_ref_from_id(self.topo, "dc:rr-dc:RR:peer-pe-hub")
        origin = self._now
        tti = TimeToImpact(
            entity=ent,
            metric="bgp_flap_penalty",
            origin=origin,
            threshold=2000.0,
            threshold_direction=Direction.INCREASES_RISK,
            eta_seconds=None,  # trending but not yet projected to breach
            confidence=0.55,
            method="trajectory_crossing",
        )
        methods = [
            MethodWeight(
                method="page_hinkley",
                family=DetectorFamily.CHANGE_POINT,
                normalized_score=0.58,
                weight=0.4,
            ),
            MethodWeight(
                method="adwin",
                family=DetectorFamily.CHANGE_POINT,
                normalized_score=0.52,
                weight=0.3,
            ),
            MethodWeight(
                method="bgp_churn_recipe",
                family=DetectorFamily.ROUTING,
                normalized_score=0.49,
                weight=0.3,
            ),
        ]
        risk = FusedRisk(
            entity=ent,
            timestamp=origin,
            risk_score=0.41,
            calibrated_confidence=0.44,
            predicted_issue=IssueType.BGP_ROUTE_FLAP,
            agreement=0.4,
            contributing_methods=methods,
            time_to_impact=tti,
        )
        signals = [
            ContributingSignal(
                signal="bgp_flap_penalty",
                shap_value=0.29,
                direction=Direction.INCREASES_RISK,
                observation="flap penalty rising: 6 flaps in 5 min",
                human_explanation="The RR-PE(hub) session is starting to flap.",
                entity=ent,
            ),
            ContributingSignal(
                signal="bgp_update_rate",
                shap_value=0.16,
                direction=Direction.INCREASES_RISK,
                observation="UPDATE churn elevated (best-path A->B->A)",
                human_explanation="Best-path is oscillating for affected prefixes.",
                entity=ent,
            ),
        ]
        blast = BlastRadius(
            affected_sites=["dc", "hub"],
            affected_devices=["rr-dc", "pe-hub", "pe-dc1"],
            affected_services_or_vpns=["CORP", "OT"],
            affected_slas=["silver-business"],
            affected_flow_count=21,
            hop_distances={"dc:rr-dc:RR:peer-pe-hub": 0, "hub:pe-hub:PE": 1},
            normalized_size=0.22,
        )
        return Incident(
            incident_id="INC-2026-0003",
            created_at=origin,
            window_start=origin - timedelta(minutes=5),
            window_end=origin,
            predicted_issue=IssueType.BGP_ROUTE_FLAP,
            severity=Severity.P3,
            risk=risk,
            root_cause_entity=ent,
            root_cause_hypothesis=(
                "Early BGP instability on the RR <-> pe-hub VPNv4 session: flap penalty "
                "and UPDATE churn are rising. Not yet service-affecting; watch for "
                "escalation to a reroute cascade."
            ),
            correlated_entities=[
                ent,
                _entity_ref_from_id(self.topo, "hub:pe-hub:PE"),
            ],
            contributing_signals=signals,
            blast_radius=blast,
            recommended_playbook=self._playbook_bgp(),
            alarm_compression_ratio=2.0,
            scenario_label=ScenarioId.B_BGP_FLAP,
        )

    def _playbook_bgp(self) -> Playbook:
        return Playbook(
            playbook_id="PB-BGP-FLAP-001",
            title="Stabilise a flapping BGP peer",
            issue_type=IssueType.BGP_ROUTE_FLAP,
            trigger_signature="rising flap penalty + UPDATE/withdraw churn",
            source_ref="runbook:bgp#PB-BGP-FLAP-001",
            actions=[
                RecommendedAction(
                    step=1,
                    description="Collect BGP/OSPF neighbor and flap statistics.",
                    command_or_guidance="show bgp vpnv4 unicast summary; show bgp flap-statistics",
                    requires_approval=False,
                    urgency=Urgency.IMMEDIATE,
                    runbook_ref="runbook:bgp#collect",
                    approval_state=ApprovalState.AUTO_OK,
                    safety_class="read_only",
                ),
                RecommendedAction(
                    step=2,
                    description="Enable/adjust BGP flap damping for the affected peer.",
                    command_or_guidance="napalm merge_config bgp/flap-damping-pe-hub.cfg",
                    requires_approval=True,
                    urgency=Urgency.MONITOR,
                    rollback="napalm rollback",
                    verification="convergence stable; penalty decays",
                    runbook_ref="runbook:bgp#damping",
                    approval_state=ApprovalState.PROPOSED,
                    safety_class="config_change",
                ),
            ],
        )

    # ----- copilot answer synthesis (template-fallback shape) -------------- #
    def _answer_for_incident(
        self,
        inc: Incident,
        request_id: str,
        query: str | None = None,
    ) -> CopilotResponse:
        """Derive a contract-valid :class:`CopilotResponse` from an incident.

        This is exactly the *deterministic template fallback* shape the real
        copilot also emits (``used_fallback=True``, confidence sourced from the
        analytics objects, never invented) — so the API/UI render identically
        whether or not an LLM is wired in via :class:`LiveProvider`.
        """
        tti = inc.risk.time_to_impact
        tti_min = (
            round(tti.eta_seconds / 60.0, 1)
            if tti and tti.eta_seconds is not None
            else None
        )
        scope = AffectedScope(
            sites=list(inc.blast_radius.affected_sites),
            devices=list(inc.blast_radius.affected_devices),
            services_or_vpns=list(inc.blast_radius.affected_services_or_vpns),
        )
        signals = [
            CopilotSignal(
                signal=s.signal,
                observation=s.observation or s.human_explanation,
                shap_contribution=s.shap_value,
            )
            for s in inc.contributing_signals
        ]
        actions: list[CopilotAction] = []
        citations: list[str] = []
        if inc.recommended_playbook:
            citations.append(inc.recommended_playbook.source_ref or inc.recommended_playbook.playbook_id)
            for a in inc.recommended_playbook.actions:
                actions.append(
                    CopilotAction(
                        step=a.description,
                        runbook_ref=a.runbook_ref,
                        urgency=a.urgency,
                        requires_approval=a.requires_approval,
                    )
                )
                if a.runbook_ref:
                    citations.append(a.runbook_ref)
        if not actions:  # contract requires >=1 action
            actions.append(
                CopilotAction(
                    step="Collect diagnostics and continue monitoring.",
                    runbook_ref=None,
                    urgency=Urgency.MONITOR,
                    requires_approval=False,
                )
            )
        # The incident window itself is a citable evidence id (closed-set safe).
        citations.append(f"telemetry:{inc.incident_id}:{inc.window_start.isoformat()}")
        citations = list(dict.fromkeys(citations))  # de-dup, preserve order

        # If the operator asked a free-text question, prepend a short grounded
        # lead-in that *answers from the incident* (the facts/confidence still come
        # from the analytics objects — nothing is fabricated). This mirrors how the
        # real template fallback frames an answer to a direct question.
        root_cause = inc.root_cause_hypothesis
        if query:
            issue_label = inc.predicted_issue.value.replace("_", " ")
            q = query.strip()
            if len(q) > 160:  # keep the lead-in tidy and within the field cap (1200)
                q = q[:157] + "…"
            entity_label = (
                inc.root_cause_entity.entity_id
                if inc.root_cause_entity
                else "the affected entity"
            )
            lead_in = (
                f"Re: “{q}” — the most relevant active signal is "
                f"{issue_label} on {entity_label}. "
            )
            # Never exceed the contract's root_cause_hypothesis max_length (1200).
            root_cause = (lead_in + inc.root_cause_hypothesis)[:1200]

        return CopilotResponse(
            request_id=request_id,
            predicted_issue=inc.predicted_issue,
            confidence_score=inc.risk.calibrated_confidence,
            time_to_impact_minutes=tti_min,
            root_cause_hypothesis=root_cause,
            contributing_signals=signals,
            affected_scope=scope,
            recommended_actions=actions,
            citations=citations,
            insufficient_context=False,
            grounding_score=0.93,
            used_fallback=True,
            model_id="template-fallback",
        )

    def _maybe_route_by_keyword(self, request: CopilotRequest) -> Incident | None:
        """Best-effort: pick an incident whose issue matches the query keywords."""
        if not request.operator_query:
            return None
        q = request.operator_query.lower()
        keyword_map = {
            IssueType.BGP_ROUTE_FLAP: ("bgp", "flap", "peer", "route"),
            IssueType.TUNNEL_DEGRADATION: ("tunnel", "ipsec", "rekey", "underlay", "mpls", "lsp"),
            IssueType.INTERFACE_CONGESTION: ("congest", "util", "uplink", "saturat", "qos", "queue"),
        }
        for inc in self._incidents:
            kws = keyword_map.get(inc.predicted_issue, ())
            if any(k in q for k in kws):
                return inc
        return None

    # ----- risk timeline + topology + live ticks --------------------------- #
    def _risk_series(self, entity_id: str) -> list[dict]:
        """A rising risk curve over the last ~20 minutes for ``entity_id``.

        Sigmoid-shaped climb so the "risk rises *before* impact" story is visible:
        flat-ish baseline, then a smooth acceleration crossing the action
        threshold a few minutes before the modelled breach.
        """
        rng = random.Random(hash(entity_id) ^ self.seed)
        points: list[dict] = []
        n = 40  # 40 samples * 30s = 20 minutes of history
        for i in range(n):
            t = self._now - timedelta(seconds=30 * (n - 1 - i))
            x = (i - n * 0.55) / (n * 0.12)
            base = 1.0 / (1.0 + math.exp(-x))  # sigmoid 0..1
            noise = (rng.random() - 0.5) * 0.04
            risk = max(0.0, min(1.0, 0.08 + 0.9 * base + noise))
            # Forecast band (conformal-style) widening with the climb.
            band = 0.05 + 0.12 * base
            points.append(
                {
                    "timestamp": t.isoformat(),
                    "risk": round(risk, 4),
                    "lower": round(max(0.0, risk - band), 4),
                    "upper": round(min(1.0, risk + band), 4),
                }
            )
        return points

    def _build_topology_graph(self) -> dict:
        """Cytoscape-style nodes/edges with per-node risk + root-cause flags."""
        # Per-entity risk: root-cause nodes carry their incident risk; blast-radius
        # members carry a decayed share; everything else a low baseline.
        device_risk: dict[str, float] = {}
        root_devices: set[str] = set()
        blast_devices: set[str] = set()
        for inc in self._incidents:
            if inc.root_cause_entity:
                rd = inc.root_cause_entity.device
                root_devices.add(rd)
                device_risk[rd] = max(device_risk.get(rd, 0.0), inc.risk.risk_score)
            for d in inc.blast_radius.affected_devices:
                blast_devices.add(d)
                share = inc.risk.risk_score * 0.5
                device_risk[d] = max(device_risk.get(d, 0.0), share)

        nodes = []
        for dev in self.topo.devices:
            rid = Topology.device_entity_id(dev)
            risk = device_risk.get(dev.name, round(self._rng.uniform(0.02, 0.12), 3))
            nodes.append(
                {
                    "data": {
                        "id": dev.name,
                        "entity_id": rid,
                        "label": dev.name,
                        "site": dev.site,
                        "site_type": dev.site_type.value,
                        "role": dev.role.value,
                        "risk": round(risk, 3),
                        "is_root_cause": dev.name in root_devices,
                        "in_blast_radius": dev.name in blast_devices,
                    }
                }
            )

        edges = []
        for ln in self.topo.links:
            # Edge risk = max of endpoint risks (so hot paths light up).
            er = max(
                device_risk.get(ln.a_device, 0.0),
                device_risk.get(ln.b_device, 0.0),
            )
            edges.append(
                {
                    "data": {
                        "id": f"{ln.a_device}-{ln.b_device}",
                        "source": ln.a_device,
                        "target": ln.b_device,
                        "kind": ln.kind,
                        "risk": round(er, 3),
                    }
                }
            )
        # Add overlay tunnel edges (hub <-> spokes, hub <-> dc) for the SD-WAN view.
        overlay_pairs = [
            ("ce-hub", "ce-br1"),
            ("ce-hub", "ce-br2"),
            ("ce-hub", "ce-br3"),
            ("ce-hub", "ce-dc"),
        ]
        for a, b in overlay_pairs:
            er = max(device_risk.get(a, 0.0), device_risk.get(b, 0.0))
            edges.append(
                {
                    "data": {
                        "id": f"ovl-{a}-{b}",
                        "source": a,
                        "target": b,
                        "kind": "overlay",
                        "risk": round(er, 3),
                    }
                }
            )

        return {
            "generated_at": self._now.isoformat(),
            "root_cause_devices": sorted(root_devices),
            "blast_radius_devices": sorted(blast_devices),
            "elements": {"nodes": nodes, "edges": edges},
        }

    def _live_entity_risks(self, headline_risk: float) -> list[dict]:
        out = [
            {"entity_id": _HUB_UPLINK_ENTITY, "device": "pe-hub", "risk": round(headline_risk, 4)}
        ]
        for inc in self._incidents[1:]:
            if inc.root_cause_entity:
                wobble = (self._rng.random() - 0.5) * 0.02
                out.append(
                    {
                        "entity_id": inc.root_cause_entity.entity_id,
                        "device": inc.root_cause_entity.device,
                        "risk": round(
                            max(0.0, min(1.0, inc.risk.risk_score + wobble)), 4
                        ),
                    }
                )
        return out


# --------------------------------------------------------------------------- #
#  Live provider — DI stub the integrator wires to the real engines           #
# --------------------------------------------------------------------------- #
class LiveProvider(SituationProvider):
    """The wired-up provider — serves REAL :class:`~netra.pipeline.NetraPipeline`
    output to the API/UI.

    Two construction modes:

      * **Wiring stub (default).** ``LiveProvider()`` with no pipeline/report
        attached raises a documented :class:`NotImplementedError` from every read
        method — the safe "not yet connected" state (so a bare ``live`` provider
        never silently serves empty data). This is what ``make_provider("live")``
        returns unless a scenario is configured.
      * **Pipeline-backed.** :meth:`LiveProvider.from_scenario` (or passing a
        prebuilt ``report=``) runs the full offline pipeline over a replayed
        synthetic scenario once and serves its :class:`~netra.pipeline.SituationReport`
        — ranked incidents, the per-entity FusedRisk timeline, the topology digital
        twin, and the grounded copilot answers — through the same shapes the
        :class:`DemoProvider` returns. ``NETRA_API_PROVIDER=live`` plus
        ``NETRA_LIVE_SCENARIO=<A|B|C|D|ALL>`` selects this mode (see
        :func:`make_provider`).

    The wiring is a thin projection: the pipeline already produced contract
    ``Incident`` / ``CopilotResponse`` objects, so the route layer is unchanged —
    only the data source differs from the demo's seeded fabrication.
    """

    def __init__(
        self,
        *,
        engines: object | None = None,
        report: object | None = None,
        seed: int = 1337,
        now: datetime | None = None,
    ) -> None:
        self.engines = engines
        self._report = report  # a netra.pipeline.SituationReport, or None (stub)
        self.seed = seed
        self._now = (now or datetime.now(UTC)).replace(microsecond=0)
        self._tick = 0
        self._rng = random.Random(seed)
        # cached copilot answers keyed by incident id (lazily extended on demand).
        self._copilot_cache: dict[str, CopilotResponse] = {}

    # -- construction from a pipeline run ----------------------------------- #
    @classmethod
    def from_scenario(
        cls,
        scenario: str | None = None,
        *,
        seed: int = 1337,
        duration_s: float = 1200.0,
        step_s: float = 10.0,
        prefer_models: bool = False,
    ) -> LiveProvider:
        """Run the offline pipeline over a replayed scenario → a wired provider.

        ``scenario`` is one of ``A`` / ``B`` / ``C`` / ``D`` (a single validation
        scenario, the clearest single-incident view) or ``ALL`` / ``None`` (all
        four injected into one run). Heavy imports are local so the module stays
        import-light for the demo/stub paths.
        """
        from netra.contracts import ScenarioId
        from netra.pipeline import NetraPipeline, PipelineConfig

        alias = {
            "A": ScenarioId.A_CONGESTION,
            "B": ScenarioId.B_BGP_FLAP,
            "C": ScenarioId.C_TUNNEL_DEGRADATION,
            "D": ScenarioId.D_POLICY_DRIFT,
        }
        sel: object | None
        key = (scenario or "").strip().upper()
        if key in alias:
            sel = alias[key]
        else:
            sel = None  # ALL / unknown -> the full four-scenario run

        pipe = NetraPipeline(PipelineConfig(step_seconds=step_s), prefer_models=prefer_models)
        report = pipe.run_scenario(sel, seed=seed, duration_s=duration_s, step_s=step_s)
        return cls(report=report, seed=seed)

    # -- guard for the unwired stub state ----------------------------------- #
    def _require_report(self, what: str):
        if self._report is None:
            raise NotImplementedError(
                f"LiveProvider.{what} is a wiring stub — no pipeline is attached. "
                f"Build it with LiveProvider.from_scenario(...) or run the API with "
                f"NETRA_API_PROVIDER=live and NETRA_LIVE_SCENARIO=<A|B|C|D|ALL> "
                f"(see netra/api/README.md 'Wiring LiveProvider'). The default "
                f"DemoProvider (NETRA_API_PROVIDER=demo) needs no engine."
            )
        return self._report

    # -- read model (served from the pipeline's SituationReport) ------------ #
    def incidents(self) -> list[Incident]:
        report = self._require_report("incidents")
        return list(report.incidents)

    def situation(self) -> dict:
        report = self._require_report("situation")
        if not report.incidents:
            return {
                "generated_at": self._now.isoformat(),
                "source": "live",
                "headline_incident": None,
                "copilot": None,
                "answers": {},
                "fleet": {"incident_count": 0, "by_severity": {}},
            }
        headline = report.incidents[0]
        copilot = self._copilot_for(headline, request_id="situation-auto")
        by_sev: dict[str, int] = {}
        for inc in report.incidents:
            by_sev[inc.severity.value] = by_sev.get(inc.severity.value, 0) + 1
        return {
            "generated_at": self._now.isoformat(),
            "source": "live",
            "headline_incident": headline.model_dump(mode="json"),
            "copilot": copilot.model_dump(mode="json"),
            "answers": {
                "q1_what_when": {
                    "predicted_issue": copilot.predicted_issue.value,
                    "time_to_impact_minutes": copilot.time_to_impact_minutes,
                    "confidence": copilot.confidence_score,
                    "affected_scope": copilot.affected_scope.model_dump(mode="json"),
                },
                "q2_why": {
                    "root_cause_hypothesis": copilot.root_cause_hypothesis,
                    "contributing_signals": [
                        s.model_dump(mode="json") for s in copilot.contributing_signals
                    ],
                },
                "q3_action": {
                    "recommended_actions": [
                        a.model_dump(mode="json") for a in copilot.recommended_actions
                    ],
                },
            },
            "fleet": {
                "incident_count": len(report.incidents),
                "by_severity": by_sev,
            },
        }

    def risk_timeline(self, entity_id: str | None = None) -> dict:
        report = self._require_report("risk_timeline")
        # default to the headline incident's root-cause entity.
        target = entity_id
        if target is None and report.incidents and report.incidents[0].root_cause_entity:
            target = report.incidents[0].root_cause_entity.entity_id
        target = target or "fleet"
        points = report.risk_history.get(target, [])
        # if the exact id has no history, fall back to any tracked entity on the
        # same device node (the pipeline keys risk by the fine-grained id).
        if not points:
            for eid, pts in report.risk_history.items():
                if target.split(":")[:2] == eid.split(":")[:2]:
                    target, points = eid, pts
                    break
        threshold = 0.7
        series = [
            {
                "timestamp": p.timestamp.isoformat(),
                "risk": round(p.risk_score, 4),
                "lower": round(max(0.0, p.risk_score - 0.05), 4),
                "upper": round(min(1.0, p.risk_score + 0.05), 4),
            }
            for p in points
        ]
        breach_idx = next(
            (i for i, p in enumerate(series) if p["risk"] >= threshold), None
        )
        return {
            "entity_id": target,
            "generated_at": self._now.isoformat(),
            "threshold": threshold,
            "breach_index": breach_idx,
            "points": series,
        }

    def topology(self) -> dict:
        report = self._require_report("topology")
        # Project the pipeline's correlation digital twin to Cytoscape elements,
        # colouring root-cause + blast-radius devices from the report's incidents.
        from netra.pipeline.topology_adapter import build_pipeline_graph

        graph = build_pipeline_graph()
        device_risk: dict[str, float] = {}
        root_devices: set[str] = set()
        blast_devices: set[str] = set()
        for inc in report.incidents:
            if inc.root_cause_entity:
                rd = inc.root_cause_entity.device
                root_devices.add(rd)
                device_risk[rd] = max(device_risk.get(rd, 0.0), inc.risk.risk_score)
            for d in inc.blast_radius.affected_devices:
                blast_devices.add(d)
                device_risk[d] = max(device_risk.get(d, 0.0), inc.risk.risk_score * 0.5)

        nodes = []
        node_ids = set()
        for nid in graph.nodes():
            ref = graph.entity_ref(nid)
            dev = ref.device
            node_ids.add(dev)
            risk = device_risk.get(dev, round(self._rng.uniform(0.02, 0.1), 3))
            nodes.append(
                {
                    "data": {
                        "id": dev,
                        "entity_id": nid,
                        "label": dev,
                        "site": ref.site,
                        "site_type": ref.site_type.value if ref.site_type else None,
                        "role": ref.role.value,
                        "risk": round(risk, 3),
                        "is_root_cause": dev in root_devices,
                        "in_blast_radius": dev in blast_devices,
                    }
                }
            )
        edges = []
        seen_edges: set[tuple[str, str]] = set()
        for u, v in graph.g.edges():
            du, dv = graph.entity_ref(u).device, graph.entity_ref(v).device
            if du == dv or (du, dv) in seen_edges or (dv, du) in seen_edges:
                continue
            seen_edges.add((du, dv))
            er = max(device_risk.get(du, 0.0), device_risk.get(dv, 0.0))
            edges.append(
                {
                    "data": {
                        "id": f"{du}-{dv}",
                        "source": du,
                        "target": dv,
                        "kind": graph.g.edges[u, v].get("kind", "link"),
                        "risk": round(er, 3),
                    }
                }
            )
        return {
            "generated_at": self._now.isoformat(),
            "root_cause_devices": sorted(root_devices),
            "blast_radius_devices": sorted(blast_devices),
            "elements": {"nodes": nodes, "edges": edges},
        }

    def copilot(self, request: CopilotRequest) -> CopilotResponse:
        report = self._require_report("copilot")
        if not report.incidents:
            raise NotImplementedError("LiveProvider.copilot: pipeline produced no incidents")
        inc = None
        if request.incident_ref:
            inc = next(
                (i for i in report.incidents if i.incident_id == request.incident_ref), None
            )
        if inc is None and request.entity_refs:
            inc = next(
                (
                    i
                    for i in report.incidents
                    if i.root_cause_entity
                    and i.root_cause_entity.entity_id in request.entity_refs
                ),
                None,
            )
        if inc is None:
            inc = report.incidents[0]
        return self._copilot_for(inc, request_id=request.request_id, query=request.operator_query)

    def risk_tick(self) -> dict:
        report = self._require_report("risk_tick")
        self._tick += 1
        ts = self._now + timedelta(seconds=10 * self._tick)
        if not report.incidents:
            return {"type": "risk_tick", "tick": self._tick, "timestamp": ts.isoformat(),
                    "entities": []}
        headline = report.incidents[0]
        tti = headline.risk.time_to_impact
        eta_min = None
        if tti and tti.eta_seconds is not None:
            eta_min = max(0.0, round((tti.eta_seconds - 10 * self._tick) / 60.0, 1))
        ent_id = (
            headline.root_cause_entity.entity_id if headline.root_cause_entity else "n/a"
        )
        entities = []
        for inc in report.incidents[:6]:
            if inc.root_cause_entity:
                wobble = (self._rng.random() - 0.5) * 0.02
                entities.append(
                    {
                        "entity_id": inc.root_cause_entity.entity_id,
                        "device": inc.root_cause_entity.device,
                        "risk": round(max(0.0, min(1.0, inc.risk.risk_score + wobble)), 4),
                    }
                )
        return {
            "type": "risk_tick",
            "tick": self._tick,
            "timestamp": ts.isoformat(),
            "headline_entity": ent_id,
            "headline_risk": round(headline.risk.risk_score, 4),
            "headline_eta_minutes": eta_min,
            "predicted_issue": headline.predicted_issue.value,
            "entities": entities,
        }

    # -- helpers ------------------------------------------------------------ #
    def _copilot_for(
        self, inc: Incident, *, request_id: str, query: str | None = None
    ) -> CopilotResponse:
        """Reuse the pipeline's grounded copilot answer; synthesise if absent.

        The pipeline already answered the copilot for its top incident(s); we serve
        that grounded answer. For an incident the pipeline did not pre-answer (or a
        free-text operator query needing a fresh request_id), we re-derive a
        contract-valid answer from the incident (the same template-fallback shape).
        """
        report = self._report
        pre = report.copilot_answers.get(inc.incident_id) if report is not None else None
        if pre is not None and query is None:
            # echo the pre-computed grounded answer under the caller's request id.
            data = pre.model_dump()
            data["request_id"] = request_id
            return CopilotResponse(**data)
        return self._synth_copilot(inc, request_id=request_id, query=query)

    @staticmethod
    def _synth_copilot(
        inc: Incident, *, request_id: str, query: str | None = None
    ) -> CopilotResponse:
        tti = inc.risk.time_to_impact
        tti_min = (
            round(tti.eta_seconds / 60.0, 1) if tti and tti.eta_seconds is not None else None
        )
        scope = AffectedScope(
            sites=list(inc.blast_radius.affected_sites),
            devices=list(inc.blast_radius.affected_devices),
            services_or_vpns=list(inc.blast_radius.affected_services_or_vpns),
        )
        signals = [
            CopilotSignal(
                signal=s.signal,
                observation=s.observation or s.human_explanation,
                shap_contribution=s.shap_value,
            )
            for s in inc.contributing_signals[:6]
        ]
        actions: list[CopilotAction] = []
        citations: list[str] = []
        if inc.recommended_playbook:
            citations.append(
                inc.recommended_playbook.source_ref or inc.recommended_playbook.playbook_id
            )
            for a in inc.recommended_playbook.actions:
                actions.append(
                    CopilotAction(
                        step=a.description,
                        runbook_ref=a.runbook_ref,
                        urgency=a.urgency,
                        requires_approval=a.requires_approval,
                    )
                )
                if a.runbook_ref:
                    citations.append(a.runbook_ref)
        if not actions:
            actions.append(
                CopilotAction(
                    step=(
                        "Collect diagnostics for the affected entities and correlate "
                        "against the predicted issue before any state-changing action."
                    ),
                    runbook_ref=None,
                    urgency=Urgency.IMMEDIATE,
                    requires_approval=False,
                )
            )
        citations.append(f"telemetry:{inc.incident_id}:{inc.window_start.isoformat()}")
        citations = list(dict.fromkeys(citations))
        root_cause = inc.root_cause_hypothesis
        if query:
            q = query.strip()
            if len(q) > 160:
                q = q[:157] + "…"
            root_cause = (f"Re: “{q}” — " + inc.root_cause_hypothesis)[:1200]
        return CopilotResponse(
            request_id=request_id,
            predicted_issue=inc.predicted_issue,
            confidence_score=inc.risk.calibrated_confidence,
            time_to_impact_minutes=tti_min,
            root_cause_hypothesis=root_cause or "See contributing signals.",
            contributing_signals=signals,
            affected_scope=scope,
            recommended_actions=actions,
            citations=citations,
            insufficient_context=False,
            grounding_score=0.9,
            used_fallback=True,
            model_id="template-fallback",
        )


def make_provider(kind: str | None = None, **kwargs) -> SituationProvider:
    """Factory: build a provider by name.

    ``kind`` defaults to the ``NETRA_API_PROVIDER`` env var, then ``"demo"``.
    Recognised: ``"demo"`` (self-contained, default) and ``"live"`` (pipeline-backed
    when ``NETRA_LIVE_SCENARIO`` is set, else a documented wiring stub).
    """
    import os

    kind = (kind or os.environ.get("NETRA_API_PROVIDER") or "demo").lower()
    if kind == "live":
        # Pipeline-backed only when a scenario is explicitly requested, so a bare
        # ``make_provider("live")`` stays the safe wiring stub (no surprise compute).
        scenario = os.environ.get("NETRA_LIVE_SCENARIO")
        if scenario:
            duration = float(os.environ.get("NETRA_LIVE_DURATION", "1200"))
            return LiveProvider.from_scenario(scenario, duration_s=duration)
        return LiveProvider(**kwargs)
    if kind == "demo":
        seed = int(os.environ.get("NETRA_API_SEED", kwargs.pop("seed", 1337)))
        return DemoProvider(seed=seed, **kwargs)
    raise ValueError(f"unknown provider kind: {kind!r} (expected 'demo' or 'live')")


__all__ = [
    "SituationProvider",
    "DemoProvider",
    "LiveProvider",
    "make_provider",
]
