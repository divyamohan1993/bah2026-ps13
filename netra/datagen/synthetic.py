"""High-fidelity, deterministic synthetic telemetry generator (the linchpin).

``SyntheticGenerator`` produces a fully labeled, time-ordered stream of the
canonical ``netra.contracts`` telemetry records — ``TelemetryRecord``,
``RoutingEvent``, ``SyslogEvent``, ``FlowRecord``, ``TunnelStat`` — for the whole
5-site reference topology (``topology.py``), with realistic diurnal baselines
(``scenarios.py``) and the four validation-scenario precursors injected so that
forecasting/drift detectors get **lead time** before each labeled fault.

Why this matters (ARCHITECTURE.md §1, §5): this generator *is* the CPU-only
promise. With no sim, no Docker, no GPU and no internet it yields the identical
record types the live Containerlab source would, plus ground-truth
``ScenarioLabel``s, so the entire downstream pipeline (streaming -> ensemble ->
fusion/correlation/risk -> copilot) is runnable, testable and reproducible.

Determinism guarantee: the full output is a pure function of
``GeneratorConfig`` (seed, start time, duration, step, enabled scenarios). The
same config produces byte-for-byte identical records on any machine — each
(entity, metric) stream draws from its own seed-derived RNG (``stream_rng``).

Usage::

    from datetime import datetime, timezone
    from netra.datagen.synthetic import SyntheticGenerator, GeneratorConfig

    cfg = GeneratorConfig(
        seed=1337,
        start=datetime(2026, 6, 20, 8, 0, tzinfo=timezone.utc),
        duration_s=3600,
        step_s=10.0,
    )
    gen = SyntheticGenerator(cfg)
    labels = gen.labels()                 # list[ScenarioLabel] (ground truth)
    for rec in gen.iter_records():        # time-ordered telemetry union
        ...                               # TelemetryRecord | RoutingEvent | ...
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from netra.contracts import (
    DeviceRole,
    FlowRecord,
    IssueType,
    MetricName,
    RoutingEvent,
    ScenarioId,
    ScenarioLabel,
    Severity,
    SyslogEvent,
    TelemetryKind,
    TelemetryRecord,
    TelemetrySourceKind,
    TunnelStat,
)

from .scenarios import (
    SITE_WEIGHT,
    ScenarioSpec,
    apply_injection,
    baseline_for,
    stream_rng,
)
from .topology import REFERENCE_TOPOLOGY, Device, Topology

# A record is any one of the five telemetry contract types.
TelemetryUnion = (
    TelemetryRecord | RoutingEvent | SyslogEvent | FlowRecord | TunnelStat
)

# Which metrics each interface-bearing role reports (drives baseline streams).
_INTERFACE_METRICS = (
    MetricName.IF_UTIL_PCT.value,
    MetricName.IF_OUT_DISCARDS.value,
    MetricName.IF_IN_ERRORS.value,
    MetricName.LATENCY_MS.value,
    MetricName.JITTER_MS.value,
    MetricName.LOSS_PCT.value,
    MetricName.QUEUE_DEPTH.value,
)

_TUNNEL_METRICS = (
    MetricName.TUNNEL_LOSS_PCT.value,
    MetricName.TUNNEL_JITTER_MS.value,
    MetricName.TUNNEL_REKEY_INTERVAL_S.value,
)

# BGP/OSPF session metrics reported per (router, peer).
_PEER_METRICS = (
    MetricName.BGP_UPDATE_RATE.value,
    MetricName.BGP_WITHDRAW_RATE.value,
    MetricName.BGP_FLAP_PENALTY.value,
    MetricName.ADJ_FLAP_COUNT.value,
    MetricName.OSPF_LSA_RATE.value,
    MetricName.OSPF_SPF_RATE.value,
    MetricName.PATH_ASYMMETRY.value,
)


@dataclass
class GeneratorConfig:
    """All knobs controlling a deterministic generator run.

    The tuple ``(seed, start, duration_s, step_s, scenarios)`` fully determines
    the output. ``start`` MUST be timezone-aware (UTC recommended); a naive
    datetime is coerced to UTC.
    """

    seed: int = 1337
    start: datetime = field(
        default_factory=lambda: datetime(2026, 6, 20, 8, 0, tzinfo=timezone.utc)
    )
    duration_s: float = 3600.0
    step_s: float = 10.0
    #: which scenarios to inject; empty => a pure healthy baseline dataset.
    scenarios: tuple[ScenarioId, ...] = (
        ScenarioId.A_CONGESTION,
        ScenarioId.B_BGP_FLAP,
        ScenarioId.C_TUNNEL_DEGRADATION,
        ScenarioId.D_POLICY_DRIFT,
    )
    #: emit NetFlow flow records (heavier); off by default for light streams.
    emit_flows: bool = True
    #: emit syslog events on routing/tunnel state changes.
    emit_syslog: bool = True
    topology: Topology = field(default_factory=lambda: REFERENCE_TOPOLOGY)

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=timezone.utc)
        else:
            self.start = self.start.astimezone(timezone.utc)
        if self.step_s <= 0:
            raise ValueError("step_s must be > 0")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be > 0")


class SyntheticGenerator:
    """Deterministic, labeled, multi-entity telemetry generator.

    The generator builds a fixed catalog of (entity, metric) baseline streams
    from the topology, schedules the requested scenarios within the run window,
    then walks time in ``step_s`` increments emitting time-ordered records with
    baseline + scenario-precursor deltas. Ground-truth ``ScenarioLabel``s are
    computed up front (windows are known before any record is produced — exactly
    the invariant the sim honours).
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.topology = self.config.topology
        self._specs: list[ScenarioSpec] = self._schedule_scenarios()
        self._labels: list[ScenarioLabel] = [self._spec_to_label(s) for s in self._specs]

    # ------------------------------------------------------------------ #
    # Scenario scheduling + ground-truth labels                          #
    # ------------------------------------------------------------------ #

    def _schedule_scenarios(self) -> list[ScenarioSpec]:
        """Lay the requested scenarios out across the run window deterministically.

        Scenarios are spaced so their fault windows don't overlap on the same
        target, leaving a healthy baseline lead-in. Offsets scale with the run
        duration so short and long runs both contain complete precursor+fault
        windows.
        """
        dur = self.config.duration_s
        topo = self.topology
        specs: list[ScenarioSpec] = []

        # Pick canonical targets from the topology.
        hub_iface = Topology.interface_entity_id(topo.device("pe-hub"), "eth3")
        rr_peer = Topology.peer_entity_id(topo.device("rr-dc"), "pe-dc1")
        br1_tunnel = Topology.tunnel_entity_id(topo.device("ce-br1"), "tunnel-hub")
        ctl_entity = Topology.device_entity_id(topo.device("sdwan-ctl"))

        # Fraction-of-duration windows (precursor must precede fault for lead time).
        # Each scenario occupies a distinct slot so labels don't collide.
        layout: dict[ScenarioId, tuple[float, float, float]] = {
            # (precursor_start, fault_start, fault_end) as fractions of duration
            ScenarioId.A_CONGESTION: (0.12, 0.28, 0.45),
            ScenarioId.B_BGP_FLAP: (0.30, 0.40, 0.55),
            ScenarioId.C_TUNNEL_DEGRADATION: (0.50, 0.62, 0.80),
            ScenarioId.D_POLICY_DRIFT: (0.70, 0.78, 0.92),
        }
        meta: dict[ScenarioId, dict] = {
            ScenarioId.A_CONGESTION: dict(
                expected_issue=IssueType.INTERFACE_CONGESTION,
                target=hub_iface,
                severity=Severity.P1,
                playbook_id="pb-congestion-qos-reroute",
                target_sites=("hub", "br1", "br2", "br3"),
                target_vpns=("CORP",),
                params={"peak_util_pct": 55.0},
            ),
            ScenarioId.B_BGP_FLAP: dict(
                expected_issue=IssueType.BGP_ROUTE_FLAP,
                target=rr_peer,
                severity=Severity.P1,
                playbook_id="pb-bgp-flap-damping",
                target_sites=("dc", "hub"),
                target_vpns=("CORP", "OT"),
                params={"flap_period_s": 80.0},
            ),
            ScenarioId.C_TUNNEL_DEGRADATION: dict(
                expected_issue=IssueType.TUNNEL_DEGRADATION,
                target=br1_tunnel,
                severity=Severity.P2,
                playbook_id="pb-tunnel-reroute-backup",
                target_sites=("br1", "hub"),
                target_vpns=("CORP",),
                params={"burst_period_s": 40.0},
            ),
            ScenarioId.D_POLICY_DRIFT: dict(
                expected_issue=IssueType.POLICY_DRIFT,
                target=ctl_entity,
                severity=Severity.P2,
                playbook_id="pb-policy-revert-golden",
                target_sites=("dc", "hub", "br3"),
                target_vpns=("CORP", "OT"),
                # scenario D fans out to multiple PEs simultaneously
                params={
                    "_fanout_ids": (
                        Topology.interface_entity_id(topo.device("pe-dc1"), "eth2"),
                        Topology.interface_entity_id(topo.device("pe-hub"), "eth2"),
                        Topology.peer_entity_id(topo.device("pe-dc1"), "pe-hub"),
                    )
                },
            ),
        }

        for scen in self.config.scenarios:
            if scen not in layout:
                continue
            f0, f1, f2 = layout[scen]
            m = meta[scen]
            specs.append(
                ScenarioSpec(
                    scenario=scen,
                    expected_issue=m["expected_issue"],
                    target_entity_id=m["target"],
                    precursor_offset_s=f0 * dur,
                    fault_offset_s=f1 * dur,
                    fault_end_offset_s=f2 * dur,
                    severity=m["severity"],
                    playbook_id=m["playbook_id"],
                    target_sites=tuple(m["target_sites"]),
                    target_vpns=tuple(m["target_vpns"]),
                    params=dict(m["params"]),
                )
            )
        return specs

    def _spec_to_label(self, spec: ScenarioSpec) -> ScenarioLabel:
        """Convert a :class:`ScenarioSpec` into a ground-truth ``ScenarioLabel``."""
        start = self.config.start
        precursor_start = start + timedelta(seconds=spec.precursor_offset_s)
        fault_start = start + timedelta(seconds=spec.fault_offset_s)
        fault_end = start + timedelta(seconds=spec.fault_end_offset_s)
        lead = spec.fault_offset_s - spec.precursor_offset_s
        # serialise injector params, dropping private/non-JSON keys
        public_params: dict[str, object] = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in spec.params.items()
            if not k.startswith("_")
        }
        public_params["precursor_window_s"] = lead
        return ScenarioLabel(
            label_id=f"syn-{spec.scenario.value}",
            scenario=spec.scenario,
            expected_issue=spec.expected_issue,
            target_entity_id=spec.target_entity_id,
            precursor_window_start=precursor_start,
            fault_window_start=fault_start,
            fault_window_end=fault_end,
            expected_lead_time_seconds=lead,
            expected_time_to_impact_seconds=lead,
            severity=spec.severity,
            target_sites=list(spec.target_sites),
            target_services_or_vpns=list(spec.target_vpns),
            expected_playbook_id=spec.playbook_id,
            injection_tool="synthetic",
            params=public_params,
            seed=self.config.seed,
        )

    def labels(self) -> list[ScenarioLabel]:
        """Return the ground-truth labels for this run (one per scenario)."""
        return list(self._labels)

    # ------------------------------------------------------------------ #
    # Stream catalog                                                     #
    # ------------------------------------------------------------------ #

    def _interface_streams(self) -> Iterator[tuple[Device, str, str, str]]:
        """Yield (device, interface, metric, entity_id) for every interface metric."""
        for d in self.topology.devices:
            if d.role == DeviceRole.CONTROLLER:
                # controller exposes config-drift, handled separately
                continue
            for iface in d.interfaces:
                ent = Topology.interface_entity_id(d, iface)
                for metric in _INTERFACE_METRICS:
                    yield d, iface, metric, ent

    def _tunnel_streams(self) -> Iterator[tuple[Device, str, str, str]]:
        for d in self.topology.devices:
            for tun in d.tunnels:
                ent = Topology.tunnel_entity_id(d, tun)
                for metric in _TUNNEL_METRICS:
                    yield d, tun, metric, ent

    def _peer_streams(self) -> Iterator[tuple[Device, str, str, str]]:
        for d in self.topology.devices:
            for peer in d.bgp_peers:
                ent = Topology.peer_entity_id(d, peer)
                for metric in _PEER_METRICS:
                    yield d, peer, metric, ent

    # ------------------------------------------------------------------ #
    # Core sampling                                                      #
    # ------------------------------------------------------------------ #

    def _value(
        self,
        entity_id: str,
        metric: str,
        ts: datetime,
        rel_t: float,
        site: str,
    ) -> float:
        """Baseline sample + summed scenario deltas for one (entity, metric, t)."""
        rng = stream_rng(self.config.seed, entity_id, metric, int(rel_t))
        base = baseline_for(metric)
        sw = SITE_WEIGHT.get(site, 0.6)
        val = base.sample(ts, rng, site_weight=sw)
        for spec in self._specs:
            val += apply_injection(spec, rel_t, entity_id, metric, rng)
        # final clamp to the metric's physical range
        return float(min(base.ceil, max(base.floor, val)))

    def _n_steps(self) -> int:
        return int(self.config.duration_s // self.config.step_s) + 1

    # ------------------------------------------------------------------ #
    # Record emission (time-ordered)                                     #
    # ------------------------------------------------------------------ #

    def iter_records(self) -> Iterator[TelemetryUnion]:
        """Yield every telemetry record for the run in strict time order.

        At each tick all per-tick records share the same timestamp; ticks are
        emitted in increasing time so a downstream streaming engine sees a clean
        monotonic stream. The record union covers all five contract types.
        """
        cfg = self.config
        n = self._n_steps()
        for i in range(n):
            rel_t = i * cfg.step_s
            ts = cfg.start + timedelta(seconds=rel_t)
            # 1) interface numeric metrics (SNMP/gNMI-style)
            yield from self._emit_interface_records(ts, rel_t)
            # 2) tunnel health (TunnelStat)
            yield from self._emit_tunnel_records(ts, rel_t)
            # 3) routing-session metrics (TelemetryRecord) + routing events
            yield from self._emit_routing_records(ts, rel_t)
            # 4) flow records (sampled, lower cadence)
            if cfg.emit_flows and i % max(1, int(30 // cfg.step_s) or 1) == 0:
                yield from self._emit_flow_records(ts, rel_t)
            # 5) controller config-drift score (scenario D)
            yield from self._emit_controller_records(ts, rel_t)

    # -- interface numeric metrics --
    def _emit_interface_records(self, ts: datetime, rel_t: float) -> Iterator[TelemetryRecord]:
        for d, iface, metric, ent in self._interface_streams():
            val = self._value(ent, metric, ts, rel_t, d.site)
            yield TelemetryRecord(
                timestamp=ts,
                site=d.site,
                device=d.name,
                role=d.role,
                metric_name=metric,
                value=val,
                kind=_metric_kind(metric),
                site_type=d.site_type,
                unit=_metric_unit(metric),
                labels={"interface": iface},
                source=TelemetrySourceKind.SYNTHETIC,
            )

    # -- tunnel health --
    def _emit_tunnel_records(self, ts: datetime, rel_t: float) -> Iterator[TelemetryUnion]:
        for d in self.topology.devices:
            for tun in d.tunnels:
                ent = Topology.tunnel_entity_id(d, tun)
                loss = self._value(ent, MetricName.TUNNEL_LOSS_PCT.value, ts, rel_t, d.site)
                jitter = self._value(ent, MetricName.TUNNEL_JITTER_MS.value, ts, rel_t, d.site)
                rekey = self._value(
                    ent, MetricName.TUNNEL_REKEY_INTERVAL_S.value, ts, rel_t, d.site
                )
                latency = self._value(ent, MetricName.LATENCY_MS.value, ts, rel_t, d.site)
                # operationally down only during severe loss bursts
                oper_up = loss < 80.0
                peer_site = tun.replace("tunnel-", "")
                yield TunnelStat(
                    timestamp=ts,
                    site=d.site,
                    device=d.name,
                    role=d.role,
                    tunnel_id=tun,
                    peer_site=peer_site,
                    loss_pct=min(100.0, max(0.0, loss)),
                    jitter_ms=max(0.0, jitter),
                    latency_ms=max(0.0, latency),
                    rekey_interval_s=max(60.0, rekey),
                    oper_up=oper_up,
                    labels={"tunnel": tun},
                )
                # rekey anomaly -> syslog precursor
                if self.config.emit_syslog and rekey < 3000.0:
                    yield SyslogEvent(
                        timestamp=ts,
                        site=d.site,
                        device=d.name,
                        role=d.role,
                        severity=4,
                        facility="local0",
                        mnemonic="%IKE-4-REKEY_ANOMALY",
                        message=(
                            f"IPSec SA {tun} rekey interval {rekey:.0f}s below "
                            f"expected 3600s (peer {peer_site})"
                        ),
                        labels={"tunnel": tun, "peer": peer_site},
                    )

    # -- routing-session metrics + control-plane events --
    def _emit_routing_records(self, ts: datetime, rel_t: float) -> Iterator[TelemetryUnion]:
        for d in self.topology.devices:
            for peer in d.bgp_peers:
                ent = Topology.peer_entity_id(d, peer)
                for metric in _PEER_METRICS:
                    val = self._value(ent, metric, ts, rel_t, d.site)
                    yield TelemetryRecord(
                        timestamp=ts,
                        site=d.site,
                        device=d.name,
                        role=d.role,
                        metric_name=metric,
                        value=val,
                        kind=_metric_kind(metric),
                        site_type=d.site_type,
                        unit=_metric_unit(metric),
                        labels={"peer": peer},
                        source=TelemetrySourceKind.SYNTHETIC,
                    )
                # discrete routing events when churn is high (scenario B precursor)
                update_rate = self._value(
                    ent, MetricName.BGP_UPDATE_RATE.value, ts, rel_t, d.site
                )
                adj_flap = self._value(
                    ent, MetricName.ADJ_FLAP_COUNT.value, ts, rel_t, d.site
                )
                if update_rate > 30.0:
                    yield RoutingEvent(
                        timestamp=ts,
                        site=d.site,
                        device=d.name,
                        role=d.role,
                        protocol=TelemetryKind.BGP,
                        event_type="route_withdraw" if adj_flap >= 0.5 else "route_announce",
                        peer=peer,
                        prefix="10.2.0.0/24",
                        vrf="CORP",
                        as_path_len=4 if adj_flap >= 0.5 else 3,
                        labels={"update_rate": f"{update_rate:.1f}"},
                    )
                if adj_flap >= 0.5:
                    if self.config.emit_syslog:
                        yield SyslogEvent(
                            timestamp=ts,
                            site=d.site,
                            device=d.name,
                            role=d.role,
                            severity=5,
                            facility="local0",
                            mnemonic="%BGP-5-ADJCHANGE",
                            message=f"neighbor {peer} Down/Up (flap)",
                            labels={"peer": peer},
                        )
                    yield RoutingEvent(
                        timestamp=ts,
                        site=d.site,
                        device=d.name,
                        role=d.role,
                        protocol=TelemetryKind.BGP,
                        event_type="adjacency_down",
                        peer=peer,
                        labels={},
                    )

    # -- flow records --
    def _emit_flow_records(self, ts: datetime, rel_t: float) -> Iterator[FlowRecord]:
        """Emit a few representative flows per PE (traffic-matrix / blast-radius)."""
        for d in self.topology.devices_by_role(DeviceRole.PE):
            ent = Topology.interface_entity_id(d, "eth2")
            util = self._value(ent, MetricName.IF_UTIL_PCT.value, ts, rel_t, d.site)
            rng = stream_rng(self.config.seed, ent, "flow", int(rel_t))
            n_flows = 2 + int(util / 25.0)
            for k in range(n_flows):
                byte_vol = int(max(1.0, util) * 12_000 * (1.0 + rng.normal(0, 0.2)))
                pkts = max(1, byte_vol // 1200)
                yield FlowRecord(
                    timestamp=ts,
                    site=d.site,
                    device=d.name,
                    src_addr=f"192.168.{abs(hash(d.site)) % 250}.{10 + k}",
                    dst_addr=f"10.0.0.{1 + (k % 8)}",
                    src_port=1024 + (k * 137) % 64000,
                    dst_port=(443, 80, 5060, 53)[k % 4],
                    protocol=("tcp", "tcp", "udp", "udp")[k % 4],
                    bytes=max(0, byte_vol),
                    packets=max(0, pkts),
                    in_iface="eth1",
                    out_iface="eth2",
                    vrf="CORP",
                    dscp=(46, 26, 0, 0)[k % 4],
                    labels={"qos_class": ("voice", "business", "bulk", "bulk")[k % 4]},
                )

    # -- controller config-drift score --
    def _emit_controller_records(self, ts: datetime, rel_t: float) -> Iterator[TelemetryUnion]:
        ctl = None
        for d in self.topology.devices:
            if d.role == DeviceRole.CONTROLLER:
                ctl = d
                break
        if ctl is None:
            return
        ent = Topology.device_entity_id(ctl)
        score = self._value(ent, MetricName.CONFIG_DRIFT_SCORE.value, ts, rel_t, ctl.site)
        yield TelemetryRecord(
            timestamp=ts,
            site=ctl.site,
            device=ctl.name,
            role=ctl.role,
            metric_name=MetricName.CONFIG_DRIFT_SCORE.value,
            value=score,
            kind=TelemetryKind.CONTROLLER,
            site_type=ctl.site_type,
            unit="score",
            labels={},
            source=TelemetrySourceKind.SYNTHETIC,
        )
        # config-change syslog at the moment drift first appears
        if self.config.emit_syslog and 0.0 < score and self._is_drift_onset(rel_t):
            yield SyslogEvent(
                timestamp=ts,
                site=ctl.site,
                device=ctl.name,
                role=ctl.role,
                severity=5,
                facility="local7",
                mnemonic="%SYS-5-CONFIG_I",
                message="Configured from controller by policy push (intent commit)",
                labels={"change": "policy_push"},
            )

    def _is_drift_onset(self, rel_t: float) -> bool:
        """True for the single tick where scenario D's config push lands."""
        for spec in self._specs:
            if spec.scenario == ScenarioId.D_POLICY_DRIFT:
                lo = spec.fault_offset_s - self.config.step_s
                hi = spec.fault_offset_s + self.config.step_s
                if lo <= rel_t <= hi:
                    return True
        return False


# --------------------------------------------------------------------------- #
# Metric -> (kind, unit) mapping helpers                                       #
# --------------------------------------------------------------------------- #


def _metric_kind(metric: str) -> TelemetryKind:
    routing = {
        MetricName.BGP_UPDATE_RATE.value,
        MetricName.BGP_WITHDRAW_RATE.value,
        MetricName.BGP_FLAP_PENALTY.value,
        MetricName.ADJ_FLAP_COUNT.value,
    }
    ospf = {MetricName.OSPF_LSA_RATE.value, MetricName.OSPF_SPF_RATE.value}
    if metric in routing or metric == MetricName.PATH_ASYMMETRY.value:
        return TelemetryKind.BGP
    if metric in ospf:
        return TelemetryKind.OSPF
    if metric in {
        MetricName.TUNNEL_LOSS_PCT.value,
        MetricName.TUNNEL_JITTER_MS.value,
        MetricName.TUNNEL_REKEY_INTERVAL_S.value,
    }:
        return TelemetryKind.TUNNEL
    if metric in {MetricName.QUEUE_DEPTH.value, MetricName.IF_OUT_DISCARDS.value}:
        return TelemetryKind.QOS
    if metric == MetricName.CONFIG_DRIFT_SCORE.value:
        return TelemetryKind.CONTROLLER
    return TelemetryKind.SNMP


def _metric_unit(metric: str) -> str | None:
    units = {
        MetricName.IF_UTIL_PCT.value: "pct",
        MetricName.LOSS_PCT.value: "pct",
        MetricName.TUNNEL_LOSS_PCT.value: "pct",
        MetricName.LATENCY_MS.value: "ms",
        MetricName.JITTER_MS.value: "ms",
        MetricName.TUNNEL_JITTER_MS.value: "ms",
        MetricName.TUNNEL_REKEY_INTERVAL_S.value: "s",
        MetricName.QUEUE_DEPTH.value: "packets",
        MetricName.IF_OUT_DISCARDS.value: "packets",
        MetricName.IF_IN_ERRORS.value: "packets",
        MetricName.PATH_ASYMMETRY.value: "score",
    }
    return units.get(metric)


__all__ = [
    "GeneratorConfig",
    "SyntheticGenerator",
    "TelemetryUnion",
]
