"""Raw telemetry contracts — the ingest boundary of NETRA.

Every signal that enters the pipeline — whether produced by the live
Containerlab sim (via gnmic/Telegraf/pmacct) or by the synthetic scenario
generator — is normalised into exactly one of these models and published onto
the NATS JetStream bus. The streaming feature engine and the historical
VictoriaMetrics writer both consume them.

These models are the contract that makes the **dual-source telemetry
abstraction** work: ``netra.datagen`` (synthetic) and the sim collectors emit
the identical types, so nothing downstream can tell — or needs to care — which
source produced a record.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from .common import EntityRef, NetraModel
from .enums import (
    DeviceRole,
    MetricName,
    SiteType,
    TelemetryKind,
    TelemetrySourceKind,
)


class TelemetryRecord(NetraModel):
    """A single numeric telemetry sample (the workhorse ingest record).

    One ``(timestamp, entity, metric)`` observation. SNMP counters, gNMI
    on-change values, derived rates and per-class QoS counters all land here.
    ``labels`` carries any extra dimensions (interface, vrf, vpn, peer, qos
    class) so high-cardinality context survives without exploding the schema.

    ``metric_name`` is a free string (not the ``MetricName`` enum) so the sim can
    pass through raw OIDs/paths; use ``MetricName`` values where possible so the
    analytics engine can map a metric to its SLA threshold.
    """

    timestamp: datetime = Field(
        ..., description="UTC sample time (absolute; never relative)."
    )
    site: str = Field(..., description="Site name.")
    device: str = Field(..., description="Device/host name.")
    role: DeviceRole = Field(..., description="Device role.")
    metric_name: str = Field(
        ...,
        description="Metric id; prefer a netra.contracts.MetricName value.",
        examples=[MetricName.IF_UTIL_PCT.value, MetricName.LATENCY_MS.value],
    )
    value: float = Field(..., description="Numeric metric value.")
    kind: TelemetryKind = Field(
        ..., description="Source/transport class of this signal."
    )
    site_type: SiteType | None = Field(default=None, description="Class of site.")
    unit: str | None = Field(
        default=None, description="Optional unit hint (e.g. 'pct', 'ms', 'octets')."
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Extra dimensions: interface, vrf, vpn, peer, qos_class, lsp.",
    )
    source: TelemetrySourceKind | None = Field(
        default=None,
        description="Optional provenance: which TelemetrySource emitted this.",
    )

    def entity(self) -> EntityRef:
        """Derive the canonical :class:`EntityRef` join key for this record."""
        sub = self.labels.get("interface") or self.labels.get("tunnel") or (
            self.labels.get("peer")
        )
        entity_id = f"{self.site}:{self.device}:{self.role.value}"
        if sub:
            entity_id = f"{entity_id}:{sub}"
        return EntityRef(
            entity_id=entity_id,
            site=self.site,
            device=self.device,
            role=self.role,
            site_type=self.site_type,
            sub=sub,
        )


class SyslogEvent(NetraModel):
    """A normalised syslog line (link up/down, rekey, config-change, etc.).

    Syslog is the precursor source for protocol flaps, IPSec rekey churn and
    controller config-change drift. ``mnemonic`` (e.g. ``%BGP-5-ADJCHANGE``) is
    kept verbatim because exact codes drive both routing detectors and RAG
    keyword retrieval.
    """

    timestamp: datetime = Field(..., description="UTC event time.")
    site: str = Field(..., description="Site name.")
    device: str = Field(..., description="Device emitting the log.")
    role: DeviceRole = Field(..., description="Device role.")
    severity: int = Field(
        ..., ge=0, le=7, description="Syslog severity 0(emerg)-7(debug)."
    )
    facility: str | None = Field(default=None, description="Syslog facility.")
    mnemonic: str | None = Field(
        default=None,
        description="Vendor mnemonic, kept verbatim.",
        examples=["%BGP-5-ADJCHANGE", "%OSPF-5-ADJCHG", "%LINK-3-UPDOWN"],
    )
    message: str = Field(..., description="Full message text.")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Parsed fields: interface, neighbor, vrf."
    )


class RoutingEvent(NetraModel):
    """A BGP/OSPF control-plane event (adjacency, route, SPF, LSA).

    Captures the routing-instability precursors the problem statement calls out:
    BGP UPDATE/withdraw churn, adjacency flaps, AS-path/path-asymmetry changes,
    OSPF LSA storms and SPF recompute stress.
    """

    timestamp: datetime = Field(..., description="UTC event time.")
    site: str = Field(..., description="Site name.")
    device: str = Field(..., description="Router emitting the event.")
    role: DeviceRole = Field(..., description="Device role.")
    protocol: TelemetryKind = Field(
        ..., description="Routing protocol kind (bgp or ospf)."
    )
    event_type: str = Field(
        ...,
        description="Event type.",
        examples=[
            "adjacency_up",
            "adjacency_down",
            "route_announce",
            "route_withdraw",
            "as_path_change",
            "spf_run",
            "lsa_regenerate",
        ],
    )
    peer: str | None = Field(default=None, description="Peer/neighbor id or IP.")
    prefix: str | None = Field(default=None, description="Affected prefix (CIDR).")
    vrf: str | None = Field(default=None, description="VRF/VPN name if applicable.")
    as_path_len: int | None = Field(
        default=None, ge=0, description="AS-path length (for asymmetry tracking)."
    )
    labels: dict[str, str] = Field(default_factory=dict, description="Extra fields.")

    @field_validator("protocol")
    @classmethod
    def _routing_protocol_only(cls, v: TelemetryKind) -> TelemetryKind:
        if v not in (TelemetryKind.BGP, TelemetryKind.OSPF):
            raise ValueError("RoutingEvent.protocol must be BGP or OSPF")
        return v


class FlowRecord(NetraModel):
    """A NetFlow/IPFIX/sFlow flow record (traffic-matrix + blast-radius input).

    Flow records give the traffic structure (top talkers, micro-bursts,
    inter-VRF leakage) and — intersected with the topology graph — turn a failing
    link into a concrete count of affected flows/SLAs/sites (research 07 A1.4).
    """

    timestamp: datetime = Field(..., description="UTC flow export time.")
    site: str = Field(..., description="Observation-point site.")
    device: str = Field(..., description="Exporter device.")
    src_addr: str = Field(..., description="Source IP.")
    dst_addr: str = Field(..., description="Destination IP.")
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_port: int | None = Field(default=None, ge=0, le=65535)
    protocol: str = Field(..., description="L4 protocol (tcp/udp/icmp/...).")
    bytes: int = Field(..., ge=0, description="Byte count for the flow.")
    packets: int = Field(..., ge=0, description="Packet count for the flow.")
    in_iface: str | None = Field(default=None, description="Ingress interface.")
    out_iface: str | None = Field(default=None, description="Egress interface.")
    vrf: str | None = Field(default=None, description="VRF/VPN the flow belongs to.")
    dscp: int | None = Field(
        default=None, ge=0, le=63, description="DSCP for QoS-class mapping."
    )
    labels: dict[str, str] = Field(default_factory=dict, description="Extra fields.")


class TunnelStat(NetraModel):
    """SD-WAN IPSec/GRE overlay tunnel health sample.

    The dedicated carrier for the 'tunnel health degradation scoring' the
    problem statement requires: loss progression, jitter trend and IPSec rekey
    anomalies (scenario C). ``rekey_interval_s`` deviating from baseline is a
    first-class precursor.
    """

    timestamp: datetime = Field(..., description="UTC sample time.")
    site: str = Field(..., description="Tunnel endpoint site.")
    device: str = Field(..., description="Tunnel endpoint device (usually a CE).")
    role: DeviceRole = Field(default=DeviceRole.CE, description="Endpoint role.")
    tunnel_id: str = Field(
        ..., description="Tunnel identifier.", examples=["tunnel-hub", "tunnel-dc"]
    )
    peer_site: str | None = Field(default=None, description="Remote endpoint site.")
    loss_pct: float = Field(..., ge=0, le=100, description="Tunnel packet loss %.")
    jitter_ms: float = Field(..., ge=0, description="Tunnel jitter (ms).")
    latency_ms: float | None = Field(default=None, ge=0, description="Tunnel RTT.")
    rekey_interval_s: float | None = Field(
        default=None, ge=0, description="Observed IPSec SA rekey interval (s)."
    )
    rekey_count: int | None = Field(
        default=None, ge=0, description="Cumulative rekeys observed."
    )
    oper_up: bool = Field(default=True, description="Tunnel operationally up?")
    labels: dict[str, str] = Field(default_factory=dict, description="Extra fields.")


__all__ = [
    "TelemetryRecord",
    "SyslogEvent",
    "RoutingEvent",
    "FlowRecord",
    "TunnelStat",
]
