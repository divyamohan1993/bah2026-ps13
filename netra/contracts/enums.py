"""Canonical enumerations shared across every NETRA module.

These enums are the controlled vocabularies for the whole pipeline. They are
deliberately defined once, here, so that the synthetic data generator, the
simulator telemetry adapter, the analytics engine, the copilot and the UI all
speak *exactly* the same strings. They also back the GBNF/JSON-schema the LLM is
constrained to (see ``netra.contracts.copilot``), so changing a value here is an
interface change that ripples to the model grammar — coordinate via the
integrator before editing.

All enums subclass ``str, Enum`` so they serialise to plain JSON strings and
compare equal to their string value (``DeviceRole.PE == "PE"``).
"""

from __future__ import annotations

from enum import Enum


class DeviceRole(str, Enum):
    """Role of a device in the SD-WAN-over-MPLS topology.

    CE  = Customer Edge (branch/site edge router, runs the IPSec overlay).
    PE  = Provider Edge (MPLS L3VPN edge; VRF + MP-BGP VPNv4).
    P   = Provider core (label-switching transport, IS-IS/SR-MPLS).
    RR  = Route Reflector (scales MP-BGP VPNv4; flap target in scenario B).
    CONTROLLER = SD-WAN controller (policy/intent source; drift in scenario D).
    HOST = Endpoint / server generating or receiving application traffic.
    """

    CE = "CE"
    PE = "PE"
    P = "P"
    RR = "RR"
    CONTROLLER = "controller"
    HOST = "host"


class SiteType(str, Enum):
    """Class of site in the multi-site enterprise topology."""

    DATACENTER = "datacenter"
    HUB = "hub"
    BRANCH = "branch"
    CORE = "core"  # provider MPLS core (no customer site, but a location bucket)


class TelemetryKind(str, Enum):
    """Source/transport class of a raw telemetry record.

    Mirrors the five signal classes the problem statement requires NETRA to
    ingest, plus derived kinds. Used to route records to the right collector
    adapter and the right family of detectors.
    """

    SNMP = "snmp"          # interface utilisation/latency/jitter/error counters
    GNMI = "gnmi"          # model-driven streaming telemetry (OpenConfig)
    NETFLOW = "netflow"    # NetFlow/IPFIX/sFlow flow records
    SYSLOG = "syslog"      # RFC5424/3164 syslog events
    BGP = "bgp"            # BGP adjacency / UPDATE / withdraw events (incl. BMP)
    OSPF = "ospf"          # OSPF adjacency / LSA / SPF events
    TUNNEL = "tunnel"      # IPSec/GRE overlay tunnel statistics + rekey events
    MPLS = "mpls"          # LSP / label / SR transport state
    QOS = "qos"            # per-class queue/drop counters
    CONTROLLER = "controller"  # SD-WAN controller config/intent change stream


class MetricName(str, Enum):
    """Canonical metric identifiers for numeric telemetry.

    Not exhaustive (``TelemetryRecord.metric_name`` is a free ``str`` so the
    sim/datagen can emit vendor-specific OIDs), but these are the metrics the
    analytics engine forecasts/scores and the copilot reasons about by name.
    Keeping them centralised lets detectors map a metric -> its SLA threshold.
    """

    IF_UTIL_PCT = "if_util_pct"                 # interface utilisation (0-100)
    IF_IN_OCTETS = "if_in_octets"               # ifHCInOctets
    IF_OUT_OCTETS = "if_out_octets"             # ifHCOutOctets
    IF_IN_ERRORS = "if_in_errors"               # ifInErrors
    IF_OUT_DISCARDS = "if_out_discards"         # ifOutDiscards (queue drops)
    LATENCY_MS = "latency_ms"                    # round-trip latency
    JITTER_MS = "jitter_ms"                      # inter-packet delay variation
    LOSS_PCT = "loss_pct"                        # packet loss ratio (0-100)
    QUEUE_DEPTH = "queue_depth"                  # egress queue depth
    BGP_UPDATE_RATE = "bgp_update_rate"          # UPDATEs per interval
    BGP_WITHDRAW_RATE = "bgp_withdraw_rate"      # withdrawals per interval
    BGP_FLAP_PENALTY = "bgp_flap_penalty"        # RFD-style decaying flap penalty
    ADJ_FLAP_COUNT = "adjacency_flap_count"      # BGP/OSPF adjacency up/down count
    OSPF_LSA_RATE = "ospf_lsa_rate"              # LSA regeneration rate
    OSPF_SPF_RATE = "ospf_spf_rate"              # SPF recomputation rate
    TUNNEL_LOSS_PCT = "tunnel_loss_pct"          # overlay tunnel loss
    TUNNEL_JITTER_MS = "tunnel_jitter_ms"        # overlay tunnel jitter
    TUNNEL_REKEY_INTERVAL_S = "tunnel_rekey_interval_s"  # IPSec rekey period
    FLOW_BYTES = "flow_bytes"                    # per-flow byte volume
    FLOW_COUNT = "flow_count"                    # active flow count
    PATH_ASYMMETRY = "path_asymmetry"            # fwd/rev path divergence score
    CONFIG_DRIFT_SCORE = "config_drift_score"    # diff vs last-known-good config


class IssueType(str, Enum):
    """Closed set of predicted/diagnosed fault classes.

    THIS ENUM IS LOAD-BEARING FOR THE COPILOT: it is the ``enum`` used in the
    GBNF grammar / JSON schema that constrains the LLM's ``predicted_issue``
    field (research 05 §4.2). Detectors, the risk engine and the copilot all
    classify into this set so a prediction can be scored against the injected
    fault label. Keep in sync with ``ScenarioId`` mappings.
    """

    INTERFACE_CONGESTION = "interface_congestion"
    LATENCY_DRIFT = "latency_drift"
    BGP_ROUTE_FLAP = "bgp_route_flap"
    OSPF_CONVERGENCE_STRESS = "ospf_convergence_stress"
    TUNNEL_DEGRADATION = "tunnel_degradation"
    MPLS_UNDERLAY_FAILURE = "mpls_underlay_failure"
    POLICY_DRIFT = "policy_drift"
    PATH_ASYMMETRY = "path_asymmetry"
    NONE = "none"  # no fault predicted / healthy


class Severity(str, Enum):
    """Incident severity / urgency class for the triage queue (Objective 4)."""

    P1 = "P1"          # imminent SLA/security breach — act now
    P2 = "P2"          # elevated risk — act soon
    P3 = "P3"          # watch / monitor
    INFO = "info"      # informational, no action


class Urgency(str, Enum):
    """Per-action urgency hint inside a recommended playbook step."""

    IMMEDIATE = "immediate"
    SOON = "soon"
    MONITOR = "monitor"


class Direction(str, Enum):
    """Sign/direction of a contributing signal's influence on risk (Q2)."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    NEUTRAL = "neutral"


class DetectorFamily(str, Enum):
    """Family grouping for the 30+ method predictive ensemble.

    Used in ``AnomalyScore`` / ``Forecast`` / ``FusedRisk`` so fusion and the
    copilot can report *which kind* of evidence fired and weight cross-family
    agreement (independent families agreeing => higher calibrated confidence).
    """

    FORECAST = "forecast"               # M1-M24 trajectory/regression forecasters
    FORECAST_RESIDUAL = "forecast_residual"  # predict-then-flag (#60)
    STATISTICAL = "statistical"         # z/EWMA/ESD/HBOS/COPOD/ECOD (#19-#29)
    ML_UNSUPERVISED = "ml_unsupervised"  # iForest/OCSVM/PCA/GMM (#30-#36)
    DEEP = "deep"                       # AE/VAE/USAD/TranAD/MTAD-GAT (#50-#58)
    CHANGE_POINT = "change_point"       # CUSUM/PH/ADWIN/BOCPD/PELT (#37-#43)
    MATRIX_PROFILE = "matrix_profile"   # STUMPY discord (#44)
    GRAPH = "graph"                     # PyGOD/centrality/correlation (#45-#48)
    ROUTING = "routing"                 # BGP/OSPF feature detectors (#61-#64)
    SURVIVAL = "survival"               # Cox/RSF time-to-event (#24, #66)


class ApprovalState(str, Enum):
    """Lifecycle state of a recommended remediation action (human-in-the-loop).

    Default safety posture is suggest -> require approval -> execute; only
    read-only diagnostics may be auto-approved (research 07 A3.2).
    """

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"
    AUTO_OK = "auto_ok"  # read-only diagnostic, no approval required


class TelemetrySourceKind(str, Enum):
    """Which backend satisfies the ``TelemetrySource`` interface.

    The dual-source abstraction that makes NETRA runnable end-to-end with no
    sim, no GPU and no internet: the same interface is served by EITHER the live
    Containerlab sim OR the synthetic scenario generator with ground-truth
    labels (architecture: Dual-Source Telemetry Abstraction).
    """

    SIM = "sim"              # live Containerlab/netlab lab via collectors
    SYNTHETIC = "synthetic"  # high-fidelity 4-scenario generator (labeled)
    REPLAY = "replay"        # replay of a previously captured/exported run


class ScenarioId(str, Enum):
    """The four required Phase-6 validation scenarios (ground-truth labels)."""

    A_CONGESTION = "A_congestion"            # progressive hub-spoke congestion
    B_BGP_FLAP = "B_bgp_flap"                # BGP route flap + reroute cascade
    C_TUNNEL_DEGRADATION = "C_tunnel_degradation"  # MPLS underlay/tunnel degrade
    D_POLICY_DRIFT = "D_policy_drift"        # controller misconfig -> policy drift
    BASELINE = "baseline"                    # healthy/no-fault baseline window


__all__ = [
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
]
