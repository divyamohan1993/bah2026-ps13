"""Canonical reference topology for the synthetic generator and sim labels.

This module is the single, deterministic source of truth for *which entities
exist* in the NETRA reference network — the 5-site, ~20-node SD-WAN-over-MPLS
topology described in ``ARCHITECTURE.md`` §3 (Phase 1) and ``research/01``. Both
the synthetic generator (``synthetic.py``) and the sim fault drivers (``sim/``)
build their entity lists, ``EntityRef`` join keys and ``ScenarioLabel`` targets
from here so the two telemetry sources are byte-for-byte aligned on identity.

It deliberately depends on **nothing heavy** — only ``netra.contracts`` enums
and stdlib dataclasses — so it can be imported anywhere (including the sim
side, which never installs numpy/pandas).

Topology (matches ``research/01`` §6 "Recommended Reference Topology"):

    Site        Role(s)                                  Scenario target
    ----        -------                                  ---------------
    DC          2x PE, 1x CE, hosts, RR                  (RR -> scenario B)
    HUB         1x PE, 1x CE, IPSec head-end             scenario A (congestion)
    BRANCH-1/2/3 1x CE each, IPSec spoke tunnels         traffic clients
    CORE        4x P (IS-IS + SR-MPLS transport)         scenario C (underlay)
    CONTROLLER  1x SD-WAN controller (policy/intent)     scenario D (drift)

Entity-id convention (from ``netra.contracts.common.EntityRef``)::

    "<site>:<device>:<role>[:<sub>]"
    e.g. "hub:pe-hub:PE:eth1"        (an interface)
         "br1:ce-br1:CE:tunnel-hub"  (an overlay tunnel)
         "core:p2:P"                 (a core router)
         "dc:rr-dc:RR:peer-pe-dc1"   (a BGP peering)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from netra.contracts import DeviceRole, SiteType

# --------------------------------------------------------------------------- #
# Static topology description (dataclasses, hashable & deterministic ordering) #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Device:
    """A single network device in the reference topology."""

    name: str
    site: str
    site_type: SiteType
    role: DeviceRole
    #: Interfaces this device exposes that carry forecastable telemetry.
    interfaces: tuple[str, ...] = ()
    #: Overlay tunnels terminated on this device (CE/Hub endpoints only).
    tunnels: tuple[str, ...] = ()
    #: BGP/VPNv4 peers (for RoutingEvent / peering entities).
    bgp_peers: tuple[str, ...] = ()
    #: VRFs configured on this device (L3VPN segmentation).
    vrfs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Link:
    """An (undirected) link between two device interfaces in the underlay."""

    a_device: str
    a_iface: str
    b_device: str
    b_iface: str
    #: 'core' (P-P / P-PE underlay), 'access' (PE-CE) or 'overlay' (tunnel path).
    kind: str = "core"


# RD/RT for the two L3VPNs (research/01 §6.1 addressing plan).
VRF_CORP = "CORP"
VRF_OT = "OT"
VRFS = (VRF_CORP, VRF_OT)


def _ifaces(*names: str) -> tuple[str, ...]:
    return tuple(names)


# --------------------------------------------------------------------------- #
#  The 5-site reference topology (≈20 nodes).                                  #
#  Ordering is fixed and stable so iteration is deterministic.                #
# --------------------------------------------------------------------------- #

_DEVICES: tuple[Device, ...] = (
    # ---- MPLS provider core: 4x P routers (IS-IS + SR-MPLS) ----
    Device("p1", "core", SiteType.CORE, DeviceRole.P, _ifaces("eth1", "eth2", "eth3")),
    Device("p2", "core", SiteType.CORE, DeviceRole.P, _ifaces("eth1", "eth2", "eth3")),
    Device("p3", "core", SiteType.CORE, DeviceRole.P, _ifaces("eth1", "eth2", "eth3")),
    Device("p4", "core", SiteType.CORE, DeviceRole.P, _ifaces("eth1", "eth2", "eth3")),
    # ---- Datacenter: dual PE, a CE, a route reflector ----
    Device(
        "pe-dc1",
        "dc",
        SiteType.DATACENTER,
        DeviceRole.PE,
        _ifaces("eth1", "eth2", "eth3"),
        bgp_peers=("rr-dc", "pe-dc2", "pe-hub"),
        vrfs=VRFS,
    ),
    Device(
        "pe-dc2",
        "dc",
        SiteType.DATACENTER,
        DeviceRole.PE,
        _ifaces("eth1", "eth2"),
        bgp_peers=("rr-dc", "pe-dc1"),
        vrfs=VRFS,
    ),
    Device(
        "ce-dc",
        "dc",
        SiteType.DATACENTER,
        DeviceRole.CE,
        _ifaces("eth1", "eth2"),
        vrfs=(VRF_CORP,),
    ),
    Device(
        "rr-dc",
        "dc",
        SiteType.DATACENTER,
        DeviceRole.RR,
        _ifaces("eth1"),
        bgp_peers=("pe-dc1", "pe-dc2", "pe-hub"),
    ),
    # ---- HQ / Hub: PE + CE + IPSec head-end (scenario A target) ----
    Device(
        "pe-hub",
        "hub",
        SiteType.HUB,
        DeviceRole.PE,
        _ifaces("eth1", "eth2", "eth3"),
        bgp_peers=("rr-dc", "pe-dc1"),
        vrfs=VRFS,
    ),
    Device(
        "ce-hub",
        "hub",
        SiteType.HUB,
        DeviceRole.CE,
        _ifaces("eth1", "eth2"),
        tunnels=("tunnel-br1", "tunnel-br2", "tunnel-br3", "tunnel-dc"),
        vrfs=(VRF_CORP,),
    ),
    # ---- Branch spokes: 1x CE each, IPSec tunnels to hub + DC ----
    Device(
        "ce-br1",
        "br1",
        SiteType.BRANCH,
        DeviceRole.CE,
        _ifaces("eth1", "eth2"),
        tunnels=("tunnel-hub", "tunnel-dc"),
        vrfs=(VRF_CORP,),
    ),
    Device(
        "ce-br2",
        "br2",
        SiteType.BRANCH,
        DeviceRole.CE,
        _ifaces("eth1", "eth2"),
        tunnels=("tunnel-hub", "tunnel-dc"),
        vrfs=(VRF_CORP,),
    ),
    Device(
        "ce-br3",
        "br3",
        SiteType.BRANCH,
        DeviceRole.CE,
        _ifaces("eth1", "eth2"),
        tunnels=("tunnel-hub", "tunnel-dc"),
        vrfs=(VRF_CORP, VRF_OT),
    ),
    # ---- SD-WAN controller (policy/intent source; scenario D drift) ----
    Device(
        "sdwan-ctl",
        "dc",
        SiteType.DATACENTER,
        DeviceRole.CONTROLLER,
        _ifaces("eth1"),
    ),
)


# Underlay links (P-P mesh + PE uplinks + PE-CE access). Used to build the
# digital-twin graph (WS4) and to make scenario blast-radius realistic.
_LINKS: tuple[Link, ...] = (
    # core mesh (partial)
    Link("p1", "eth1", "p2", "eth1", "core"),
    Link("p2", "eth2", "p3", "eth1", "core"),
    Link("p3", "eth2", "p4", "eth1", "core"),
    Link("p4", "eth2", "p1", "eth2", "core"),
    Link("p1", "eth3", "p3", "eth3", "core"),  # diagonal for redundancy
    # PE uplinks into the core
    Link("pe-dc1", "eth1", "p1", "eth1", "core"),
    Link("pe-dc2", "eth1", "p2", "eth3", "core"),
    Link("pe-hub", "eth1", "p3", "eth3", "core"),
    # PE-CE access links
    Link("ce-dc", "eth1", "pe-dc1", "eth2", "access"),
    Link("ce-hub", "eth1", "pe-hub", "eth2", "access"),
    Link("rr-dc", "eth1", "pe-dc1", "eth3", "core"),
)


# --------------------------------------------------------------------------- #
#  Public accessors                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Topology:
    """Immutable view over the reference topology with convenient lookups."""

    devices: tuple[Device, ...] = field(default_factory=lambda: _DEVICES)
    links: tuple[Link, ...] = field(default_factory=lambda: _LINKS)

    # -- device lookups --
    def device(self, name: str) -> Device:
        """Return the :class:`Device` with ``name`` (raises if unknown)."""
        for d in self.devices:
            if d.name == name:
                return d
        raise KeyError(f"unknown device: {name!r}")

    def devices_by_role(self, role: DeviceRole) -> list[Device]:
        """All devices that play ``role`` (stable order)."""
        return [d for d in self.devices if d.role == role]

    def sites(self) -> list[str]:
        """Distinct site names in stable first-seen order."""
        seen: dict[str, None] = {}
        for d in self.devices:
            seen.setdefault(d.site, None)
        return list(seen)

    # -- entity-id helpers (mirror EntityRef convention) --
    @staticmethod
    def interface_entity_id(d: Device, iface: str) -> str:
        return f"{d.site}:{d.name}:{d.role.value}:{iface}"

    @staticmethod
    def tunnel_entity_id(d: Device, tunnel: str) -> str:
        return f"{d.site}:{d.name}:{d.role.value}:{tunnel}"

    @staticmethod
    def peer_entity_id(d: Device, peer: str) -> str:
        return f"{d.site}:{d.name}:{d.role.value}:peer-{peer}"

    @staticmethod
    def device_entity_id(d: Device) -> str:
        return f"{d.site}:{d.name}:{d.role.value}"


#: A ready-to-use singleton; callers may also build their own ``Topology()``.
REFERENCE_TOPOLOGY = Topology()


__all__ = [
    "Device",
    "Link",
    "Topology",
    "REFERENCE_TOPOLOGY",
    "VRF_CORP",
    "VRF_OT",
    "VRFS",
]
