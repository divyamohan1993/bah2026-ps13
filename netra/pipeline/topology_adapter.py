"""Build the correlation :class:`TopologyGraph` from the datagen reference topology.

The synthetic generator (``netra.datagen``) and the correlation digital twin
(``netra.analytics.correlation.graph``) both speak the ``EntityRef`` id
convention ``"<site>:<device>:<role>[:<sub>]"`` but they are *separately* built —
datagen from :data:`netra.datagen.topology.REFERENCE_TOPOLOGY`, correlation from a
plain spec dict. This adapter bridges them so the graph nodes are exactly the
*device-level* entity ids the generator's per-(interface/tunnel/peer) telemetry
maps onto (``TopologyGraph.map_to_node`` strips the trailing sub-segment), and the
failure-propagation edges match the real topology (core P → PE → CE/branch, the
route reflector and SD-WAN controller fanning out to every PE).

Keeping this in the *pipeline* layer (not editing either module) is the intended
adapter-over-invasive-edit approach for integration.
"""

from __future__ import annotations

from netra.analytics.correlation import TopologyGraph
from netra.analytics.correlation.graph import default_criticality
from netra.contracts import DeviceRole
from netra.datagen import REFERENCE_TOPOLOGY, Topology


def topology_spec_from_reference(topo: Topology | None = None) -> dict:
    """Render :data:`REFERENCE_TOPOLOGY` as a correlation topology spec dict.

    Nodes are device-level entity ids (``site:device:role``) carrying role,
    site_type, criticality, services (VRFs) and SLAs. Edges encode
    failure-propagation direction (upstream → downstream):

      * provider-core P-routers meshed (bidirectional links);
      * each P → the PEs it carries (directed downstream);
      * the route reflector → every PE (VPNv4 reflection; directed);
      * the SD-WAN controller → every PE (policy push; directed);
      * each hub/branch PE/CE → its overlay-tunnel spokes (directed downstream).
    """
    topo = topo or REFERENCE_TOPOLOGY
    nodes: list[dict] = []
    for d in topo.devices:
        node_id = Topology.device_entity_id(d)
        nodes.append(
            {
                "id": node_id,
                "site": d.site,
                "device": d.name,
                "role": d.role.value,
                "site_type": d.site_type.value,
                "criticality": default_criticality(d.role, d.site_type),
                "services": list(d.vrfs),
                "slas": _slas_for(d),
            }
        )

    node_ids = {n["id"] for n in nodes}

    def nid(name: str) -> str | None:
        d = _device_or_none(topo, name)
        return Topology.device_entity_id(d) if d is not None else None

    edges: list[dict] = []

    # 1) underlay links from the reference Link list (bidirectional unless P->PE).
    for ln in topo.links:
        a, b = nid(ln.a_device), nid(ln.b_device)
        if a is None or b is None or a == b:
            continue
        a_role = _device_or_none(topo, ln.a_device)
        b_role = _device_or_none(topo, ln.b_device)
        # A P->PE / P->CE uplink propagates downstream (directed); P-P core mesh and
        # PE-CE access are kept bidirectional so reachability is symmetric there.
        directed = bool(
            a_role
            and b_role
            and a_role.role == DeviceRole.P
            and b_role.role in (DeviceRole.PE, DeviceRole.CE)
        )
        edges.append({"source": a, "target": b, "kind": "link", "directed": directed})

    pes = [Topology.device_entity_id(d) for d in topo.devices_by_role(DeviceRole.PE)]

    # 2) route reflector -> every PE (VPNv4 reflection: an RR flap reaches all PEs).
    for rr in topo.devices_by_role(DeviceRole.RR):
        rr_id = Topology.device_entity_id(rr)
        for pe in pes:
            edges.append({"source": rr_id, "target": pe, "kind": "session", "directed": True})

    # 3) SD-WAN controller -> every PE (policy drift fans out from here).
    for ctl in topo.devices_by_role(DeviceRole.CONTROLLER):
        ctl_id = Topology.device_entity_id(ctl)
        for pe in pes:
            edges.append({"source": ctl_id, "target": pe, "kind": "session", "directed": True})

    # 4) overlay tunnels: a tunnel-terminating device -> the spoke it names.
    #    e.g. ce-hub 'tunnel-br1' -> ce-br1 ; ce-br1 'tunnel-hub' -> ce-hub.
    by_site = _devices_by_site(topo)
    for d in topo.devices:
        for tun in d.tunnels:
            peer_site = tun.replace("tunnel-", "")
            src = Topology.device_entity_id(d)
            # pick a CE/PE terminating the tunnel at the peer site.
            dst_dev = _tunnel_peer_device(by_site.get(peer_site, []))
            if dst_dev is None:
                continue
            dst = Topology.device_entity_id(dst_dev)
            if dst in node_ids and src != dst:
                edges.append({"source": src, "target": dst, "kind": "tunnel", "directed": True})

    return {"nodes": nodes, "edges": edges}


def build_pipeline_graph(topo: Topology | None = None) -> TopologyGraph:
    """Convenience: build the correlation :class:`TopologyGraph` for the pipeline."""
    return TopologyGraph.from_spec(topology_spec_from_reference(topo))


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _device_or_none(topo: Topology, name: str):
    try:
        return topo.device(name)
    except KeyError:
        return None


def _devices_by_site(topo: Topology) -> dict[str, list]:
    out: dict[str, list] = {}
    for d in topo.devices:
        out.setdefault(d.site, []).append(d)
    return out


def _tunnel_peer_device(devices: list):
    """Pick the device at a site that terminates an overlay tunnel (CE, else PE)."""
    ces = [d for d in devices if d.role == DeviceRole.CE]
    if ces:
        return ces[0]
    pes = [d for d in devices if d.role == DeviceRole.PE]
    return pes[0] if pes else None


def _slas_for(d) -> list[str]:
    """A small, deterministic SLA list per role/site so blast radius is non-empty."""
    out: list[str] = []
    if "CORP" in d.vrfs:
        out.append("sla-corp")
    if "OT" in d.vrfs:
        out.append("sla-ot")
    return out


__all__ = ["topology_spec_from_reference", "build_pipeline_graph"]
