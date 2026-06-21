"""Topology digital twin — a ``networkx`` graph of the SD-WAN-over-MPLS network.

This is the *authoritative* graph NETRA owns (architecture §Phase 4): CE / PE / P /
RR / controller / host nodes plus links, tunnels and adjacencies as edges. Every
downstream correlation, RCA and blast-radius computation traverses it, so it is
built deterministically from a plain topology spec — never inferred by an LLM.

A topology spec is a small dict (or JSON), shaped to match the simulator / synthetic
topology::

    {
      "nodes": [
        {"id": "core:p1:P", "site": "core", "device": "p1", "role": "P",
         "site_type": "core", "criticality": 0.9},
        ...
      ],
      "edges": [
        {"source": "dc1:pe-dc1:PE", "target": "core:p1:P", "kind": "link"},
        {"source": "br1:ce-br1:CE", "target": "hub1:pe-hub1:PE",
         "kind": "tunnel", "directed": true},
        ...
      ]
    }

Edges carry a ``kind`` (``link`` / ``tunnel`` / ``adjacency`` / ``session``). For
blast-radius the graph is treated as a *directed* digital twin where an edge
``A -> B`` means "a failure at A propagates to / affects B" (A is upstream of B).
Undirected physical links are expanded into both directions so reachability works
in either sense; a link/tunnel/adjacency may be marked ``directed: true`` to keep a
single propagation direction (e.g. core -> edge -> spoke).

The module is pure ``networkx`` + stdlib (CPU-only, offline). It imports types only
from :mod:`netra.contracts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from netra.contracts import DeviceRole, EntityRef, SiteType

# ---------------------------------------------------------------------------
# Node / edge attribute keys (kept as module constants so callers don't guess).
# ---------------------------------------------------------------------------
ATTR_SITE = "site"
ATTR_DEVICE = "device"
ATTR_ROLE = "role"
ATTR_SITE_TYPE = "site_type"
ATTR_CRITICALITY = "criticality"
ATTR_SERVICES = "services"
ATTR_SLAS = "slas"

EDGE_KIND = "kind"
EDGE_LINK = "link"
EDGE_TUNNEL = "tunnel"
EDGE_ADJACENCY = "adjacency"
EDGE_SESSION = "session"

# Default asset criticality by role when a node spec does not provide one.
# DC/core/RR infrastructure outranks hub PEs, which outrank branch CEs (research
# 07 A2.1: "DC-PE > hub-PE > branch-CE").
_DEFAULT_ROLE_CRITICALITY: dict[str, float] = {
    DeviceRole.RR.value: 0.95,
    DeviceRole.P.value: 0.90,
    DeviceRole.CONTROLLER.value: 0.95,
    DeviceRole.PE.value: 0.75,
    DeviceRole.CE.value: 0.45,
    DeviceRole.HOST.value: 0.30,
}

# Site-type multiplier layered on top of the role baseline.
_SITE_TYPE_WEIGHT: dict[str, float] = {
    SiteType.DATACENTER.value: 1.00,
    SiteType.CORE.value: 1.00,
    SiteType.HUB.value: 0.85,
    SiteType.BRANCH.value: 0.60,
}


def default_criticality(role: str | DeviceRole, site_type: str | SiteType | None) -> float:
    """Return a sensible [0,1] asset criticality for a node lacking an explicit one.

    Combines a role baseline with a site-type multiplier. Used by the risk engine
    as the ``AssetCriticality`` factor (research 07 A2.1).
    """
    role_v = role.value if isinstance(role, DeviceRole) else str(role)
    st_v = (
        site_type.value if isinstance(site_type, SiteType) else site_type
    )
    base = _DEFAULT_ROLE_CRITICALITY.get(role_v, 0.5)
    mult = _SITE_TYPE_WEIGHT.get(st_v, 1.0) if st_v else 1.0
    return round(min(1.0, base * mult), 4)


class TopologyGraph:
    """A directed topology digital twin with convenience accessors.

    Wraps a ``networkx.DiGraph`` whose nodes are entity ids and whose directed
    edges encode failure-propagation direction (``upstream -> downstream``). The
    correlation, RCA and blast-radius modules operate on this object.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.g: nx.DiGraph = graph

    # -- construction ------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "TopologyGraph":
        """Build a :class:`TopologyGraph` from a topology dict (see module docstring)."""
        g = nx.DiGraph()
        for node in spec.get("nodes", []):
            cls._add_node(g, node)
        for edge in spec.get("edges", []):
            cls._add_edge(g, edge)
        return cls(g)

    @classmethod
    def from_json(cls, path: str | Path) -> "TopologyGraph":
        """Load a topology spec from a JSON file and build the graph."""
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        return cls.from_spec(spec)

    @staticmethod
    def _add_node(g: nx.DiGraph, node: Mapping[str, Any]) -> None:
        node_id = node.get("id") or node.get("entity_id")
        if not node_id:
            raise ValueError(f"topology node missing 'id': {node!r}")
        role = node.get("role", DeviceRole.HOST.value)
        site_type = node.get("site_type")
        crit = node.get("criticality")
        if crit is None:
            crit = default_criticality(role, site_type)
        g.add_node(
            node_id,
            **{
                ATTR_SITE: node.get("site", node_id.split(":")[0]),
                ATTR_DEVICE: node.get("device", ""),
                ATTR_ROLE: role.value if isinstance(role, DeviceRole) else str(role),
                ATTR_SITE_TYPE: site_type.value
                if isinstance(site_type, SiteType)
                else site_type,
                ATTR_CRITICALITY: float(crit),
                ATTR_SERVICES: list(node.get("services", []) or []),
                ATTR_SLAS: list(node.get("slas", []) or []),
            },
        )

    @staticmethod
    def _add_edge(g: nx.DiGraph, edge: Mapping[str, Any]) -> None:
        src = edge.get("source") or edge.get("src") or edge.get("u")
        dst = edge.get("target") or edge.get("dst") or edge.get("v")
        if not src or not dst:
            raise ValueError(f"topology edge missing source/target: {edge!r}")
        # Auto-add endpoints not previously declared (keeps loaders forgiving).
        for endpoint in (src, dst):
            if endpoint not in g:
                g.add_node(
                    endpoint,
                    **{
                        ATTR_SITE: endpoint.split(":")[0],
                        ATTR_DEVICE: "",
                        ATTR_ROLE: DeviceRole.HOST.value,
                        ATTR_SITE_TYPE: None,
                        ATTR_CRITICALITY: 0.5,
                        ATTR_SERVICES: [],
                        ATTR_SLAS: [],
                    },
                )
        kind = edge.get(EDGE_KIND, EDGE_LINK)
        attrs = {EDGE_KIND: kind}
        # carry through any extra metadata (e.g. capacity, vrf) for the UI.
        for k, v in edge.items():
            if k not in {"source", "target", "src", "dst", "u", "v", "directed"}:
                attrs.setdefault(k, v)
        g.add_edge(src, dst, **attrs)
        # Undirected by default → also add the reverse so reachability is
        # symmetric on physical media; a ``directed: true`` edge keeps one way.
        if not edge.get("directed", False):
            g.add_edge(dst, src, **dict(attrs))

    # -- accessors ---------------------------------------------------------
    def has_node(self, entity_id: str) -> bool:
        return entity_id in self.g

    def nodes(self) -> Iterable[str]:
        return self.g.nodes

    def criticality(self, entity_id: str) -> float:
        """Asset criticality [0,1] for an entity (default 0.5 if unknown)."""
        if entity_id not in self.g:
            return 0.5
        return float(self.g.nodes[entity_id].get(ATTR_CRITICALITY, 0.5))

    def site_of(self, entity_id: str) -> str | None:
        if entity_id not in self.g:
            return None
        return self.g.nodes[entity_id].get(ATTR_SITE)

    def services_of(self, entity_id: str) -> list[str]:
        if entity_id not in self.g:
            return []
        return list(self.g.nodes[entity_id].get(ATTR_SERVICES, []))

    def slas_of(self, entity_id: str) -> list[str]:
        if entity_id not in self.g:
            return []
        return list(self.g.nodes[entity_id].get(ATTR_SLAS, []))

    def entity_ref(self, entity_id: str) -> EntityRef:
        """Reconstruct an :class:`EntityRef` from a graph node's attributes.

        Falls back to parsing the colon-delimited id when attributes are missing,
        so an entity that only appeared in events (not the spec) still resolves.
        """
        if entity_id in self.g:
            data = self.g.nodes[entity_id]
            role_v = data.get(ATTR_ROLE, DeviceRole.HOST.value)
            st_v = data.get(ATTR_SITE_TYPE)
            return EntityRef(
                entity_id=entity_id,
                site=data.get(ATTR_SITE) or entity_id.split(":")[0],
                device=data.get(ATTR_DEVICE) or _device_from_id(entity_id),
                role=_coerce_role(role_v),
                site_type=_coerce_site_type(st_v),
                sub=_sub_from_id(entity_id),
            )
        return entity_ref_from_id(entity_id)

    def map_to_node(self, entity_id: str) -> str | None:
        """Resolve an (often sub-entity) id to the nearest graph node.

        Events fire on interfaces/tunnels/peers (e.g. ``hub1:pe-hub1:PE:eth1``)
        but the topology graph is usually keyed at device granularity
        (``hub1:pe-hub1:PE``). This maps a fine-grained id onto the device node it
        belongs to, so per-interface anomalies attach to the right graph vertex.
        Returns ``None`` if nothing matches.
        """
        if entity_id in self.g:
            return entity_id
        # progressively strip trailing ``:sub`` segments until a node matches.
        parts = entity_id.split(":")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = ":".join(parts[:cut])
            if candidate in self.g:
                return candidate
        # last resort: a node sharing the same site+device prefix.
        if len(parts) >= 2:
            prefix = ":".join(parts[:2])
            for n in self.g.nodes:
                if n.startswith(prefix):
                    return n
        return None


# ---------------------------------------------------------------------------
# Small id-parsing helpers (EntityRef convention: "<site>:<device>:<role>[:<sub>]").
# ---------------------------------------------------------------------------
def _device_from_id(entity_id: str) -> str:
    parts = entity_id.split(":")
    return parts[1] if len(parts) >= 2 else parts[0]


def _sub_from_id(entity_id: str) -> str | None:
    parts = entity_id.split(":")
    return ":".join(parts[3:]) if len(parts) >= 4 else None


def _coerce_role(role_v: Any) -> DeviceRole:
    if isinstance(role_v, DeviceRole):
        return role_v
    try:
        return DeviceRole(role_v)
    except ValueError:
        return DeviceRole.HOST


def _coerce_site_type(st_v: Any) -> SiteType | None:
    if st_v is None:
        return None
    if isinstance(st_v, SiteType):
        return st_v
    try:
        return SiteType(st_v)
    except ValueError:
        return None


def entity_ref_from_id(entity_id: str) -> EntityRef:
    """Best-effort :class:`EntityRef` purely from a colon-delimited entity id."""
    parts = entity_id.split(":")
    site = parts[0] if parts else entity_id
    device = parts[1] if len(parts) >= 2 else site
    role = _coerce_role(parts[2]) if len(parts) >= 3 else DeviceRole.HOST
    sub = ":".join(parts[3:]) if len(parts) >= 4 else None
    return EntityRef(entity_id=entity_id, site=site, device=device, role=role, sub=sub)


# ---------------------------------------------------------------------------
# Built-in demo topology — a compact 5-site SD-WAN-over-MPLS network matching the
# reference topology in ARCHITECTURE.md §Phase 1 (DC + hub + 3 branches + core).
# Used by tests and the API demo when no sim/synthetic topology is supplied.
# ---------------------------------------------------------------------------
def demo_topology_spec() -> dict[str, Any]:
    """Return the built-in demo topology spec (dict).

    Layout (directed edges point *downstream* in failure-propagation sense):

        controller ─┐
        RR ─────────┤ (control plane reflected to PEs)
        DC PEs ── core (P1,P2,P3) ── HUB PE ── {BR1,BR2,BR3} CEs
                                       │
                              overlay tunnels hub→spoke

    A failing core P-router therefore reaches the hub PE and all three branch
    CEs (large blast radius); a flapping RR reaches every PE; a branch CE reaches
    nothing downstream (leaf).
    """
    nodes = [
        # provider core
        {"id": "core:p1:P", "site": "core", "device": "p1", "role": "P",
         "site_type": "core", "criticality": 0.92},
        {"id": "core:p2:P", "site": "core", "device": "p2", "role": "P",
         "site_type": "core", "criticality": 0.90},
        {"id": "core:p3:P", "site": "core", "device": "p3", "role": "P",
         "site_type": "core", "criticality": 0.88},
        # route reflector + controller
        {"id": "core:rr1:RR", "site": "core", "device": "rr1", "role": "RR",
         "site_type": "core", "criticality": 0.95},
        {"id": "dc1:ctl1:controller", "site": "dc1", "device": "ctl1",
         "role": "controller", "site_type": "datacenter", "criticality": 0.97},
        # datacenter PEs
        {"id": "dc1:pe-dc1:PE", "site": "dc1", "device": "pe-dc1", "role": "PE",
         "site_type": "datacenter", "criticality": 0.85,
         "services": ["CORP", "OT"], "slas": ["sla-corp-gold", "sla-ot-gold"]},
        {"id": "dc1:pe-dc2:PE", "site": "dc1", "device": "pe-dc2", "role": "PE",
         "site_type": "datacenter", "criticality": 0.84,
         "services": ["CORP"], "slas": ["sla-corp-gold"]},
        # hub PE (scenario A target)
        {"id": "hub1:pe-hub1:PE", "site": "hub1", "device": "pe-hub1", "role": "PE",
         "site_type": "hub", "criticality": 0.80,
         "services": ["CORP", "OT"], "slas": ["sla-corp-silver"]},
        # branch CEs (spokes)
        {"id": "br1:ce-br1:CE", "site": "br1", "device": "ce-br1", "role": "CE",
         "site_type": "branch", "criticality": 0.50,
         "services": ["CORP"], "slas": ["sla-corp-bronze"]},
        {"id": "br2:ce-br2:CE", "site": "br2", "device": "ce-br2", "role": "CE",
         "site_type": "branch", "criticality": 0.48,
         "services": ["CORP"], "slas": ["sla-corp-bronze"]},
        {"id": "br3:ce-br3:CE", "site": "br3", "device": "ce-br3", "role": "CE",
         "site_type": "branch", "criticality": 0.46,
         "services": ["OT"], "slas": ["sla-ot-bronze"]},
    ]
    edges = [
        # controller pushes policy to all PEs (drift fans out from here).
        {"source": "dc1:ctl1:controller", "target": "dc1:pe-dc1:PE", "kind": "session", "directed": True},
        {"source": "dc1:ctl1:controller", "target": "dc1:pe-dc2:PE", "kind": "session", "directed": True},
        {"source": "dc1:ctl1:controller", "target": "hub1:pe-hub1:PE", "kind": "session", "directed": True},
        # RR reflects VPNv4 to every PE (RR flap → all PEs).
        {"source": "core:rr1:RR", "target": "dc1:pe-dc1:PE", "kind": "session", "directed": True},
        {"source": "core:rr1:RR", "target": "dc1:pe-dc2:PE", "kind": "session", "directed": True},
        {"source": "core:rr1:RR", "target": "hub1:pe-hub1:PE", "kind": "session", "directed": True},
        # MPLS core mesh (physical links, bidirectional).
        {"source": "core:p1:P", "target": "core:p2:P", "kind": "link"},
        {"source": "core:p2:P", "target": "core:p3:P", "kind": "link"},
        {"source": "core:p1:P", "target": "core:p3:P", "kind": "link"},
        # PEs attach to the core (directed downstream: core → edge).
        {"source": "core:p1:P", "target": "dc1:pe-dc1:PE", "kind": "link", "directed": True},
        {"source": "core:p1:P", "target": "dc1:pe-dc2:PE", "kind": "link", "directed": True},
        {"source": "core:p2:P", "target": "hub1:pe-hub1:PE", "kind": "link", "directed": True},
        {"source": "core:p3:P", "target": "hub1:pe-hub1:PE", "kind": "link", "directed": True},
        # hub → spoke overlay tunnels (directed downstream: hub → branch).
        {"source": "hub1:pe-hub1:PE", "target": "br1:ce-br1:CE", "kind": "tunnel", "directed": True},
        {"source": "hub1:pe-hub1:PE", "target": "br2:ce-br2:CE", "kind": "tunnel", "directed": True},
        {"source": "hub1:pe-hub1:PE", "target": "br3:ce-br3:CE", "kind": "tunnel", "directed": True},
    ]
    return {"nodes": nodes, "edges": edges}


def build_demo_graph() -> TopologyGraph:
    """Convenience: build the built-in demo :class:`TopologyGraph`."""
    return TopologyGraph.from_spec(demo_topology_spec())


__all__ = [
    "TopologyGraph",
    "build_demo_graph",
    "demo_topology_spec",
    "default_criticality",
    "entity_ref_from_id",
    # attribute key constants
    "ATTR_SITE",
    "ATTR_DEVICE",
    "ATTR_ROLE",
    "ATTR_SITE_TYPE",
    "ATTR_CRITICALITY",
    "ATTR_SERVICES",
    "ATTR_SLAS",
    "EDGE_KIND",
    "EDGE_LINK",
    "EDGE_TUNNEL",
    "EDGE_ADJACENCY",
    "EDGE_SESSION",
]
