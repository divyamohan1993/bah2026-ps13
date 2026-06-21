"""GraphRAG-lite — deterministic topology + blast-radius facts for the prompt.

The topology is a graph; the copilot must reason about *where* and *how bad*
(affected sites/VPNs/devices) — and that scope is computed **deterministically**,
never guessed by the LLM (research 06 §6, architecture §4 Q1). This module turns
two deterministic sources into grounded, citable context lines:

  1. The :class:`~netra.contracts.BlastRadius` already computed by the
     correlation/risk workstream (WS4) — we **consume the contract type as an
     input**; we do NOT import WS4 code. This is the authoritative affected scope.
  2. The corpus topology (``corpus/topology/*.json``) loaded into a graph, used
     to (a) describe the root-cause device and (b) derive a fallback blast radius
     by reachability when an explicit :class:`BlastRadius` is not supplied.

A ``networkx`` digital twin is built when available (core tier); a tiny
pure-Python reachability fallback keeps it working with zero deps. The output is
a list of ``(fact_id, text)`` pairs the orchestrator injects into the prompt and
adds to the citation universe, so the copilot's affected-scope claims are always
backed by a deterministic graph fact.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from netra.contracts import AffectedScope, BlastRadius, EntityRef

from .ingest import DEFAULT_CORPUS_DIR


class TopologyGraph:
    """An in-memory topology digital twin (networkx if present, else pure-python).

    Built from the corpus topology JSON. Provides device lookup and BFS
    reachability (the deterministic blast-radius primitive). It is read-only and
    fully offline.
    """

    def __init__(self) -> None:
        self._nx = None
        self._graph = None
        self._adj: dict[str, list[str]] = {}
        self._device_meta: dict[str, dict] = {}
        self._device_site: dict[str, str] = {}
        self._device_vrfs: dict[str, list[str]] = {}

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_corpus(cls, corpus_dir: str | Path | None = None) -> TopologyGraph:
        """Load every ``topology/*.json`` under the corpus into one graph."""
        g = cls()
        root = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        topo_dir = root / "topology"
        if not topo_dir.exists():
            return g
        for path in sorted(topo_dir.glob("*.json")):
            try:
                g._ingest(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        g._finalize()
        return g

    def _ingest(self, data: dict) -> None:
        for dev in data.get("devices", []):
            eid = dev.get("entity_id") or dev.get("device")
            if not eid:
                continue
            self._device_meta[eid] = dev
            self._device_site[eid] = dev.get("site", "")
            self._device_vrfs[eid] = list(dev.get("vrfs", []) or [])
            self._adj.setdefault(eid, [])
        for link in data.get("links", []):
            a, b = link.get("from"), link.get("to")
            if not a or not b:
                continue
            self._adj.setdefault(a, []).append(b)
            self._adj.setdefault(b, []).append(a)  # treat topology as undirected

    def _finalize(self) -> None:
        """Build a networkx graph mirror when the library is available."""
        try:  # core tier, but guard anyway
            import networkx as nx  # type: ignore

            self._nx = nx
            self._graph = nx.Graph()
            for eid, meta in self._device_meta.items():
                self._graph.add_node(eid, **{k: v for k, v in meta.items() if k != "entity_id"})
            for a, neighbours in self._adj.items():
                for b in neighbours:
                    self._graph.add_edge(a, b)
        except Exception:
            self._nx = None
            self._graph = None

    # -- queries ----------------------------------------------------------------
    def has_device(self, entity_id: str) -> bool:
        return entity_id in self._device_meta

    def describe_device(self, entity_id: str) -> str | None:
        """One-line grounded description of a device, or None if unknown."""
        dev = self._device_meta.get(entity_id)
        if not dev:
            return None
        vrfs = self._device_vrfs.get(entity_id) or []
        vrf_txt = f", VRFs {', '.join(vrfs)}" if vrfs else ""
        neighbours = self._adj.get(entity_id, [])
        nbr_txt = (
            f"; directly connected to {', '.join(sorted(set(neighbours))[:6])}"
            if neighbours
            else ""
        )
        return (
            f"{dev.get('device')} is a {dev.get('role')} at site "
            f"{dev.get('site')}{vrf_txt}{nbr_txt}."
        )

    def reachable(self, entity_id: str, max_hops: int = 2) -> dict[str, int]:
        """BFS reachable set from ``entity_id`` -> {entity_id: hop_distance}.

        The deterministic blast-radius primitive: which devices are within
        ``max_hops`` of the failing node. Uses networkx if present, else a small
        BFS over the adjacency map.
        """
        if entity_id not in self._adj:
            return {}
        if self._graph is not None and self._nx is not None:
            try:
                lengths = self._nx.single_source_shortest_path_length(
                    self._graph, entity_id, cutoff=max_hops
                )
                return {k: int(v) for k, v in lengths.items() if k != entity_id}
            except Exception:
                pass
        # Pure-python BFS fallback.
        dist: dict[str, int] = {entity_id: 0}
        q: deque[str] = deque([entity_id])
        while q:
            cur = q.popleft()
            if dist[cur] >= max_hops:
                continue
            for nb in self._adj.get(cur, []):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        dist.pop(entity_id, None)
        return dist

    def site_of(self, entity_id: str) -> str:
        return self._device_site.get(entity_id, "")

    def vrfs_of(self, entity_id: str) -> list[str]:
        return self._device_vrfs.get(entity_id, [])

    def __len__(self) -> int:
        return len(self._device_meta)


def affected_scope_from_blast_radius(br: BlastRadius | None) -> AffectedScope:
    """Project a contract :class:`BlastRadius` into the copilot ``AffectedScope``.

    Deterministic mapping (no inference): the copilot reports exactly what WS4
    computed.
    """
    if br is None:
        return AffectedScope()
    return AffectedScope(
        sites=list(br.affected_sites),
        devices=list(br.affected_devices),
        services_or_vpns=list(br.affected_services_or_vpns),
    )


def graph_facts(
    *,
    root_cause_entity: EntityRef | None = None,
    blast_radius: BlastRadius | None = None,
    topology: TopologyGraph | None = None,
    max_hops: int = 2,
) -> list[tuple[str, str]]:
    """Return deterministic ``(fact_id, text)`` topology facts for the prompt.

    Combines the supplied :class:`BlastRadius` (authoritative, from WS4) and the
    corpus topology graph (for the root-cause device description and, if no blast
    radius was supplied, a reachability-derived fallback scope). Every fact gets a
    stable ``graph:*`` id so it can be cited.
    """
    facts: list[tuple[str, str]] = []

    # 1) Root-cause device description from the topology graph.
    if root_cause_entity is not None and topology is not None:
        desc = topology.describe_device(root_cause_entity.entity_id)
        if desc:
            facts.append(("graph:root-cause", f"Root-cause node: {desc}"))

    # 2) Authoritative affected scope from the contract BlastRadius.
    if blast_radius is not None:
        scope_bits = []
        if blast_radius.affected_sites:
            scope_bits.append(f"sites {', '.join(blast_radius.affected_sites)}")
        if blast_radius.affected_devices:
            scope_bits.append(f"devices {', '.join(blast_radius.affected_devices)}")
        if blast_radius.affected_services_or_vpns:
            scope_bits.append(
                f"services/VPNs {', '.join(blast_radius.affected_services_or_vpns)}"
            )
        if blast_radius.affected_slas:
            scope_bits.append(f"SLAs at risk: {', '.join(blast_radius.affected_slas)}")
        if blast_radius.affected_flow_count is not None:
            scope_bits.append(f"{blast_radius.affected_flow_count} flows traversing")
        if scope_bits:
            facts.append(
                (
                    "graph:blast-radius",
                    "Deterministic blast radius (graph reachability ∩ NetFlow): "
                    + "; ".join(scope_bits)
                    + ".",
                )
            )

    # 3) Fallback reachability scope when no BlastRadius was supplied.
    elif root_cause_entity is not None and topology is not None:
        reach = topology.reachable(root_cause_entity.entity_id, max_hops=max_hops)
        if reach:
            within = ", ".join(
                f"{eid}({hops}h)" for eid, hops in sorted(reach.items(), key=lambda t: t[1])
            )
            facts.append(
                (
                    "graph:reachability",
                    f"Devices within {max_hops} hops of "
                    f"{root_cause_entity.entity_id}: {within}.",
                )
            )

    return facts


__all__ = [
    "TopologyGraph",
    "graph_facts",
    "affected_scope_from_blast_radius",
]
