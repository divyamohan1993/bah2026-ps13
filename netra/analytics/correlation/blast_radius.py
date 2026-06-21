"""Deterministic blast-radius computation via graph reachability ∩ NetFlow.

"Blast radius" = the set of sites / services / SLAs / flows reachable *downstream*
of a failing entity under the failure-propagation relation. This is a graph
traversal / reachability problem of complexity O(V+E) (research 07 A1.4) — it is
**computed here, deterministically**, and the copilot must NOT re-guess it.

Mechanism (research 07 A1.4):
  * ``nx.descendants(G, root)`` → all downstream-reachable nodes (the affected set).
  * ``nx.single_source_shortest_path_length(G, root)`` → hop distance per node,
    which doubles as a propagation-time / urgency proxy.
  * Intersect with the NetFlow flow set (flows whose ingress/egress touch the
    failure or any affected device) → a concrete count of affected flows / SLAs /
    sites — the impact number operators care about.

Produces a :class:`netra.contracts.BlastRadius`. CPU-only, pure ``networkx``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import networkx as nx

from netra.contracts import BlastRadius, FlowRecord

from .graph import (
    ATTR_DEVICE,
    ATTR_SITE,
    ATTR_SLAS,
    ATTR_SERVICES,
    TopologyGraph,
)


def compute_blast_radius(
    topology: TopologyGraph,
    root_entity_id: str,
    *,
    flows: Sequence[FlowRecord] | None = None,
    max_hops: int | None = None,
    include_root: bool = False,
    normalization_basis: int | None = None,
) -> BlastRadius:
    """Compute the deterministic blast radius downstream of ``root_entity_id``.

    Parameters
    ----------
    topology:
        The :class:`TopologyGraph` digital twin.
    root_entity_id:
        The (root-cause) entity whose downstream impact is measured. A
        fine-grained id (interface/tunnel) is mapped to its device node first.
    flows:
        Optional NetFlow/IPFIX records; flows touching the failure or any affected
        device are counted and their VRFs/services folded into the affected set.
    max_hops:
        If set, only count nodes within this many hops (bounds an enormous graph).
    include_root:
        Whether the root node itself is listed among affected devices/sites.
    normalization_basis:
        Denominator for ``normalized_size`` (default: total node count of the
        graph). Lets the risk engine scale blast radius consistently to [0,1].

    Returns
    -------
    BlastRadius
        Affected sites / devices / services / SLAs / flow count + per-node hop
        distances + a normalized [0,1] size for risk scoring.
    """
    node = topology.map_to_node(root_entity_id) or root_entity_id
    g: nx.DiGraph = topology.g

    if node not in g:
        # Unknown root: empty, well-formed blast radius (size 0).
        return BlastRadius(normalized_size=0.0)

    # 1) reachable downstream set + hop distances ---------------------------
    if max_hops is not None:
        hop_map = nx.single_source_shortest_path_length(g, node, cutoff=max_hops)
    else:
        hop_map = nx.single_source_shortest_path_length(g, node)
    # drop the root itself from the *distance* map's "affected" view unless asked.
    affected_ids = {n for n in hop_map if n != node or include_root}

    affected_sites: set[str] = set()
    affected_devices: set[str] = set()
    affected_services: set[str] = set()
    affected_slas: set[str] = set()
    hop_distances: dict[str, int] = {}

    for n in affected_ids:
        data = g.nodes[n]
        hop_distances[n] = int(hop_map[n])
        site = data.get(ATTR_SITE)
        if site:
            affected_sites.add(site)
        dev = data.get(ATTR_DEVICE) or n
        affected_devices.add(dev)
        for svc in data.get(ATTR_SERVICES, []) or []:
            affected_services.add(svc)
        for sla in data.get(ATTR_SLAS, []) or []:
            affected_slas.add(sla)

    # 2) intersect with NetFlow --------------------------------------------
    flow_count: int | None = None
    if flows is not None:
        affected_site_set = set(affected_sites)
        if include_root:
            affected_site_set.add(g.nodes[node].get(ATTR_SITE))
        # Always consider the failure's own site as part of the flow footprint.
        root_site = g.nodes[node].get(ATTR_SITE)
        affected_device_set = set(affected_devices)
        root_device = g.nodes[node].get(ATTR_DEVICE) or node
        affected_device_set.add(root_device)

        count = 0
        for fr in flows:
            touches = (
                fr.site == root_site
                or fr.site in affected_site_set
                or fr.device == root_device
                or fr.device in affected_device_set
            )
            if touches:
                count += 1
                if fr.vrf:
                    affected_services.add(fr.vrf)
        flow_count = count

    # 3) normalized size ----------------------------------------------------
    basis = normalization_basis if normalization_basis else max(g.number_of_nodes() - 1, 1)
    normalized = min(1.0, len(affected_ids) / basis) if basis else 0.0

    return BlastRadius(
        affected_sites=sorted(affected_sites),
        affected_devices=sorted(affected_devices),
        affected_services_or_vpns=sorted(affected_services),
        affected_slas=sorted(affected_slas),
        affected_flow_count=flow_count,
        hop_distances=hop_distances,
        normalized_size=round(normalized, 4),
    )


def blast_urgency_factor(blast: BlastRadius) -> float:
    """Map a :class:`BlastRadius` to a [0,1] urgency contribution for risk scoring.

    Uses the normalized size when present; otherwise derives one from the count of
    affected sites with a saturating curve (so impact stops growing linearly once
    a handful of sites are hit). This is the ``BlastRadius`` factor of the
    product-form risk score (research 07 A2.1).
    """
    if blast.normalized_size is not None:
        return float(blast.normalized_size)
    n_sites = len(blast.affected_sites)
    if n_sites <= 0:
        return 0.0
    # 1 - exp(-n/3): ~0.28 at 1 site, ~0.49 at 2, ~0.86 at 6.
    return round(1.0 - math.exp(-n_sites / 3.0), 4)


__all__ = ["compute_blast_radius", "blast_urgency_factor"]
