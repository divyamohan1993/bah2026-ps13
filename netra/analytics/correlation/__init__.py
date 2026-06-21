"""netra.analytics.correlation — graph event-correlation + blast-radius (WS4).

The NetworkX digital twin of the topology, temporal+topological event
correlation (sliding window + WCC/SCC over the failure subgraph, with alarm
compression), root-cause ranking (centrality x earliest-onset x Granger), and
deterministic blast-radius via BFS reachability intersected with NetFlow.
Feeds ``netra.contracts.Incident`` (BlastRadius, root_cause_entity).

Builder: ``graph.py`` (digital twin), ``correlate.py`` (grouping + RCA),
``rca.py`` (root-cause ranking), ``blast_radius.py`` (reachability -> BlastRadius).
Pure ``networkx`` + Granger, CPU-only. The copilot must NOT recompute blast radius
— it is computed here.
"""

from __future__ import annotations

from .blast_radius import blast_urgency_factor, compute_blast_radius
from .correlate import (
    CorrelationEvent,
    IncidentGroup,
    assemble_incident,
    build_contributing_signals,
    correlate_events,
    correlate_to_incidents,
    dedup_events,
    normalize_events,
)
from .graph import (
    TopologyGraph,
    build_demo_graph,
    default_criticality,
    demo_topology_spec,
    entity_ref_from_id,
)
from .rca import (
    RootCauseCandidate,
    build_hypothesis,
    granger_causal_scores,
    onset_scores,
    rank_root_causes,
    topology_centrality,
)

__all__ = [
    # graph
    "TopologyGraph",
    "build_demo_graph",
    "demo_topology_spec",
    "default_criticality",
    "entity_ref_from_id",
    # blast radius
    "compute_blast_radius",
    "blast_urgency_factor",
    # rca
    "RootCauseCandidate",
    "rank_root_causes",
    "topology_centrality",
    "onset_scores",
    "granger_causal_scores",
    "build_hypothesis",
    # correlate
    "CorrelationEvent",
    "IncidentGroup",
    "normalize_events",
    "dedup_events",
    "correlate_events",
    "assemble_incident",
    "build_contributing_signals",
    "correlate_to_incidents",
]
