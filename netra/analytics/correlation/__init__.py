"""netra.analytics.correlation — graph event-correlation + blast-radius (WS4).

The NetworkX digital twin of the topology, temporal+topological event
correlation (sliding window + WCC/SCC over the failure subgraph, with alarm
compression), root-cause ranking (centrality x earliest-onset x Granger), and
deterministic blast-radius via BFS reachability intersected with NetFlow.
Feeds ``netra.contracts.Incident`` (BlastRadius, root_cause_entity).

Builder: ``graph.py`` (digital twin), ``correlate.py`` (grouping + RCA),
``blast_radius.py`` (reachability -> BlastRadius). Pure ``networkx`` + Granger,
CPU-only. The copilot must NOT recompute blast radius — it is computed here.
"""
