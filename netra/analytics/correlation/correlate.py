"""Temporal + topological event correlation → one ranked :class:`Incident`.

Modern AIOps does not treat alarms as independent rows: it correlates along **two
axes simultaneously** (research 07 A1.1):

  * **temporal** — events inside a sliding window (``window_seconds``) are
    candidates to belong to the same incident.
  * **topological** — events on nodes adjacent/reachable in the topology digital
    twin are candidates to belong to the same incident.

This module:
  1. **Deduplicates / compresses** raw events (``node|kind|severity`` hash within
     the window) — alarm compression (research 07 A1.5).
  2. **Groups** co-occurring ``AnomalyScore`` / ``FusedRisk`` events into incident
     groups using a sliding time window + **connected components over the failure
     subgraph** (so N symptoms collapse to 1 incident and two unrelated incidents
     do not merge).
  3. **Ranks the root cause** inside each group (delegates to :mod:`rca`:
     centrality × earliest-onset × Granger causal score).
  4. **Assembles** an :class:`netra.contracts.Incident` (correlated entities,
     root-cause hypothesis, contributing signals, blast radius, compression ratio).

The risk *severity bucketing* and *flap suppression* live in
:mod:`netra.analytics.risk`; this module produces the structured incident and its
``FusedRisk`` (the group's representative risk). CPU-only, pure ``networkx``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import networkx as nx

from netra.contracts import (
    AnomalyScore,
    BlastRadius,
    ContributingSignal,
    Direction,
    EntityRef,
    FlowRecord,
    FusedRisk,
    Incident,
    IssueType,
    Severity,
)

from .blast_radius import compute_blast_radius
from .graph import TopologyGraph
from .rca import build_hypothesis, rank_root_causes

# ---------------------------------------------------------------------------
# Event abstraction — a thin, uniform view over the heterogeneous inputs so the
# grouping logic does not care whether a signal came from an AnomalyScore or a
# FusedRisk.
# ---------------------------------------------------------------------------
@dataclass
class CorrelationEvent:
    """A uniform correlation event derived from an AnomalyScore or FusedRisk."""

    entity_id: str
    timestamp: datetime
    score: float
    metric: str
    method: str
    kind: str  # dedup bucket label: "<metric>|<family>" or similar
    issue: IssueType = IssueType.NONE
    entity_ref: EntityRef | None = None
    fused: FusedRisk | None = None
    raw: object | None = None


@dataclass
class IncidentGroup:
    """An intermediate correlated group prior to severity/playbook finalisation."""

    events: list[CorrelationEvent] = field(default_factory=list)
    node_ids: set[str] = field(default_factory=set)  # graph nodes touched
    raw_alarm_count: int = 0  # pre-dedup alarm count (for compression ratio)

    @property
    def window_start(self) -> datetime:
        return min(e.timestamp for e in self.events)

    @property
    def window_end(self) -> datetime:
        return max(e.timestamp for e in self.events)

    @property
    def entity_ids(self) -> list[str]:
        # stable, de-duplicated by earliest onset
        first_seen: dict[str, datetime] = {}
        for e in self.events:
            t = first_seen.get(e.entity_id)
            if t is None or e.timestamp < t:
                first_seen[e.entity_id] = e.timestamp
        return sorted(first_seen, key=lambda k: first_seen[k])


# ---------------------------------------------------------------------------
# Step 1 — normalise heterogeneous inputs into CorrelationEvents (+ dedup)
# ---------------------------------------------------------------------------
def _event_from_anomaly(a: AnomalyScore) -> CorrelationEvent:
    return CorrelationEvent(
        entity_id=a.entity.entity_id,
        timestamp=_as_utc(a.timestamp),
        score=float(a.normalized_score),
        metric=a.metric,
        method=a.method,
        kind=f"{a.metric}|{a.family.value}",
        entity_ref=a.entity,
        raw=a,
    )


def _event_from_fused(f: FusedRisk) -> CorrelationEvent:
    return CorrelationEvent(
        entity_id=f.entity.entity_id,
        timestamp=_as_utc(f.timestamp),
        score=float(f.risk_score),
        metric="fused_risk",
        method="fusion",
        kind=f"fused|{f.predicted_issue.value}",
        issue=f.predicted_issue,
        entity_ref=f.entity,
        fused=f,
        raw=f,
    )


def normalize_events(
    anomalies: Sequence[AnomalyScore] | None = None,
    fused: Sequence[FusedRisk] | None = None,
    *,
    min_anomaly_score: float = 0.0,
) -> list[CorrelationEvent]:
    """Flatten AnomalyScore + FusedRisk inputs into a single event list.

    Only anomalies with ``is_anomaly`` true (or ``normalized_score`` above
    ``min_anomaly_score``) and fused risks with ``risk_score>0`` are considered —
    healthy readings are not events.
    """
    events: list[CorrelationEvent] = []
    for a in anomalies or []:
        if a.is_anomaly or a.normalized_score > min_anomaly_score:
            events.append(_event_from_anomaly(a))
    for f in fused or []:
        if f.risk_score > 0:
            events.append(_event_from_fused(f))
    return events


def dedup_events(events: Sequence[CorrelationEvent]) -> tuple[list[CorrelationEvent], int]:
    """Compress identical alarms (same ``entity|kind``) to the earliest occurrence.

    Returns ``(deduped_events, raw_count)`` where ``raw_count`` is the pre-dedup
    total used for the alarm compression ratio (research 07 A1.5).
    """
    raw_count = len(events)
    best: dict[tuple[str, str], CorrelationEvent] = {}
    for e in events:
        key = (e.entity_id, e.kind)
        cur = best.get(key)
        if cur is None or e.timestamp < cur.timestamp:
            best[key] = e
    return list(best.values()), raw_count


# ---------------------------------------------------------------------------
# Step 2 — temporal + topological grouping
# ---------------------------------------------------------------------------
def correlate_events(
    events: Sequence[CorrelationEvent],
    topology: TopologyGraph,
    *,
    window_seconds: float = 300.0,
    max_topo_distance: int = 2,
) -> list[IncidentGroup]:
    """Group events into incidents by time proximity AND topological proximity.

    Two events join the same incident iff their onset times are within
    ``window_seconds`` **and** their mapped graph nodes are within
    ``max_topo_distance`` hops (in the undirected sense) of each other. This is
    implemented as connected components of a graph whose edges encode "temporally
    close + topologically close", so a chain of related symptoms forms one
    component while independent incidents stay separate (research 07 A1.4 WCC/SCC).
    """
    if not events:
        return []

    deduped, raw_count = dedup_events(events)
    # Map each event to a topology node (fine-grained ids → device node).
    node_of: dict[int, str | None] = {}
    for idx, e in enumerate(deduped):
        node_of[idx] = topology.map_to_node(e.entity_id)

    # Precompute pairwise topological closeness using an undirected view for
    # "are these symptoms near each other on the network" (direction-agnostic).
    undirected = topology.g.to_undirected(as_view=True)

    link = nx.Graph()
    link.add_nodes_from(range(len(deduped)))
    for i in range(len(deduped)):
        for j in range(i + 1, len(deduped)):
            dt = abs((deduped[i].timestamp - deduped[j].timestamp).total_seconds())
            if dt > window_seconds:
                continue
            ni, nj = node_of[i], node_of[j]
            if _topologically_close(undirected, ni, nj, max_topo_distance):
                link.add_edge(i, j)

    groups: list[IncidentGroup] = []
    for comp in nx.connected_components(link):
        grp = IncidentGroup()
        for idx in comp:
            grp.events.append(deduped[idx])
            n = node_of[idx]
            if n is not None:
                grp.node_ids.add(n)
        # apportion the raw-alarm total across groups proportionally to size so
        # the per-incident compression ratio is meaningful.
        grp.raw_alarm_count = 0
        groups.append(grp)

    # distribute the original raw alarm count across groups by their event share.
    _apportion_raw_counts(groups, deduped, events, raw_count)
    # largest / most-severe groups first
    groups.sort(key=lambda g: (len(g.events), _group_peak(g)), reverse=True)
    return groups


def _apportion_raw_counts(
    groups: list[IncidentGroup],
    deduped: Sequence[CorrelationEvent],
    original: Sequence[CorrelationEvent],
    raw_count: int,
) -> None:
    """Assign each group the count of *original* (pre-dedup) events it represents."""
    # map (entity,kind) -> group index
    owner: dict[tuple[str, str], int] = {}
    for gi, g in enumerate(groups):
        for e in g.events:
            owner[(e.entity_id, e.kind)] = gi
    counts = [0] * len(groups)
    for e in original:
        gi = owner.get((e.entity_id, e.kind))
        if gi is not None:
            counts[gi] += 1
    for gi, g in enumerate(groups):
        # at least the number of deduped events it holds.
        g.raw_alarm_count = max(counts[gi], len(g.events))


def _topologically_close(
    undirected: nx.Graph, a: str | None, b: str | None, max_distance: int
) -> bool:
    """True if nodes a,b are within ``max_distance`` hops (or either is unknown)."""
    if a is None or b is None:
        # An event we could not place on the graph still correlates temporally
        # (better to over-group an unplaceable symptom than to drop it).
        return True
    if a == b:
        return True
    if a not in undirected or b not in undirected:
        return True
    try:
        d = nx.shortest_path_length(undirected, a, b)
    except nx.NetworkXNoPath:
        return False
    return d <= max_distance


def _group_peak(g: IncidentGroup) -> float:
    return max((e.score for e in g.events), default=0.0)


# ---------------------------------------------------------------------------
# Step 3/4 — assemble an Incident from a group
# ---------------------------------------------------------------------------
def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _series_for_group(group: IncidentGroup) -> dict[str, list[float]]:
    """Build a per-entity time-ordered score series for Granger causality.

    Uses all events (pre-dedup view from the group) so each entity has a short
    trajectory of its anomaly/risk scores ordered by time.
    """
    by_entity: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for e in group.events:
        by_entity[e.entity_id].append((e.timestamp, e.score))
    out: dict[str, list[float]] = {}
    for ent, pts in by_entity.items():
        pts.sort(key=lambda p: p[0])
        out[ent] = [v for _, v in pts]
    return out


def _dominant_issue(group: IncidentGroup) -> IssueType:
    """Pick the group's predicted issue: the most common non-NONE issue, else NONE."""
    counts: dict[IssueType, int] = defaultdict(int)
    for e in group.events:
        if e.issue != IssueType.NONE:
            counts[e.issue] += 1
    if not counts:
        return IssueType.NONE
    return max(counts, key=lambda k: counts[k])


def _representative_fused(group: IncidentGroup, root_id: str) -> FusedRisk | None:
    """Choose the FusedRisk that best represents the incident.

    Prefers a fused risk on the root-cause entity; otherwise the highest-risk
    fused event in the group.
    """
    fused_events = [e for e in group.events if e.fused is not None]
    if not fused_events:
        return None
    on_root = [e for e in fused_events if e.entity_id == root_id]
    pool = on_root or fused_events
    best = max(pool, key=lambda e: e.fused.risk_score)  # type: ignore[union-attr]
    return best.fused


def assemble_incident(
    group: IncidentGroup,
    topology: TopologyGraph,
    *,
    flows: Sequence[FlowRecord] | None = None,
    incident_id: str | None = None,
    now: datetime | None = None,
    fallback_fused_factory=None,
) -> Incident:
    """Build a full :class:`Incident` from a correlated group.

    Computes RCA (root-cause entity + grounded hypothesis), the deterministic
    blast radius, contributing signals (from per-entity events), the alarm
    compression ratio, and attaches a representative :class:`FusedRisk`. Severity
    is initialised from the fused risk score (the risk module may re-bucket it).
    """
    if not group.events:
        raise ValueError("cannot assemble an incident from an empty group")

    created = _as_utc(now) if now else datetime.now(timezone.utc)
    onsets = _entity_onsets(group)
    series = _series_for_group(group)
    issue = _dominant_issue(group)

    # --- RCA -----------------------------------------------------------
    candidates = group.entity_ids
    ranked = rank_root_causes(topology, candidates, onsets=onsets, series=series)
    top = ranked[0]
    root_node = topology.map_to_node(top.entity_id) or top.entity_id
    root_ref = topology.entity_ref(root_node)

    # --- blast radius (from the root) ----------------------------------
    blast = compute_blast_radius(topology, root_node, flows=flows)

    # --- representative fused risk -------------------------------------
    fused = _representative_fused(group, top.entity_id)
    if fused is None:
        fused = (
            fallback_fused_factory(group, root_ref, issue)
            if fallback_fused_factory
            else _synthesize_fused(group, root_ref, issue)
        )

    # --- contributing signals ------------------------------------------
    signals = build_contributing_signals(group, root_id=top.entity_id)

    # --- hypothesis ----------------------------------------------------
    hypothesis = build_hypothesis(
        top,
        topology,
        issue=issue,
        n_correlated=len(candidates),
        affected_sites=blast.affected_sites,
    )

    # --- compression ratio ---------------------------------------------
    n_incident_signals = max(len(group.events), 1)
    ratio = round(group.raw_alarm_count / n_incident_signals, 3)
    ratio = max(ratio, 1.0)

    severity = _provisional_severity(fused.risk_score)

    return Incident(
        incident_id=incident_id or f"INC-{uuid.uuid4().hex[:10]}",
        created_at=created,
        window_start=group.window_start,
        window_end=group.window_end,
        predicted_issue=issue,
        severity=severity,
        risk=fused,
        root_cause_entity=root_ref,
        root_cause_hypothesis=hypothesis,
        correlated_entities=[topology.entity_ref(c) for c in candidates],
        contributing_signals=signals,
        blast_radius=blast,
        alarm_compression_ratio=ratio,
    )


def _entity_onsets(group: IncidentGroup) -> dict[str, datetime]:
    onsets: dict[str, datetime] = {}
    for e in group.events:
        t = onsets.get(e.entity_id)
        if t is None or e.timestamp < t:
            onsets[e.entity_id] = e.timestamp
    return onsets


def build_contributing_signals(
    group: IncidentGroup, *, root_id: str
) -> list[ContributingSignal]:
    """Derive ranked :class:`ContributingSignal` entries from a group's events.

    One signal per (entity, metric), scored by peak normalized score; the root
    cause's own signals are surfaced first. This is a *correlation-level* view of
    Q2; richer SHAP attributions come from :mod:`netra.analytics.explain` and are
    merged by the risk/copilot layer.
    """
    best: dict[tuple[str, str], CorrelationEvent] = {}
    for e in group.events:
        key = (e.entity_id, e.metric)
        cur = best.get(key)
        if cur is None or e.score > cur.score:
            best[key] = e

    sigs: list[ContributingSignal] = []
    for (ent, metric), e in best.items():
        direction = Direction.INCREASES_RISK if e.score > 0 else Direction.NEUTRAL
        name = f"{metric}:{_short_entity(ent)}"
        sigs.append(
            ContributingSignal(
                signal=name,
                shap_value=round(e.score, 4),
                direction=direction,
                observation=f"{e.method} normalized score {e.score:.2f}",
                human_explanation=_signal_explanation(metric, e.score, ent == root_id),
                entity=e.entity_ref,
            )
        )
    # root-cause signals first, then by descending magnitude.
    sigs.sort(key=lambda s: (s.entity is not None and s.entity.entity_id == root_id,
                             s.shap_value or 0.0), reverse=True)
    return sigs


def _signal_explanation(metric: str, score: float, is_root: bool) -> str:
    where = "the root-cause entity" if is_root else "a correlated entity"
    return (
        f"Detector evidence on {metric} for {where} "
        f"(strength {score:.2f}) contributes to the elevated risk."
    )


def _short_entity(entity_id: str) -> str:
    parts = entity_id.split(":")
    if len(parts) >= 4:
        return f"{parts[1]}:{parts[3]}"
    if len(parts) >= 2:
        return parts[1]
    return entity_id


def _synthesize_fused(
    group: IncidentGroup, root_ref: EntityRef, issue: IssueType
) -> FusedRisk:
    """Build a minimal FusedRisk when the group carried only AnomalyScores.

    The contract forbids ``risk_score>0`` without a contributing method, so we
    record the firing detectors as provenance. Confidence/agreement are derived
    conservatively from the peak score and family diversity.
    """
    from netra.contracts import DetectorFamily, MethodWeight

    peak = _group_peak(group)
    methods: list[MethodWeight] = []
    seen: set[tuple[str, str]] = set()
    families: set[str] = set()
    for e in group.events:
        if e.raw is not None and isinstance(e.raw, AnomalyScore):
            fam = e.raw.family
            key = (e.method, e.metric)
            if key in seen:
                continue
            seen.add(key)
            families.add(fam.value)
            methods.append(
                MethodWeight(
                    method=e.method,
                    family=fam,
                    normalized_score=float(e.raw.normalized_score),
                    weight=1.0,
                )
            )
    if not methods:
        methods.append(
            MethodWeight(
                method="correlation",
                family=DetectorFamily.GRAPH,
                normalized_score=peak,
                weight=1.0,
            )
        )
        families.add(DetectorFamily.GRAPH.value)
    agreement = min(1.0, len(families) / 3.0)  # 3 independent families ⇒ full
    return FusedRisk(
        entity=root_ref,
        timestamp=group.window_end,
        risk_score=round(peak, 4),
        calibrated_confidence=round(0.5 * peak + 0.5 * agreement * peak, 4),
        predicted_issue=issue,
        agreement=round(agreement, 4),
        contributing_methods=methods,
    )


def _provisional_severity(risk_score: float) -> Severity:
    if risk_score >= 0.75:
        return Severity.P1
    if risk_score >= 0.45:
        return Severity.P2
    if risk_score > 0.0:
        return Severity.P3
    return Severity.INFO


# ---------------------------------------------------------------------------
# Convenience one-shot pipeline
# ---------------------------------------------------------------------------
def correlate_to_incidents(
    topology: TopologyGraph,
    *,
    anomalies: Sequence[AnomalyScore] | None = None,
    fused: Sequence[FusedRisk] | None = None,
    flows: Sequence[FlowRecord] | None = None,
    window_seconds: float = 300.0,
    max_topo_distance: int = 2,
    now: datetime | None = None,
) -> list[Incident]:
    """End-to-end: events → dedup → correlate → RCA → assemble incidents.

    The single entry point the risk/prioritisation layer and the API call. Returns
    incidents ordered largest/most-severe first (the risk layer re-ranks by
    calibrated risk and applies flap suppression).
    """
    events = normalize_events(anomalies, fused)
    groups = correlate_events(
        events,
        topology,
        window_seconds=window_seconds,
        max_topo_distance=max_topo_distance,
    )
    incidents = [
        assemble_incident(g, topology, flows=flows, now=now) for g in groups if g.events
    ]
    return incidents


__all__ = [
    "CorrelationEvent",
    "IncidentGroup",
    "normalize_events",
    "dedup_events",
    "correlate_events",
    "assemble_incident",
    "build_contributing_signals",
    "correlate_to_incidents",
]
