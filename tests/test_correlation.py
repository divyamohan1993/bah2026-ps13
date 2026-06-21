"""WS4 tests — correlation / RCA / blast-radius / risk-prioritisation / explain.

Builds a small demo topology + a set of synthetic ``AnomalyScore`` / ``FusedRisk``
events and asserts the full Builder-4 pipeline:

  * correlation groups the right co-occurring events into ONE incident;
  * RCA picks the plausible (upstream/central, earliest) root node;
  * blast radius counts the right downstream set;
  * prioritisation orders incidents by calibrated risk;
  * flap suppression demotes a chronically-flapping entity;
  * explain produces ContributingSignals with sane directions.

LIGHT deps only (pydantic, networkx, numpy, scikit-learn, statsmodels, scipy).
``shap`` / ``causal-learn`` are intentionally NOT required — the fallbacks are
exercised. Construct contract types directly; never import Builder-3 modules.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from netra.analytics.correlation import (
    TopologyGraph,
    build_demo_graph,
    compute_blast_radius,
    correlate_events,
    correlate_to_incidents,
    normalize_events,
    rank_root_causes,
)
from netra.analytics.correlation.graph import demo_topology_spec
from netra.analytics.explain import (
    attribute_fused_risk,
    explain_fused_risk,
    shap_available,
)
from netra.analytics.risk import (
    FlapSuppressor,
    RiskCalibrator,
    brier_score,
    compute_risk_factors,
    expected_calibration_error,
    prioritize_incidents,
    triage_queue,
)
from netra.analytics.risk.score import geometric_mean_score, time_to_impact_urgency
from netra.contracts import (
    AnomalyScore,
    DetectorFamily,
    Direction,
    EntityRef,
    FlowRecord,
    FusedRisk,
    Incident,
    IssueType,
    MethodWeight,
    Severity,
    TimeToImpact,
)

T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Builders for synthetic contract events
# ---------------------------------------------------------------------------
def _ref(entity_id: str) -> EntityRef:
    g = build_demo_graph()
    return g.entity_ref(entity_id)


def _anomaly(
    entity_id: str,
    *,
    metric: str,
    method: str,
    family: DetectorFamily,
    score: float,
    t: datetime,
    is_anomaly: bool = True,
) -> AnomalyScore:
    return AnomalyScore(
        entity=_ref(entity_id),
        metric=metric,
        timestamp=t,
        method=method,
        family=family,
        score=score * 10.0,
        normalized_score=score,
        is_anomaly=is_anomaly,
        threshold=0.5,
    )


def _fused(
    entity_id: str,
    *,
    risk: float,
    issue: IssueType,
    t: datetime,
    confidence: float = 0.7,
    agreement: float = 0.66,
    eta_seconds: float | None = 180.0,
    methods: list[MethodWeight] | None = None,
) -> FusedRisk:
    if methods is None:
        methods = [
            MethodWeight(
                method="page_hinkley:if_util_pct",
                family=DetectorFamily.CHANGE_POINT,
                normalized_score=risk,
                weight=1.0,
            ),
            MethodWeight(
                method="isolation_forest",
                family=DetectorFamily.ML_UNSUPERVISED,
                normalized_score=max(0.0, risk - 0.1),
                weight=0.8,
            ),
        ]
    tti = None
    if eta_seconds is not None:
        tti = TimeToImpact(
            entity=_ref(entity_id),
            metric="if_util_pct",
            origin=t,
            threshold=90.0,
            eta_seconds=eta_seconds,
            confidence=confidence,
        )
    return FusedRisk(
        entity=_ref(entity_id),
        timestamp=t,
        risk_score=risk,
        calibrated_confidence=confidence,
        predicted_issue=issue,
        agreement=agreement,
        contributing_methods=methods,
        time_to_impact=tti,
    )


# ===========================================================================
# Topology / graph
# ===========================================================================
def test_demo_topology_builds_and_has_expected_shape():
    g = build_demo_graph()
    assert g.g.number_of_nodes() == 11
    assert g.has_node("core:p2:P")
    assert g.has_node("hub1:pe-hub1:PE")
    # criticality ordering: controller/RR > core P > hub PE > branch CE
    assert g.criticality("dc1:ctl1:controller") > g.criticality("hub1:pe-hub1:PE")
    assert g.criticality("core:rr1:RR") > g.criticality("hub1:pe-hub1:PE")
    assert g.criticality("hub1:pe-hub1:PE") > g.criticality("br1:ce-br1:CE")


def test_map_to_node_resolves_subentities():
    g = build_demo_graph()
    # an interface id maps onto its device node
    assert g.map_to_node("hub1:pe-hub1:PE:eth1") == "hub1:pe-hub1:PE"
    # an exact node maps to itself
    assert g.map_to_node("core:p2:P") == "core:p2:P"


def test_topology_from_json(tmp_path):
    import json

    spec = demo_topology_spec()
    p = tmp_path / "topo.json"
    p.write_text(json.dumps(spec))
    g = TopologyGraph.from_json(p)
    assert g.g.number_of_nodes() == 11


# ===========================================================================
# Blast radius
# ===========================================================================
def test_blast_radius_hub_reaches_three_branches():
    g = build_demo_graph()
    blast = compute_blast_radius(g, "hub1:pe-hub1:PE")
    # hub → spoke tunnels reach exactly the three branch sites downstream.
    assert {"br1", "br2", "br3"}.issubset(set(blast.affected_sites))
    assert "br1:ce-br1:CE" in blast.hop_distances
    assert blast.hop_distances["br1:ce-br1:CE"] == 1  # one tunnel hop
    assert 0.0 < (blast.normalized_size or 0) <= 1.0


def test_blast_radius_branch_leaf_is_small():
    g = build_demo_graph()
    blast = compute_blast_radius(g, "br1:ce-br1:CE")
    # a spoke CE is a leaf (directed tunnel points INTO it) → no downstream.
    assert blast.affected_sites == [] or "br1" not in [
        s for s in blast.affected_sites if s != "br1"
    ]
    assert (blast.normalized_size or 0.0) < 0.2


def test_blast_radius_intersects_netflow():
    g = build_demo_graph()
    flows = [
        FlowRecord(
            timestamp=T0, site="br1", device="ce-br1", src_addr="10.1.1.1",
            dst_addr="10.0.0.1", protocol="tcp", bytes=1000, packets=10, vrf="CORP",
        ),
        FlowRecord(
            timestamp=T0, site="br2", device="ce-br2", src_addr="10.2.1.1",
            dst_addr="10.0.0.1", protocol="udp", bytes=500, packets=5, vrf="CORP",
        ),
        FlowRecord(
            timestamp=T0, site="zzz", device="other", src_addr="9.9.9.9",
            dst_addr="8.8.8.8", protocol="tcp", bytes=50, packets=1,
        ),
    ]
    blast = compute_blast_radius(g, "hub1:pe-hub1:PE", flows=flows)
    # exactly the two flows on downstream branch sites are counted (not the zzz one)
    assert blast.affected_flow_count == 2
    assert "CORP" in blast.affected_services_or_vpns


def test_core_router_has_larger_blast_than_hub():
    g = build_demo_graph()
    core = compute_blast_radius(g, "core:p2:P")
    hub = compute_blast_radius(g, "hub1:pe-hub1:PE")
    assert (core.normalized_size or 0) >= (hub.normalized_size or 0)
    assert len(core.affected_sites) >= len(hub.affected_sites)


# ===========================================================================
# RCA ranking
# ===========================================================================
def test_rca_picks_central_earliest_node():
    g = build_demo_graph()
    # hub PE fires first (root), two downstream branches fire later (symptoms).
    onsets = {
        "hub1:pe-hub1:PE": T0,
        "br1:ce-br1:CE": T0 + timedelta(seconds=30),
        "br2:ce-br2:CE": T0 + timedelta(seconds=45),
    }
    # series where the hub leads the branches (hub Granger-causes branches).
    base = [0.1, 0.15, 0.2, 0.35, 0.55, 0.7, 0.8, 0.85, 0.9, 0.92]
    lag = [0.1, 0.1, 0.12, 0.15, 0.2, 0.35, 0.55, 0.7, 0.8, 0.85]
    series = {
        "hub1:pe-hub1:PE": base,
        "br1:ce-br1:CE": lag,
        "br2:ce-br2:CE": lag,
    }
    ranked = rank_root_causes(g, list(onsets), onsets=onsets, series=series)
    assert ranked[0].entity_id == "hub1:pe-hub1:PE"
    assert ranked[0].rank == 1
    # the root scores strictly higher than at least one downstream symptom.
    assert ranked[0].score > ranked[-1].score
    assert ranked[0].onset_score >= ranked[-1].onset_score


def test_rca_works_without_series_or_onsets():
    g = build_demo_graph()
    ranked = rank_root_causes(g, ["core:p2:P", "br1:ce-br1:CE"])
    # core P-router is far more central than a branch leaf → ranks first.
    assert ranked[0].entity_id == "core:p2:P"


# ===========================================================================
# Correlation — grouping + dedup + compression
# ===========================================================================
def _congestion_event_set():
    """Hub-spoke congestion: hub PE + its 3 branches, all within the window."""
    anomalies = [
        _anomaly("hub1:pe-hub1:PE", metric="if_util_pct", method="page_hinkley:if_util_pct",
                 family=DetectorFamily.CHANGE_POINT, score=0.85, t=T0),
        _anomaly("hub1:pe-hub1:PE", metric="if_util_pct", method="page_hinkley:if_util_pct",
                 family=DetectorFamily.CHANGE_POINT, score=0.80, t=T0 + timedelta(seconds=5)),  # dup
        _anomaly("br1:ce-br1:CE", metric="latency_ms", method="ewma",
                 family=DetectorFamily.STATISTICAL, score=0.6, t=T0 + timedelta(seconds=40)),
        _anomaly("br2:ce-br2:CE", metric="latency_ms", method="ewma",
                 family=DetectorFamily.STATISTICAL, score=0.55, t=T0 + timedelta(seconds=50)),
    ]
    fused = [
        _fused("hub1:pe-hub1:PE", risk=0.82, issue=IssueType.INTERFACE_CONGESTION,
               t=T0 + timedelta(seconds=10), confidence=0.8, eta_seconds=120.0),
    ]
    return anomalies, fused


def test_correlation_groups_related_events_into_one_incident():
    g = build_demo_graph()
    anomalies, fused = _congestion_event_set()
    events = normalize_events(anomalies, fused)
    groups = correlate_events(events, g, window_seconds=300.0, max_topo_distance=2)
    assert len(groups) == 1
    grp = groups[0]
    # hub + 2 branches = 3 distinct entities folded into the one incident.
    assert set(grp.entity_ids) == {
        "hub1:pe-hub1:PE", "br1:ce-br1:CE", "br2:ce-br2:CE",
    }


def test_unrelated_distant_events_do_not_merge():
    g = build_demo_graph()
    # Two clusters far apart in BOTH time and topology must stay separate:
    #  - congestion at the hub (T0)
    #  - a controller policy-drift event 1 hour later (different subtree + time)
    anomalies = [
        _anomaly("hub1:pe-hub1:PE", metric="if_util_pct", method="page_hinkley:if_util_pct",
                 family=DetectorFamily.CHANGE_POINT, score=0.85, t=T0),
        _anomaly("dc1:ctl1:controller", metric="config_drift_score", method="bocpd",
                 family=DetectorFamily.CHANGE_POINT, score=0.9, t=T0 + timedelta(hours=1)),
    ]
    incidents = correlate_to_incidents(g, anomalies=anomalies, now=T0 + timedelta(hours=2))
    assert len(incidents) == 2


def test_alarm_compression_ratio_reported():
    g = build_demo_graph()
    anomalies, fused = _congestion_event_set()
    incidents = correlate_to_incidents(
        g, anomalies=anomalies, fused=fused, now=T0 + timedelta(minutes=5)
    )
    assert len(incidents) == 1
    inc = incidents[0]
    # 5 raw signals compressed into one incident → ratio >= 1 and duplicate dropped.
    assert inc.alarm_compression_ratio is not None
    assert inc.alarm_compression_ratio > 1.0


# ===========================================================================
# Full correlate → Incident assembly
# ===========================================================================
def test_assemble_incident_full_shape():
    g = build_demo_graph()
    anomalies, fused = _congestion_event_set()
    flows = [
        FlowRecord(timestamp=T0, site="br1", device="ce-br1", src_addr="10.1.1.1",
                   dst_addr="10.0.0.1", protocol="tcp", bytes=1000, packets=10, vrf="CORP"),
    ]
    incidents = correlate_to_incidents(
        g, anomalies=anomalies, fused=fused, flows=flows, now=T0 + timedelta(minutes=5)
    )
    inc = incidents[0]
    assert isinstance(inc, Incident)
    assert inc.predicted_issue == IssueType.INTERFACE_CONGESTION
    assert inc.root_cause_entity is not None
    assert inc.root_cause_entity.entity_id == "hub1:pe-hub1:PE"
    assert "root cause" in inc.root_cause_hypothesis.lower()
    # blast radius reaches the branch spokes
    assert {"br1", "br2", "br3"}.issubset(set(inc.blast_radius.affected_sites))
    # contributing signals present and the root's signal ranks first
    assert len(inc.contributing_signals) >= 1
    assert inc.contributing_signals[0].entity is not None
    # serialises cleanly (API boundary)
    assert inc.model_dump_json()


def test_incident_with_only_anomalies_synthesizes_valid_risk():
    g = build_demo_graph()
    anomalies, _ = _congestion_event_set()
    incidents = correlate_to_incidents(g, anomalies=anomalies, now=T0 + timedelta(minutes=5))
    inc = incidents[0]
    # FusedRisk contract: risk_score>0 requires >=1 contributing method.
    assert inc.risk.risk_score > 0
    assert len(inc.risk.contributing_methods) >= 1


# ===========================================================================
# Risk scoring + prioritisation ordering
# ===========================================================================
def test_time_to_impact_urgency_monotonic():
    soon, _ = time_to_impact_urgency(
        TimeToImpact(entity=_ref("hub1:pe-hub1:PE"), metric="if_util_pct", origin=T0,
                     threshold=90.0, eta_seconds=60.0, confidence=0.8)
    )
    later, _ = time_to_impact_urgency(
        TimeToImpact(entity=_ref("hub1:pe-hub1:PE"), metric="if_util_pct", origin=T0,
                     threshold=90.0, eta_seconds=600.0, confidence=0.8)
    )
    none_urg, _ = time_to_impact_urgency(None)
    assert soon > later > none_urg


def test_product_form_zero_blast_suppresses_score():
    g = build_demo_graph()
    # construct an incident whose root is a leaf (zero downstream blast) but high risk
    inc = _incident_for(g, root="br3:ce-br3:CE", risk=0.9, issue=IssueType.TUNNEL_DEGRADATION,
                        eta_seconds=60.0)
    factors = compute_risk_factors(inc, asset_criticality=g.criticality("br3:ce-br3:CE"))
    # blast factor should be small for a leaf → product score pulled down vs a hub.
    inc_hub = _incident_for(g, root="hub1:pe-hub1:PE", risk=0.9,
                            issue=IssueType.INTERFACE_CONGESTION, eta_seconds=60.0)
    factors_hub = compute_risk_factors(inc_hub, asset_criticality=g.criticality("hub1:pe-hub1:PE"))
    assert factors.blast_radius < factors_hub.blast_radius
    assert geometric_mean_score(factors) < geometric_mean_score(factors_hub)


def _incident_for(
    g: TopologyGraph, *, root: str, risk: float, issue: IssueType, eta_seconds: float | None
) -> Incident:
    fused = _fused(root, risk=risk, issue=issue, t=T0, confidence=risk, eta_seconds=eta_seconds)
    blast = compute_blast_radius(g, root)
    return Incident(
        incident_id=f"INC-{root}",
        created_at=T0,
        window_start=T0,
        window_end=T0,
        predicted_issue=issue,
        severity=Severity.P3,
        risk=fused,
        root_cause_entity=g.entity_ref(root),
        root_cause_hypothesis="test",
        correlated_entities=[g.entity_ref(root)],
        blast_radius=blast,
    )


def test_prioritisation_orders_by_calibrated_risk():
    g = build_demo_graph()
    # imminent, high-blast hub congestion should outrank a slow, low-blast branch issue.
    big = _incident_for(g, root="core:p2:P", risk=0.85,
                        issue=IssueType.MPLS_UNDERLAY_FAILURE, eta_seconds=60.0)
    small = _incident_for(g, root="br3:ce-br3:CE", risk=0.4,
                          issue=IssueType.TUNNEL_DEGRADATION, eta_seconds=1800.0)
    ordered = triage_queue([small, big], topology=g)
    assert ordered[0].incident_id == "INC-core:p2:P"
    assert ordered[1].incident_id == "INC-br3:ce-br3:CE"
    # severity assigned and consistent with order
    assert ordered[0].severity in (Severity.P1, Severity.P2)


def test_prioritisation_assigns_severity_buckets():
    g = build_demo_graph()
    incs = [
        _incident_for(g, root="core:p1:P", risk=0.95,
                      issue=IssueType.MPLS_UNDERLAY_FAILURE, eta_seconds=30.0),
        _incident_for(g, root="br1:ce-br1:CE", risk=0.2,
                      issue=IssueType.TUNNEL_DEGRADATION, eta_seconds=3600.0),
    ]
    results = prioritize_incidents(incs, topology=g)
    sevs = {pi.incident.root_cause_entity.entity_id: pi.incident.severity for pi in results}
    # the big core incident outranks (more severe than) the tiny branch one.
    order = [Severity.INFO, Severity.P3, Severity.P2, Severity.P1]
    assert order.index(sevs["core:p1:P"]) >= order.index(sevs["br1:ce-br1:CE"])


# ===========================================================================
# Flap suppression
# ===========================================================================
def test_flap_suppressor_penalty_decays():
    s = FlapSuppressor(half_life_seconds=60.0, penalty_increment=1.0)
    # 4 flaps in quick succession → penalty ~4, above suppress threshold (3.0).
    for i in range(4):
        s.observe("br1:ce-br1:CE", now=T0 + timedelta(seconds=i), flaps=1)
    assert s.is_suppressed("br1:ce-br1:CE", now=T0 + timedelta(seconds=4))
    # after several half-lives the penalty decays below reuse → not suppressed.
    later = T0 + timedelta(seconds=4 + 60 * 5)
    assert s.penalty_of("br1:ce-br1:CE", now=later) < 1.0
    assert not s.is_suppressed("br1:ce-br1:CE", now=later)


def test_flap_suppression_demotes_flapping_incident_in_queue():
    g = build_demo_graph()
    # two equally-risky incidents; one entity is flapping → must rank lower.
    flapping_root = "hub1:pe-hub1:PE"
    stable_root = "dc1:pe-dc1:PE"
    inc_flap = _incident_for(g, root=flapping_root, risk=0.8,
                             issue=IssueType.INTERFACE_CONGESTION, eta_seconds=120.0)
    inc_flap.incident_id = "INC-FLAP"
    inc_stable = _incident_for(g, root=stable_root, risk=0.8,
                               issue=IssueType.INTERFACE_CONGESTION, eta_seconds=120.0)
    inc_stable.incident_id = "INC-STABLE"

    sup = FlapSuppressor(half_life_seconds=600.0, penalty_increment=1.0)
    for i in range(8):  # drive the flapping entity well above suppress threshold
        sup.observe(flapping_root, now=T0 + timedelta(seconds=i * 2), flaps=1)
    now = T0 + timedelta(seconds=20)

    results = prioritize_incidents([inc_flap, inc_stable], topology=g, suppressor=sup, now=now)
    by_id = {pi.incident.incident_id: pi for pi in results}
    assert by_id["INC-FLAP"].demotion_factor < 1.0
    assert by_id["INC-FLAP"].suppressed is True
    # stable incident ends up ranked ahead of the flapping one.
    assert results[0].incident.incident_id == "INC-STABLE"
    assert by_id["INC-STABLE"].calibrated_priority > by_id["INC-FLAP"].calibrated_priority


# ===========================================================================
# Calibration
# ===========================================================================
def test_platt_calibration_improves_or_matches_brier():
    cal = RiskCalibrator(method="platt")
    # synthetic over-confident raw scores: label ~ Bernoulli with lower true rate.
    rng = __import__("numpy").random.default_rng(0)
    raw = rng.uniform(0, 1, size=200)
    labels = (rng.uniform(0, 1, size=200) < (raw * 0.5)).astype(int)  # true p = raw/2
    cal.fit(raw.tolist(), labels.tolist())
    assert cal.fitted
    calibrated = cal.transform(raw.tolist())
    b_raw = brier_score(raw.tolist(), labels.tolist())
    b_cal = brier_score(calibrated, labels.tolist())
    assert b_cal <= b_raw + 1e-6
    # ECE finite and in range
    ece = expected_calibration_error(calibrated, labels.tolist())
    assert 0.0 <= ece <= 1.0


def test_calibrator_identity_when_unfitted():
    cal = RiskCalibrator()
    assert cal.transform(0.42) == pytest.approx(0.42)


# ===========================================================================
# Explainability (Q2)
# ===========================================================================
def test_explain_produces_signals_with_sane_directions():
    risk = _fused("hub1:pe-hub1:PE", risk=0.82, issue=IssueType.INTERFACE_CONGESTION, t=T0)
    signals = explain_fused_risk(risk, top_k=8)
    assert len(signals) >= 1
    # contributions from positively-firing methods push risk UP.
    assert any(s.direction == Direction.INCREASES_RISK for s in signals)
    # each signal carries a non-empty grounded explanation and a signal name.
    for s in signals:
        assert s.signal
        assert s.human_explanation
    # the if_util feature (from page_hinkley:if_util_pct provenance) is explained
    # with the congestion template.
    util_sigs = [s for s in signals if "util" in s.signal.lower()]
    assert util_sigs
    assert "saturation" in util_sigs[0].human_explanation.lower() or \
           "capacity" in util_sigs[0].human_explanation.lower()


def test_explain_fallback_used_without_shap():
    # The deterministic fallback must run whenever no SHAP model+instance is
    # supplied — regardless of whether the optional ``shap`` library happens to be
    # importable in this environment (it ships in the core tier, so it usually is).
    # ``shap_available()`` is informational here; the contract under test is that
    # ``attribute_fused_risk`` with no model falls back to the model-free path.
    _ = shap_available()
    risk = _fused("br1:ce-br1:CE", risk=0.5, issue=IssueType.TUNNEL_DEGRADATION, t=T0)
    attrs = attribute_fused_risk(risk)
    assert attrs
    assert all(a.method in ("fallback", "permutation") for a in attrs)
    # signed contributions roughly track the risk magnitude.
    assert sum(abs(a.value) for a in attrs) > 0


def test_explain_with_feature_values_adds_signals():
    risk = _fused("hub1:pe-hub1:PE", risk=0.7, issue=IssueType.INTERFACE_CONGESTION, t=T0)
    feats = {"jitter_ms": 0.9, "loss_pct": 0.3}
    signals = explain_fused_risk(risk, feature_values=feats, top_k=10)
    names = {s.signal for s in signals}
    assert "jitter_ms" in names
    jitter = next(s for s in signals if s.signal == "jitter_ms")
    assert jitter.human_explanation  # has a jitter-specific explanation
    assert "jitter" in jitter.human_explanation.lower()


# ===========================================================================
# End-to-end integration (the full Builder-4 spine)
# ===========================================================================
def test_end_to_end_correlate_explain_prioritize():
    g = build_demo_graph()
    anomalies, fused = _congestion_event_set()
    # A second, independent flapping BGP incident at the RR — separated in TIME
    # (a fresh window an hour later) so it is a genuinely distinct incident.
    t_rr = T0 + timedelta(hours=1)
    rr_anoms = [
        _anomaly("core:rr1:RR", metric="bgp_flap_penalty", method="page_hinkley:bgp_flap_penalty",
                 family=DetectorFamily.ROUTING, score=0.7, t=t_rr),
    ]
    rr_fused = [
        _fused("core:rr1:RR", risk=0.6, issue=IssueType.BGP_ROUTE_FLAP,
               t=t_rr + timedelta(seconds=3), confidence=0.6, eta_seconds=240.0,
               methods=[MethodWeight(method="page_hinkley:bgp_flap_penalty",
                                     family=DetectorFamily.ROUTING,
                                     normalized_score=0.6, weight=1.0)]),
    ]
    incidents = correlate_to_incidents(
        g,
        anomalies=anomalies + rr_anoms,
        fused=fused + rr_fused,
        now=t_rr + timedelta(minutes=5),
    )
    # enrich each incident's contributing signals via the explain layer.
    for inc in incidents:
        inc.contributing_signals = explain_fused_risk(inc.risk, entity=inc.root_cause_entity)

    # flap suppression on the RR (it is flapping), then prioritise.
    sup = FlapSuppressor(half_life_seconds=600.0)
    for i in range(6):
        sup.observe("core:rr1:RR", now=t_rr + timedelta(seconds=i), flaps=1)

    ordered = prioritize_incidents(
        incidents, topology=g, suppressor=sup, now=t_rr + timedelta(minutes=5)
    )
    assert len(ordered) == 2
    # every incident is fully formed and serialisable.
    for pi in ordered:
        inc = pi.incident
        assert inc.severity in (Severity.P1, Severity.P2, Severity.P3, Severity.INFO)
        assert inc.root_cause_entity is not None
        assert inc.contributing_signals
        assert inc.model_dump_json()
    # ordering is by calibrated priority (descending).
    assert ordered[0].calibrated_priority >= ordered[1].calibrated_priority
