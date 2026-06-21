"""CPU-only smoke + contract tests for the offline copilot (Workstream 5).

These tests exercise the **light path only** — ``TemplateClient`` +
TF-IDF/in-memory retriever + lexical faithfulness — so they pass with just the
core deps (pydantic, numpy, scikit-learn) and **no heavy model, no llama-server,
no internet, no sim**. They assert the contract guarantees that earn the
"Copilot Effectiveness" (grounded, schema-valid, no hallucination) score:

  * the produced ``CopilotResponse`` validates against the canonical contract;
  * citations are present and drawn from the retrieved context when it exists;
  * ``insufficient_context=True`` (abstain) when the corpus/context is empty;
  * Q1/Q2/Q3 fields are populated from the analytics inputs;
  * the closed-set citation gate and the localhost-only no-egress guard hold.

Run: ``pytest -q tests/test_copilot.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from netra.contracts import (
    AffectedScope,
    BlastRadius,
    ContributingSignal,
    CopilotAction,
    CopilotRequest,
    CopilotResponse,
    CopilotSignal,
    DetectorFamily,
    DeviceRole,
    Direction,
    EntityRef,
    FusedRisk,
    Incident,
    IssueType,
    MethodWeight,
    Playbook,
    RecommendedAction,
    Severity,
    TimeToImpact,
    Urgency,
)
from netra.copilot import AnalyticsContext, Copilot
from netra.copilot.grounding import (
    FaithfulnessScorer,
    enforce_citations,
    should_abstain,
    validate_citations,
)
from netra.copilot.llm import (
    CopilotGrounding,
    CopilotPrompt,
    LlamaCppClient,
    TemplateClient,
    copilot_gbnf,
    copilot_json_schema,
    select_llm_client,
)
from netra.copilot.rag import Chunk, HybridRetriever, TopologyGraph, build_retriever

# --------------------------------------------------------------------------- #
# Fixtures: analytics inputs (contract objects only) for scenario A.          #
# --------------------------------------------------------------------------- #
NOW = datetime.now(UTC)
ENT_HUB = EntityRef(
    entity_id="hub1:pe-hub1:PE:eth1",
    site="hub1",
    device="pe-hub1",
    role=DeviceRole.PE,
    sub="eth1",
)


def _fused_risk(issue: IssueType, confidence: float = 0.83) -> FusedRisk:
    return FusedRisk(
        entity=ENT_HUB,
        timestamp=NOW,
        risk_score=0.86,
        calibrated_confidence=confidence,
        predicted_issue=issue,
        agreement=0.7,
        contributing_methods=[
            MethodWeight(
                method="page_hinkley",
                family=DetectorFamily.CHANGE_POINT,
                normalized_score=0.8,
                weight=1.0,
            )
        ],
    )


def _tti(eta_seconds: float | None = 360.0) -> TimeToImpact:
    return TimeToImpact(
        entity=ENT_HUB,
        metric="if_util_pct",
        origin=NOW,
        threshold=90.0,
        eta_seconds=eta_seconds,
        confidence=0.8,
    )


def _signals() -> list[ContributingSignal]:
    return [
        ContributingSignal(
            signal="if_util_pct:eth1",
            shap_value=0.41,
            direction=Direction.INCREASES_RISK,
            observation="rising 4%/min, now 78%",
            human_explanation="Hub-spoke uplink trending toward saturation.",
            entity=ENT_HUB,
        ),
        ContributingSignal(
            signal="if_out_discards",
            shap_value=0.22,
            direction=Direction.INCREASES_RISK,
            observation="creeping from zero",
            human_explanation="Egress queue drops beginning.",
        ),
    ]


def _blast_radius() -> BlastRadius:
    return BlastRadius(
        affected_sites=["hub1", "br1", "br2", "br3"],
        affected_devices=["pe-hub1"],
        affected_services_or_vpns=["CORP"],
        affected_slas=["CORP-VOICE"],
        affected_flow_count=42,
    )


def _playbook() -> Playbook:
    return Playbook(
        playbook_id="PB-CONGESTION-001",
        title="Relieve progressive interface congestion",
        issue_type=IssueType.INTERFACE_CONGESTION,
        source_ref="RB-CONGESTION-001",
        actions=[
            RecommendedAction(
                step=1,
                description="Collect interface and egress-queue statistics.",
                command_or_guidance="show interface eth1 | include rate|drops",
                requires_approval=False,
                urgency=Urgency.IMMEDIATE,
                runbook_ref="RB-CONGESTION-001",
            ),
            RecommendedAction(
                step=2,
                description="Raise QoS priority for the business class.",
                command_or_guidance="napalm load_merge policy-map QOS-UPLINK",
                requires_approval=True,
                urgency=Urgency.IMMEDIATE,
                rollback="napalm rollback",
                verification="business latency holds",
                runbook_ref="RB-CONGESTION-001",
            ),
        ],
    )


def _analytics_context() -> AnalyticsContext:
    return AnalyticsContext(
        fused_risk=_fused_risk(IssueType.INTERFACE_CONGESTION),
        time_to_impact=_tti(),
        contributing_signals=_signals(),
        blast_radius=_blast_radius(),
        playbook=_playbook(),
        root_cause_entity=ENT_HUB,
    )


@pytest.fixture(scope="module")
def copilot() -> Copilot:
    """A copilot on the light path (template fallback, TF-IDF retriever)."""
    return Copilot(prefer_models=False, llm=TemplateClient())


@pytest.fixture(scope="module")
def request_a() -> CopilotRequest:
    return CopilotRequest(
        request_id="req-A-1",
        created_at=NOW,
        operator_query="Why is the Mumbai hub uplink at risk and what do I do?",
        max_context_chunks=5,
    )


# --------------------------------------------------------------------------- #
# 1. Schema validity — the headline contract guarantee.                       #
# --------------------------------------------------------------------------- #
def test_response_is_schema_valid(copilot: Copilot, request_a: CopilotRequest) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    assert isinstance(resp, CopilotResponse)
    # Round-trips through the canonical contract (extra='forbid' enforced).
    revalidated = CopilotResponse.model_validate(resp.model_dump())
    assert revalidated == resp
    # JSON serialisation works (API boundary).
    assert CopilotResponse.model_validate_json(resp.model_dump_json())


def test_response_uses_template_fallback(
    copilot: Copilot, request_a: CopilotRequest
) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    assert resp.used_fallback is True
    assert resp.model_id == "template-fallback"
    assert resp.request_id == request_a.request_id


# --------------------------------------------------------------------------- #
# 2. Citations present (and grounded) when context exists.                    #
# --------------------------------------------------------------------------- #
def test_citations_present_when_context_exists(
    copilot: Copilot, request_a: CopilotRequest
) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    assert resp.insufficient_context is False
    assert len(resp.citations) >= 1
    # Every citation is a real retrieved/graph id, not "no-context".
    assert "no-context" not in resp.citations
    # The congestion runbook/playbook should be retrievable for this query.
    assert any("CONGESTION" in c.upper() for c in resp.citations)


def test_action_runbook_refs_are_grounded(
    copilot: Copilot, request_a: CopilotRequest
) -> None:
    """Playbook step runbook_refs survive the closed-set gate (doc-level id)."""
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    refs = [a.runbook_ref for a in resp.recommended_actions if a.runbook_ref]
    assert "RB-CONGESTION-001" in refs


# --------------------------------------------------------------------------- #
# 3. Abstain when the corpus / context is empty.                              #
# --------------------------------------------------------------------------- #
def test_abstains_on_empty_corpus() -> None:
    # Build a copilot whose corpus is empty -> no retrievable context.
    empty_retriever = build_retriever(chunks=[])
    empty_topology = TopologyGraph()  # no devices loaded
    cop = Copilot(
        retriever=empty_retriever,
        topology=empty_topology,
        llm=TemplateClient(),
        faithfulness=FaithfulnessScorer(prefer_model=False),
    )
    req = CopilotRequest(
        request_id="req-empty",
        created_at=NOW,
        operator_query="What is failing?",
        max_context_chunks=5,
    )
    resp = cop.answer(req, analytics_context=_analytics_context())
    assert isinstance(resp, CopilotResponse)
    assert resp.insufficient_context is True
    # Still schema-valid: >=1 citation, >=1 action, and a low confidence.
    assert len(resp.citations) >= 1
    assert len(resp.recommended_actions) >= 1
    assert resp.confidence_score <= 0.2


def test_abstains_with_no_analytics_and_no_context() -> None:
    cop = Copilot(
        retriever=build_retriever(chunks=[]),
        topology=TopologyGraph(),
        llm=TemplateClient(),
    )
    req = CopilotRequest(request_id="req-bare", created_at=NOW)
    resp = cop.answer(req)  # no analytics_context at all
    assert resp.insufficient_context is True
    assert resp.predicted_issue == IssueType.NONE
    CopilotResponse.model_validate(resp.model_dump())


# --------------------------------------------------------------------------- #
# 4. Q1/Q2/Q3 fields populated from the analytics inputs.                      #
# --------------------------------------------------------------------------- #
def test_q1_fields_from_analytics(copilot: Copilot, request_a: CopilotRequest) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    # Q1: predicted issue, time-to-impact (seconds->minutes), affected scope.
    assert resp.predicted_issue == IssueType.INTERFACE_CONGESTION
    assert resp.time_to_impact_minutes == pytest.approx(6.0)  # 360s -> 6 min
    # Confidence is the analytics calibrated_confidence, NOT invented.
    assert resp.confidence_score == pytest.approx(0.83)
    # Affected scope is the deterministic blast radius (reported, not guessed).
    assert resp.affected_scope.sites == ["hub1", "br1", "br2", "br3"]
    assert resp.affected_scope.devices == ["pe-hub1"]
    assert resp.affected_scope.services_or_vpns == ["CORP"]


def test_q2_fields_from_analytics(copilot: Copilot, request_a: CopilotRequest) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    # Q2: root-cause hypothesis non-empty + contributing signals carried through.
    assert len(resp.root_cause_hypothesis) >= 1
    names = {s.signal for s in resp.contributing_signals}
    assert "if_util_pct:eth1" in names
    assert "if_out_discards" in names
    # SHAP values preserved.
    by_name = {s.signal: s for s in resp.contributing_signals}
    assert by_name["if_util_pct:eth1"].shap_contribution == pytest.approx(0.41)


def test_q3_actions_from_playbook(copilot: Copilot, request_a: CopilotRequest) -> None:
    resp = copilot.answer(request_a, analytics_context=_analytics_context())
    # Q3: recommended actions come from the supplied playbook (>=1 required).
    assert len(resp.recommended_actions) >= 1
    steps = [a.step for a in resp.recommended_actions]
    assert any("QoS" in s for s in steps)
    # State-changing step keeps requires_approval=True (human-in-the-loop).
    qos = next(a for a in resp.recommended_actions if "QoS" in a.step)
    assert qos.requires_approval is True
    assert qos.urgency == Urgency.IMMEDIATE


# --------------------------------------------------------------------------- #
# 5. The copilot can also run from a single Incident object.                  #
# --------------------------------------------------------------------------- #
def test_answer_from_incident_object(copilot: Copilot) -> None:
    incident = Incident(
        incident_id="INC-TEST-1",
        created_at=NOW,
        window_start=NOW,
        window_end=NOW,
        predicted_issue=IssueType.BGP_ROUTE_FLAP,
        severity=Severity.P1,
        risk=_fused_risk(IssueType.BGP_ROUTE_FLAP, confidence=0.77),
        root_cause_entity=EntityRef(
            entity_id="rr1:rr1:RR:peer-pe-dc1",
            site="rr1",
            device="rr1",
            role=DeviceRole.RR,
            sub="peer-pe-dc1",
        ),
        root_cause_hypothesis="RR VPNv4 session flapping after a core link fault.",
        contributing_signals=[
            ContributingSignal(
                signal="bgp_flap_penalty",
                shap_value=0.5,
                direction=Direction.INCREASES_RISK,
                observation="accumulating toward suppress limit",
                human_explanation="Route-flap penalty rising.",
            )
        ],
        blast_radius=BlastRadius(
            affected_sites=["dc1", "hub1"], affected_devices=["rr1"]
        ),
    )
    req = CopilotRequest(
        request_id="req-bgp",
        created_at=NOW,
        operator_query="Why is BGP unstable on the route reflector?",
        max_context_chunks=5,
    )
    resp = copilot.answer(req, analytics_context=AnalyticsContext(incident=incident))
    CopilotResponse.model_validate(resp.model_dump())
    assert resp.predicted_issue == IssueType.BGP_ROUTE_FLAP
    assert resp.confidence_score == pytest.approx(0.77)
    assert any("BGPFLAP" in c.upper() or "bgp" in c.lower() for c in resp.citations)


# --------------------------------------------------------------------------- #
# 6. All four validation scenarios route to the right corpus + issue.         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "issue,query,needle",
    [
        (IssueType.INTERFACE_CONGESTION, "hub uplink utilisation rising", "CONGESTION"),
        (IssueType.BGP_ROUTE_FLAP, "bgp route flap damping peer", "BGPFLAP"),
        (IssueType.TUNNEL_DEGRADATION, "tunnel jitter loss rekey LSP", "TUNNEL"),
        (IssueType.POLICY_DRIFT, "controller policy drift golden config", "POLICYDRIFT"),
    ],
)
def test_each_scenario_grounds_to_its_corpus(
    copilot: Copilot, issue: IssueType, query: str, needle: str
) -> None:
    ctx = AnalyticsContext(
        fused_risk=_fused_risk(issue),
        time_to_impact=_tti(),
        contributing_signals=_signals(),
        blast_radius=_blast_radius(),
        root_cause_entity=ENT_HUB,
    )
    req = CopilotRequest(
        request_id=f"req-{issue.value}",
        created_at=NOW,
        operator_query=query,
        max_context_chunks=5,
    )
    resp = copilot.answer(req, analytics_context=ctx)
    assert resp.predicted_issue == issue
    assert not resp.insufficient_context
    assert any(needle in c.upper() for c in resp.citations), (
        f"{issue.value}: expected a {needle} citation, got {resp.citations}"
    )


# --------------------------------------------------------------------------- #
# 7. Auto-selection degrades to the template fallback with no server.         #
# --------------------------------------------------------------------------- #
def test_select_llm_defaults_to_template(monkeypatch) -> None:
    monkeypatch.delenv("NETRA_LLAMA_URL", raising=False)
    client = select_llm_client()  # no env, no server -> template
    assert isinstance(client, TemplateClient)
    assert client.is_fallback is True


def test_select_llm_force_template() -> None:
    assert isinstance(select_llm_client(prefer_llm=False), TemplateClient)


def test_unreachable_llama_falls_back_without_probe(monkeypatch) -> None:
    # Force LLM selection without probing; a down server must still degrade at
    # call time to a valid response (never an exception, never invalid output).
    client = select_llm_client(
        prefer_llm=True, base_url="http://127.0.0.1:1", probe=False
    )
    # It may be the LlamaCppClient, but completing must fall back to template.
    grounding = CopilotGrounding(
        request_id="req-x",
        predicted_issue=IssueType.INTERFACE_CONGESTION,
        confidence_score=0.5,
        citation_universe=["RB-CONGESTION-001#0"],
    )
    prompt = CopilotPrompt(system="s", user="u")
    resp = client.complete_copilot(prompt, grounding)
    CopilotResponse.model_validate(resp.model_dump())
    assert len(resp.citations) >= 1


# --------------------------------------------------------------------------- #
# 8. No-egress guard: the llama client refuses non-loopback URLs.             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_url",
    [
        "http://1.1.1.1:8080",
        "https://api.openai.com/v1",
        "http://example.com:8080",
        "http://8.8.8.8",
    ],
)
def test_llama_client_refuses_non_loopback(bad_url: str) -> None:
    with pytest.raises(ValueError):
        LlamaCppClient(base_url=bad_url)


@pytest.mark.parametrize(
    "ok_url",
    ["http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080"],
)
def test_llama_client_allows_loopback(ok_url: str) -> None:
    client = LlamaCppClient(base_url=ok_url)  # must not raise
    assert client.is_fallback is False


# --------------------------------------------------------------------------- #
# 9. Grammar generation from the contract (GBNF + JSON schema).               #
# --------------------------------------------------------------------------- #
def test_gbnf_grammar_covers_issue_enum() -> None:
    gbnf = copilot_gbnf()
    for issue in IssueType:
        assert issue.value in gbnf, f"{issue.value} missing from GBNF"
    for urgency in Urgency:
        assert urgency.value in gbnf
    # Structural fields the model must emit.
    for field in ("predicted_issue", "confidence_score", "recommended_actions",
                  "citations", "insufficient_context"):
        assert field in gbnf


def test_json_schema_subset_matches_contract() -> None:
    schema = copilot_json_schema()
    assert schema["additionalProperties"] is False
    for field in ("predicted_issue", "confidence_score", "root_cause_hypothesis",
                  "recommended_actions", "citations", "insufficient_context"):
        assert field in schema["required"]


def test_grammar_gbnf_file_exists_and_matches() -> None:
    from pathlib import Path

    gbnf_path = Path(__file__).resolve().parents[1] / "netra/copilot/llm/grammar.gbnf"
    assert gbnf_path.exists(), "static grammar.gbnf must be bundled"
    text = gbnf_path.read_text(encoding="utf-8")
    # Every issue-class literal is present in the bundled grammar.
    for issue in IssueType:
        assert issue.value in text


# --------------------------------------------------------------------------- #
# 10. Grounding utilities: closed-set citations, abstain, faithfulness.        #
# --------------------------------------------------------------------------- #
def _valid_response(citations: list[str]) -> CopilotResponse:
    return CopilotResponse(
        request_id="r",
        predicted_issue=IssueType.INTERFACE_CONGESTION,
        confidence_score=0.8,
        root_cause_hypothesis="Uplink saturating.",
        contributing_signals=[CopilotSignal(signal="x", observation="y")],
        affected_scope=AffectedScope(sites=["hub1"], devices=["pe-hub1"]),
        recommended_actions=[
            CopilotAction(step="do it", urgency=Urgency.IMMEDIATE, requires_approval=True)
        ],
        citations=citations,
        insufficient_context=False,
    )


def test_validate_citations_partitions_closed_set() -> None:
    check = validate_citations(["A", "B", "ghost"], {"A", "B", "C"})
    assert check.valid_citations == ["A", "B"]
    assert check.dropped_citations == ["ghost"]
    assert check.all_valid is False


def test_enforce_citations_drops_unknown_ids() -> None:
    resp = _valid_response(["RB-1#0", "HALLUCINATED-99"])
    out = enforce_citations(resp, universe={"RB-1#0", "RB-1#1"}, grounding_score=0.9)
    assert out.citations == ["RB-1#0"]
    assert out.grounding_score == pytest.approx(0.9)
    assert out.insufficient_context is False


def test_enforce_citations_forces_abstain_when_all_ungrounded() -> None:
    resp = _valid_response(["GHOST-1", "GHOST-2"])
    out = enforce_citations(resp, universe={"REAL-1"})
    assert out.insufficient_context is True
    assert out.citations == ["no-context"]
    assert out.confidence_score <= 0.2
    assert len(out.recommended_actions) >= 1
    CopilotResponse.model_validate(out.model_dump())


def test_should_abstain_logic() -> None:
    assert should_abstain(n_context_chunks=0) is True
    assert should_abstain(n_context_chunks=3) is False
    assert should_abstain(n_context_chunks=3, top_score=0.1, min_score=0.5) is True


def test_faithfulness_grounded_vs_ungrounded() -> None:
    scorer = FaithfulnessScorer(prefer_model=False, threshold=0.5)
    context = (
        "Interface utilisation on pe-hub1 eth1 is rising toward saturation; "
        "egress queue discards are creeping and latency is drifting."
    )
    grounded = scorer.score_text(
        answer="Utilisation on pe-hub1 eth1 is rising toward saturation.",
        context=context,
    )
    ungrounded = scorer.score_text(
        answer="A meteorite struck the datacenter cooling system in Antarctica.",
        context=context,
    )
    assert grounded.score > ungrounded.score
    assert grounded.backend == "lexical_overlap"


# --------------------------------------------------------------------------- #
# 11. TemplateClient direct unit behaviour (deterministic, model-free).        #
# --------------------------------------------------------------------------- #
def test_template_client_is_deterministic() -> None:
    client = TemplateClient()
    grounding = CopilotGrounding(
        request_id="req-det",
        predicted_issue=IssueType.TUNNEL_DEGRADATION,
        confidence_score=0.6,
        time_to_impact_minutes=4.0,
        contributing_signals=[CopilotSignal(signal="tunnel_jitter_ms", observation="spiking")],
        affected_scope=AffectedScope(sites=["br3"], devices=["ce-br3"]),
        recommended_actions=[
            CopilotAction(step="reroute LSP", urgency=Urgency.IMMEDIATE, requires_approval=True)
        ],
        citation_universe=["RB-TUNNEL-001#0", "PB-TUNNEL-001"],
    )
    prompt = CopilotPrompt(system="s", user="u")
    r1 = client.complete_copilot(prompt, grounding)
    r2 = client.complete_copilot(prompt, grounding)
    assert r1.model_dump() == r2.model_dump()  # byte-identical
    assert r1.predicted_issue == IssueType.TUNNEL_DEGRADATION
    assert r1.time_to_impact_minutes == 4.0


def test_retriever_hybrid_finds_literal_token() -> None:
    """BM25 half of hybrid retrieval finds a rare literal token (NOC jargon)."""
    chunks = [
        Chunk(chunk_id="c1", text="The quick brown fox jumps lazily."),
        Chunk(
            chunk_id="c2",
            text="Interface GigabitEthernet0/0/1 on AS64512 saw %BGP-5-ADJCHANGE.",
        ),
    ]
    retriever = HybridRetriever().index(chunks)
    hits = retriever.retrieve("AS64512 %BGP-5-ADJCHANGE", top_k=1)
    assert hits and hits[0].chunk_id == "c2"
