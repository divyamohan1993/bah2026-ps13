"""Tests for the NETRA operator API + UI air-gap conformance (Workstream 6).

Two concerns:
  1. **API contract conformance** — a FastAPI ``TestClient`` hits every endpoint
     with the bundled ``DemoProvider`` and asserts (a) HTTP 200 and (b) the
     payload validates against the canonical ``netra.contracts`` models (the API
     response schema *is* the contract).
  2. **UI air-gap check** — greps ``ui/`` for external ``http(s)://`` references
     in authored assets (html/js/css) and asserts there are none (every asset is
     vendored locally), which is a hard requirement for the offline deployment.

Light deps only: fastapi + starlette TestClient (httpx) + pydantic. No uvicorn,
no network, no GPU. Run::

    pytest -q tests/test_api.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# --- locate the repo + ensure it is importable -----------------------------
REPO = Path(__file__).resolve().parents[1]
UI_DIR = REPO / "ui"

import sys  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Skip the whole API suite gracefully if FastAPI isn't installed (the air-gap
# grep test below does not need it and is collected separately).
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from netra.api import deps  # noqa: E402
from netra.api.app import create_app  # noqa: E402
from netra.contracts import (  # noqa: E402
    CopilotResponse,
    Incident,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient over a fresh app backed by the seeded DemoProvider."""
    deps.reset_provider()  # ensure the default (demo) provider, fresh seed
    app = create_app()
    with TestClient(app) as c:
        yield c
    deps.reset_provider()


# --------------------------------------------------------------------------- #
#  Health                                                                     #
# --------------------------------------------------------------------------- #
def test_health_ok(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "netra-api"
    assert body["provider"] == "demo"


# --------------------------------------------------------------------------- #
#  Incidents — contract-conformant Incident[]                                 #
# --------------------------------------------------------------------------- #
def test_incidents_are_contract_valid(client: TestClient) -> None:
    r = client.get("/api/incidents")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) >= 1
    incidents = [Incident.model_validate(d) for d in data]  # raises on drift
    # Prioritised: severities are non-decreasing in rank (P1 before P2 before P3).
    rank = {"P1": 0, "P2": 1, "P3": 2, "info": 3}
    sev_ranks = [rank[i.severity.value] for i in incidents]
    assert sev_ranks == sorted(sev_ranks), "incident queue must be prioritised"
    # Every incident must carry the fields the 3-answer card renders.
    for inc in incidents:
        assert inc.root_cause_hypothesis
        assert inc.recommended_playbook is not None
        assert len(inc.recommended_playbook.actions) >= 1
        # FusedRisk invariant: risk>0 => at least one contributing method.
        if inc.risk.risk_score > 0:
            assert inc.risk.contributing_methods


def test_incident_by_id_and_404(client: TestClient) -> None:
    # fetch a known id from the list
    first_id = client.get("/api/incidents").json()[0]["incident_id"]
    r = client.get(f"/api/incidents/{first_id}")
    assert r.status_code == 200
    Incident.model_validate(r.json())
    # unknown id -> 404
    assert client.get("/api/incidents/does-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
#  Situation snapshot — embeds Incident + CopilotResponse                     #
# --------------------------------------------------------------------------- #
def test_situation_snapshot(client: TestClient) -> None:
    r = client.get("/api/situation")
    assert r.status_code == 200
    body = r.json()
    # embedded contract objects validate
    Incident.model_validate(body["headline_incident"])
    copilot = CopilotResponse.model_validate(body["copilot"])
    # the 3-answer projection is present and consistent
    answers = body["answers"]
    assert answers["q1_what_when"]["predicted_issue"] == copilot.predicted_issue.value
    assert "root_cause_hypothesis" in answers["q2_why"]
    assert len(answers["q3_action"]["recommended_actions"]) >= 1
    assert body["fleet"]["incident_count"] >= 1


# --------------------------------------------------------------------------- #
#  Risk timeline                                                              #
# --------------------------------------------------------------------------- #
def test_risk_timeline_shape_and_lead_time(client: TestClient) -> None:
    r = client.get("/api/risk/timeline")
    assert r.status_code == 200
    tl = r.json()
    assert tl["points"], "timeline must have points"
    # monotone-ish rising curve: last risk clearly exceeds first (lead-time story)
    assert tl["points"][-1]["risk"] > tl["points"][0]["risk"]
    # each point is in [0,1] with a band
    for p in tl["points"]:
        assert 0.0 <= p["risk"] <= 1.0
        assert p["lower"] <= p["risk"] <= p["upper"]
    # a breach index is identified (risk crosses the threshold before the end)
    assert tl["breach_index"] is not None
    assert 0 <= tl["breach_index"] < len(tl["points"])
    # explicit entity selection works too
    ent = client.get("/api/incidents").json()[0]["root_cause_entity"]["entity_id"]
    r2 = client.get("/api/risk/timeline", params={"entity_id": ent})
    assert r2.status_code == 200
    assert r2.json()["entity_id"] == ent


# --------------------------------------------------------------------------- #
#  Topology graph                                                             #
# --------------------------------------------------------------------------- #
def test_topology_graph(client: TestClient) -> None:
    r = client.get("/api/topology")
    assert r.status_code == 200
    topo = r.json()
    nodes = topo["elements"]["nodes"]
    edges = topo["elements"]["edges"]
    assert len(nodes) >= 10  # the ~20-node reference topology
    assert len(edges) >= 5
    # every node carries the fields the UI colours/flags on
    ids = set()
    for n in nodes:
        d = n["data"]
        assert {"id", "entity_id", "risk", "is_root_cause", "in_blast_radius"} <= d.keys()
        assert 0.0 <= d["risk"] <= 1.0
        ids.add(d["id"])
    # edges reference existing nodes
    for e in edges:
        d = e["data"]
        assert d["source"] in ids and d["target"] in ids
    # at least one root-cause node is flagged (matches an incident)
    assert any(n["data"]["is_root_cause"] for n in nodes)
    assert topo["root_cause_devices"]


# --------------------------------------------------------------------------- #
#  Copilot — CopilotRequest -> CopilotResponse                                #
# --------------------------------------------------------------------------- #
def test_copilot_query_contract(client: TestClient) -> None:
    body = {
        "request_id": "test-req-1",
        "created_at": "2026-06-20T14:00:00Z",
        "operator_query": "Why is the hub uplink at risk and what do I do?",
    }
    r = client.post("/api/copilot/query", json=body)
    assert r.status_code == 200
    resp = CopilotResponse.model_validate(r.json())
    assert resp.request_id == "test-req-1"
    # contract guarantees: >=1 action, >=1 citation, confidence in [0,1]
    assert len(resp.recommended_actions) >= 1
    assert len(resp.citations) >= 1
    assert 0.0 <= resp.confidence_score <= 1.0
    # demo provider uses the deterministic template fallback shape
    assert resp.used_fallback is True


def test_copilot_routes_by_keyword(client: TestClient) -> None:
    # a BGP-flavoured question should resolve to the BGP incident
    r = client.post("/api/copilot/chat", json={"operator_query": "what is the bgp flap status?"})
    assert r.status_code == 200
    resp = CopilotResponse.model_validate(r.json())
    assert resp.predicted_issue.value == "bgp_route_flap"
    # a tunnel question -> tunnel degradation
    r2 = client.post("/api/copilot/chat", json={"operator_query": "tell me about the ipsec tunnel"})
    assert CopilotResponse.model_validate(r2.json()).predicted_issue.value == "tunnel_degradation"


def test_copilot_chat_minimal_body(client: TestClient) -> None:
    # the UI helper accepts just {operator_query}
    r = client.post("/api/copilot/chat", json={"operator_query": "status?"})
    assert r.status_code == 200
    resp = CopilotResponse.model_validate(r.json())
    assert resp.request_id  # server filled one in


def test_copilot_query_by_incident_ref(client: TestClient) -> None:
    inc_id = client.get("/api/incidents").json()[0]["incident_id"]
    body = {
        "request_id": "by-ref",
        "created_at": "2026-06-20T14:00:00Z",
        "auto_trigger": True,
        "incident_ref": inc_id,
    }
    r = client.post("/api/copilot/query", json=body)
    assert r.status_code == 200
    CopilotResponse.model_validate(r.json())


def test_copilot_rejects_bad_body(client: TestClient) -> None:
    # extra='forbid' on the contract => unknown field is a 422
    r = client.post(
        "/api/copilot/query",
        json={"request_id": "x", "created_at": "2026-06-20T14:00:00Z", "bogus": 1},
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
#  SSE live risk stream                                                       #
# --------------------------------------------------------------------------- #
def test_stream_risk_sse(client: TestClient) -> None:
    # bounded with limit so the response terminates deterministically
    r = client.get("/api/stream/risk", params={"limit": 3, "interval": 0.05})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    frames = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            frames.append(json.loads(line[len("data:") :].strip()))
    risk_frames = [f for f in frames if f.get("type") == "risk_tick"]
    assert len(risk_frames) == 3
    for f in risk_frames:
        assert 0.0 <= f["headline_risk"] <= 1.0
        assert isinstance(f["entities"], list)
    # the ETA should count down across frames (lead-time shrinking)
    etas = [f["headline_eta_minutes"] for f in risk_frames if f["headline_eta_minutes"] is not None]
    if len(etas) >= 2:
        assert etas[0] >= etas[-1]


# --------------------------------------------------------------------------- #
#  UI serving                                                                 #
# --------------------------------------------------------------------------- #
def test_ui_index_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "NETRA" in r.text
    # references the vendored lib, not a CDN
    assert "./vendor/cytoscape.min.js" in r.text


def test_ui_static_assets_served(client: TestClient) -> None:
    for path, ctype in [
        ("/app.js", "javascript"),
        ("/style.css", "css"),
        ("/vendor/cytoscape.min.js", "javascript"),
    ]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert ctype in r.headers["content-type"], (path, r.headers["content-type"])


def test_ui_vendor_path_traversal_blocked(client: TestClient) -> None:
    assert client.get("/vendor/nonexistent.js").status_code == 404


# --------------------------------------------------------------------------- #
#  Provider factory / live stub                                               #
# --------------------------------------------------------------------------- #
def test_live_provider_is_a_wiring_stub() -> None:
    from netra.api.providers import LiveProvider, make_provider

    lp = make_provider("live")
    assert isinstance(lp, LiveProvider)
    # every method is a documented NotImplementedError until the integrator wires it
    with pytest.raises(NotImplementedError):
        lp.incidents()
    with pytest.raises(NotImplementedError):
        lp.situation()


def test_demo_provider_is_deterministic() -> None:
    from netra.api.providers import DemoProvider

    a = DemoProvider(seed=42)
    b = DemoProvider(seed=42)
    # same seed => same incident ids, risks and ordering
    ia = [(i.incident_id, round(i.risk.risk_score, 6)) for i in a.incidents()]
    ib = [(i.incident_id, round(i.risk.risk_score, 6)) for i in b.incidents()]
    assert ia == ib


# --------------------------------------------------------------------------- #
#  AIR-GAP: no external references in the authored UI assets                  #
# --------------------------------------------------------------------------- #
# Allowed substrings in any URL-looking match (not external network deps):
#   - w3.org            : the XML/SVG/HTML namespace URI (not fetched)
#   - localhost/127.*   : same-host references
_ALLOWED_URL_SUBSTR = ("w3.org", "localhost", "127.0.0.1")

_URL_RE = re.compile(r"https?://[^\s\"'`)>}]+", re.IGNORECASE)


def _authored_ui_files() -> list[Path]:
    files: list[Path] = []
    for ext in ("*.html", "*.js", "*.css"):
        for p in UI_DIR.rglob(ext):
            # exclude vendored minified bundles — those legitimately carry
            # license-header comment URLs; they are checked separately below.
            if p.name.endswith(".min.js") or "vendor" in p.relative_to(UI_DIR).parts:
                continue
            files.append(p)
    return files


def test_ui_directory_exists() -> None:
    assert UI_DIR.is_dir(), f"ui/ not found at {UI_DIR}"
    assert (UI_DIR / "index.html").is_file()
    assert (UI_DIR / "app.js").is_file()
    assert (UI_DIR / "style.css").is_file()
    assert (UI_DIR / "vendor" / "cytoscape.min.js").is_file(), "Cytoscape must be vendored"


def test_ui_authored_assets_have_no_external_refs() -> None:
    """No CDNs / Google Fonts / remote scripts in authored html/js/css."""
    offenders: list[str] = []
    for p in _authored_ui_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in _URL_RE.findall(text):
            if not any(s in m for s in _ALLOWED_URL_SUBSTR):
                offenders.append(f"{p.relative_to(REPO)}: {m}")
    assert not offenders, "external references found in authored UI assets:\n" + "\n".join(offenders)


def test_ui_html_has_no_cdn_link_or_script_tags() -> None:
    """Belt-and-suspenders: <link>/<script> src/href must be local (./ or /)."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    # capture src= and href= attribute values
    for attr in ("src", "href"):
        for val in re.findall(rf'{attr}\s*=\s*"([^"]+)"', html):
            assert not val.lower().startswith(("http://", "https://", "//")), (
                f"non-local {attr}={val!r} in index.html"
            )


def test_vendored_min_js_only_has_license_urls() -> None:
    """The vendored minified lib may contain URLs, but only in license headers."""
    vendor = UI_DIR / "vendor"
    if not vendor.is_dir():
        pytest.skip("no vendor dir")
    license_hosts = ("opensource.org", "engelschall.com", "en.wikipedia.org", "cytoscape.org", "js.cytoscape.org")
    for p in vendor.rglob("*.min.js"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in _URL_RE.findall(text):
            assert any(h in m for h in license_hosts), (
                f"unexpected non-license URL in vendored {p.name}: {m}"
            )
