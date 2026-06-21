/* NETRA operator console — vanilla JS (no build step, no external runtime deps).
 *
 * Talks to the FastAPI backend (same origin) and renders:
 *   - a Cytoscape.js topology graph (root-cause highlighted, blast-radius shaded)
 *   - a hand-rolled <canvas> risk timeline (risk rising before impact)
 *   - the 3-answer incident card (Q1 what/when · Q2 why · Q3 action)
 *   - the prioritised incident queue
 *   - a copilot chat box that POSTs to /api/copilot/chat
 *   - a live SSE stream (/api/stream/risk) recolouring the graph + ticking the card
 *
 * Everything is offline: the only third-party code is the vendored Cytoscape.js.
 */
"use strict";

const API = ""; // same-origin; the API serves this page.

const state = {
  incidents: [],
  selectedId: null,
  topology: null,
  cy: null,
  timeline: null,
  evtSource: null,
  liveRisk: {}, // entity_id -> latest risk from the stream
};

/* ----------------------------- helpers --------------------------------- */
async function getJSON(path) {
  const r = await fetch(API + path, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
async function postJSON(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}
function el(id) {
  return document.getElementById(id);
}
function riskColor(r) {
  // green (low) -> yellow -> red (high)
  r = Math.max(0, Math.min(1, r || 0));
  const stops = [
    [0.0, [46, 111, 78]],
    [0.5, [241, 196, 15]],
    [1.0, [255, 91, 91]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (r >= stops[i][0] && r <= stops[i + 1][0]) {
      a = stops[i]; b = stops[i + 1]; break;
    }
  }
  const t = (r - a[0]) / (b[0] - a[0] || 1);
  const c = a[1].map((v, i) => Math.round(v + (b[1][i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function fmtMinutes(m) {
  if (m === null || m === undefined) return "no breach forecast";
  if (m < 1) return "< 1 min to impact";
  return `~${m} min to impact`;
}

/* ----------------------------- topology -------------------------------- */
function initTopology(topo) {
  state.topology = topo;
  if (typeof cytoscape === "undefined") {
    el("topo-hint").textContent =
      "Cytoscape failed to load from ./vendor — topology unavailable.";
    return;
  }
  const elements = [
    ...topo.elements.nodes.map((n) => ({ data: n.data })),
    ...topo.elements.edges.map((e) => ({ data: e.data })),
  ];
  state.cy = cytoscape({
    container: el("cy"),
    elements,
    wheelSensitivity: 0.2,
    style: [
      {
        selector: "node",
        style: {
          "background-color": (n) => riskColor(n.data("risk")),
          label: "data(label)",
          color: "#e6edf7",
          "font-size": 9,
          "text-valign": "bottom",
          "text-margin-y": 3,
          width: (n) => 18 + 26 * (n.data("risk") || 0),
          height: (n) => 18 + 26 * (n.data("risk") || 0),
          "border-width": 2,
          "border-color": "#0b1020",
        },
      },
      {
        selector: 'node[?is_root_cause]',
        style: {
          "border-width": 4,
          "border-color": "#ff3b6b",
          "background-color": "#ff3b6b",
        },
      },
      {
        selector: 'node[?in_blast_radius]',
        style: { "border-color": "#ffb24d", "border-width": 3 },
      },
      {
        selector: "node.dim",
        style: { opacity: 0.25 },
      },
      {
        selector: "edge",
        style: {
          width: (e) => 1 + 4 * (e.data("risk") || 0),
          "line-color": (e) => riskColor(e.data("risk")),
          "curve-style": "bezier",
          opacity: 0.7,
          "line-style": (e) => (e.data("kind") === "overlay" ? "dashed" : "solid"),
        },
      },
      { selector: "edge.dim", style: { opacity: 0.12 } },
    ],
    layout: { name: "cose", animate: false, padding: 20, nodeRepulsion: 9000, idealEdgeLength: 70 },
  });

  state.cy.on("tap", "node", (evt) => {
    const dev = evt.target.data("device") || evt.target.id();
    // Find an incident whose root cause or blast radius includes this device.
    const inc =
      state.incidents.find(
        (i) => i.root_cause_entity && i.root_cause_entity.device === dev
      ) ||
      state.incidents.find((i) =>
        (i.blast_radius.affected_devices || []).includes(dev)
      );
    if (inc) selectIncident(inc.incident_id, { fromGraph: true });
  });
}

function highlightTopologyForIncident(inc) {
  if (!state.cy) return;
  const focus = new Set();
  if (inc.root_cause_entity) focus.add(inc.root_cause_entity.device);
  (inc.blast_radius.affected_devices || []).forEach((d) => focus.add(d));
  state.cy.batch(() => {
    state.cy.elements().removeClass("dim");
    if (focus.size === 0) return;
    state.cy.nodes().forEach((n) => {
      if (!focus.has(n.data("device"))) n.addClass("dim");
    });
    state.cy.edges().forEach((e) => {
      if (!focus.has(e.data("source")) || !focus.has(e.data("target"))) e.addClass("dim");
    });
  });
}

function applyLiveRiskToTopology() {
  if (!state.cy) return;
  state.cy.batch(() => {
    Object.entries(state.liveRisk).forEach(([entityId, risk]) => {
      const device = entityId.split(":")[1];
      const node = state.cy.getElementById(device);
      if (node && node.nonempty()) node.data("risk", risk);
    });
  });
}

/* --------------------------- risk timeline ----------------------------- */
async function loadTimeline(entityId) {
  const q = entityId ? `?entity_id=${encodeURIComponent(entityId)}` : "";
  const tl = await getJSON(`/api/risk/timeline${q}`);
  state.timeline = tl;
  el("timeline-entity").textContent = tl.entity_id ? `· ${tl.entity_id}` : "";
  drawTimeline(tl);
}

function drawTimeline(tl) {
  const canvas = el("risk-chart");
  const ctx = canvas.getContext("2d");
  // Handle HiDPI + responsive width.
  const cssW = canvas.clientWidth || 600;
  const cssH = canvas.clientHeight || 180;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 34, r: 12, t: 12, b: 20 };
  const W = cssW - pad.l - pad.r;
  const H = cssH - pad.t - pad.b;
  const pts = tl.points || [];
  if (pts.length === 0) return;
  const x = (i) => pad.l + (W * i) / (pts.length - 1);
  const y = (v) => pad.t + H * (1 - Math.max(0, Math.min(1, v)));

  // grid + y labels (0, .5, 1)
  ctx.strokeStyle = "#233152";
  ctx.fillStyle = "#93a3c0";
  ctx.font = "10px ui-monospace, monospace";
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach((g) => {
    ctx.globalAlpha = g === 0 || g === 1 ? 0.5 : 0.22;
    ctx.beginPath();
    ctx.moveTo(pad.l, y(g));
    ctx.lineTo(pad.l + W, y(g));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(g.toFixed(2), 4, y(g) + 3);
  });

  // threshold line
  if (tl.threshold !== undefined && tl.threshold !== null) {
    ctx.strokeStyle = "#f1c40f";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, y(tl.threshold));
    ctx.lineTo(pad.l + W, y(tl.threshold));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f1c40f";
    ctx.fillText("action threshold", pad.l + 4, y(tl.threshold) - 4);
  }

  // confidence band (upper/lower)
  ctx.beginPath();
  pts.forEach((p, i) => {
    const yy = y(p.upper !== undefined ? p.upper : p.risk);
    i === 0 ? ctx.moveTo(x(i), yy) : ctx.lineTo(x(i), yy);
  });
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    ctx.lineTo(x(i), y(p.lower !== undefined ? p.lower : p.risk));
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(78,161,255,0.14)";
  ctx.fill();

  // risk line (gradient-ish: colour each segment by its risk)
  ctx.lineWidth = 2;
  for (let i = 1; i < pts.length; i++) {
    ctx.strokeStyle = riskColor((pts[i].risk + pts[i - 1].risk) / 2);
    ctx.beginPath();
    ctx.moveTo(x(i - 1), y(pts[i - 1].risk));
    ctx.lineTo(x(i), y(pts[i].risk));
    ctx.stroke();
  }

  // breach marker
  if (tl.breach_index !== null && tl.breach_index !== undefined && pts[tl.breach_index]) {
    const bi = tl.breach_index;
    ctx.strokeStyle = "#ff5b5b";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x(bi), pad.t);
    ctx.lineTo(x(bi), pad.t + H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff5b5b";
    ctx.beginPath();
    ctx.arc(x(bi), y(pts[bi].risk), 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText("breach", Math.min(x(bi) + 4, pad.l + W - 36), pad.t + 10);
  }

  // "now" dot at the end
  const last = pts[pts.length - 1];
  ctx.fillStyle = "#e6edf7";
  ctx.beginPath();
  ctx.arc(x(pts.length - 1), y(last.risk), 3, 0, Math.PI * 2);
  ctx.fill();
}

/* --------------------------- 3-answer card ----------------------------- */
function renderAnswers(inc, copilot) {
  // Severity tag
  const sevTag = el("answer-sev");
  sevTag.textContent = inc.severity;
  sevTag.className = "sev-tag sev-" + inc.severity;

  // Q1 — what & when
  el("q1-issue").textContent = (copilot.predicted_issue || inc.predicted_issue).replace(/_/g, " ");
  el("q1-eta").textContent = fmtMinutes(copilot.time_to_impact_minutes);
  const conf = Math.round((copilot.confidence_score || 0) * 100);
  el("q1-conf-fill").style.width = conf + "%";
  el("q1-conf-label").textContent = conf + "% confidence";
  const scope = el("q1-scope");
  scope.innerHTML = "";
  (copilot.affected_scope.sites || []).forEach((s) =>
    scope.appendChild(chip(s, "site"))
  );
  (copilot.affected_scope.services_or_vpns || []).forEach((v) =>
    scope.appendChild(chip(v, "vpn"))
  );
  if (inc.blast_radius.affected_flow_count != null)
    scope.appendChild(chip(`${inc.blast_radius.affected_flow_count} flows`, ""));

  // Q2 — why
  el("q2-root").textContent = copilot.root_cause_hypothesis;
  const sigUl = el("q2-signals");
  sigUl.innerHTML = "";
  const maxShap = Math.max(
    0.01,
    ...copilot.contributing_signals.map((s) => Math.abs(s.shap_contribution || 0))
  );
  copilot.contributing_signals.forEach((s) => {
    const li = document.createElement("li");
    const bar = document.createElement("div");
    bar.className = "sig-bar";
    const span = document.createElement("span");
    span.style.width = `${Math.round((Math.abs(s.shap_contribution || 0) / maxShap) * 100)}%`;
    bar.appendChild(span);
    const txt = document.createElement("div");
    txt.className = "sig-text";
    txt.innerHTML = `<strong>${escapeHTML(s.signal)}</strong><br><span class="obs">${escapeHTML(
      s.observation || ""
    )}</span>`;
    li.appendChild(bar);
    li.appendChild(txt);
    sigUl.appendChild(li);
  });

  // Q3 — actions
  const ol = el("q3-actions");
  ol.innerHTML = "";
  copilot.recommended_actions.forEach((a) => {
    const li = document.createElement("li");
    const step = document.createElement("div");
    step.className = "act-step";
    step.textContent = a.step;
    const meta = document.createElement("div");
    meta.className = "act-meta";
    meta.appendChild(badge(a.urgency, "u-" + a.urgency));
    meta.appendChild(
      a.requires_approval ? badge("approval", "approve") : badge("auto-ok", "auto")
    );
    if (a.runbook_ref) meta.appendChild(badge(a.runbook_ref, "ref"));
    li.appendChild(step);
    li.appendChild(meta);
    ol.appendChild(li);
  });
}
function chip(text, cls) {
  const s = document.createElement("span");
  s.className = "chip " + (cls || "");
  s.textContent = text;
  return s;
}
function badge(text, cls) {
  const s = document.createElement("span");
  s.className = "badge " + (cls || "");
  s.textContent = text;
  return s;
}
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* --------------------------- incident queue ---------------------------- */
function renderQueue() {
  const ul = el("incident-queue");
  ul.innerHTML = "";
  state.incidents.forEach((inc) => {
    const li = document.createElement("li");
    li.dataset.id = inc.incident_id;
    if (inc.incident_id === state.selectedId) li.classList.add("selected");
    const dot = document.createElement("span");
    dot.className = "riskdot";
    dot.style.background = riskColor(inc.risk.risk_score);
    const mid = document.createElement("div");
    const rc = inc.root_cause_entity ? inc.root_cause_entity.device : "—";
    mid.innerHTML =
      `<div class="q-issue">${escapeHTML(inc.predicted_issue.replace(/_/g, " "))} ` +
      `<span class="sev-tag sev-${inc.severity}">${inc.severity}</span></div>` +
      `<div class="q-sub">${escapeHTML(inc.incident_id)} · root: ${escapeHTML(rc)}</div>`;
    const risk = document.createElement("div");
    risk.className = "q-risk";
    const eta = inc.risk.time_to_impact && inc.risk.time_to_impact.eta_seconds != null
      ? `${Math.round(inc.risk.time_to_impact.eta_seconds / 60)}m`
      : "—";
    risk.innerHTML = `${(inc.risk.risk_score * 100).toFixed(0)}%<br><span class="q-sub">${eta}</span>`;
    li.appendChild(dot);
    li.appendChild(mid);
    li.appendChild(risk);
    li.addEventListener("click", () => selectIncident(inc.incident_id));
    ul.appendChild(li);
  });
}

/* ----------------------- selection orchestration ----------------------- */
async function selectIncident(id, opts = {}) {
  const inc = state.incidents.find((i) => i.incident_id === id);
  if (!inc) return;
  state.selectedId = id;
  renderQueue();

  // Ask the copilot for this incident (so the card matches an LLM/fallback answer).
  let copilot;
  try {
    copilot = await postJSON("/api/copilot/query", {
      request_id: `card-${id}-${Date.now()}`,
      created_at: new Date().toISOString(),
      auto_trigger: true,
      incident_ref: id,
    });
  } catch (e) {
    copilot = synthCopilotFromIncident(inc); // resilient offline fallback
  }
  renderAnswers(inc, copilot);
  highlightTopologyForIncident(inc);
  // Centre the root-cause node.
  if (state.cy && inc.root_cause_entity) {
    const node = state.cy.getElementById(inc.root_cause_entity.device);
    if (node && node.nonempty()) state.cy.animate({ center: { eles: node }, zoom: 1.1 }, { duration: 250 });
  }
  // Load the risk timeline for the root-cause entity.
  const ent = inc.root_cause_entity ? inc.root_cause_entity.entity_id : null;
  try { await loadTimeline(ent); } catch (e) { /* keep prior chart */ }
}

// If the copilot endpoint is unreachable, build a card straight from the incident.
function synthCopilotFromIncident(inc) {
  const tti = inc.risk.time_to_impact;
  return {
    predicted_issue: inc.predicted_issue,
    confidence_score: inc.risk.calibrated_confidence,
    time_to_impact_minutes:
      tti && tti.eta_seconds != null ? Math.round((tti.eta_seconds / 60) * 10) / 10 : null,
    root_cause_hypothesis: inc.root_cause_hypothesis,
    contributing_signals: (inc.contributing_signals || []).map((s) => ({
      signal: s.signal,
      observation: s.observation || s.human_explanation,
      shap_contribution: s.shap_value,
    })),
    affected_scope: {
      sites: inc.blast_radius.affected_sites,
      devices: inc.blast_radius.affected_devices,
      services_or_vpns: inc.blast_radius.affected_services_or_vpns,
    },
    recommended_actions: (inc.recommended_playbook
      ? inc.recommended_playbook.actions
      : []
    ).map((a) => ({
      step: a.description,
      runbook_ref: a.runbook_ref,
      urgency: a.urgency,
      requires_approval: a.requires_approval,
    })),
    used_fallback: true,
    model_id: "ui-local-fallback",
  };
}

/* ------------------------------- copilot ------------------------------- */
function addMsg(cls, html) {
  const log = el("chat-log");
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

async function askCopilot(query) {
  addMsg("user", escapeHTML(query));
  const thinking = addMsg("bot thinking", "consulting analytics + runbooks…");
  try {
    const resp = await postJSON("/api/copilot/chat", {
      operator_query: query,
      incident_ref: state.selectedId || undefined,
    });
    thinking.remove();
    renderCopilotMessage(resp);
    // If the answer is about a specific incident, sync the card/graph.
    const match = state.incidents.find((i) => i.predicted_issue === resp.predicted_issue);
    if (match && match.incident_id !== state.selectedId) {
      // soft-sync without re-querying copilot
      state.selectedId = match.incident_id;
      renderQueue();
      renderAnswers(match, resp);
      highlightTopologyForIncident(match);
    }
  } catch (e) {
    thinking.remove();
    addMsg("bot", `Copilot unavailable (${escapeHTML(e.message)}).`);
  }
}

function renderCopilotMessage(resp) {
  const issue = (resp.predicted_issue || "").replace(/_/g, " ");
  const conf = Math.round((resp.confidence_score || 0) * 100);
  const tti = fmtMinutes(resp.time_to_impact_minutes);
  const sigs = (resp.contributing_signals || [])
    .slice(0, 3)
    .map((s) => `• ${escapeHTML(s.signal)} — ${escapeHTML(s.observation || "")}`)
    .join("\n");
  const acts = (resp.recommended_actions || [])
    .slice(0, 4)
    .map(
      (a, i) =>
        `${i + 1}. ${escapeHTML(a.step)}${a.requires_approval ? "  [approval]" : "  [auto]"}`
    )
    .join("\n");
  const cites = (resp.citations || []).map((c) => escapeHTML(c)).join(", ");
  const modelTag = resp.used_fallback
    ? `template-fallback`
    : escapeHTML(resp.model_id || "llm");
  el("copilot-mode").textContent = resp.used_fallback ? "mode: template fallback" : "mode: LLM";
  const body =
    `<b>Q1 ·</b> ${escapeHTML(issue)} — ${tti} (${conf}% conf)\n\n` +
    `<b>Q2 ·</b> ${escapeHTML(resp.root_cause_hypothesis)}\n${sigs ? sigs + "\n" : ""}\n` +
    `<b>Q3 ·</b>\n${acts || "Gather diagnostics and monitor."}`;
  const div = addMsg("bot", body);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML =
    `model: ${modelTag}` +
    (resp.grounding_score != null ? ` · grounding ${(resp.grounding_score * 100).toFixed(0)}%` : "") +
    (resp.insufficient_context ? ` · <span style="color:#f1c40f">abstained</span>` : "") +
    (cites ? `<br><span class="cites">citations: ${cites}</span>` : "");
  div.appendChild(meta);
}

function initCopilot() {
  el("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = el("chat-input");
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    askCopilot(q);
  });
  const suggestions = [
    "Why is the hub uplink at risk and what do I do?",
    "What's the BGP situation?",
    "Tell me about the tunnel degradation",
  ];
  const wrap = el("chat-suggestions");
  suggestions.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s;
    b.addEventListener("click", () => askCopilot(s));
    wrap.appendChild(b);
  });
}

/* ------------------------------ live SSE ------------------------------- */
function initStream() {
  if (typeof EventSource === "undefined") {
    setLive(false);
    return;
  }
  try {
    const es = new EventSource(API + "/api/stream/risk?interval=2");
    state.evtSource = es;
    es.addEventListener("risk", (ev) => {
      const frame = JSON.parse(ev.data);
      setLive(true);
      // update live risk map + recolour graph
      (frame.entities || []).forEach((e) => {
        state.liveRisk[e.entity_id] = e.risk;
      });
      if (frame.headline_entity != null)
        state.liveRisk[frame.headline_entity] = frame.headline_risk;
      applyLiveRiskToTopology();
      // tick the headline ETA on the card if the headline incident is selected
      const headline = state.incidents[0];
      if (headline && state.selectedId === headline.incident_id && frame.headline_eta_minutes != null) {
        el("q1-eta").textContent = fmtMinutes(frame.headline_eta_minutes);
      }
      el("pill-clock").textContent = new Date(frame.timestamp).toLocaleTimeString();
    });
    es.onerror = () => setLive(false);
  } catch (e) {
    setLive(false);
  }
}
function setLive(on) {
  const pill = el("pill-live");
  pill.classList.toggle("stale", !on);
  pill.lastChild.textContent = on ? " live" : " offline-stream";
}

/* ------------------------------- bootstrap ----------------------------- */
async function boot() {
  initCopilot();
  // header status
  try {
    const h = await getJSON("/api/health");
    el("pill-provider").textContent = `provider: ${h.provider}`;
  } catch (e) {
    el("pill-provider").textContent = "provider: offline";
  }

  // topology + incidents in parallel
  const [topo, incidents] = await Promise.all([
    getJSON("/api/topology").catch(() => ({ elements: { nodes: [], edges: [] } })),
    getJSON("/api/incidents").catch(() => []),
  ]);
  state.incidents = incidents;
  el("pill-incidents").textContent = `incidents: ${incidents.length}`;
  initTopology(topo);
  renderQueue();

  // select the top (headline) incident
  if (incidents.length) {
    await selectIncident(incidents[0].incident_id);
  } else {
    el("q2-root").textContent = "No incidents — network healthy.";
  }

  initStream();
  window.addEventListener("resize", () => {
    if (state.timeline) drawTimeline(state.timeline);
  });
}

document.addEventListener("DOMContentLoaded", boot);
