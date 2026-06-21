"""Root-cause hypothesis ranking — centrality × earliest-onset × causal score.

Within a correlated group of anomalous entities, this module ranks candidate
root causes and emits a grounded one-paragraph hypothesis for the
:class:`netra.contracts.Incident` (architecture §Phase 4; research 07 A1.3/A1.5).

Ranking score per candidate node (research 07 A1.5):

    rca_score = centrality × earliest_onset × causal_score

where:
  * **centrality** — structural importance in the topology digital twin. We blend
    betweenness (bridge nodes whose failure fragments the network) and eigenvector
    centrality (influence), both from ``networkx`` (#47). Falls back to degree
    centrality on graphs too small/disconnected for eigenvector to converge.
  * **earliest_onset** — the earliest-firing entity is more likely the cause;
    later symptoms are effects. Scored as a [0,1] recency-inverse over the group's
    onset times.
  * **causal_score** — pairwise **Granger causality** (#49, ``statsmodels``):
    "does the candidate's past improve prediction of the other series?" Averaged
    over the other group members. Optional **PC / constraint-based** discovery via
    ``causal-learn`` refines the ranking when installed (try/except fallback).

Granger is a *ranking hint*, cross-checked by graph correlation + onset — never
presented as certainty (research 04 §14). CPU-only; the heavy ``causal-learn``
dependency is import-guarded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
import numpy as np

from netra.contracts import IssueType

from .graph import TopologyGraph

# ---------------------------------------------------------------------------
# Optional Granger causality (statsmodels is a core dep, but guard anyway so the
# module imports on a bare install).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by environment, not unit-tested both ways
    from statsmodels.tsa.stattools import grangercausalitytests as _granger_test

    _HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    _granger_test = None  # type: ignore[assignment]
    _HAVE_STATSMODELS = False

# Optional PC algorithm (causal-learn) — heavy/optional, import-guarded.
try:  # pragma: no cover
    from causallearn.search.ConstraintBased.PC import pc as _pc_search  # type: ignore

    _HAVE_CAUSAL_LEARN = True
except Exception:  # pragma: no cover
    _pc_search = None  # type: ignore[assignment]
    _HAVE_CAUSAL_LEARN = False


@dataclass
class RootCauseCandidate:
    """A scored root-cause hypothesis for one entity in a correlated group."""

    entity_id: str
    score: float
    centrality: float
    onset_score: float
    causal_score: float
    earliest_onset: datetime | None = None
    rank: int = 0
    rationale: str = ""
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------
def topology_centrality(
    topology: TopologyGraph, candidate_ids: Sequence[str]
) -> dict[str, float]:
    """Blend betweenness + eigenvector centrality for the candidate nodes → [0,1].

    Computed over the full topology (a node's importance is global, not just
    within the failing subgraph), then min-max normalised across the candidates so
    the most structurally central candidate scores ~1. Robust fallbacks keep this
    total on tiny or degenerate graphs.
    """
    g = topology.g
    # Restrict to candidates that actually map onto graph nodes.
    nodes = [topology.map_to_node(c) or c for c in candidate_ids]
    present = [n for n in nodes if n in g]
    if not present:
        return {c: 0.5 for c in candidate_ids}

    try:
        betw = nx.betweenness_centrality(g, normalized=True)
    except Exception:
        betw = {n: 0.0 for n in g.nodes}
    try:
        eig = nx.eigenvector_centrality_numpy(g)
    except Exception:
        try:
            eig = nx.eigenvector_centrality(g, max_iter=500, tol=1e-04)
        except Exception:
            # final fallback: degree centrality (always defined).
            eig = nx.degree_centrality(g)

    raw: dict[str, float] = {}
    for cid, n in zip(candidate_ids, nodes, strict=False):
        if n in g:
            raw[cid] = 0.5 * float(betw.get(n, 0.0)) + 0.5 * float(eig.get(n, 0.0))
        else:
            raw[cid] = 0.0

    return _minmax_norm(raw, floor=0.05)


def _minmax_norm(raw: Mapping[str, float], floor: float = 0.0) -> dict[str, float]:
    """Min-max normalise a score map to [floor,1]; constant maps → all 1.0."""
    if not raw:
        return {}
    vals = list(raw.values())
    lo, hi = min(vals), max(vals)
    if math.isclose(hi, lo):
        return {k: 1.0 for k in raw}
    span = hi - lo
    return {k: round(floor + (1 - floor) * (v - lo) / span, 4) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Onset
# ---------------------------------------------------------------------------
def onset_scores(onsets: Mapping[str, datetime]) -> dict[str, float]:
    """Score entities by earliest onset → [0,1] (earliest = 1.0).

    Linear in time between the group's first and last onset; the earliest-firing
    entity (the likely cause) scores 1.0, the latest scores ~0 (a pure symptom).
    """
    if not onsets:
        return {}
    times = list(onsets.values())
    t0 = min(times)
    t1 = max(times)
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return {k: 1.0 for k in onsets}
    out: dict[str, float] = {}
    for k, t in onsets.items():
        frac = (t - t0).total_seconds() / span
        out[k] = round(1.0 - frac, 4)  # earliest → 1, latest → 0
    return out


# ---------------------------------------------------------------------------
# Causality (Granger + optional PC)
# ---------------------------------------------------------------------------
def granger_causal_scores(
    series: Mapping[str, Sequence[float]],
    *,
    max_lag: int = 2,
) -> dict[str, float]:
    """Average pairwise Granger-causality strength per entity → [0,1].

    For each ordered pair (X → Y), ``grangercausalitytests`` answers "does X's past
    improve prediction of Y?". We convert the best p-value across lags to a strength
    ``1 - p`` and average over all Y for each candidate X. A high mean means "this
    series Granger-causes many others" → a more likely root cause.

    Degrades gracefully: too-short / constant / NaN series, or a missing
    ``statsmodels``, yield neutral 0.5 scores so RCA still works on centrality +
    onset alone.
    """
    keys = list(series.keys())
    if len(keys) < 2 or not _HAVE_STATSMODELS:
        return {k: 0.5 for k in keys}

    arrs: dict[str, np.ndarray] = {}
    for k in keys:
        a = np.asarray(list(series[k]), dtype=float)
        arrs[k] = a

    raw: dict[str, float] = {}
    for cause in keys:
        x = arrs[cause]
        strengths: list[float] = []
        for effect in keys:
            if effect == cause:
                continue
            y = arrs[effect]
            s = _granger_strength(cause_series=x, effect_series=y, max_lag=max_lag)
            if s is not None:
                strengths.append(s)
        raw[cause] = float(np.mean(strengths)) if strengths else 0.5

    # normalise across candidates so the relatively-most-causal stands out,
    # but keep a floor so causality never zeroes the product on its own.
    return _minmax_norm(raw, floor=0.25)


def _granger_strength(
    *, cause_series: np.ndarray, effect_series: np.ndarray, max_lag: int
) -> float | None:
    """Return ``1 - min_p`` for "cause Granger-causes effect", or None if undefined.

    ``grangercausalitytests`` expects a 2-column array ``[effect, cause]`` and tests
    whether the *second* column helps predict the *first*.
    """
    n = min(len(cause_series), len(effect_series))
    if n < (max_lag * 2 + 4):
        return None
    cause = cause_series[-n:]
    effect = effect_series[-n:]
    # Constant series carry no predictive information → undefined.
    if np.nanstd(cause) < 1e-9 or np.nanstd(effect) < 1e-9:
        return None
    if not (np.all(np.isfinite(cause)) and np.all(np.isfinite(effect))):
        return None
    data = np.column_stack([effect, cause])
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _granger_test(data, maxlag=max_lag, verbose=False)
    except Exception:
        return None
    pvals: list[float] = []
    for lag in range(1, max_lag + 1):
        try:
            p = res[lag][0]["ssr_ftest"][1]
            if np.isfinite(p):
                pvals.append(float(p))
        except Exception:
            continue
    if not pvals:
        return None
    return max(0.0, min(1.0, 1.0 - min(pvals)))


def pc_causal_scores(series: Mapping[str, Sequence[float]]) -> dict[str, float] | None:
    """Optional PC-algorithm out-degree score per entity via ``causal-learn``.

    Returns ``None`` when ``causal-learn`` is unavailable or the data is unfit, so
    callers transparently fall back to Granger. A node with more outgoing causal
    edges in the discovered DAG is a more likely root cause.
    """
    if not _HAVE_CAUSAL_LEARN:
        return None
    keys = list(series.keys())
    if len(keys) < 3:
        return None
    try:
        n = min(len(series[k]) for k in keys)
        if n < 10:
            return None
        mat = np.column_stack([np.asarray(list(series[k])[-n:], dtype=float) for k in keys])
        if not np.all(np.isfinite(mat)):
            return None
        cg = _pc_search(mat, 0.05, show_progress=False)
        graph = cg.G.graph  # adjacency matrix (causal-learn encoding)
        out_deg: dict[str, float] = {}
        m = len(keys)
        for i, k in enumerate(keys):
            # count directed edges i -> j (encoding: graph[j,i]==1 and graph[i,j]==-1)
            deg = 0
            for j in range(m):
                if i == j:
                    continue
                if graph[j, i] == 1 and graph[i, j] == -1:
                    deg += 1
            out_deg[k] = float(deg)
        return _minmax_norm(out_deg, floor=0.25)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level ranking
# ---------------------------------------------------------------------------
def rank_root_causes(
    topology: TopologyGraph,
    candidate_ids: Sequence[str],
    *,
    onsets: Mapping[str, datetime] | None = None,
    series: Mapping[str, Sequence[float]] | None = None,
    max_lag: int = 2,
) -> list[RootCauseCandidate]:
    """Rank candidate entities as root causes, highest score first.

    Combines topology centrality, earliest-onset and Granger (or PC) causal score
    multiplicatively (a product so a near-zero factor demotes a candidate). Every
    factor has a small floor so a single missing signal cannot zero the product.
    """
    candidates = list(dict.fromkeys(candidate_ids))  # de-dup, preserve order
    if not candidates:
        return []

    cen = topology_centrality(topology, candidates)
    ons = onset_scores(onsets) if onsets else {}
    if series:
        caus = pc_causal_scores(series) or granger_causal_scores(series, max_lag=max_lag)
    else:
        caus = {}

    out: list[RootCauseCandidate] = []
    for cid in candidates:
        c = cen.get(cid, 0.5)
        o = ons.get(cid, 1.0) if ons else 1.0
        k = caus.get(cid, 0.5) if caus else 0.5
        # floors prevent any single 0 from collapsing the product entirely.
        c_f, o_f, k_f = max(c, 0.05), max(o, 0.05), max(k, 0.05)
        score = round(c_f * o_f * k_f, 6)
        out.append(
            RootCauseCandidate(
                entity_id=cid,
                score=score,
                centrality=round(c, 4),
                onset_score=round(o, 4),
                causal_score=round(k, 4),
                earliest_onset=onsets.get(cid) if onsets else None,
                components={"centrality": c, "onset": o, "causal": k},
            )
        )

    out.sort(key=lambda rc: rc.score, reverse=True)
    for i, rc in enumerate(out):
        rc.rank = i + 1
        rc.rationale = _rationale(rc, topology)
    return out


def _rationale(rc: RootCauseCandidate, topology: TopologyGraph) -> str:
    """Build a short, grounded justification string for a candidate."""
    parts = []
    if rc.centrality >= 0.66:
        parts.append("high topology centrality (a structural bridge node)")
    elif rc.centrality >= 0.33:
        parts.append("moderate topology centrality")
    if rc.onset_score >= 0.8:
        parts.append("earliest anomaly onset in the correlated group")
    elif rc.onset_score <= 0.3:
        parts.append("late onset (more likely a downstream symptom)")
    if rc.causal_score >= 0.66:
        parts.append("strong causal influence (Granger) over the other signals")
    if not parts:
        parts.append("balanced centrality/onset/causal evidence")
    return ", ".join(parts)


def build_hypothesis(
    top: RootCauseCandidate,
    topology: TopologyGraph,
    *,
    issue: IssueType,
    n_correlated: int,
    affected_sites: Sequence[str] | None = None,
) -> str:
    """Compose a one-paragraph, grounded root-cause hypothesis for the Incident.

    Strictly templated from computed facts (entity, issue, centrality/onset/causal
    rationale, correlated count, affected sites) so it is auditable and never
    fabricated. The copilot may quote/expand this, grounded.
    """
    ref = topology.entity_ref(top.entity_id)
    where = f"{ref.device or top.entity_id} at site {ref.site} (role {ref.role.value})"
    issue_phrase = _ISSUE_PHRASE.get(issue, "anomalous behaviour")
    sites_txt = ""
    if affected_sites:
        shown = ", ".join(sorted(set(affected_sites))[:5])
        sites_txt = f" Downstream sites at risk: {shown}."
    return (
        f"Most likely root cause: {where}, exhibiting {issue_phrase}. "
        f"Selected by {top.rationale} "
        f"(centrality={top.centrality:.2f}, onset={top.onset_score:.2f}, "
        f"causal={top.causal_score:.2f}). "
        f"It correlates {n_correlated} co-occurring symptom signal(s) into one "
        f"incident.{sites_txt} "
        f"Granger/PC causality is a ranking hint cross-checked by graph "
        f"correlation and onset ordering; confirm before remediation."
    )


_ISSUE_PHRASE: dict[IssueType, str] = {
    IssueType.INTERFACE_CONGESTION: "interface utilisation trending toward saturation",
    IssueType.LATENCY_DRIFT: "a sustained latency drift",
    IssueType.BGP_ROUTE_FLAP: "BGP route-flap churn driving a reroute cascade",
    IssueType.OSPF_CONVERGENCE_STRESS: "OSPF LSA/SPF convergence stress",
    IssueType.TUNNEL_DEGRADATION: "overlay tunnel loss/jitter degradation",
    IssueType.MPLS_UNDERLAY_FAILURE: "an intermittent MPLS underlay fault",
    IssueType.POLICY_DRIFT: "a controller-driven policy drift fanning out to many sites",
    IssueType.PATH_ASYMMETRY: "forward/reverse path asymmetry",
    IssueType.NONE: "anomalous behaviour",
}


__all__ = [
    "RootCauseCandidate",
    "rank_root_causes",
    "topology_centrality",
    "onset_scores",
    "granger_causal_scores",
    "pc_causal_scores",
    "build_hypothesis",
]
