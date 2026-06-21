"""Analytics surface: situation snapshot, incident queue, risk timeline, topology.

Endpoints serialise ``netra.contracts`` types directly. ``GET /api/incidents``
declares ``response_model=list[Incident]`` so the OpenAPI schema *is* the
contract; the composite endpoints (situation/topology/timeline) return JSON
objects that embed contract-serialised sub-objects (documented per route).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from netra.api.deps import get_provider
from netra.api.providers import SituationProvider
from netra.contracts import Incident

router = APIRouter(tags=["analytics"])


@router.get("/situation")
def get_situation(provider: SituationProvider = Depends(get_provider)) -> dict:
    """Combined Q1/Q2/Q3 snapshot for the headline incident + fleet rollup.

    Shape: ``{generated_at, source, headline_incident: Incident, copilot:
    CopilotResponse, answers: {q1_what_when, q2_why, q3_action}, fleet}``.
    """
    return provider.situation()


@router.get("/incidents", response_model=list[Incident])
def get_incidents(
    provider: SituationProvider = Depends(get_provider),
) -> list[Incident]:
    """Prioritised triage queue (P1 first), each a contract ``Incident``."""
    return provider.incidents()


@router.get("/incidents/{incident_id}", response_model=Incident)
def get_incident(
    incident_id: str,
    provider: SituationProvider = Depends(get_provider),
) -> Incident:
    """A single incident by id (404 if unknown)."""
    from fastapi import HTTPException

    for inc in provider.incidents():
        if inc.incident_id == incident_id:
            return inc
    raise HTTPException(status_code=404, detail=f"unknown incident: {incident_id}")


@router.get("/risk/timeline")
def get_risk_timeline(
    entity_id: str | None = Query(
        default=None, description="Entity to chart; defaults to the headline uplink."
    ),
    provider: SituationProvider = Depends(get_provider),
) -> dict:
    """Risk-over-time series (with conformal band + breach marker).

    Shape: ``{entity_id, threshold, breach_index, points: [{timestamp, risk,
    lower, upper}]}``. The climb crossing ``threshold`` *before* the modelled
    breach is the visual proof of predictive lead time.
    """
    return provider.risk_timeline(entity_id)


@router.get("/topology")
def get_topology(provider: SituationProvider = Depends(get_provider)) -> dict:
    """Nodes/edges + per-node risk for the Cytoscape graph.

    Shape: ``{root_cause_devices, blast_radius_devices, elements: {nodes, edges}}``
    where each node/edge is a Cytoscape ``{data: {...}}`` element carrying
    ``risk``, ``is_root_cause`` and ``in_blast_radius`` flags.
    """
    return provider.topology()


__all__ = ["router"]
