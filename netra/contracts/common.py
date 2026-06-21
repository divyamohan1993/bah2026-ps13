"""Shared base model and small reusable value objects for NETRA contracts.

Keeping a single ``NetraModel`` base centralises Pydantic v2 model config
(immutability-friendly defaults, populate-by-name, enum-by-value serialisation)
so every contract behaves consistently when (de)serialised across the NATS bus,
the API boundary and the LLM context.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import DeviceRole, SiteType


class NetraModel(BaseModel):
    """Base class for all NETRA contract models.

    Config choices and why:
      * ``use_enum_values=False`` — keep enum *members* in-process (so code can
        compare against ``IssueType.X``); they still serialise to their string
        value because every NETRA enum subclasses ``str``.
      * ``populate_by_name=True`` — allow constructing by field name even when a
        serialisation ``alias`` is set.
      * ``extra="forbid"`` — reject unknown fields. Contracts are the interface;
        silently swallowing typos hides integration bugs between workstreams.
      * ``validate_assignment=True`` — keep instances valid after mutation.
      * ``ser_json_timedelta="float"`` — stable numeric durations on the wire.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
        ser_json_timedelta="float",
        str_strip_whitespace=True,
    )


class EntityRef(NetraModel):
    """Canonical, stable identifier for a network entity.

    Almost every analytics artifact is *about* an entity (an interface, a tunnel,
    a peer, a device, a site). A single ``entity_id`` string is the join key used
    everywhere (features, forecasts, anomaly scores, incidents, graph nodes), and
    the optional structured fields let the UI/graph render and group without
    re-parsing the id.

    Convention for ``entity_id`` (colon-delimited, stable, sortable)::

        "<site>:<device>:<role>[:<sub>]"
        e.g. "hub1:pe-hub1:PE:eth1"            (an interface)
             "br3:ce-br3:CE:tunnel-hub"        (an overlay tunnel)
             "core:p2:P"                        (a core router)
             "rr1:rr1:RR:peer-pe-dc1"           (a BGP peering)
    """

    entity_id: str = Field(
        ...,
        description="Stable colon-delimited entity id; the universal join key.",
        examples=["hub1:pe-hub1:PE:eth1", "br3:ce-br3:CE:tunnel-hub"],
    )
    site: str = Field(..., description="Site name this entity belongs to.")
    device: str = Field(..., description="Device/host name.")
    role: DeviceRole = Field(..., description="Device role in the topology.")
    site_type: SiteType | None = Field(
        default=None, description="Class of the site (datacenter/hub/branch/core)."
    )
    sub: str | None = Field(
        default=None,
        description="Optional sub-entity: interface, tunnel, peer, vrf, lsp.",
        examples=["eth1", "tunnel-hub", "peer-pe-dc1", "vrf:CORP"],
    )


__all__ = ["NetraModel", "EntityRef"]
