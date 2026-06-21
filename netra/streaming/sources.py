"""Transport adapters feeding the :class:`FeatureEngine` (Workstream 2).

The engine itself is transport-agnostic (it consumes any iterable of
``TelemetryRecord``). This module supplies the two concrete feeds the build plan
calls for, each import-light and contract-only:

  * :func:`iter_telemetry_source` — adapt a ``netra.datagen`` ``TelemetrySource``
    (or *any* object exposing a compatible ``stream()/__iter__``) into a plain
    iterator of ``TelemetryRecord``. **We do not import ``netra.datagen``** —
    the source is dependency-injected, so the streaming module stays decoupled
    from the generator (dual-source abstraction) and the CPU-only smoke test can
    pass a hand-built list.
  * :class:`NatsTelemetrySource` — an optional NATS JetStream consumer that
    deserialises ``telemetry.>`` messages into ``TelemetryRecord`` and drives the
    engine. ``nats-py`` is import-guarded so the module imports (and the CPU-only
    path runs) even when NATS is not installed.

Both yield records the engine folds in O(1). Only non-numeric carriers
(``SyslogEvent`` / ``RoutingEvent`` for routing-rate features) need bespoke
mapping; helpers for that live in :func:`routing_event_to_records`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from netra.contracts import (
    MetricName,
    RoutingEvent,
    TelemetryRecord,
)

from .engine import FeatureEngine

__all__ = [
    "SupportsTelemetryStream",
    "iter_telemetry_source",
    "drive_engine",
    "routing_event_to_records",
    "NatsTelemetrySource",
]


@runtime_checkable
class SupportsTelemetryStream(Protocol):
    """Structural type for anything that can produce a ``TelemetryRecord`` stream.

    Satisfied by a ``netra.datagen`` ``TelemetrySource`` (which exposes
    ``stream()``), or any iterable/generator of records. We only depend on this
    *shape*, never on the concrete class, so streaming and datagen stay decoupled.
    """

    def stream(self) -> Iterable[TelemetryRecord]:  # pragma: no cover - protocol
        ...


def iter_telemetry_source(
    source: SupportsTelemetryStream | Iterable[TelemetryRecord],
) -> Iterator[TelemetryRecord]:
    """Yield ``TelemetryRecord``s from a source object or a plain iterable.

    Accepts (in order of preference): an object with a ``stream()`` method, an
    object with a ``records()`` method, or any iterable of records. This is the
    CPU-only entry point — no NATS, no sim — used by the smoke test.
    """
    stream_fn: Callable[[], Iterable[TelemetryRecord]] | None = None
    if hasattr(source, "stream") and callable(source.stream):  # type: ignore[attr-defined]
        stream_fn = source.stream  # type: ignore[assignment]
    elif hasattr(source, "records") and callable(source.records):  # type: ignore[attr-defined]
        stream_fn = source.records  # type: ignore[assignment]
    iterable: Iterable[TelemetryRecord] = stream_fn() if stream_fn else source  # type: ignore[assignment]
    for rec in iterable:
        if isinstance(rec, TelemetryRecord):
            yield rec


def drive_engine(
    engine: FeatureEngine,
    source: SupportsTelemetryStream | Iterable[TelemetryRecord],
    on_feature: Callable[[Any], None] | None = None,
) -> int:
    """Drive ``engine`` from ``source`` to exhaustion (CPU-only path).

    Returns the number of ``FeatureVector``s emitted. If ``on_feature`` is given
    it is called with each emitted vector (e.g. publish to ``features.>`` or hand
    to the analytics layer); otherwise vectors are produced and discarded
    (useful for the throughput smoke test).
    """
    emitted = 0
    for record in iter_telemetry_source(source):
        fv = engine.process(record)
        if fv is not None:
            emitted += 1
            if on_feature is not None:
                on_feature(fv)
    return emitted


def routing_event_to_records(
    event: RoutingEvent, *, value: float = 1.0
) -> list[TelemetryRecord]:
    """Map a :class:`RoutingEvent` to numeric ``TelemetryRecord``(s) the engine folds.

    Routing churn/flap features are rate-based, so each discrete control-plane
    event becomes a unit observation on the appropriate rate metric:

      * ``route_announce`` / ``as_path_change``  -> ``bgp_update_rate``
      * ``route_withdraw``                       -> ``bgp_withdraw_rate``
      * ``adjacency_up`` / ``adjacency_down``    -> ``adjacency_flap_count``
      * ``spf_run``                              -> ``ospf_spf_rate``
      * ``lsa_regenerate``                       -> ``ospf_lsa_rate``

    The engine's churn/flap computers consume these as per-event unit counts.
    """
    mapping = {
        "route_announce": MetricName.BGP_UPDATE_RATE.value,
        "as_path_change": MetricName.BGP_UPDATE_RATE.value,
        "route_withdraw": MetricName.BGP_WITHDRAW_RATE.value,
        "adjacency_up": MetricName.ADJ_FLAP_COUNT.value,
        "adjacency_down": MetricName.ADJ_FLAP_COUNT.value,
        "spf_run": MetricName.OSPF_SPF_RATE.value,
        "lsa_regenerate": MetricName.OSPF_LSA_RATE.value,
    }
    metric = mapping.get(event.event_type)
    if metric is None:
        return []
    rec = TelemetryRecord(
        timestamp=event.timestamp,
        site=event.site,
        device=event.device,
        role=event.role,
        metric_name=metric,
        value=value,
        kind=event.protocol,
        labels={k: v for k, v in {"peer": event.peer, "prefix": event.prefix}.items() if v},
    )
    return [rec]


# ---------------------------------------------------------------------------
# Optional NATS JetStream consumer (import-guarded — heavy/extra transport).
# ---------------------------------------------------------------------------

try:  # nats-py is a CORE dep, but guard so the module imports without it.
    import nats  # type: ignore

    _HAS_NATS = True
except Exception:  # pragma: no cover - exercised only without nats-py
    nats = None  # type: ignore[assignment]
    _HAS_NATS = False


def _record_from_json(payload: bytes) -> TelemetryRecord | None:
    """Deserialise a NATS message body into a ``TelemetryRecord`` (best effort)."""
    try:
        data = json.loads(payload)
    except Exception:
        return None
    # parse timestamp if it arrived as an ISO string
    if isinstance(data.get("timestamp"), str):
        try:
            data["timestamp"] = datetime.fromisoformat(
                data["timestamp"].replace("Z", "+00:00")
            )
        except Exception:
            return None
    try:
        return TelemetryRecord.model_validate(data)
    except Exception:
        return None


class NatsTelemetrySource:
    """NATS JetStream consumer that drives a :class:`FeatureEngine` (optional).

    Subscribes to a telemetry subject (default ``telemetry.>``) on a durable,
    explicit-ack consumer, deserialises each message into a ``TelemetryRecord``,
    folds it into the engine, and (optionally) publishes the emitted
    ``FeatureVector`` to a features subject (default ``features.>``).

    **Delivery semantics (honours the research correction):** the consumer acks
    *after* the record is folded in. Because the bus is **at-least-once** (a
    crash between fold and ack redelivers the message), the downstream alerting
    must be idempotent — see ``alerts.py`` for the dedup-key pattern. The feature
    fold itself is effectively idempotent for monotone running stats but a
    duplicate sample can still perturb EWMA slightly, so for strict correctness
    publishers SHOULD set ``Nats-Msg-Id`` (record identity) to let JetStream's
    duplicate window drop dupes within it.

    This class is import-guarded: constructing it without ``nats-py`` raises a
    clear error, but importing the module never fails — the CPU-only path that
    uses :func:`drive_engine` does not need NATS at all.
    """

    def __init__(
        self,
        engine: FeatureEngine,
        servers: str | list[str] = "nats://127.0.0.1:4222",
        *,
        subject: str = "telemetry.>",
        durable: str = "river-scorer",
        publish_features_subject: str | None = "features.scored",
    ) -> None:
        if not _HAS_NATS:  # pragma: no cover - depends on optional import
            raise RuntimeError(
                "nats-py is not installed; NatsTelemetrySource is unavailable. "
                "Use drive_engine(engine, source) for the CPU-only path, or "
                "`pip install nats-py`."
            )
        self.engine = engine
        self.servers = servers
        self.subject = subject
        self.durable = durable
        self.publish_features_subject = publish_features_subject
        self._nc: Any = None
        self._js: Any = None

    async def connect(self) -> None:  # pragma: no cover - requires a live broker
        self._nc = await nats.connect(self.servers)
        self._js = self._nc.jetstream()

    async def run(self, max_messages: int | None = None) -> int:  # pragma: no cover
        """Consume telemetry, drive the engine, optionally publish features.

        Returns the number of messages processed. ``max_messages`` bounds the run
        (useful for tests against an ephemeral broker); ``None`` runs until
        cancelled.
        """
        if self._js is None:
            await self.connect()
        sub = await self._js.subscribe(self.subject, durable=self.durable)
        processed = 0
        async for msg in sub.messages:
            record = _record_from_json(msg.data)
            if record is not None:
                fv = self.engine.process(record)
                if (
                    fv is not None
                    and self.publish_features_subject is not None
                ):
                    await self._js.publish(
                        self.publish_features_subject,
                        fv.model_dump_json().encode("utf-8"),
                    )
            # ack AFTER folding (at-least-once; downstream must be idempotent)
            await msg.ack()
            processed += 1
            if max_messages is not None and processed >= max_messages:
                break
        return processed

    async def close(self) -> None:  # pragma: no cover - requires a live broker
        if self._nc is not None:
            await self._nc.drain()
