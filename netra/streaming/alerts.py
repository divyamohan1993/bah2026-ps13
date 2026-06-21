"""Idempotent alert emitter — the dedup-key pattern (Workstream 2).

This module implements the **PR-review correction (P2)** directly: NATS
JetStream WorkQueue retention is **at-least-once, NOT exactly-once**. A dropped
ack or a consumer restart between processing and ack will *redeliver* a message.
"Exactly-once-effective" requires BOTH:

  (a) **publisher-side dedup** — set a stable ``Nats-Msg-Id`` on publish so
      JetStream rejects duplicates within the stream's duplicate window; AND
  (b) **confirmed / double acknowledgement** on the consumer (``AckSync``).

Absent both, the bus must be treated as at-least-once and every consumer made
**idempotent on a stable key** so a redelivered alert does not double-fire.

:class:`AlertEmitter` is that idempotent consumer-side guard. It derives a
**stable alert key** from the alert's identity (``scenario+entity+window+detector``,
per the research note) — which doubles as the ``Nats-Msg-Id`` for publisher-side
dedup — and suppresses any alert whose key it has already emitted within a
bounded dedup window. Duplicate deliveries are therefore harmless: the second
(and Nth) attempt is dropped.

The emitter is transport-agnostic: it returns the :class:`Alert` to publish (and
the ``Nats-Msg-Id`` to stamp) but does not itself talk to NATS, so it is trivially
unit-testable and reusable by the analytics/copilot consumers.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from netra.contracts import EntityRef, FeatureVector

__all__ = [
    "Alert",
    "AlertEmitter",
    "make_alert_key",
]


def _floor_window(ts: datetime, window_seconds: float) -> int:
    """Bucket a timestamp into a discrete window index (stabilises the key).

    Two near-simultaneous firings of the same precursor on the same entity should
    collapse to ONE alert; flooring the timestamp into a ``window_seconds`` bucket
    makes the derived key identical for both, so the second is deduped.
    """
    epoch = ts.timestamp()
    return int(epoch // window_seconds)


def make_alert_key(
    *,
    detector: str,
    entity_id: str,
    scenario: str | None = None,
    window_index: int | None = None,
) -> str:
    """Build the **stable alert key** = the ``Nats-Msg-Id`` for dedup.

    Mirrors the research note's recommended key composition
    ``scenario+link+window+detector`` so the same logical alert always hashes to
    the same id regardless of how many times it is (re)delivered. The result is a
    short hex digest safe to use as a NATS ``Nats-Msg-Id`` header.
    """
    parts = [
        scenario or "-",
        entity_id,
        str(window_index) if window_index is not None else "-",
        detector,
    ]
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"alert-{digest}"


@dataclass
class Alert:
    """A deduplicated precursor alert ready to publish to ``alerts.>``.

    ``key`` (== ``nats_msg_id``) is the stable identity used for both
    publisher-side dedup (the ``Nats-Msg-Id`` header) and consumer-side
    idempotency (the dedupe set). ``first_seen`` is when this key was *first*
    emitted; redeliveries reuse it.
    """

    key: str
    entity_id: str
    detector: str
    timestamp: datetime
    scenario: str | None = None
    window_index: int | None = None
    features: dict[str, float] = field(default_factory=dict)
    message: str | None = None

    @property
    def nats_msg_id(self) -> str:
        """The header a publisher MUST set so JetStream drops in-window dupes."""
        return self.key


class AlertEmitter:
    """Idempotent alert emitter: dedupes by stable key so dupes don't double-fire.

    Drop-in consumer-side guard for the at-least-once bus. Feed it the precursor
    triggers from a :class:`~netra.contracts.FeatureVector` (or call
    :meth:`emit` directly with a detector + entity); it returns a new
    :class:`Alert` the *first* time a key is seen and ``None`` for every duplicate
    within the dedup window.

    The dedupe store is an ``OrderedDict`` acting as a bounded LRU: it holds the
    last ``max_keys`` emitted keys (constant memory) and additionally expires keys
    older than ``dedup_window_seconds`` so a genuinely *new* occurrence in a later
    window can fire again. This is exactly the "make the consumer idempotent on a
    stable alert key" behaviour the correction mandates.

    Parameters
    ----------
    window_seconds:
        Bucketing window used to derive the key's ``window_index`` (collapses
        near-simultaneous firings of the same precursor into one alert).
    dedup_window_seconds:
        How long a key remains suppressed. ``None`` = suppress forever (until
        evicted by ``max_keys``).
    max_keys:
        LRU cap on remembered keys (bounds memory).
    """

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        dedup_window_seconds: float | None = 600.0,
        max_keys: int = 8192,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.dedup_window_seconds = dedup_window_seconds
        self.max_keys = int(max_keys)
        # key -> first-seen epoch seconds (insertion-ordered for LRU eviction)
        self._seen: "OrderedDict[str, float]" = OrderedDict()
        self._emitted_count = 0
        self._suppressed_count = 0

    # -- stats -----------------------------------------------------------------
    @property
    def emitted_count(self) -> int:
        """Number of *unique* alerts emitted (post-dedup)."""
        return self._emitted_count

    @property
    def suppressed_count(self) -> int:
        """Number of duplicate deliveries suppressed."""
        return self._suppressed_count

    @property
    def active_keys(self) -> int:
        return len(self._seen)

    # -- core ------------------------------------------------------------------
    def _expire(self, now_epoch: float) -> None:
        if self.dedup_window_seconds is None:
            return
        cutoff = now_epoch - self.dedup_window_seconds
        # OrderedDict is insertion-ordered; pop from the front while expired.
        while self._seen:
            key, first = next(iter(self._seen.items()))
            if first < cutoff:
                self._seen.popitem(last=False)
            else:
                break

    def _remember(self, key: str, now_epoch: float) -> None:
        self._seen[key] = now_epoch
        self._seen.move_to_end(key)
        while len(self._seen) > self.max_keys:
            self._seen.popitem(last=False)

    def seen(self, key: str) -> bool:
        """True if ``key`` is currently suppressed (already emitted, not expired)."""
        return key in self._seen

    def emit(
        self,
        *,
        detector: str,
        entity_id: str,
        timestamp: datetime,
        scenario: str | None = None,
        features: dict[str, float] | None = None,
        message: str | None = None,
    ) -> Alert | None:
        """Return a new :class:`Alert` if unseen, else ``None`` (deduped).

        Idempotent: calling repeatedly with the same identity inside the dedup
        window yields one ``Alert`` then ``None`` — so redelivered messages and
        repeated precursor firings never double-fire.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        now_epoch = timestamp.timestamp()
        self._expire(now_epoch)
        window_index = _floor_window(timestamp, self.window_seconds)
        key = make_alert_key(
            detector=detector,
            entity_id=entity_id,
            scenario=scenario,
            window_index=window_index,
        )
        if key in self._seen:
            self._suppressed_count += 1
            return None
        self._remember(key, now_epoch)
        self._emitted_count += 1
        return Alert(
            key=key,
            entity_id=entity_id,
            detector=detector,
            timestamp=timestamp,
            scenario=scenario,
            window_index=window_index,
            features=dict(features or {}),
            message=message,
        )

    def emit_from_feature_vector(
        self,
        fv: FeatureVector,
        *,
        scenario: str | None = None,
    ) -> list[Alert]:
        """Emit (deduped) alerts for every drift/anomaly trigger in ``fv``.

        Each name in ``fv.triggered_drift`` (e.g. ``"page_hinkley:latency_ms"``)
        becomes a candidate alert keyed on that detector + the entity + the time
        window; duplicates across ticks within the window are suppressed. Returns
        only the *newly* emitted alerts.
        """
        out: list[Alert] = []
        entity: EntityRef = fv.entity
        for trigger in fv.triggered_drift:
            detector = trigger.split(":", 1)[0]
            alert = self.emit(
                detector=detector,
                entity_id=entity.entity_id,
                timestamp=fv.timestamp,
                scenario=scenario,
                features=fv.features,
                message=f"{trigger} fired on {entity.entity_id}",
            )
            if alert is not None:
                out.append(alert)
        return out

    def dedupe(self, alerts: Iterable[Alert]) -> list[Alert]:
        """Filter an existing stream of ``Alert``s, dropping ones already seen.

        Models the consumer reading from the at-least-once bus: feed it the
        deserialised alerts (which already carry their stable ``key``) and it
        returns only those not yet processed — the redelivered ones are dropped.
        """
        out: list[Alert] = []
        for alert in alerts:
            now_epoch = alert.timestamp.timestamp()
            self._expire(now_epoch)
            if alert.key in self._seen:
                self._suppressed_count += 1
                continue
            self._remember(alert.key, now_epoch)
            self._emitted_count += 1
            out.append(alert)
        return out
