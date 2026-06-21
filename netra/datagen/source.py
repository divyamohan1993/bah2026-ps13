"""The ``TelemetrySource`` interface — NETRA's dual-source telemetry abstraction.

This module defines the single seam on which the whole "runs-anywhere" promise
hangs (ARCHITECTURE.md §5). Everything downstream of ingest depends ONLY on this
interface and the ``netra.contracts`` record types, never on *how* the records
were produced. Three backends satisfy it:

  * :class:`SyntheticSource` (``SYNTHETIC``) — wraps :class:`SyntheticGenerator`;
    the CPU-only / no-sim / no-internet default, with ground-truth labels.
  * :class:`ReplaySource` (``REPLAY``) — re-emits a previously captured run
    (records + labels) for deterministic regression testing.
  * :class:`ContainerlabSource` (``SIM``) — a thin, documented stub describing how
    a live deployment would read the NATS JetStream / VictoriaMetrics pipeline fed
    by the Containerlab lab in ``sim/``. NOT runnable in the CPU-only container.

A source yields records in **time order** and exposes the ground-truth
``ScenarioLabel``s for the run (empty for a live source until faults are
injected). ``stream()`` optionally paces emission in real time (or accelerated)
for driving the live pipeline; ``iter_records()`` is the fast, unpaced path used
by batch generation, the streaming engine's direct adapter, and tests.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from datetime import datetime

from netra.contracts import (
    FlowRecord,
    RoutingEvent,
    ScenarioLabel,
    SyslogEvent,
    TelemetryRecord,
    TelemetrySourceKind,
    TunnelStat,
)

from .synthetic import GeneratorConfig, SyntheticGenerator, TelemetryUnion


def record_timestamp(rec: TelemetryUnion) -> datetime:
    """Return the ``timestamp`` of any telemetry record (all five carry one)."""
    return rec.timestamp  # every contract record type has a ``timestamp`` field


class TelemetrySource(ABC):
    """Abstract producer of time-ordered NETRA telemetry records.

    Implementations MUST yield records (``TelemetryRecord``, ``RoutingEvent``,
    ``SyslogEvent``, ``FlowRecord``, ``TunnelStat``) in non-decreasing timestamp
    order. The ``kind`` property identifies the backend; ``labels()`` returns the
    ground-truth ``ScenarioLabel``s (used for training/eval/scoring; possibly
    empty for a live source).
    """

    #: which backend this source is (SIM / SYNTHETIC / REPLAY).
    kind: TelemetrySourceKind

    @abstractmethod
    def iter_records(self) -> Iterator[TelemetryUnion]:
        """Yield all records for the run in non-decreasing timestamp order."""
        raise NotImplementedError

    @abstractmethod
    def labels(self) -> list[ScenarioLabel]:
        """Return ground-truth scenario labels for the run (may be empty)."""
        raise NotImplementedError

    # -- convenience: real-time (or accelerated) paced streaming --------- #

    def stream(
        self,
        *,
        realtime: bool = False,
        speed: float = 1.0,
        sleep: bool = True,
    ) -> Iterator[TelemetryUnion]:
        """Yield records, optionally paced by their inter-record time deltas.

        Parameters
        ----------
        realtime:
            If ``True``, sleep between records proportionally to the gap between
            their timestamps so the consumer sees a live-like cadence. If
            ``False`` (default) records are yielded as fast as possible.
        speed:
            Wall-clock acceleration factor when ``realtime`` is set
            (``speed=60`` => 1 simulated minute per real second). Must be > 0.
        sleep:
            Set ``False`` to compute pacing but skip the actual ``time.sleep``
            (used by tests to exercise the pacing path without waiting).
        """
        if speed <= 0:
            raise ValueError("speed must be > 0")
        prev_ts: datetime | None = None
        for rec in self.iter_records():
            if realtime and prev_ts is not None:
                gap = (record_timestamp(rec) - prev_ts).total_seconds()
                if gap > 0 and sleep:
                    time.sleep(gap / speed)
            prev_ts = record_timestamp(rec)
            yield rec

    def __iter__(self) -> Iterator[TelemetryUnion]:
        return self.iter_records()


class SyntheticSource(TelemetrySource):
    """``SYNTHETIC`` backend — the high-fidelity, labeled, CPU-only default.

    Thin adapter over :class:`SyntheticGenerator`. Construct either from a
    :class:`GeneratorConfig` or directly with generator kwargs::

        SyntheticSource(seed=1337, duration_s=3600, step_s=10)
        SyntheticSource(config=GeneratorConfig(...))
    """

    kind = TelemetrySourceKind.SYNTHETIC

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        **generator_kwargs: object,
    ) -> None:
        if config is not None and generator_kwargs:
            raise ValueError("pass either a config or kwargs, not both")
        if config is None:
            config = GeneratorConfig(**generator_kwargs)  # type: ignore[arg-type]
        self.generator = SyntheticGenerator(config)

    def iter_records(self) -> Iterator[TelemetryUnion]:
        return self.generator.iter_records()

    def labels(self) -> list[ScenarioLabel]:
        return self.generator.labels()


class ReplaySource(TelemetrySource):
    """``REPLAY`` backend — deterministically re-emit a captured run.

    Takes an in-memory (or lazily-iterable) sequence of records plus their
    labels and yields them back in timestamp order. Use it to reproduce a
    specific incident or to drive regression tests from a frozen capture (the
    CLI's ``generate`` command writes exactly the artifacts a ``ReplaySource``
    can re-read via :meth:`from_records`).
    """

    kind = TelemetrySourceKind.REPLAY

    def __init__(
        self,
        records: Iterable[TelemetryUnion],
        labels: Iterable[ScenarioLabel] = (),
        *,
        assume_sorted: bool = False,
    ) -> None:
        self._records: list[TelemetryUnion] = list(records)
        if not assume_sorted:
            self._records.sort(key=record_timestamp)
        self._labels: list[ScenarioLabel] = list(labels)

    def iter_records(self) -> Iterator[TelemetryUnion]:
        return iter(self._records)

    def labels(self) -> list[ScenarioLabel]:
        return list(self._labels)

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict | TelemetryUnion],
        labels: Iterable[dict | ScenarioLabel] = (),
    ) -> ReplaySource:
        """Build a replay source from records that may be dicts or models.

        Dicts are parsed back into the correct contract type by inspecting their
        discriminating fields — the inverse of how the CLI serialises a dataset
        to JSONL/parquet. This lets a captured run round-trip without the caller
        knowing each row's concrete type.
        """
        parsed: list[TelemetryUnion] = [
            r if not isinstance(r, dict) else _parse_record(r) for r in records
        ]
        parsed_labels: list[ScenarioLabel] = [
            lbl if isinstance(lbl, ScenarioLabel) else ScenarioLabel(**lbl)
            for lbl in labels
        ]
        return cls(parsed, parsed_labels)


class ContainerlabSource(TelemetrySource):
    """``SIM`` backend — live Containerlab lab adapter (documented stub).

    In a full deployment this source reads the **live** telemetry the
    Containerlab/netlab lab in ``sim/`` produces, after it has flowed through the
    Phase-2 pipeline (ARCHITECTURE.md §3 Phase 2):

        FRR / SR Linux nodes
          -> gnmic (gNMI on-change + sample) + Telegraf (SNMP/syslog/NetFlow)
          -> NATS JetStream subjects ``telemetry.>``
          -> (this adapter) durable JetStream consumer, decode each message into
             the matching ``netra.contracts`` record, yield in arrival order.

    Ground-truth ``labels()`` come from the JSONL the sim fault orchestrator
    (``sim/scenarios/*.py``) writes *before* each injection (see ``sim/README``).

    It is intentionally **not runnable** in the air-gapped CPU-only container
    (there is no NATS, no lab). It raises a clear, actionable error so callers
    know to use :class:`SyntheticSource` for the CPU-only path. The constructor
    records the connection parameters a real implementation would need.
    """

    kind = TelemetrySourceKind.SIM

    def __init__(
        self,
        nats_url: str = "nats://127.0.0.1:4222",
        subjects: tuple[str, ...] = ("telemetry.>",),
        labels_path: str | None = None,
        *,
        durable: str = "netra-datagen",
    ) -> None:
        self.nats_url = nats_url
        self.subjects = subjects
        self.labels_path = labels_path
        self.durable = durable

    def iter_records(self) -> Iterator[TelemetryUnion]:
        raise RuntimeError(
            "ContainerlabSource (SIM) requires a live NATS JetStream pipeline fed "
            "by the Containerlab lab in sim/, which is unavailable in the "
            "air-gapped CPU-only container. Use SyntheticSource for the CPU-only "
            "path. See netra/datagen/README.md and sim/README.md for how to bring "
            "the live source up. (Configured: nats_url=%r subjects=%r)"
            % (self.nats_url, self.subjects)
        )

    def labels(self) -> list[ScenarioLabel]:
        # A live deployment would parse self.labels_path (JSONL written by the
        # sim fault orchestrator). Empty here since no lab is running.
        return []


# --------------------------------------------------------------------------- #
# dict -> contract record parsing (for ReplaySource.from_records)             #
# --------------------------------------------------------------------------- #


def _parse_record(row: dict) -> TelemetryUnion:
    """Parse a serialised telemetry row back into its contract model.

    Discriminates by a ``_type`` hint if present (written by the CLI), else by
    characteristic fields. Keeps the captured-run round-trip lossless.
    """
    rtype = row.get("_type")
    payload = {k: v for k, v in row.items() if k != "_type"}
    table: dict[str, type] = {
        "TelemetryRecord": TelemetryRecord,
        "RoutingEvent": RoutingEvent,
        "SyslogEvent": SyslogEvent,
        "FlowRecord": FlowRecord,
        "TunnelStat": TunnelStat,
    }
    if rtype and rtype in table:
        return table[rtype](**payload)  # type: ignore[return-value]
    # structural fallback (order matters: most specific fields first)
    if "tunnel_id" in payload:
        return TunnelStat(**payload)
    if "event_type" in payload and "protocol" in payload:
        return RoutingEvent(**payload)
    if "mnemonic" in payload or ("message" in payload and "severity" in payload):
        return SyslogEvent(**payload)
    if "src_addr" in payload and "dst_addr" in payload:
        return FlowRecord(**payload)
    return TelemetryRecord(**payload)


__all__ = [
    "TelemetrySource",
    "SyntheticSource",
    "ReplaySource",
    "ContainerlabSource",
    "record_timestamp",
]
