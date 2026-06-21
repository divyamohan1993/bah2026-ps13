"""netra.datagen — synthetic 4-scenario TelemetrySource (Workstream 1).

Defines the ``TelemetrySource`` interface and the high-fidelity synthetic
generator that replays the four validation scenarios with ground-truth
``ScenarioLabel``s, emitting the canonical ``netra.contracts`` telemetry types.
This is the CPU-only / no-sim / no-internet default source that makes NETRA
always-runnable (the live Containerlab backend in ``sim/`` is the alternative).

Public API (import these; they are stable)::

    from netra.datagen import (
        TelemetrySource,        # the ABC every backend satisfies
        SyntheticSource,        # CPU-only default (labeled)
        ReplaySource,           # re-emit a captured run
        ContainerlabSource,     # live SIM adapter (documented stub)
        SyntheticGenerator,     # the underlying generator
        GeneratorConfig,        # deterministic run config (seed/start/duration)
        REFERENCE_TOPOLOGY,     # the 5-site reference topology
    )

Quickstart::

    src = SyntheticSource(seed=1337, duration_s=3600, step_s=10)
    labels = src.labels()                  # list[ScenarioLabel] ground truth
    for rec in src.iter_records():         # time-ordered telemetry union
        ...

Determinism: the entire output is a pure function of ``GeneratorConfig`` — the
same config yields byte-for-byte identical records on any machine.
"""

from __future__ import annotations

from .scenarios import ScenarioSpec, diurnal_multiplier
from .source import (
    ContainerlabSource,
    ReplaySource,
    SyntheticSource,
    TelemetrySource,
    record_timestamp,
)
from .synthetic import GeneratorConfig, SyntheticGenerator, TelemetryUnion
from .topology import REFERENCE_TOPOLOGY, Device, Link, Topology

__all__ = [
    # source interface + backends
    "TelemetrySource",
    "SyntheticSource",
    "ReplaySource",
    "ContainerlabSource",
    "record_timestamp",
    # generator
    "SyntheticGenerator",
    "GeneratorConfig",
    "TelemetryUnion",
    # topology
    "REFERENCE_TOPOLOGY",
    "Topology",
    "Device",
    "Link",
    # scenario helpers
    "ScenarioSpec",
    "diurnal_multiplier",
]
