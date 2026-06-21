"""netra.datagen — synthetic 4-scenario TelemetrySource (Workstream 1).

Defines the ``TelemetrySource`` interface and the high-fidelity synthetic
generator that replays the four validation scenarios with ground-truth
``ScenarioLabel``s, emitting the canonical ``netra.contracts`` telemetry types.
This is the CPU-only / no-sim / no-internet default source that makes NETRA
always-runnable (the live Containerlab backend in ``sim/`` is the alternative).

Builder: implement ``source.py`` (TelemetrySource ABC + replay backend),
``synthetic.py`` (the generator), ``scenarios.py`` (per-scenario signal models).
Keep this package CPU-light (core tier deps only).
"""
