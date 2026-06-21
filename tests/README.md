# `tests/` — Test suites

**What goes here:**
- `airgap/` (Workstream 7) — the **air-gap conformance suite**: a self-contained
  pytest that actively tries every common egress path (TCP to public IPs on
  53/80/443/22/21/123, external DNS resolution, HTTPS fetch, UDP/53) and
  **passes only if every attempt fails/times out**. This is the runnable,
  judge-facing proof of "verifiably zero outbound dependency during runtime":
  `pytest -q tests/airgap`.
- `contracts/` (integrator) — round-trip / validation tests for
  `netra.contracts` (import-light check, JSON serialisation, validator
  rejection).
- Per-workstream smoke tests live alongside their modules and run against the
  synthetic `TelemetrySource` (no GPU / no internet / no sim).

See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §8.
