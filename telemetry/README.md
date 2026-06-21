# `telemetry/` — Phase 2: Telemetry collection + bus + store configs (Workstream 2)

Collector, bus and time-series-store configuration for the offline telemetry
pipeline. The O(1) streaming feature *engine* (Python) lives in
[`../netra/streaming/`](../netra/streaming); this directory holds the
infrastructure configs.

**What goes here:**
- `gnmic.yaml` — gNMI `on-change` + `sample(1s)` subscriptions → NATS + Prometheus.
- `telegraf.conf` — SNMP / syslog / NetFlow-IPFIX / sFlow inputs → VictoriaMetrics
  (`remote_write`) and NATS.
- `pmacct/` — optional high-volume NetFlow/IPFIX accounting config.
- `nats-streams.sh` — JetStream stream/consumer provisioning (`telemetry.>`,
  `alerts.>`).
- `victoriametrics.yaml` / `vmagent.yaml` — single-node TSDB + on-disk buffering.

**Contracts:** the collectors emit raw signals that map onto `TelemetryRecord` /
`SyslogEvent` / `RoutingEvent` / `FlowRecord` / `TunnelStat` on the bus. See
[`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS2.

**Air-gap:** every component is a single static binary; no cloud endpoints.
