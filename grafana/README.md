# `grafana/` — Telemetry & alert dashboards (Workstream 6)

Grafana provisioned **entirely as code** for the NOC telemetry/alert wall, fully
offline (all plugins/assets vendored; anonymous local auth).

**What goes here:**
- `provisioning/datasources/*.yaml` — VictoriaMetrics (PromQL/MetricsQL) +
  optional Loki (syslog) datasources.
- `provisioning/dashboards/*.yaml` — dashboard providers (load from `dashboards/`).
- `dashboards/*.json` — dashboards: interface utilisation/latency/jitter/loss,
  BGP/OSPF churn, tunnel health, the **risk timeline**, and the **air-gap
  blocked-egress counter** panel (from the nftables counter — live proof of
  zero egress).
- `plugins/` — vendored plugin zips (set `GF_PLUGINS_PREINSTALL_SYNC`,
  `allow_loading_unsigned_plugins` as needed).

**Air-gap:** no plugin downloads at runtime; no external image/font/CDN refs.
See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS6.
