# `grafana/` — Telemetry & risk dashboards (Workstream 6)

Grafana provisioned **entirely as code** for the NOC wall — fully offline
(local-only datasource, anonymous viewer auth, no plugin downloads, no external
fonts/CDN).

## Files

| Path | Role |
|---|---|
| `grafana.ini` | Air-gap config: localhost bind, **all phone-home / update checks off**, anonymous read-only viewer, no runtime plugin fetch. Mount at `/etc/grafana/grafana.ini`. |
| `provisioning/datasources/victoriametrics.yaml` | The **VictoriaMetrics** datasource (Prometheus/PromQL-compatible) on the internal bridge, `${GF_VM_URL:http://victoriametrics:8428}`. Default + non-editable. (A Loki block is included, commented.) |
| `provisioning/dashboards/netra.yaml` | Dashboard provider — auto-loads every JSON from `/etc/grafana/dashboards` at boot (no manual import). |
| `dashboards/netra-telemetry.json` | **Telemetry wall:** interface utilisation/latency/jitter/loss + output-discards, BGP churn & flap penalty, tunnel loss/jitter/rekey. Entity template variable. |
| `dashboards/netra-risk-overview.json` | **Risk & air-gap overview:** top risk, open incidents, min time-to-impact, the **risk timeline** (risk rising before impact, with action threshold), predicted-issue/confidence table, detector-family agreement, **and the air-gap proof panels** — the nftables `EGRESS-DROP` **blocked-attempt counter** and an "external ESTABLISHED connections (must stay 0)" panel. |

## Metrics expected (from the analytics/security exporters)

The dashboards query these series (produced by WS2/WS3/WS4 `remote_write` to
VictoriaMetrics and the WS7 security exporters):

- Telemetry: `if_util_pct`, `latency_ms`, `jitter_ms`, `loss_pct`,
  `if_out_discards`, `bgp_update_rate`, `bgp_flap_penalty`, `tunnel_loss_pct`,
  `tunnel_jitter_ms`, `tunnel_rekey_interval_s` (all labelled `entity_id`).
- Risk: `netra_fused_risk_score`, `netra_fused_risk_calibrated_confidence`,
  `netra_fused_risk_agreement`, `netra_time_to_impact_seconds`,
  `netra_incident_open` (labelled `entity_id` / `predicted_issue`).
- Air-gap proof: `airgap_egress_blocked_total`,
  `airgap_external_established_connections` (from `security/` — WS7).

## Run (with docker-compose; integrator)

Mount this directory read-only into the Grafana container (see
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §9.1):

```yaml
grafana:
  image: grafana/grafana:<pinned>
  environment:
    - GF_VM_URL=http://victoriametrics:8428
  volumes:
    - ./grafana/grafana.ini:/etc/grafana/grafana.ini:ro
    - ./grafana/provisioning:/etc/grafana/provisioning:ro
    - ./grafana/dashboards:/etc/grafana/dashboards:ro
  networks: [airgap_net]      # internal: true
  ports: ["127.0.0.1:3000:3000"]
```

Grafana auto-provisions the datasource + both dashboards on startup; open
`http://127.0.0.1:3000/`. Any vendored plugin zips go in a `plugins/` dir here
(set `allow_loading_unsigned_plugins` in `grafana.ini` if unsigned).

**Air-gap:** no plugin/image/font/CDN fetch at runtime. The `http://...` in the
datasource URL is the *internal-bridge* VictoriaMetrics service, not an external
dependency. See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS6.
