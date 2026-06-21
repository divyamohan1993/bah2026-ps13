# `telemetry/` — Phase 2: Telemetry collection + bus + store (Workstream 2)

Declarative, **offline** configuration for the NETRA telemetry pipeline:
collectors → NATS JetStream bus → VictoriaMetrics history. The O(1) streaming
feature *engine* (Python) lives in [`../netra/streaming/`](../netra/streaming);
this directory holds the infrastructure configs the integrator merges into the
top-level compose.

```
 SR Linux nodes ──gNMI──▶ gnmic ─┐
 FRR / classic ──SNMP/syslog─────┤
   NetFlow/IPFIX/sFlow ─────────▶ Telegraf ─┬─▶ NATS JetStream (telemetry.>) ─▶ netra.streaming ─▶ features.>
   heavy flow volume ──────────▶ nfacctd ──┘                 │
                                                              └─────────────────▶ vmagent ─▶ VictoriaMetrics (history)
```

## Files

| File | Role |
|---|---|
| [`gnmic.yaml`](gnmic.yaml) | gNMI `on-change` + `sample(1s)` subscriptions (interface counters, BGP/OSPF state, MPLS LSP) → NATS + Prometheus. Sub-second deltas = lead time. |
| [`telegraf.conf`](telegraf.conf) | SNMP (interface/error/tunnel counters), syslog (flaps/rekey/config-drift), NetFlow/IPFIX, sFlow → VictoriaMetrics (`remote_write`) **and** NATS. One binary covers the signal classes gNMI can't. |
| [`pmacct/nfacctd.conf`](pmacct/nfacctd.conf) | *Optional* high-volume NetFlow/IPFIX accounting with in-collector aggregation + BGP correlation → `telemetry.flows`. Add only when flow volume is heavy. |
| [`nats-streams.sh`](nats-streams.sh) | JetStream stream/consumer provisioning (`TELEMETRY`/`FEATURES`/`ALERTS`). Encodes the **delivery-semantics contract** (see below). |
| [`victoriametrics.yaml`](victoriametrics.yaml) | Single-node TSDB launch flags + scrape config. High-cardinality inverted index, ~70× less disk than alternatives. |
| [`vmagent.yaml`](vmagent.yaml) | vmagent with **on-disk buffering** so a VM restart/burst never loses data. |
| [`scrape.yml`](scrape.yml) | Prometheus-style scrape config mounted into vmagent/VM (single source of truth for pulled targets). |
| [`compose.telemetry.yml`](compose.telemetry.yml) | The collector/bus/store services on the `airgap_net` `internal: true` bridge. Integrator merges into the top-level compose; WS7 adds hardening. |

## Contracts on the wire

Collectors emit raw signals that map onto the `netra.contracts` ingest types the
streaming engine consumes:

| Signal | Collector | Contract type | `metric_name` examples |
|---|---|---|---|
| Interface counters | gnmic / Telegraf SNMP | `TelemetryRecord` | `if_util_pct`, `if_in_errors`, `if_out_discards` |
| Latency / jitter / loss | Telegraf / gnmic | `TelemetryRecord` | `latency_ms`, `jitter_ms`, `loss_pct` |
| Syslog | Telegraf syslog | `SyslogEvent` | mnemonic `%BGP-5-ADJCHANGE`, `%LINK-3-UPDOWN` |
| BGP/OSPF events | gnmic on-change | `RoutingEvent` | `bgp_update_rate`, `adjacency_flap_count` |
| Flow records | Telegraf netflow / nfacctd | `FlowRecord` | src/dst/proto/bytes/packets |
| Tunnel health | Telegraf SNMP | `TunnelStat` | `tunnel_loss_pct`, `tunnel_rekey_interval_s` |

## Delivery semantics (PR-review correction, P2) — IMPORTANT

NATS JetStream **WorkQueue retention alone is AT-LEAST-ONCE, not exactly-once.**
A dropped ack or a consumer restart between processing and ack **redelivers** the
message.

**"Exactly-once-effective" delivery requires BOTH:**
1. **Publisher-side dedup** — set the `Nats-Msg-Id` header on publish (we use the
   stable alert key `scenario+entity+window+detector`, produced by
   `netra.streaming.alerts.make_alert_key`) so JetStream rejects duplicates that
   arrive inside the stream's `--dupe-window`; **AND**
2. **Confirmed / double acknowledgement** on the consumer (`AckSync` — wait for
   the server to confirm the ack before treating the message as done).

Absent **both**, treat the bus as at-least-once and make every consumer
**idempotent on a stable key**. The streaming alert emitter
([`../netra/streaming/alerts.py`](../netra/streaming/alerts.py),
`AlertEmitter`) does exactly this: it dedupes by the stable key so a redelivered
alert never double-fires. The stream/consumer settings in
[`nats-streams.sh`](nats-streams.sh) configure WorkQueue retention **plus** a
duplicate window, and the comments document the full contract:

| Subject | Retention | Semantics |
|---|---|---|
| `telemetry.>` | limits | at-least-once; idempotent metric writes (VM dedups) — optional `Nats-Msg-Id` for strict |
| `features.>` | limits | at-least-once; idempotent on `entity_id+timestamp` |
| `alerts.>` | **work** | at-least-once → **exactly-once-effective only with (a) `Nats-Msg-Id` + (b) `AckSync` + idempotent consumer** |

## Running (offline)

```bash
# 1) bring up bus + store (+ collectors for the live-sim source)
docker compose -f telemetry/compose.telemetry.yml up -d            # core
docker compose -f telemetry/compose.telemetry.yml --profile sim up -d   # + collectors

# 2) provision JetStream streams/consumers (also runs as the nats-init service)
NATS_URL=nats://127.0.0.1:4222 ./telemetry/nats-streams.sh

# 3) start the O(1) feature engine consuming telemetry.> -> features.>
#    (see ../netra/streaming/README.md)
```

**CPU-only / no-sim path:** none of the collectors are needed — the
`FeatureEngine` reads the synthetic `TelemetrySource` directly via
`netra.streaming.sources.drive_engine(engine, source)`; NATS/VM are optional.

## Air-gap

Every component is a single static binary with no phone-home. All targets are
RFC1918/localhost; VictoriaMetrics anonymous usage stats are disabled
(`-usagecollector.disable=true`). No cloud endpoint appears in any config here.
