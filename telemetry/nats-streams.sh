#!/usr/bin/env bash
# nats-streams.sh — NATS JetStream stream/consumer provisioning (NETRA Phase 2, WS2)
# ============================================================================
# Provisions the durable JetStream streams + consumers the telemetry pipeline
# uses. NATS JetStream is a single ~10-15 MB Go binary that gives persistent
# streams, durable consumers and replay with no JVM/ZooKeeper (research 02 §2.1).
#
# Run (offline, inside the air-gap), with a JetStream-enabled server already up
#   (`nats-server -js -sd /data/jetstream`):
#     ./telemetry/nats-streams.sh
# Override the endpoint with NATS_URL=nats://host:4222 ./telemetry/nats-streams.sh
#
# Requires the `nats` CLI (single static binary) on PATH. Idempotent: re-running
# updates existing streams/consumers in place (`add ... ` with the same name).
# ============================================================================
set -euo pipefail

NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
NATS="nats --server ${NATS_URL}"

echo ">> Provisioning NATS JetStream on ${NATS_URL}"

# ----------------------------------------------------------------------------
# 1) TELEMETRY stream — the raw signal bus (telemetry.>).
# ----------------------------------------------------------------------------
# Delivery semantics: AT-LEAST-ONCE. `retention=limits` + explicit-ack durable
# consumers mean a sample may be redelivered after a consumer restart or a lost
# ack. This is the PRAGMATIC default for telemetry: metric writes are idempotent
# (VictoriaMetrics dedupes identical (series,timestamp) points) and the O(1)
# feature engine tolerates an occasional duplicate sample (a duplicate barely
# perturbs an EWMA). For STRICT exactly-once-effective ingest, publishers SHOULD
# additionally set the `Nats-Msg-Id` header to the record identity so JetStream's
# duplicate window drops in-window dupes (see the ALERTS note below).
${NATS} stream add TELEMETRY \
  --subjects "telemetry.>" \
  --storage file \
  --retention limits \
  --max-age 24h \
  --max-bytes 8GB \
  --max-msgs-per-subject 1000000 \
  --discard old \
  --dupe-window 2m \
  --replicas 1 \
  --defaults || \
${NATS} stream edit TELEMETRY \
  --subjects "telemetry.>" --max-age 24h --dupe-window 2m --defaults -f

# Durable pull consumer for the streaming feature engine. Explicit ack + bounded
# redelivery; the engine acks AFTER folding a record into its running stats.
${NATS} consumer add TELEMETRY river-scorer \
  --pull \
  --ack explicit \
  --max-deliver 5 \
  --wait 30s \
  --deliver all \
  --replay instant \
  --defaults || true

# Push consumer that fans telemetry into VictoriaMetrics (vmagent reads this).
${NATS} consumer add TELEMETRY vm-writer \
  --pull \
  --ack explicit \
  --max-deliver 3 \
  --deliver all \
  --defaults || true

# ----------------------------------------------------------------------------
# 2) FEATURES stream — the engine's FeatureVector output (features.>).
# ----------------------------------------------------------------------------
# The analytics layer (Phase 3) subscribes here. At-least-once; FeatureVectors
# carry (entity_id, timestamp) so downstream is naturally idempotent on that key.
${NATS} stream add FEATURES \
  --subjects "features.>" \
  --storage file \
  --retention limits \
  --max-age 6h \
  --max-bytes 4GB \
  --discard old \
  --dupe-window 2m \
  --replicas 1 \
  --defaults || true

# ----------------------------------------------------------------------------
# 3) ALERTS stream — the precursor/alert bus (alerts.>).
# ----------------------------------------------------------------------------
# CORRECTION (PR review, P2): WorkQueue retention is **at-least-once, NOT
# exactly-once**. A dropped ack or a consumer restart between processing and ack
# WILL redeliver the message. "Exactly-once-EFFECTIVE" delivery requires BOTH:
#   (a) publisher-side dedup — set `Nats-Msg-Id` on publish (we use the stable
#       alert key `scenario+entity+window+detector`, produced by
#       netra.streaming.alerts.make_alert_key) so JetStream rejects duplicates
#       that arrive inside `--dupe-window`; AND
#   (b) confirmed / double acknowledgement on the consumer (AckSync — wait for
#       the server to confirm the ack before treating the message as done).
# Absent BOTH, treat the bus as at-least-once and make the alert/copilot consumer
# IDEMPOTENT on that stable key (netra.streaming.alerts.AlertEmitter does exactly
# this — it dedupes by key so a redelivered alert never double-fires).
#
# We therefore configure WorkQueue retention AND a duplicate window, and document
# that (a)+(b)+idempotent-consumer together deliver the exactly-once-effective
# guarantee — WorkQueue alone does not.
${NATS} stream add ALERTS \
  --subjects "alerts.>" \
  --storage file \
  --retention work \
  --max-age 72h \
  --max-bytes 1GB \
  --discard old \
  --dupe-window 5m \
  --replicas 1 \
  --defaults || \
${NATS} stream edit ALERTS --dupe-window 5m --defaults -f || true

# Durable consumer for the copilot/incident pipeline. `--ack explicit` and the
# client using AckSync (confirmed ack) is half of the exactly-once-effective
# contract; the idempotent AlertEmitter dedup is the belt-and-suspenders other
# half. max-deliver>1 so a transient consumer failure retries (at-least-once).
${NATS} consumer add ALERTS copilot-consumer \
  --pull \
  --ack explicit \
  --max-deliver 10 \
  --wait 30s \
  --deliver all \
  --defaults || true

echo ">> Streams:"
${NATS} stream ls || true
echo ">> Done. Delivery-semantics contract:"
echo "   telemetry.>  : at-least-once  (idempotent writes; optional Nats-Msg-Id)"
echo "   features.>   : at-least-once  (idempotent on entity_id+timestamp)"
echo "   alerts.>     : at-least-once  -> exactly-once-EFFECTIVE only with"
echo "                  (a) Nats-Msg-Id dedup + (b) AckSync + idempotent consumer"
