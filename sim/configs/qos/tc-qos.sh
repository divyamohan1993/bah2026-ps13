#!/usr/bin/env bash
# tc HTB QoS classes — voice / business / bulk (Workstream 1)
# ===========================================================
# Installs a 3-class HTB hierarchy on a CE/PE egress interface so traffic hits
# per-class queues (research/01 §2.4). This is BOTH the baseline QoS that makes
# class-specific drops/latency observable AND the mechanism scenario A drives
# (the congestion fault steps the root ceiling down so the business/bulk classes
# queue and drop while voice is protected).
#
# Usage:  tc-qos.sh <iface> [ceil_mbit]
#   e.g.  docker exec clab-netra-ce-hub bash /qos/tc-qos.sh eth1 100
#
# DSCP -> class mapping (set by the traffic generators):
#   EF  (46) voice    -> 1:10 (priority, low latency)
#   AF31(26) business -> 1:20
#   BE  (0)  bulk     -> 1:30
set -euo pipefail
IFACE="${1:?usage: tc-qos.sh <iface> [ceil_mbit]}"
CEIL="${2:-100}"

# Reset any existing qdisc.
tc qdisc del dev "$IFACE" root 2>/dev/null || true

# Root HTB; default unclassified -> bulk (30).
tc qdisc add dev "$IFACE" root handle 1: htb default 30
tc class add dev "$IFACE" parent 1: classid 1:1 htb rate "${CEIL}mbit" ceil "${CEIL}mbit"

# voice: 20% guaranteed, can burst, highest priority, shallow fq_codel.
tc class add dev "$IFACE" parent 1:1 classid 1:10 htb \
    rate "$((CEIL * 20 / 100))mbit" ceil "${CEIL}mbit" prio 0
tc qdisc add dev "$IFACE" parent 1:10 fq_codel target 5ms

# business: 50% guaranteed.
tc class add dev "$IFACE" parent 1:1 classid 1:20 htb \
    rate "$((CEIL * 50 / 100))mbit" ceil "${CEIL}mbit" prio 1
tc qdisc add dev "$IFACE" parent 1:20 fq_codel

# bulk: 30% guaranteed, lowest priority (first to drop under congestion).
tc class add dev "$IFACE" parent 1:1 classid 1:30 htb \
    rate "$((CEIL * 30 / 100))mbit" ceil "${CEIL}mbit" prio 2
tc qdisc add dev "$IFACE" parent 1:30 fq_codel

# DSCP -> class filters.
tc filter add dev "$IFACE" parent 1: protocol ip prio 1 u32 \
    match ip dsfield 0xb8 0xfc flowid 1:10   # EF (46<<2 = 0xb8)
tc filter add dev "$IFACE" parent 1: protocol ip prio 2 u32 \
    match ip dsfield 0x68 0xfc flowid 1:20   # AF31 (26<<2 = 0x68)

echo "tc HTB QoS installed on $IFACE (ceil ${CEIL}mbit: voice/business/bulk)"
