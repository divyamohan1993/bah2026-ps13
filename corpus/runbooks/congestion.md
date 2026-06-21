---
runbook_id: RB-CONGESTION-001
issue_type: interface_congestion
scenario_id: A_congestion
title: Progressive interface congestion on hub-spoke uplinks
sites: [hub1]
devices: [pe-hub1]
metrics: [if_util_pct, if_out_discards, queue_depth, latency_ms, jitter_ms]
severity: P2
---

# Runbook: Progressive Interface Congestion (Hub-Spoke Uplink)

## Summary
This runbook covers **progressive congestion** on a hub aggregation uplink
(typically `pe-hub1` egress toward the MPLS core or toward the spokes). The
defining signature is a **monotonically rising interface-utilisation slope**
with **queue-drop creep** (`if_out_discards`) and **latency/jitter drift** that
appear *before* any packet loss is reported. NETRA's drift detectors
(EWMA / CUSUM / Page-Hinkley) and the LightGBM/MSTL forecast fire while the
metric is still below the SLA threshold, giving roughly **2-10 minutes** of lead
time.

## Precursor signature
| Signal | Behaviour before impact |
|---|---|
| `if_util_pct` | Monotonic upward slope (e.g. +3-5%/min toward saturation). |
| `if_out_discards` | Slow creep from zero as egress queues begin to fill. |
| `queue_depth` | Rising egress queue occupancy. |
| `latency_ms` / `jitter_ms` | Drift upward as buffering increases. |
| `loss_pct` | **Lagging** — only rises once buffers overflow (too late). |

## Diagnosis (read-only, auto-approved)
1. Collect interface and queue statistics on the suspect uplink:
   `show interface eth1 | include rate|drops` and the per-class queue counters.
2. Identify the **top-talker flows** from NetFlow/IPFIX over the last 5 minutes
   to see which application classes are driving the ramp.
3. Confirm the trend is real (not a transient burst) by checking the
   utilisation slope over the trailing 10-minute window.

## Remediation (suggest -> approve -> execute)
Order of preference (least disruptive first):
1. **Raise QoS priority for the business/voice class** on the congested
   interface so latency-sensitive traffic is protected as utilisation climbs.
2. **Shift bulk/scavenger traffic to an alternate SD-WAN path** (secondary
   transport) to relieve the primary uplink.
3. **Rate-limit the bulk class** (HTB/policer) if no alternate path has capacity.

Push the chosen change via NAPALM on operator approval. Each change is
state-changing and therefore requires approval and carries a rollback.

## Verification & rollback
- **Verify:** utilisation slope flattens and `if_out_discards` stop incrementing
  within 2-3 minutes; latency/jitter return toward baseline.
- **Rollback:** restore the previous QoS policy / remove the policer / move
  traffic back to the primary path if the SLA does not recover or the alternate
  path degrades.

## Related
- Playbook: `PB-CONGESTION-001`
- Past incident: `INC-2026-0007` (HQ uplink saturation during quarterly close).
