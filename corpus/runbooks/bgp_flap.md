---
runbook_id: RB-BGPFLAP-001
issue_type: bgp_route_flap
scenario_id: B_bgp_flap
title: BGP route-flap cascade and reroute storm
sites: [dc1, rr1]
devices: [rr1, pe-dc1]
metrics: [bgp_update_rate, bgp_withdraw_rate, bgp_flap_penalty, adjacency_flap_count, ospf_spf_rate]
severity: P1
---

# Runbook: BGP Route-Flap Cascade

## Summary
This runbook covers a **BGP route-flap cascade** originating at a route
reflector (`rr1`) or a PE-RR VPNv4 session. The signature is **bursty BGP
UPDATE/withdraw churn**, **adjacency flaps**, repeated **best-path A->B->A**
oscillation, and downstream **OSPF SPF storms**. NETRA's route-flap penalty
(RFD-style decaying score) plus CUSUM / Page-Hinkley / ADWIN on the update-rate,
combined with graph event-correlation, fire roughly **1-5 minutes** before mass
reachability loss.

## Precursor signature
| Signal | Behaviour before impact |
|---|---|
| `bgp_update_rate` | Bursty spikes well above the diurnal baseline. |
| `bgp_withdraw_rate` | Withdrawals tracking the updates (flap, not growth). |
| `bgp_flap_penalty` | RFD-style penalty accumulating toward the suppress limit. |
| `adjacency_flap_count` | Neighbor up/down transitions incrementing. |
| `ospf_spf_rate` | SPF recomputations rising as the IGP reacts. |

## Diagnosis (read-only, auto-approved)
1. Collect BGP and OSPF neighbor + flap statistics:
   `show ip bgp summary`, `show ip bgp flap-statistics`,
   `show ip ospf neighbor`.
2. Identify the **flapping prefix and originating peer** (the peer with the
   highest flap penalty / churn) using Granger ranking against the update stream.
3. Determine the cause: an unstable physical link, a misbehaving CE, or an
   underlay path that is itself flapping (cross-check the tunnel runbook).

## Remediation (suggest -> approve -> execute)
Order of preference:
1. **Enable / tighten BGP route-flap damping** on the affected session so the
   oscillating prefix is suppressed instead of re-advertised on every flip.
2. **Pin the next-hop / install a static fallback** for the critical prefix so
   reachability is preserved while the flap is contained.
3. **Administratively shut the unstable peer** if it is the confirmed source and
   a redundant path exists.

Apply via NAPALM on approval. All three are state-changing.

## Verification & rollback
- **Verify:** `bgp_update_rate` and `bgp_withdraw_rate` fall back to baseline,
  the flap penalty decays below the reuse limit, and the routing table is stable
  (no best-path oscillation) for 5 minutes.
- **Rollback:** remove damping / static fallback or no-shut the peer once the
  underlying instability is fixed, to restore normal convergence behaviour.

## Related
- Playbook: `PB-BGPFLAP-001`
- Past incident: `INC-2026-0011` (RR VPNv4 session flap after a core link fault).
