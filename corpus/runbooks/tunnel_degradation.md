---
runbook_id: RB-TUNNEL-001
issue_type: tunnel_degradation
scenario_id: C_tunnel_degradation
title: Intermittent MPLS underlay / IPSec tunnel degradation
sites: [br3]
devices: [ce-br3]
metrics: [tunnel_loss_pct, tunnel_jitter_ms, tunnel_rekey_interval_s, loss_pct]
severity: P1
---

# Runbook: Intermittent MPLS Underlay / Tunnel Degradation

## Summary
This runbook covers **intermittent degradation of an IPSec-over-GRE overlay
tunnel** riding an MPLS underlay (typically `ce-br3:tunnel-hub`). The signature
is **intermittent tunnel loss/jitter spikes**, **LSP path churn**, **IPSec
rekey-interval anomalies**, and **BFD flaps** — bursty, not monotonic. NETRA's
Half-Space-Trees / RRCF / Spectral-Residual detectors plus COPOD/ECOD on the
`{loss, jitter, rekey}` vector and the streaming Matrix Profile (stumpi) fire
roughly **1-8 minutes** before sustained SLA loss.

## Precursor signature
| Signal | Behaviour before impact |
|---|---|
| `tunnel_loss_pct` | Intermittent spikes above baseline (bursty). |
| `tunnel_jitter_ms` | Jitter spikes correlated with the loss bursts. |
| `tunnel_rekey_interval_s` | Rekey period drifts / becomes irregular (anomaly). |
| BFD state | Short flaps that precede a hard tunnel-down. |

## Diagnosis (read-only, auto-approved)
1. Collect tunnel, LSP and BFD statistics plus the IPSec rekey log:
   `show interface tunnel-hub`, `show mpls lsp`, `show bfd session`,
   IPSec SA / rekey history.
2. **Localise the faulty LSP / underlay hop** — correlate the loss bursts with a
   specific P-router or label-switched path using the topology graph.
3. Distinguish underlay loss from a rekey-induced micro-outage by aligning the
   loss timestamps with the rekey events.

## Remediation (suggest -> approve -> execute)
Order of preference:
1. **Reroute the LSP onto a healthy TE path** that avoids the degraded underlay
   hop.
2. **Fail the tunnel over to the backup transport** (secondary SD-WAN path) if
   the primary underlay cannot be made healthy quickly.
3. **Trigger a controlled IPSec rekey** to clear a stuck/anomalous SA if the
   rekey interval is the confirmed driver.

Apply via NAPALM / controller on approval; all are state-changing.

## Verification & rollback
- **Verify:** `tunnel_loss_pct` and `tunnel_jitter_ms` recover to baseline and
  stay there for 5 minutes; BFD is stable; rekey interval is regular again.
- **Rollback:** restore the original LSP path / fail back to the primary
  transport if the alternate path does not improve loss/jitter.

## Related
- Playbook: `PB-TUNNEL-001`
- Past incident: `INC-2026-0014` (branch-3 overlay jitter from a flapping core LSP).
