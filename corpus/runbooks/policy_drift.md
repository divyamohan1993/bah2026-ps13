---
runbook_id: RB-POLICYDRIFT-001
issue_type: policy_drift
scenario_id: D_policy_drift
title: Controller misconfiguration / policy drift across sites
sites: [hub1, br1, br2, br3]
devices: [sdwan-ctl, ce-br1, ce-br2, ce-br3]
metrics: [config_drift_score, path_asymmetry]
severity: P1
---

# Runbook: Controller Misconfiguration -> Policy Drift

## Summary
This runbook covers **policy drift** caused by an SD-WAN **controller
misconfiguration or bad commit** (`sdwan-ctl`). The defining discriminator is
**simultaneous multi-site behaviour change with NO physical fault** — many sites
change at once, no hardware alarm fires. The earliest signal is the
**config-version diff vs golden** (`config_drift_score`), which fans out
instantly. NETRA's BOCPD/PELT step detectors plus DDM/HDDM/ADWIN and
PCA-recon / Isolation-Forest fire within **seconds to minutes**, and Granger ties
the drift to the controller push.

## Precursor signature
| Signal | Behaviour before impact |
|---|---|
| `config_drift_score` | Step increase vs last-known-good across multiple sites. |
| `path_asymmetry` | Forwarding paths diverge from intent simultaneously. |
| Hardware alarms | **None** — this is the key discriminator from A/B/C. |
| Onset | Step change (config commit), not a ramp or a burst. |

## Diagnosis (read-only, auto-approved)
1. **Diff running vs golden config** across all affected sites to confirm the
   drift is configuration, not a physical fault.
2. **Attribute the drift to the controller push/commit** — identify the commit
   id, the approver, and the timestamp from the controller audit log.
3. Confirm the change is the cause by correlating the commit time with the
   simultaneous multi-site metric change (Granger / change-point onset).

## Remediation (suggest -> approve -> execute)
Order of preference:
1. **Revert to the last-known-good policy** on the controller (roll back the
   offending commit).
2. **Roll back the controller change** for the specific affected policy if a
   full revert is too broad.
3. **Re-push the golden configuration** to the affected sites via the controller
   API or NAPALM `replace`.

Execute on approval; keep an **audit record of the approver**. All are
state-changing.

## Verification & rollback
- **Verify:** `config_drift_score` returns to zero across all sites and the
  affected metrics normalise; intent and forwarding paths match again.
- **Rollback:** if the revert itself causes issues, restore the prior commit and
  escalate to the controller/change-management owner.

## Related
- Playbook: `PB-POLICYDRIFT-001`
- Past incident: `INC-2026-0019` (golden-config drift after an out-of-window push).
