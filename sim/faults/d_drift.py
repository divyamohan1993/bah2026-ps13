"""Scenario D — controller misconfiguration -> policy drift (WS1).

Pushes a bad config delta to a PE (as if from the SD-WAN controller): a WRONG
route-target import that makes VRF CORP and OT leak into each other. There is NO
physical fault — the discriminator (research/01 §4.3 D) is that anomalous
behaviour appears at MULTIPLE sites simultaneously with no hardware alarm. The
earliest signal is the config-change event itself; reachability/flow patterns
then drift before a hard SLA/security breach. Ground-truth root cause entity is
the controller ``dc:sdwan-ctl:controller``.

Injection backends (``--tool``):
  * ``vtysh`` — push the bad RT import directly via the PE's integrated vtysh
    (``docker exec ... vtysh -c ...``); the simplest offline mechanism.
  * ``napalm`` — emit a NAPALM merge candidate (``drift.napalm.cfg``) representing
    the controller push, for environments wired to NAPALM/Nornir.

The driver writes a golden snapshot first, applies the drift, then (on cleanup)
restores — so the impairment is bounded and reversible. Default ``--dry-run``.

    python sim/faults/d_drift.py --labels labels/run.jsonl              # dry-run
    python sim/faults/d_drift.py --labels labels/run.jsonl --run        # live (vtysh)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _labels import ScenarioClock, open_label

from netra.contracts import IssueType, ScenarioId, Severity

# Push the bad RT import onto pe-dc2 (which holds the OT VRF for ce-br3).
PE_CONTAINER = "clab-netra-pe-dc2"
TARGET_ENTITY = "dc:sdwan-ctl:controller"

# The DRIFT: import CORP's RT (100:1) into the OT VRF -> CORP routes leak into OT.
DRIFT_COMMANDS = [
    "configure terminal",
    "router bgp 65000 vrf OT",
    " address-family ipv4 unicast",
    "  rt vpn import 100:1",        # <-- WRONG: leaks CORP into OT
    " exit-address-family",
    "end",
]
# The revert removes the bad import (restores isolation).
REVERT_COMMANDS = [
    "configure terminal",
    "router bgp 65000 vrf OT",
    " address-family ipv4 unicast",
    "  no rt vpn import 100:1",
    " exit-address-family",
    "end",
]

NAPALM_DELTA = """\
! NAPALM merge candidate (scenario D controller push) — wrong RT import.
router bgp 65000 vrf OT
 address-family ipv4 unicast
  rt vpn import 100:1
 exit-address-family
"""


def _vtysh(container: str, commands: list[str], run: bool) -> None:
    args = ["docker", "exec", container, "vtysh"]
    for c in commands:
        args += ["-c", c]
    print("  $", " ".join(f'"{a}"' if " " in a else a for a in args))
    if run:
        subprocess.run(args, check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/run.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--precursor", type=float, default=20.0,
                    help="Short lead: the config push is itself the earliest event.")
    ap.add_argument("--hold", type=float, default=300.0, help="Seconds the drift persists.")
    ap.add_argument("--tool", choices=["vtysh", "napalm"], default="vtysh")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    run = args.run

    clock = ScenarioClock(precursor_s=args.precursor, fault_s=args.precursor, hold_s=args.hold)

    label = open_label(
        args.labels,
        scenario=ScenarioId.D_POLICY_DRIFT,
        expected_issue=IssueType.POLICY_DRIFT,
        target_entity_id=TARGET_ENTITY,
        clock=clock,
        severity=Severity.P2,
        seed=args.seed,
        injection_tool=f"{args.tool}_config_push",
        params={
            "pe": PE_CONTAINER, "drift": "rt_import_leak_CORP_into_OT",
            "rt": "100:1", "vrf": "OT",
        },
        target_sites=["dc", "hub", "br3"],
        target_vpns=["CORP", "OT"],
        expected_playbook_id="pb-policy-revert-golden",
    )
    print(f"[D_drift] label {label.label_id} written -> {args.labels}")

    # Snapshot golden (for the revert / diff-vs-golden detector input).
    print("[D_drift] snapshot golden running-config")
    _vtysh(PE_CONTAINER, ["show running-config"], run=run)

    if args.tool == "napalm":
        Path("drift.napalm.cfg").write_text(NAPALM_DELTA, encoding="utf-8")
        print("[D_drift] wrote drift.napalm.cfg (apply via NAPALM merge to represent the push)")
    else:
        print("[D_drift] pushing bad RT import (controller misconfig)")
        _vtysh(PE_CONTAINER, DRIFT_COMMANDS, run=run)

    if run:
        time.sleep(args.hold)

    # Revert (bounds the impairment; label window already fixed).
    print("[D_drift] reverting drift (restore VRF isolation)")
    _vtysh(PE_CONTAINER, REVERT_COMMANDS, run=run)

    if not run:
        print("[D_drift] dry-run complete. Use --run for a live lab.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
