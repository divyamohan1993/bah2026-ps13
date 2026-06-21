"""Scenario B — BGP route flap + downstream reroute cascade (WS1).

Repeatedly announces/withdraws a VPNv4 prefix (and bounces the RR<->PE session)
so downstream PEs reconverge, best-path churns and traffic reroutes. The
precursors — BGP UPDATE/withdraw-rate spikes, adjacency flaps in syslog,
path-attribute churn and transient path asymmetry — rise *before* mass
reachability loss (research/01 §4.3 B). The ground-truth root cause is the
RR's peering with pe-dc1 (entity ``dc:rr-dc:RR:peer-pe-dc1``).

Two injection backends (pick with ``--tool``):
  * ``exabgp``  — drive an ExaBGP speaker that announces/withdraws the prefix on
    a cadence (the cleanest, most reproducible route-flap source).
  * ``clab``    — bounce the session by toggling the RR's peering interface
    (``ip link set ... down/up``) on a cadence (no extra speaker needed).

Determinism: fixed seed, absolute timestamps, label written BEFORE injection,
flap cadence fixed. Default ``--dry-run``; ``--run`` executes.

    python sim/faults/b_bgp_flap.py --labels labels/run.jsonl                  # dry-run
    python sim/faults/b_bgp_flap.py --labels labels/run.jsonl --tool clab --run

ExaBGP process template (announce/withdraw loop) is emitted to
``exabgp.flap.conf`` for reference when ``--tool exabgp``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _labels import ScenarioClock, open_label

from netra.contracts import IssueType, ScenarioId, Severity

RR_CONTAINER = "clab-netra-rr-dc"
RR_PEER_IFACE = "eth1"            # rr-dc <-> pe-dc1 peering link
TARGET_ENTITY = "dc:rr-dc:RR:peer-pe-dc1"
FLAP_PREFIX = "10.2.99.0/24"      # the prefix we flap inside the CORP VRF
FLAP_RD = "100:1"

EXABGP_TEMPLATE = """\
# ExaBGP route-flap process (scenario B). Announces then withdraws {prefix}
# every {up}s/{down}s to churn the VPNv4 best-path. Run inside an ExaBGP peer
# container attached to the RR. Image: pre-staged exabgp (offline).
neighbor 10.0.0.8 {{
    router-id 10.0.0.250;
    local-address 10.0.0.250;
    local-as 65000;
    peer-as 65000;
    family {{ ipv4 mpls-vpn; }}
    api {{ processes [ flap ]; }}
}}
process flap {{
    run /usr/bin/env python3 -c "
import sys, time
while True:
    sys.stdout.write('announce route {prefix} rd {rd} next-hop self label 116384\\n'); sys.stdout.flush()
    time.sleep({up})
    sys.stdout.write('withdraw route {prefix} rd {rd} next-hop self label 116384\\n'); sys.stdout.flush()
    time.sleep({down})
";
    encoder text;
}}
"""


def _exec(*cmd: str, run: bool) -> None:
    print("  $", " ".join(cmd))
    if run:
        subprocess.run(list(cmd), check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/run.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--precursor", type=float, default=90.0)
    ap.add_argument("--cycles", type=int, default=6, help="Number of flap cycles.")
    ap.add_argument("--up", type=float, default=60.0, help="Seconds announced.")
    ap.add_argument("--down", type=float, default=20.0, help="Seconds withdrawn.")
    ap.add_argument("--tool", choices=["exabgp", "clab"], default="clab")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    run = args.run

    hold = args.cycles * (args.up + args.down)
    clock = ScenarioClock(precursor_s=args.precursor, fault_s=args.precursor, hold_s=hold)

    label = open_label(
        args.labels,
        scenario=ScenarioId.B_BGP_FLAP,
        expected_issue=IssueType.BGP_ROUTE_FLAP,
        target_entity_id=TARGET_ENTITY,
        clock=clock,
        severity=Severity.P1,
        seed=args.seed,
        injection_tool=args.tool,
        params={
            "prefix": FLAP_PREFIX, "rd": FLAP_RD,
            "cycles": args.cycles, "up_s": args.up, "down_s": args.down,
        },
        target_sites=["dc", "hub"],
        target_vpns=["CORP", "OT"],
        expected_playbook_id="pb-bgp-flap-damping",
    )
    print(f"[B_bgp_flap] label {label.label_id} written -> {args.labels}")

    if args.tool == "exabgp":
        conf = EXABGP_TEMPLATE.format(prefix=FLAP_PREFIX, rd=FLAP_RD, up=args.up, down=args.down)
        Path("exabgp.flap.conf").write_text(conf, encoding="utf-8")
        print("[B_bgp_flap] wrote exabgp.flap.conf (run it inside the ExaBGP peer container):")
        print("  $ docker exec clab-netra-exabgp exabgp /exabgp.flap.conf")
        if run:
            print("[B_bgp_flap] (live ExaBGP orchestration is environment-specific; "
                  "start the speaker with the emitted config)", file=sys.stderr)
    else:
        # clab session bounce: toggle the RR peering interface on the flap cadence.
        for cyc in range(args.cycles):
            print(f"[B_bgp_flap] cycle {cyc + 1}/{args.cycles}: peer up({args.up}s)/down({args.down}s)")
            _exec("docker", "exec", RR_CONTAINER, "ip", "link", "set", RR_PEER_IFACE, "up", run=run)
            if run:
                time.sleep(args.up)
            _exec("docker", "exec", RR_CONTAINER, "ip", "link", "set", RR_PEER_IFACE, "down", run=run)
            if run:
                time.sleep(args.down)
        # restore
        _exec("docker", "exec", RR_CONTAINER, "ip", "link", "set", RR_PEER_IFACE, "up", run=run)

    if not run:
        print("[B_bgp_flap] dry-run complete. Use --run for a live lab.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
