"""Scenario C — intermittent MPLS underlay failure / tunnel degradation (WS1).

Intermittently impairs a core (P-P) link so LSPs / SR transport paths flap, and
the SD-WAN GRE-over-IPSec overlay riding on top (branch ce-br1's tunnel-hub) sees
rising loss/jitter and IKE rekey churn. The precursors — transport-label path
changes, an increasing tunnel loss/jitter trend and IPSec rekey/SA-rebuild
anomalies — precede SLA loss (research/01 §4.3 C). Ground-truth root cause entity
is ``br1:ce-br1:CE:tunnel-hub``.

Primary injector is **Pumba** (container-targeted chaos that wraps tc/netem),
using the CORRECTED image ``ghcr.io/alexei-led/pumba:latest`` (NOT
``alexei-led/pumba`` or ``alexeiled/pumba``). Pumba's sidekick-tc model means even
a minimal FRR/P container can be impaired without modifying the image.

Determinism: fixed seed, absolute timestamps, label written BEFORE injection,
burst cadence fixed. Default ``--dry-run``; ``--run`` executes.

    python sim/faults/c_tunnel.py --labels labels/run.jsonl                # dry-run
    python sim/faults/c_tunnel.py --labels labels/run.jsonl --run          # live (Pumba)

Each burst (every ``--burst-period`` s for ``--burst-len`` s) applies high loss to
the core link; between bursts the link recovers — the "intermittent" signature.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from _labels import ScenarioClock, open_label

from netra.contracts import IssueType, ScenarioId, Severity

# CORRECTED Pumba image (PR review): ghcr.io/alexei-led/pumba:latest
PUMBA_IMAGE = "ghcr.io/alexei-led/pumba:latest"
# Impair a core P-P link carrying the LSP that the br1 overlay rides over.
CORE_CONTAINER = "clab-netra-p3"
CORE_IFACE = "eth1"               # p3 <-> p2 core link
TARGET_ENTITY = "br1:ce-br1:CE:tunnel-hub"


def _pumba_netem(loss_pct: float, secs: float, run: bool) -> None:
    """One time-boxed Pumba netem loss burst on the core link container."""
    cmd = [
        "docker", "run", "--rm",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        PUMBA_IMAGE,
        "netem", "--duration", f"{int(secs)}s", "--interface", CORE_IFACE,
        "loss", "--percent", str(loss_pct), "--correlation", "25",
        CORE_CONTAINER,
    ]
    print("  $", " ".join(cmd))
    if run:
        subprocess.run(cmd, check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/run.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--precursor", type=float, default=120.0)
    ap.add_argument("--bursts", type=int, default=6)
    ap.add_argument("--burst-period", type=float, default=40.0, help="Seconds between burst starts.")
    ap.add_argument("--burst-len", type=float, default=12.0, help="Seconds each burst lasts.")
    ap.add_argument("--loss", type=float, default=60.0, help="Loss %% during a burst.")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)
    run = args.run

    hold = args.bursts * args.burst_period
    clock = ScenarioClock(precursor_s=args.precursor, fault_s=args.precursor, hold_s=hold)

    label = open_label(
        args.labels,
        scenario=ScenarioId.C_TUNNEL_DEGRADATION,
        expected_issue=IssueType.TUNNEL_DEGRADATION,
        target_entity_id=TARGET_ENTITY,
        clock=clock,
        severity=Severity.P2,
        seed=args.seed,
        injection_tool="pumba",
        params={
            "pumba_image": PUMBA_IMAGE,
            "core_link": f"{CORE_CONTAINER}:{CORE_IFACE}",
            "bursts": args.bursts, "burst_period_s": args.burst_period,
            "burst_len_s": args.burst_len, "loss_pct": args.loss,
        },
        target_sites=["br1", "hub"],
        target_vpns=["CORP"],
        expected_playbook_id="pb-tunnel-reroute-backup",
    )
    print(f"[C_tunnel] label {label.label_id} written -> {args.labels}")
    print(f"  intermittent {args.loss}% loss x{args.bursts} on {CORE_CONTAINER}:{CORE_IFACE} "
          f"(image {PUMBA_IMAGE})")

    for b in range(args.bursts):
        print(f"[C_tunnel] burst {b + 1}/{args.bursts}")
        _pumba_netem(args.loss, args.burst_len, run=run)
        if run:
            # gap between burst starts (burst itself is time-boxed by Pumba)
            time.sleep(max(0.0, args.burst_period - args.burst_len))

    if not run:
        print("[C_tunnel] dry-run complete. Use --run for a live lab (needs Pumba + Docker socket).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
