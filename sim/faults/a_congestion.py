"""Scenario A — progressive congestion buildup on a hub-spoke link (WS1).

Injects a deterministic, labeled congestion fault on the hub PE's CE-facing link
(``clab-netra-pe-hub:eth3``) by stepping a ``tc`` HTB/rate ceiling **downward**
over time while diurnal + TRex/iperf3 load stays high. Queues fill, latency and
jitter climb and discards creep up *before* loss crosses the SLA — the precursor
the predictive engine must catch (research/01 §4.3 A).

Determinism: a single parameterised script, fixed seed, absolute timestamps, and
the ``ScenarioLabel`` written *before* injection starts (and the window fixed up
front). Default is ``--dry-run`` (prints the exact commands + writes the label)
so it is inspectable/testable with no live lab; ``--run`` actually executes.

    python sim/faults/a_congestion.py --labels labels/run.jsonl            # dry-run
    python sim/faults/a_congestion.py --labels labels/run.jsonl --run      # live

Mechanism per step: inside the target container's netns,
    tc qdisc replace dev eth3 root netem rate <mbit>   # congestion ceiling
plus a small rising netem delay. Cleared at the end (qdisc del).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from _labels import ScenarioClock, open_label

from netra.contracts import IssueType, ScenarioId, Severity

TARGET_CONTAINER = "clab-netra-pe-hub"
TARGET_IFACE = "eth3"
TARGET_ENTITY = "hub:pe-hub:PE:eth3"
# Rate ceiling schedule (Mbit): high -> progressively throttled (the buildup).
RATE_SCHEDULE_MBIT = [100, 60, 35, 18, 8]


def _exec(container: str, *cmd: str, run: bool) -> None:
    full = ["docker", "exec", container, *cmd]
    print("  $", " ".join(full))
    if run:
        subprocess.run(full, check=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/run.jsonl", help="ScenarioLabel JSONL path.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--precursor", type=float, default=120.0, help="Precursor lead (s).")
    ap.add_argument("--step", type=float, default=90.0, help="Seconds per rate step.")
    ap.add_argument("--run", action="store_true", help="Execute (default is dry-run).")
    args = ap.parse_args(argv)
    run = args.run

    hold = args.step * len(RATE_SCHEDULE_MBIT)
    clock = ScenarioClock(precursor_s=args.precursor, fault_s=args.precursor, hold_s=hold)

    # 1) Write the ground-truth label BEFORE injecting.
    label = open_label(
        args.labels,
        scenario=ScenarioId.A_CONGESTION,
        expected_issue=IssueType.INTERFACE_CONGESTION,
        target_entity_id=TARGET_ENTITY,
        clock=clock,
        severity=Severity.P1,
        seed=args.seed,
        injection_tool="tc+netem",
        params={"rate_schedule_mbit": RATE_SCHEDULE_MBIT, "step_s": args.step},
        target_sites=["hub", "br1", "br2", "br3"],
        target_vpns=["CORP"],
        expected_playbook_id="pb-congestion-qos-reroute",
    )
    print(f"[A_congestion] label {label.label_id} written -> {args.labels}")
    print(f"  fault window {label.fault_window_start.isoformat()} .. "
          f"{label.fault_window_end.isoformat()} on {TARGET_ENTITY}")

    # 2) Step the rate ceiling down (the progressive congestion).
    for mbit in RATE_SCHEDULE_MBIT:
        print(f"[A_congestion] step -> {mbit} mbit ceiling")
        _exec(TARGET_CONTAINER, "tc", "qdisc", "replace", "dev", TARGET_IFACE,
              "root", "netem", "rate", f"{mbit}mbit", "delay",
              f"{int(120 / max(mbit, 1))}ms", "5ms", run=run)
        if run:
            time.sleep(args.step)

    # 3) Clear the impairment (closes the impairment; label window already fixed).
    print("[A_congestion] clearing impairment")
    _exec(TARGET_CONTAINER, "tc", "qdisc", "del", "dev", TARGET_IFACE, "root", run=run)
    if not run:
        print("[A_congestion] dry-run complete (no commands executed). Use --run for a live lab.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
