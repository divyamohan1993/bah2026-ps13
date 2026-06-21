"""Run all four validation-scenario fault drivers in sequence (WS1 orchestrator).

Convenience wrapper that invokes ``a_congestion`` -> ``b_bgp_flap`` ->
``c_tunnel`` -> ``d_drift`` back-to-back, each appending its ``ScenarioLabel`` to
a shared JSONL. Default is ``--dry-run`` (prints every command, writes the
labels) so the full labeled run is inspectable/testable without a live lab;
``--run`` executes against a deployed Containerlab lab.

    python sim/faults/run_all.py --labels labels/run.jsonl            # dry-run
    python sim/faults/run_all.py --labels labels/run.jsonl --run      # live

The labels JSONL this produces is the sim-side equivalent of what
``netra.datagen``'s ``SyntheticSource.labels()`` returns — the same ground truth
the predictive ensemble (Phase 3) and the scoring (Phase 6) consume.
"""

from __future__ import annotations

import argparse

import a_congestion
import b_bgp_flap
import c_tunnel
import d_drift


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="labels/run.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args(argv)

    common = ["--labels", args.labels, "--seed", str(args.seed)]
    if args.run:
        common.append("--run")

    print("==== scenario A: progressive congestion ====")
    a_congestion.main(common)
    print("\n==== scenario B: BGP route flap ====")
    b_bgp_flap.main(common)
    print("\n==== scenario C: tunnel degradation ====")
    c_tunnel.main(common)
    print("\n==== scenario D: policy drift ====")
    d_drift.main(common)
    print(f"\n[run_all] all four scenarios labeled -> {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
