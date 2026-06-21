#!/usr/bin/env python3
"""NETRA end-to-end demo — the four validation scenarios, offline, on CPU.

Runs the full :class:`netra.pipeline.NetraPipeline` over each of the four labeled
validation scenarios produced by the synthetic ``TelemetrySource`` and prints an
operator-style report per scenario:

  * **Q1 — what fails next & when**  predicted issue + entity + time-to-impact,
    with the analytics-sourced confidence.
  * **Q2 — why / which signals**     the ranked contributing signals (grounded).
  * **Q3 — what action**             the recommended remediation actions (copilot).
  * **EVAL**                         did NETRA raise risk during the precursor
    window BEFORE the labeled fault? measured LEAD TIME + which methods fired.

Then a final summary table across the four scenarios (detected? / lead time / top
method / copilot confidence).

It uses the deterministic template-fallback copilot (no model, no RAG heavy deps)
so it runs with ZERO heavy dependencies — synthetic data + the core analytics
tier + the template copilot. No GPU, no internet, no sim.

Usage::

    PYTHONPATH=. python scripts/demo.py                 # all four scenarios
    PYTHONPATH=. python scripts/demo.py --duration 900  # shorter/faster run
    PYTHONPATH=. python scripts/demo.py --scenario A    # one scenario only
    PYTHONPATH=. python scripts/demo.py --json out.json # also dump a JSON summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Make the repo importable when run directly (PYTHONPATH=. is the documented path,
# but be forgiving if invoked as `python scripts/demo.py`).
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from netra.contracts import ScenarioId  # noqa: E402
from netra.pipeline import NetraPipeline, PipelineConfig  # noqa: E402
from netra.pipeline.report import ScenarioEval, SituationReport  # noqa: E402

# Optional/heavy backends (statsmodels, sklearn) emit benign convergence warnings
# on these short synthetic series — silence them so the operator report stays clean.
warnings.filterwarnings("ignore")

# Friendly short aliases for the CLI + a stable print order.
_SCENARIOS: dict[str, ScenarioId] = {
    "A": ScenarioId.A_CONGESTION,
    "B": ScenarioId.B_BGP_FLAP,
    "C": ScenarioId.C_TUNNEL_DEGRADATION,
    "D": ScenarioId.D_POLICY_DRIFT,
}

_SCENARIO_TITLE: dict[ScenarioId, str] = {
    ScenarioId.A_CONGESTION: "A — Progressive hub-spoke congestion",
    ScenarioId.B_BGP_FLAP: "B — BGP route-flap cascade",
    ScenarioId.C_TUNNEL_DEGRADATION: "C — Intermittent MPLS/tunnel degradation",
    ScenarioId.D_POLICY_DRIFT: "D — Controller policy drift",
}

# ---- tiny ANSI styling (auto-disabled when not a TTY) ---------------------- #
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(s: str) -> str:
    return _c(s, "1")


def _green(s: str) -> str:
    return _c(s, "32")


def _red(s: str) -> str:
    return _c(s, "31")


def _yellow(s: str) -> str:
    return _c(s, "33")


def _cyan(s: str) -> str:
    return _c(s, "36")


def _rule(char: str = "─", width: int = 78) -> str:
    return char * width


def _fmt_minutes(m: float | None) -> str:
    if m is None:
        return "n/a"
    return f"{m:.1f} min"


# --------------------------------------------------------------------------- #
# per-scenario operator report                                               #
# --------------------------------------------------------------------------- #
def print_scenario_report(scen: ScenarioId, report: SituationReport, elapsed: float) -> None:
    title = _SCENARIO_TITLE.get(scen, scen.value)
    print()
    print(_bold(_cyan("╔" + _rule("═", 76) + "╗")))
    print(_bold(_cyan("║ ")) + _bold(f"{title:<74}") + _bold(_cyan(" ║")))
    print(_bold(_cyan("╚" + _rule("═", 76) + "╝")))

    inc = report.headline_incident
    ev = report.eval_for(scen)
    copilot = report.copilot_for(inc.incident_id) if inc else None

    if inc is None or copilot is None:
        print(_yellow("  No incident was raised for this scenario."))
        if ev is not None:
            _print_eval(ev)
        return

    # ---- Q1 — what fails next & when --------------------------------------- #
    print(_bold("\n  Q1 — What is likely to fail next, and when?"))
    print(
        f"      predicted issue : {_yellow(copilot.predicted_issue.value)}"
        f"   (confidence {copilot.confidence_score:.0%})"
    )
    root = inc.root_cause_entity.entity_id if inc.root_cause_entity else "n/a"
    print(f"      root-cause node : {root}")
    print(f"      time-to-impact  : {_fmt_minutes(copilot.time_to_impact_minutes)}")
    sc = copilot.affected_scope
    if sc.sites or sc.devices:
        print(
            f"      affected scope  : sites={sc.sites or '-'} "
            f"services={sc.services_or_vpns or '-'}"
        )

    # ---- Q2 — why / which signals ----------------------------------------- #
    print(_bold("\n  Q2 — Why is risk elevated, which signals contributed?"))
    print(f"      {inc.root_cause_hypothesis}")
    if copilot.contributing_signals:
        print("      top contributing signals:")
        for s in copilot.contributing_signals[:5]:
            shap = f"  [shap {s.shap_contribution:+.2f}]" if s.shap_contribution is not None else ""
            print(f"        • {s.signal}: {s.observation}{shap}")

    # ---- Q3 — what action -------------------------------------------------- #
    print(_bold("\n  Q3 — What corrective action should be taken?"))
    for i, a in enumerate(copilot.recommended_actions, 1):
        appr = "approval-gated" if a.requires_approval else "auto"
        ref = f"  (runbook {a.runbook_ref})" if a.runbook_ref else ""
        print(f"      {i}. [{a.urgency.value}/{appr}] {a.step}{ref}")
    if copilot.citations:
        print(f"      citations: {', '.join(copilot.citations[:4])}"
              + (" …" if len(copilot.citations) > 4 else ""))
    print(
        f"      copilot backend : {copilot.model_id or 'template-fallback'} "
        f"(fallback={copilot.used_fallback})"
    )

    # ---- EVAL — lead time vs the labeled fault ----------------------------- #
    if ev is not None:
        _print_eval(ev)

    print(_c(f"\n  (pipeline run: {elapsed:.1f}s, "
             f"{int(report.stats.get('records_processed', 0))} records, "
             f"{int(report.stats.get('streams_tracked', 0))} metric streams)", "90"))


def _print_eval(ev: ScenarioEval) -> None:
    print(_bold("\n  EVAL — did NETRA warn BEFORE the labeled fault?"))
    if ev.detected:
        verdict = _green("YES — risk raised in the precursor window")
        lead = _green(f"{_fmt_minutes(ev.lead_time_minutes)}")
    else:
        verdict = _red("NO — no in-window early warning")
        lead = _red("—")
    target = f"{_fmt_minutes((ev.expected_lead_time_seconds or 0) / 60.0)}"
    print(f"      early warning   : {verdict}")
    print(f"      LEAD TIME       : {lead}   (label target ≈ {target})")
    print(f"      peak risk       : {ev.peak_risk:.2f}")
    issue = _green("correct") if ev.predicted_issue_correct else _yellow("differs")
    print(f"      predicted issue : {issue} (expected {ev.expected_issue.value})")
    if ev.methods_fired:
        print(f"      methods fired   : {', '.join(ev.methods_fired[:6])}"
              + (" …" if len(ev.methods_fired) > 6 else ""))


# --------------------------------------------------------------------------- #
# summary table                                                              #
# --------------------------------------------------------------------------- #
def print_summary_table(rows: list[tuple[ScenarioId, SituationReport, float]]) -> None:
    print()
    print(_bold(_cyan("┌" + _rule("─", 92) + "┐")))
    print(_bold(_cyan("│ ")) + _bold(f"{'NETRA — 4-scenario validation summary':<90}") + _bold(_cyan(" │")))
    print(_bold(_cyan("├" + _rule("─", 92) + "┤")))
    header = (
        f"│ {'Scenario':<26} {'Detected':<9} {'Lead time':<11} "
        f"{'Top method':<18} {'Issue':<9} {'Conf':<6} │"
    )
    print(_bold(header))
    print(_cyan("├" + _rule("─", 92) + "┤"))
    n_detected = 0
    for scen, report, _elapsed in rows:
        ev = report.eval_for(scen)
        inc = report.headline_incident
        cp = report.copilot_for(inc.incident_id) if inc else None
        det = "YES" if (ev and ev.detected) else "no"
        if ev and ev.detected:
            n_detected += 1
        lead = _fmt_minutes(ev.lead_time_minutes) if ev else "n/a"
        top = (ev.top_method if ev and ev.top_method else "-")[:18]
        issue_ok = "ok" if (ev and ev.predicted_issue_correct) else "~"
        conf = f"{cp.confidence_score:.0%}" if cp else "-"
        det_col = _green(f"{det:<9}") if det == "YES" else _red(f"{det:<9}")
        line = (
            f"│ {scen.value:<26} {det_col} {lead:<11} "
            f"{top:<18} {issue_ok:<9} {conf:<6} │"
        )
        print(line)
    print(_cyan("└" + _rule("─", 92) + "┘"))
    verdict = _green(f"{n_detected}/4 scenarios detected with lead time") if n_detected == 4 \
        else _yellow(f"{n_detected}/4 scenarios detected with lead time")
    print(_bold(f"\n  RESULT: {verdict} — fully offline, CPU-only, template-fallback copilot.\n"))


# --------------------------------------------------------------------------- #
# JSON dump (optional)                                                        #
# --------------------------------------------------------------------------- #
def build_json_summary(rows: list[tuple[ScenarioId, SituationReport, float]]) -> dict:
    out: dict = {"scenarios": []}
    for scen, report, elapsed in rows:
        ev = report.eval_for(scen)
        inc = report.headline_incident
        cp = report.copilot_for(inc.incident_id) if inc else None
        out["scenarios"].append(
            {
                "scenario": scen.value,
                "detected": bool(ev.detected) if ev else False,
                "lead_time_minutes": ev.lead_time_minutes if ev else None,
                "peak_risk": ev.peak_risk if ev else None,
                "predicted_issue_correct": bool(ev.predicted_issue_correct) if ev else False,
                "top_method": ev.top_method if ev else None,
                "methods_fired": ev.methods_fired if ev else [],
                "headline_incident": {
                    "incident_id": inc.incident_id,
                    "predicted_issue": inc.predicted_issue.value,
                    "severity": inc.severity.value,
                    "risk_score": round(inc.risk.risk_score, 4),
                    "root_cause_entity": (
                        inc.root_cause_entity.entity_id if inc.root_cause_entity else None
                    ),
                }
                if inc
                else None,
                "copilot": {
                    "predicted_issue": cp.predicted_issue.value,
                    "confidence_score": round(cp.confidence_score, 4),
                    "time_to_impact_minutes": cp.time_to_impact_minutes,
                    "n_recommended_actions": len(cp.recommended_actions),
                    "n_citations": len(cp.citations),
                    "used_fallback": cp.used_fallback,
                    "model_id": cp.model_id,
                }
                if cp
                else None,
                "runtime_seconds": round(elapsed, 2),
            }
        )
    out["detected_count"] = sum(1 for s in out["scenarios"] if s["detected"])
    return out


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        choices=sorted(_SCENARIOS),
        action="append",
        help="run only this scenario (A/B/C/D); repeatable. Default: all four.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=900.0,
        help="telemetry duration per scenario in seconds (default 900).",
    )
    parser.add_argument(
        "--step", type=float, default=10.0, help="sample period seconds (default 10)."
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "full"),
        default="fast",
        help="pipeline speed/fidelity profile (default fast: lightweight O(n) "
        "forecasters + anomaly pre-screen). 'full' runs the heavy ensemble.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="generator seed.")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="also write a machine-readable JSON summary to this path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the summary table (skip per-scenario reports).",
    )
    args = parser.parse_args(argv)

    chosen = (
        [_SCENARIOS[s] for s in args.scenario]
        if args.scenario
        else [_SCENARIOS[k] for k in ("A", "B", "C", "D")]
    )

    print(_bold(_cyan("\n" + _rule("═", 78))))
    print(_bold("  NETRA — air-gapped predictive NOC copilot (PS-13) — end-to-end demo"))
    print("  Source: synthetic TelemetrySource (labeled) · Copilot: template fallback")
    print(f"  Offline · CPU-only · seed={args.seed} · duration={args.duration:.0f}s/scenario"
          f" · profile={args.profile}")
    print(_bold(_cyan(_rule("═", 78))))

    rows: list[tuple[ScenarioId, SituationReport, float]] = []
    for scen in chosen:
        # A fresh pipeline per scenario isolates each fault morphology (its own
        # incident + clean root cause), which is the clearest operator view.
        pipe = NetraPipeline(PipelineConfig(step_seconds=args.step, profile=args.profile))
        t0 = time.time()
        report = pipe.run_scenario(scen, seed=args.seed, duration_s=args.duration, step_s=args.step)
        elapsed = time.time() - t0
        rows.append((scen, report, elapsed))
        if not args.quiet:
            print_scenario_report(scen, report, elapsed)

    print_summary_table(rows)

    if args.json:
        summary = build_json_summary(rows)
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(_c(f"  wrote JSON summary to {args.json}", "90"))

    # exit non-zero if any scenario failed to detect (useful in CI smoke checks)
    all_detected = all(
        (report.eval_for(scen) and report.eval_for(scen).detected) for scen, report, _ in rows
    )
    return 0 if all_detected else 1


if __name__ == "__main__":
    raise SystemExit(main())
