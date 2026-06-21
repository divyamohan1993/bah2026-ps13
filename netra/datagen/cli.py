"""``netra-datagen`` CLI — generate labeled datasets and stream telemetry.

Two subcommands cover the two ways the rest of NETRA consumes the synthetic
source:

  * ``generate`` — materialise a deterministic, labeled dataset to disk:
      - ``<out>/telemetry.parquet``  (all records, flat columnar; JSONL fallback
        if pyarrow/pandas are unavailable)
      - ``<out>/telemetry.jsonl``    (always written — newline-delimited records,
        one JSON object per line, with a ``_type`` discriminator)
      - ``<out>/labels.jsonl``       (ground-truth ``ScenarioLabel``s)
      - ``<out>/manifest.json``      (the exact config + record counts, so a run
        is fully reproducible / auditable)
    These are exactly the artifacts a :class:`netra.datagen.ReplaySource` can
    re-read, and the parquet/JSONL the streaming + analytics workstreams load.

  * ``stream`` — emit records to stdout (NDJSON) or count them, optionally paced
    in real time (``--realtime --speed N``) to drive the live pipeline.

Everything is offline and CPU-only. ``pyarrow``/``pandas`` are optional: if they
are not importable the parquet step is skipped (a warning is printed) and the
JSONL artifacts — which are sufficient for replay and downstream loading — are
still written.

Run::

    python -m netra.datagen.cli generate --out ./data --seed 1337 \
        --duration 3600 --step 10
    python -m netra.datagen.cli stream --duration 600 --step 5 --format ndjson
    python -m netra.datagen.cli stream --count-only --duration 3600
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from netra.contracts import ScenarioId, ScenarioLabel

from .source import SyntheticSource
from .synthetic import GeneratorConfig, TelemetryUnion

# scenario shorthand accepted on the CLI
_SCENARIO_ALIASES: dict[str, ScenarioId] = {
    "a": ScenarioId.A_CONGESTION,
    "b": ScenarioId.B_BGP_FLAP,
    "c": ScenarioId.C_TUNNEL_DEGRADATION,
    "d": ScenarioId.D_POLICY_DRIFT,
    "A_congestion": ScenarioId.A_CONGESTION,
    "B_bgp_flap": ScenarioId.B_BGP_FLAP,
    "C_tunnel_degradation": ScenarioId.C_TUNNEL_DEGRADATION,
    "D_policy_drift": ScenarioId.D_POLICY_DRIFT,
    "baseline": ScenarioId.BASELINE,
    "none": ScenarioId.BASELINE,
}


# --------------------------------------------------------------------------- #
# Serialisation helpers                                                       #
# --------------------------------------------------------------------------- #


def record_to_row(rec: TelemetryUnion) -> dict:
    """Serialise a record to a JSON-able dict with a ``_type`` discriminator."""
    row = rec.model_dump(mode="json")
    row["_type"] = type(rec).__name__
    return row


def _iter_rows(records: Iterable[TelemetryUnion]) -> Iterator[dict]:
    for rec in records:
        yield record_to_row(rec)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":"), default=str))
            fh.write("\n")
            n += 1
    return n


def _flatten_for_parquet(row: dict) -> dict:
    """Flatten the nested ``labels`` dict and stringify it so parquet stays flat.

    Columnar formats dislike per-row variable-key maps; we JSON-encode ``labels``
    and ``params`` into a single string column. ``timestamp`` is left as an ISO
    string (pandas parses it on read).
    """
    flat = dict(row)
    for key in ("labels", "params"):
        if isinstance(flat.get(key), (dict, list)):
            flat[key] = json.dumps(flat[key], default=str)
    return flat


def _try_write_parquet(path: Path, rows: list[dict]) -> bool:
    """Write rows to parquet via pandas/pyarrow; return False if unavailable."""
    try:
        import pandas as pd  # noqa: PLC0415  (optional, import-guarded)
    except Exception:
        return False
    try:
        df = pd.DataFrame([_flatten_for_parquet(r) for r in rows])
        # parquet needs a single engine; pyarrow is the core-tier choice.
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:  # pragma: no cover - depends on optional engine
        print(f"[netra-datagen] parquet write skipped ({exc})", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Config construction                                                         #
# --------------------------------------------------------------------------- #


def _parse_scenarios(values: list[str] | None) -> tuple[ScenarioId, ...]:
    if not values:
        return (
            ScenarioId.A_CONGESTION,
            ScenarioId.B_BGP_FLAP,
            ScenarioId.C_TUNNEL_DEGRADATION,
            ScenarioId.D_POLICY_DRIFT,
        )
    out: list[ScenarioId] = []
    for v in values:
        key = v.strip()
        if key in _SCENARIO_ALIASES:
            sid = _SCENARIO_ALIASES[key]
        else:
            try:
                sid = ScenarioId(key)
            except ValueError as exc:
                raise SystemExit(
                    f"unknown scenario {v!r}; choose from a,b,c,d or "
                    f"{[s.value for s in ScenarioId]}"
                ) from exc
        if sid != ScenarioId.BASELINE and sid not in out:
            out.append(sid)
    return tuple(out)


def _parse_start(value: str | None) -> datetime:
    if not value:
        return datetime(2026, 6, 20, 8, 0, tzinfo=UTC)
    ts = datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        seed=args.seed,
        start=_parse_start(getattr(args, "start", None)),
        duration_s=float(args.duration),
        step_s=float(args.step),
        scenarios=_parse_scenarios(getattr(args, "scenario", None)),
        emit_flows=not getattr(args, "no_flows", False),
        emit_syslog=not getattr(args, "no_syslog", False),
    )


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> int:
    """Materialise a labeled dataset to ``--out``."""
    cfg = _config_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = SyntheticSource(config=cfg)

    # Collect records once (we need them twice: JSONL + parquet) — a run is
    # bounded by duration/step so this is intentionally in-memory.
    records = list(source.iter_records())
    rows = [record_to_row(r) for r in records]

    n_jsonl = _write_jsonl(out / "telemetry.jsonl", rows)
    parquet_ok = _try_write_parquet(out / "telemetry.parquet", rows)

    labels: list[ScenarioLabel] = source.labels()
    n_labels = _write_jsonl(
        out / "labels.jsonl", (lbl.model_dump(mode="json") for lbl in labels)
    )

    manifest = {
        "product": "netra.datagen",
        "seed": cfg.seed,
        "start": cfg.start.isoformat(),
        "duration_s": cfg.duration_s,
        "step_s": cfg.step_s,
        "scenarios": [s.value for s in cfg.scenarios],
        "emit_flows": cfg.emit_flows,
        "emit_syslog": cfg.emit_syslog,
        "record_count": n_jsonl,
        "label_count": n_labels,
        "parquet_written": parquet_ok,
        "record_type_counts": _type_counts(rows),
        "files": {
            "telemetry_jsonl": "telemetry.jsonl",
            "telemetry_parquet": "telemetry.parquet" if parquet_ok else None,
            "labels_jsonl": "labels.jsonl",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"[netra-datagen] wrote {n_jsonl} records "
        f"({'parquet+jsonl' if parquet_ok else 'jsonl only'}), "
        f"{n_labels} labels to {out}/"
    )
    for lbl in labels:
        print(
            f"  label {lbl.label_id}: {lbl.scenario.value} -> "
            f"{lbl.expected_issue.value} @ {lbl.target_entity_id} "
            f"(precursor {lbl.precursor_window_start.isoformat()} .. "
            f"fault {lbl.fault_window_start.isoformat()})"
        )
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    """Stream records to stdout (NDJSON) or count them, optionally real-time."""
    cfg = _config_from_args(args)
    source = SyntheticSource(config=cfg)

    if args.count_only:
        n = sum(1 for _ in source.iter_records())
        print(json.dumps({"record_count": n, "labels": len(source.labels())}))
        return 0

    paced = source.stream(realtime=args.realtime, speed=args.speed)
    out = sys.stdout
    written = 0
    try:
        for rec in paced:
            out.write(json.dumps(record_to_row(rec), separators=(",", ":"), default=str))
            out.write("\n")
            written += 1
            if args.limit and written >= args.limit:
                break
            if args.realtime:
                out.flush()
    except BrokenPipeError:  # pragma: no cover - downstream closed the pipe
        return 0
    return 0


def _type_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        t = r.get("_type", "?")
        counts[t] = counts.get(t, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Argument parser                                                             #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="netra-datagen",
        description=(
            "Deterministic synthetic SD-WAN/MPLS telemetry generator "
            "(the CPU-only TelemetrySource for NETRA)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--seed", type=int, default=1337, help="RNG seed.")
        sp.add_argument(
            "--start",
            type=str,
            default=None,
            help="ISO-8601 UTC start time (default 2026-06-20T08:00:00Z).",
        )
        sp.add_argument(
            "--duration", type=float, default=3600.0, help="Run duration in seconds."
        )
        sp.add_argument("--step", type=float, default=10.0, help="Sample step (s).")
        sp.add_argument(
            "--scenario",
            action="append",
            help="Scenario to inject (a/b/c/d or full id); repeatable. "
            "Default: all four.",
        )
        sp.add_argument("--no-flows", action="store_true", help="Skip NetFlow records.")
        sp.add_argument("--no-syslog", action="store_true", help="Skip syslog events.")

    g = sub.add_parser("generate", help="Write a labeled dataset to disk.")
    add_common(g)
    g.add_argument("--out", required=True, help="Output directory.")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("stream", help="Stream records to stdout (NDJSON).")
    add_common(s)
    s.add_argument(
        "--realtime",
        action="store_true",
        help="Pace emission by inter-record time deltas.",
    )
    s.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Real-time acceleration factor (e.g. 60 = 1 min/s).",
    )
    s.add_argument("--limit", type=int, default=0, help="Stop after N records (0=all).")
    s.add_argument(
        "--count-only",
        action="store_true",
        help="Only print the record/label counts (no NDJSON).",
    )
    s.set_defaults(func=cmd_stream)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
