"""Tests for ``netra.datagen`` — the synthetic 4-scenario TelemetrySource (WS1).

These assert the four properties the rest of NETRA relies on:

  1. **Determinism** — same ``GeneratorConfig`` (seed) ⇒ byte-for-byte identical
     output; different seed ⇒ different output.
  2. **Schema conformance** — every emitted record is a valid
     ``netra.contracts`` model and round-trips through JSON; labels validate
     their window ordering.
  3. **Label correctness** — one ``ScenarioLabel`` per requested scenario, with
     the right ``IssueType``, target entity, severity, and ordered windows.
  4. **Precursor precedes fault** — for each scenario the injected signal is
     measurably elevated/trending during the precursor window
     (``[precursor_window_start, fault_window_start)``) and stronger inside the
     fault window — i.e. a forecaster/drift detector gets real lead time.

The suite is CPU-only and offline: it needs only ``pydantic`` + ``numpy`` (and
optionally ``pandas``/``pyarrow`` for the parquet round-trip, which is skipped if
those are absent). No sim, no GPU, no network.
"""

from __future__ import annotations

import json
import statistics as st
from datetime import UTC, datetime

import pytest

from netra.contracts import (
    FlowRecord,
    IssueType,
    MetricName,
    RoutingEvent,
    ScenarioId,
    ScenarioLabel,
    SyslogEvent,
    TelemetryRecord,
    TelemetrySourceKind,
    TunnelStat,
)
from netra.datagen import (
    ContainerlabSource,
    GeneratorConfig,
    ReplaySource,
    SyntheticGenerator,
    SyntheticSource,
)
from netra.datagen.scenarios import diurnal_multiplier

# A small, fast, fully-deterministic config reused across tests.
START = datetime(2026, 6, 20, 8, 0, tzinfo=UTC)


def _cfg(seed: int = 1337, duration: float = 1200.0, step: float = 10.0) -> GeneratorConfig:
    return GeneratorConfig(seed=seed, start=START, duration_s=duration, step_s=step)


def _records(cfg: GeneratorConfig) -> list:
    return list(SyntheticGenerator(cfg).iter_records())


def _series(records, *, metric=None, device=None, sub_key=None, sub_val=None, value_attr="value"):
    """Extract a time-ordered (rel_seconds, value) series for one stream."""
    out = []
    base_ts = None
    for r in records:
        if metric is not None and getattr(r, "metric_name", None) != metric:
            continue
        if device is not None and getattr(r, "device", None) != device:
            continue
        if sub_key is not None:
            if isinstance(getattr(r, "labels", None), dict):
                if r.labels.get(sub_key) != sub_val:
                    continue
            elif getattr(r, sub_key, None) != sub_val:
                continue
        if base_ts is None:
            base_ts = r.timestamp
        out.append(((r.timestamp - base_ts).total_seconds(), getattr(r, value_attr)))
    return out


# --------------------------------------------------------------------------- #
# 1. Determinism                                                              #
# --------------------------------------------------------------------------- #


def test_same_seed_is_byte_identical():
    a = [r.model_dump_json() for r in _records(_cfg(seed=1337))]
    b = [r.model_dump_json() for r in _records(_cfg(seed=1337))]
    assert a == b
    assert len(a) > 1000  # the run actually produced a substantial stream


def test_different_seed_differs():
    a = [r.model_dump_json() for r in _records(_cfg(seed=1337))]
    c = [r.model_dump_json() for r in _records(_cfg(seed=2024))]
    assert a != c
    # the noise differs but the record COUNT/skeleton is identical (same topology)
    assert len(a) == len(c)


def test_source_iteration_is_repeatable():
    src = SyntheticSource(config=_cfg(seed=7, duration=600))
    first = [r.model_dump_json() for r in src.iter_records()]
    second = [r.model_dump_json() for r in src.iter_records()]
    assert first == second  # iterating twice yields the same stream


def test_diurnal_multiplier_is_deterministic_and_bounded():
    for h in range(0, 24):
        ts = datetime(2026, 6, 20, h, 0, tzinfo=UTC)
        m1 = diurnal_multiplier(ts)
        m2 = diurnal_multiplier(ts)
        assert m1 == m2
        assert 0.1 <= m1 <= 1.1
    # mid-afternoon busier than pre-dawn
    busy = diurnal_multiplier(datetime(2026, 6, 20, 15, tzinfo=UTC))
    quiet = diurnal_multiplier(datetime(2026, 6, 20, 4, tzinfo=UTC))
    assert busy > quiet


# --------------------------------------------------------------------------- #
# 2. Schema conformance                                                      #
# --------------------------------------------------------------------------- #


def test_all_records_are_valid_contract_models():
    records = _records(_cfg(duration=900))
    allowed = (TelemetryRecord, RoutingEvent, SyslogEvent, FlowRecord, TunnelStat)
    assert records, "generator produced no records"
    for r in records:
        assert isinstance(r, allowed), f"unexpected record type {type(r)}"
        # round-trips through JSON and re-validates (extra='forbid' contracts)
        dumped = r.model_dump_json()
        type(r).model_validate_json(dumped)


def test_records_are_time_ordered():
    records = _records(_cfg(duration=900))
    ts = [r.timestamp for r in records]
    assert ts == sorted(ts), "records are not in non-decreasing timestamp order"


def test_all_five_record_types_are_emitted():
    records = _records(_cfg(duration=3600))
    kinds = {type(r).__name__ for r in records}
    assert {"TelemetryRecord", "TunnelStat", "RoutingEvent", "SyslogEvent", "FlowRecord"} <= kinds


def test_telemetry_records_carry_synthetic_provenance():
    records = _records(_cfg(duration=300))
    tele = [r for r in records if isinstance(r, TelemetryRecord)]
    assert tele
    assert all(r.source == TelemetrySourceKind.SYNTHETIC for r in tele)


def test_metric_values_respect_physical_bounds():
    records = _records(_cfg(duration=3600))
    for r in records:
        if isinstance(r, TelemetryRecord) and r.metric_name == MetricName.IF_UTIL_PCT.value:
            assert 0.0 <= r.value <= 100.0
        if isinstance(r, TunnelStat):
            assert 0.0 <= r.loss_pct <= 100.0
            assert r.jitter_ms >= 0.0
            assert r.rekey_interval_s is None or r.rekey_interval_s >= 60.0


def test_entity_ref_derivation_round_trips():
    records = _records(_cfg(duration=120))
    tele = next(r for r in records if isinstance(r, TelemetryRecord))
    ent = tele.entity()
    assert ent.entity_id.startswith(f"{tele.site}:{tele.device}:{tele.role.value}")
    assert ent.site == tele.site and ent.device == tele.device


# --------------------------------------------------------------------------- #
# 3. Label correctness                                                       #
# --------------------------------------------------------------------------- #


def test_one_label_per_requested_scenario():
    src = SyntheticSource(config=_cfg())
    labels = src.labels()
    got = {lbl.scenario for lbl in labels}
    assert got == {
        ScenarioId.A_CONGESTION,
        ScenarioId.B_BGP_FLAP,
        ScenarioId.C_TUNNEL_DEGRADATION,
        ScenarioId.D_POLICY_DRIFT,
    }
    assert all(isinstance(lbl, ScenarioLabel) for lbl in labels)


def test_label_windows_are_ordered_and_within_run():
    cfg = _cfg(duration=2000)
    src = SyntheticSource(config=cfg)
    end = START.timestamp() + cfg.duration_s
    for lbl in src.labels():
        assert lbl.precursor_window_start < lbl.fault_window_start <= lbl.fault_window_end
        assert lbl.fault_window_end.timestamp() <= end + cfg.step_s
        assert lbl.precursor_window_start >= START
        assert lbl.seed == cfg.seed


def test_label_issue_types_and_targets():
    src = SyntheticSource(config=_cfg())
    by_scen = {lbl.scenario: lbl for lbl in src.labels()}
    assert by_scen[ScenarioId.A_CONGESTION].expected_issue == IssueType.INTERFACE_CONGESTION
    assert by_scen[ScenarioId.B_BGP_FLAP].expected_issue == IssueType.BGP_ROUTE_FLAP
    assert by_scen[ScenarioId.C_TUNNEL_DEGRADATION].expected_issue == IssueType.TUNNEL_DEGRADATION
    assert by_scen[ScenarioId.D_POLICY_DRIFT].expected_issue == IssueType.POLICY_DRIFT
    # every target entity id is non-empty and colon-delimited
    for lbl in src.labels():
        assert lbl.target_entity_id.count(":") >= 2
        assert lbl.expected_playbook_id  # Q3 scoring needs a playbook ref
        assert lbl.expected_lead_time_seconds and lbl.expected_lead_time_seconds > 0


def test_scenario_subset_only_emits_requested_labels():
    cfg = GeneratorConfig(
        seed=1, start=START, duration_s=1200, step_s=10,
        scenarios=(ScenarioId.A_CONGESTION, ScenarioId.C_TUNNEL_DEGRADATION),
    )
    labels = SyntheticSource(config=cfg).labels()
    assert {lbl.scenario for lbl in labels} == {
        ScenarioId.A_CONGESTION,
        ScenarioId.C_TUNNEL_DEGRADATION,
    }


def test_empty_scenarios_gives_baseline_only():
    cfg = GeneratorConfig(seed=1, start=START, duration_s=600, step_s=10, scenarios=())
    src = SyntheticSource(config=cfg)
    assert src.labels() == []
    # a pure baseline still produces records
    assert len(list(src.iter_records())) > 100


# --------------------------------------------------------------------------- #
# 4. Precursor precedes (and is detectable before) the fault                 #
# --------------------------------------------------------------------------- #


def _window_stats(records, label, value_pred, value_attr="value"):
    """Mean value of a stream in the baseline / precursor / fault windows."""
    base_ts = records[0].timestamp
    pre_s = (label.precursor_window_start - base_ts).total_seconds()
    fault_s = (label.fault_window_start - base_ts).total_seconds()
    fault_end_s = (label.fault_window_end - base_ts).total_seconds()
    baseline, precursor, fault = [], [], []
    for r in records:
        if not value_pred(r):
            continue
        t = (r.timestamp - base_ts).total_seconds()
        v = getattr(r, value_attr)
        if t < pre_s - 30:
            baseline.append(v)
        elif pre_s <= t < fault_s:
            precursor.append(v)
        elif fault_s <= t < fault_end_s:
            fault.append(v)
    return baseline, precursor, fault


def test_scenario_A_congestion_precursor_precedes_fault():
    cfg = _cfg(duration=3600)
    records = _records(cfg)
    label = next(l for l in SyntheticSource(config=cfg).labels() if l.scenario == ScenarioId.A_CONGESTION)

    def pred(r):
        return (
            isinstance(r, TelemetryRecord)
            and r.metric_name == MetricName.IF_UTIL_PCT.value
            and r.device == "pe-hub"
            and isinstance(r.labels, dict)
            and r.labels.get("interface") == "eth3"
        )

    baseline, precursor, fault = _window_stats(records, label, pred)
    assert baseline and precursor and fault
    # utilisation is elevated in the precursor window vs baseline, and higher still in fault
    assert st.mean(precursor) > st.mean(baseline) + 2.0
    assert st.mean(fault) > st.mean(precursor)
    # and it is *trending up* across the precursor window (lead-time signal)
    pw = [(t, r.value) for r in records if pred(r)
          for t in [(r.timestamp - records[0].timestamp).total_seconds()]
          if (label.precursor_window_start - records[0].timestamp).total_seconds()
          <= t < (label.fault_window_start - records[0].timestamp).total_seconds()]
    times = [t for t, _ in pw]
    vals = [v for _, v in pw]
    # simple least-squares slope > 0
    n = len(times)
    mt = sum(times) / n
    mv = sum(vals) / n
    slope = sum((t - mt) * (v - mv) for t, v in pw) / sum((t - mt) ** 2 for t in times)
    assert slope > 0


def test_scenario_B_bgp_flap_precursor_precedes_fault():
    cfg = _cfg(duration=3600)
    records = _records(cfg)
    label = next(l for l in SyntheticSource(config=cfg).labels() if l.scenario == ScenarioId.B_BGP_FLAP)

    def pred(r):
        return (
            isinstance(r, TelemetryRecord)
            and r.metric_name == MetricName.BGP_FLAP_PENALTY.value
            and r.device == "rr-dc"
            and isinstance(r.labels, dict)
            and r.labels.get("peer") == "pe-dc1"
        )

    baseline, precursor, fault = _window_stats(records, label, pred)
    assert precursor and fault
    assert st.mean(precursor) > st.mean(baseline or [0.0]) + 1.0
    assert st.mean(fault) > st.mean(precursor)


def test_scenario_C_tunnel_precursor_precedes_fault():
    cfg = _cfg(duration=3600)
    records = _records(cfg)
    label = next(l for l in SyntheticSource(config=cfg).labels() if l.scenario == ScenarioId.C_TUNNEL_DEGRADATION)

    def loss_pred(r):
        return isinstance(r, TunnelStat) and r.device == "ce-br1" and r.tunnel_id == "tunnel-hub"

    baseline, precursor, fault = _window_stats(records, label, loss_pred, value_attr="loss_pct")
    assert precursor and fault
    # tunnel loss rises through the precursor and is worse in the fault
    assert st.mean(precursor) > st.mean(baseline or [0.0])
    assert st.mean(fault) > st.mean(precursor)

    # rekey interval is an ANOMALY: it shrinks below the ~3600s baseline as a precursor
    b2, p2, f2 = _window_stats(records, label, loss_pred, value_attr="rekey_interval_s")
    assert b2 and p2
    assert st.mean(p2) < st.mean(b2) - 50.0


def test_scenario_D_policy_drift_is_step_and_fans_out():
    cfg = _cfg(duration=3600)
    records = _records(cfg)
    label = next(l for l in SyntheticSource(config=cfg).labels() if l.scenario == ScenarioId.D_POLICY_DRIFT)

    def drift_pred(r):
        return (
            isinstance(r, TelemetryRecord)
            and r.metric_name == MetricName.CONFIG_DRIFT_SCORE.value
            and r.device == "sdwan-ctl"
        )

    baseline, precursor, fault = _window_stats(records, label, drift_pred)
    assert fault
    # near-zero before, a clear step up at/after the config push
    assert st.mean(baseline or [0.0]) < 0.1
    assert st.mean(fault) > 0.3
    # config-change syslog is emitted around the drift onset (the earliest signal)
    cfg_syslogs = [
        r for r in records
        if isinstance(r, SyslogEvent) and r.mnemonic == "%SYS-5-CONFIG_I"
    ]
    assert cfg_syslogs, "expected a %SYS-5-CONFIG_I config-push syslog for scenario D"


def test_non_target_entities_stay_healthy_during_scenarios():
    """An untouched interface keeps a sane utilisation across the whole run."""
    cfg = _cfg(duration=3600)
    records = _records(cfg)
    # p4 core router eth1 is not any scenario's target
    util = [
        r.value for r in records
        if isinstance(r, TelemetryRecord)
        and r.metric_name == MetricName.IF_UTIL_PCT.value
        and r.device == "p4"
        and isinstance(r.labels, dict)
        and r.labels.get("interface") == "eth1"
    ]
    assert util
    # healthy utilisation never saturates (well under congestion levels)
    assert max(util) < 80.0
    assert st.mean(util) < 60.0


# --------------------------------------------------------------------------- #
# Source interface behaviours                                                 #
# --------------------------------------------------------------------------- #


def test_replay_source_round_trips_records_and_labels():
    cfg = _cfg(duration=600)
    src = SyntheticSource(config=cfg)
    records = list(src.iter_records())
    rows = []
    for r in records:
        d = r.model_dump(mode="json")
        d["_type"] = type(r).__name__
        rows.append(d)
    labels = [lbl.model_dump(mode="json") for lbl in src.labels()]

    replay = ReplaySource.from_records(rows, labels)
    replayed = list(replay.iter_records())
    assert len(replayed) == len(records)
    assert replay.kind == TelemetrySourceKind.REPLAY
    assert len(replay.labels()) == len(src.labels())
    # types reconstructed correctly from the serialised rows
    assert [type(x).__name__ for x in replayed] == [type(x).__name__ for x in records]


def test_replay_source_sorts_unsorted_input():
    cfg = _cfg(duration=300)
    records = list(SyntheticSource(config=cfg).iter_records())
    shuffled = list(reversed(records))
    replay = ReplaySource(shuffled)
    ts = [r.timestamp for r in replay.iter_records()]
    assert ts == sorted(ts)


def test_stream_pacing_path_without_sleep():
    src = SyntheticSource(config=_cfg(duration=120))
    paced = list(src.stream(realtime=True, speed=1000.0, sleep=False))
    assert len(paced) == len(list(src.iter_records()))


def test_stream_rejects_bad_speed():
    src = SyntheticSource(config=_cfg(duration=60))
    with pytest.raises(ValueError):
        list(src.stream(realtime=True, speed=0))


def test_containerlab_source_is_documented_stub():
    src = ContainerlabSource()
    assert src.kind == TelemetrySourceKind.SIM
    assert src.labels() == []
    with pytest.raises(RuntimeError, match="SyntheticSource"):
        list(src.iter_records())


def test_generator_config_validates_inputs():
    with pytest.raises(ValueError):
        GeneratorConfig(step_s=0)
    with pytest.raises(ValueError):
        GeneratorConfig(duration_s=0)
    # naive start is coerced to UTC
    cfg = GeneratorConfig(start=datetime(2026, 1, 1, 0, 0))
    assert cfg.start.tzinfo is UTC


def test_source_rejects_config_and_kwargs_together():
    with pytest.raises(ValueError):
        SyntheticSource(config=_cfg(), seed=5)


# --------------------------------------------------------------------------- #
# CLI smoke (dataset materialisation)                                         #
# --------------------------------------------------------------------------- #


def test_cli_generate_writes_dataset(tmp_path):
    from netra.datagen.cli import main

    out = tmp_path / "ds"
    rc = main(["generate", "--out", str(out), "--seed", "1337",
               "--duration", "300", "--step", "10"])
    assert rc == 0
    assert (out / "telemetry.jsonl").exists()
    assert (out / "labels.jsonl").exists()
    assert (out / "manifest.json").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == 1337
    assert manifest["record_count"] > 0
    assert manifest["label_count"] == 4

    # labels.jsonl re-parses into valid ScenarioLabel models
    for line in (out / "labels.jsonl").read_text().splitlines():
        ScenarioLabel.model_validate_json(line)

    # telemetry.jsonl rows re-parse via the replay loader
    rows = [json.loads(l) for l in (out / "telemetry.jsonl").read_text().splitlines()[:50]]
    replay = ReplaySource.from_records(rows)
    assert len(list(replay.iter_records())) == len(rows)


def test_cli_stream_count_only(capsys):
    from netra.datagen.cli import main

    rc = main(["stream", "--duration", "200", "--step", "10", "--count-only"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["record_count"] > 0
    assert payload["labels"] == 4
