# `security/` — Air-gap enforcement (Workstream 7)

Defense-in-depth zero-egress controls. Any single layer here blocks outbound
traffic; together they are belt-and-suspenders. Pairs with the verification test
in [`../tests/airgap/`](../tests/airgap) and the bundling in
[`../scripts/`](../scripts).

**What goes here:**
- `nftables.conf` — host firewall: `policy drop` on OUTPUT, allow only `lo` +
  the internal docker bridge subnet (+ optional LAN telemetry sources); `log +
  counter` every blocked egress attempt (surfaced in Grafana).
- `falco-rules.yaml` — runtime monitor: a CRITICAL rule firing on **any**
  outbound `connect` from a NETRA container (excludes the internal bridge CIDR).
- `seccomp-llm.json` — seccomp profile forbidding `socket`/`connect` for the LLM
  process (run under firejail/bubblewrap `--unshare-net`).
- `docker-network.md` — the `internal: true` bridge (`airgap_net`) + per-service
  hardening (`--cap-drop NET_RAW NET_ADMIN`, `--security-opt no-new-privileges`).
- `.env.example` — offline env vars (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `DO_NOT_TRACK=1`, `GRADIO_ANALYTICS_ENABLED=False`).

**Standards:** maps to NIST SP 800-53 SC-7 (boundary protection) / SC-7(21)
(component isolation). See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS7
and [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §8.
