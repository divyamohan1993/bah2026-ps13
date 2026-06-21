# `sim/` — Phase 1: Simulated SD-WAN/MPLS environment (Workstream 1)

Topology-as-code for a reproducible, fully offline, multi-site SD-WAN-over-MPLS
lab with labeled fault injection. This is the live (`SIM`) backend of the
dual-source telemetry abstraction; the synthetic backend lives in
[`../netra/datagen/`](../netra/datagen).

**What goes here:**
- `topology.clab.yml` (+ `netlab/topology.yml`) — Containerlab/netlab IaC for the
  ~20-node, 5-site topology (DC, HQ/Hub, Branch-1/2/3, MPLS core, RR) with
  FRRouting + Nokia SR Linux nodes, IS-IS + SR-MPLS underlay, VRF `CORP`/`OT`
  L3VPN, and strongSwan GRE-over-IPSec overlay.
- `configs/` — FRR / SR Linux / strongSwan / `tc` QoS config render templates.
- `scenarios/{a_congestion,b_bgp_flap,c_tunnel,d_drift}.py` — seeded fault
  drivers (tc/netem + Pumba + ExaBGP/GoBGP + config push) that write
  `ScenarioLabel` JSONL *before* injection starts.
- `traffic/` — iperf3 / TRex ASTF / Scapy seeded traffic generators (diurnal).

**Contracts:** produces `TelemetryRecord`, `SyslogEvent`, `RoutingEvent`,
`FlowRecord`, `TunnelStat`, `ScenarioLabel`. See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS1.

**Air-gap:** all container images are pre-staged via `docker save`/local
registry; host kernel MPLS modules (`mpls_router`) are a documented prerequisite.

> Optional for the demo: the CPU-only path uses the synthetic generator instead.
