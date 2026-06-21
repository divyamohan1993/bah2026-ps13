# `sim/` — Phase 1: Simulated SD-WAN/MPLS environment (Workstream 1)

Topology-as-code for a reproducible, **fully offline**, multi-site
SD-WAN-over-MPLS lab with labeled fault injection. This is the live (`SIM`)
backend of the dual-source telemetry abstraction; the always-runnable
CPU-only synthetic backend lives in [`../netra/datagen/`](../netra/datagen) and
emits the **same** `netra.contracts` record types and `ScenarioLabel`s, so the
rest of NETRA is identical whichever source feeds it.

> **This lab does not run inside the air-gapped CPU-only container** (no Docker,
> no kernel MPLS, no NATS here). It is the high-fidelity "real" source that
> proves protocol behaviour on a capable Containerlab host. For the demo/eval on
> a plain CPU box, use the synthetic source. Everything here is written to run
> offline on such a host once the images are pre-staged (below).

---

## What's here

```
sim/
├── topology.clab.yml          # Containerlab IaC (the frozen, air-gapped artifact)
├── netlab/topology.yml        # netlab 26.06 source (renders clab + device configs)
├── Makefile                   # up / down / scenario-* / images-save|load
├── configs/
│   ├── frr/daemons            # shared FRR daemons file
│   ├── frr/<node>/frr.conf    # per-node FRR configs (p1, pe-dc2, rr-dc, ce-br1, …)
│   ├── srlinux/<node>.cfg     # Nokia SR Linux startup configs (gNMI telemetry nodes)
│   ├── strongswan/*.ipsec.conf# GRE-over-IPSec overlay (hub head-end + branch spokes)
│   ├── qos/tc-qos.sh          # tc HTB voice/business/bulk QoS classes
│   ├── templates/*.j2         # Jinja render templates (topology-as-code)
│   └── render_configs.py      # deterministic config renderer (offline)
└── faults/
    ├── _labels.py             # shared ScenarioLabel writer (contract-conformant)
    ├── a_congestion.py        # scenario A — progressive congestion (tc/netem)
    ├── b_bgp_flap.py          # scenario B — BGP route flap (ExaBGP / clab bounce)
    ├── c_tunnel.py            # scenario C — tunnel degradation (Pumba)
    ├── d_drift.py             # scenario D — policy drift (vtysh/NAPALM config push)
    └── run_all.py             # orchestrator: run all four, append labels
```

---

## Reference topology (5 sites, ~18 nodes)

Matches `research/01` §6 and — node-for-node — the entity catalog in
[`../netra/datagen/topology.py`](../netra/datagen/topology.py).

| Site | Role(s) | Devices (kind) | Scenario target |
|---|---|---|---|
| **DC** | dual PE + CE + RR | `pe-dc1` (SR Linux), `pe-dc2` (FRR), `ce-dc`, `rr-dc` | RR → **B** |
| **HQ/Hub** | PE + CE + IPSec head-end | `pe-hub` (SR Linux), `ce-hub` | hub link → **A** |
| **Branch-1/2/3** | CE spokes + IPSec tunnels | `ce-br1/2/3` | br1 tunnel → **C** |
| **MPLS core** | 4× P (IS-IS + SR-MPLS) | `p1`,`p3`,`p4` (FRR), `p2` (SR Linux) | core link → **C** |
| **Controller** | SD-WAN policy/intent | `sdwan-ctl` | → **D** |

- **Underlay:** IS-IS L2 + **SR-MPLS** (SRGB base 16000; LDP variant is a second
  profile). Real MPLS forwarding via the host kernel MPLS modules.
- **L3VPN:** VRFs `CORP` (RD/RT `100:1`) and `OT` (`100:2`); MP-BGP **VPNv4** via
  the route reflector, `send-community both` enforced (mandatory — else RTs are
  stripped and L3VPN silently fails).
- **Overlay (SD-WAN):** strongSwan **GRE-over-IPSec**, hub-spoke + partial mesh,
  with **OSPF inside the tunnels** so overlay paths reroute; short rekey lifetime
  so scenario C's rekey-anomaly precursor is observable.
- **QoS:** `tc` HTB classes voice/business/bulk (`configs/qos/tc-qos.sh`).

A few PE/P nodes are **Nokia SR Linux** (free image) purely for best-in-class
native **gNMI** on-change/sample streaming telemetry into Phase 2; everything
else is **FRRouting 10.3**.

### netlab vs Containerlab

`netlab/topology.yml` is the higher-level **source of truth**: one declarative
YAML with the routing modules (`isis`, `sr`, `mpls`, `bgp`, `vrf`, `ospf`).
`netlab create -p clab netlab/topology.yml` renders the Containerlab file **and**
the per-device configs; the rendered bundle then runs **without netlab
installed** — the air-gap artifact. `topology.clab.yml` is the hand-maintained
equivalent so the lab is usable directly from Containerlab too.

> **netlab link syntax (note):** links use `interfaces:` (**plural**) two-endpoint
> lists / peer node keys — proper two-endpoint links — **not** `interface`
> (singular) and **not** nested endpoint objects.

---

## Bringing it up — OFFLINE

### 0. Host prerequisites (one-time, on the Containerlab host)

```bash
# Containerlab + Docker installed. Load kernel MPLS modules for the real
# MPLS dataplane (SR-MPLS/LDP forwarding programs the LFIB through these):
sudo modprobe mpls_router mpls_gso mpls_iptunnel
echo -e "mpls_router\nmpls_gso\nmpls_iptunnel" | sudo tee /etc/modules-load.d/mpls.conf
```

### 1. Pre-stage the images across the air-gap (the hard offline requirement)

Containerlab pulls missing images at deploy time, so for a true air-gap **every**
image must be staged first. On an internet-connected **staging** host:

```bash
make -C sim images-save        # docker pull + docker save | gzip -> tarball + sha256
# (uses the CORRECTED Pumba image ghcr.io/alexei-led/pumba:latest)
```

Transfer `sim/clab-images-airgap.tar.gz` across the gap, then on the **offline**
host:

```bash
make -C sim images-load        # gunzip | docker load
```

(Optionally stand up a local `registry:2` inside the air-gap and retag — clab
then "pulls" locally with zero outbound traffic.) Pinned image set:
`frrouting/frr:v10.3.0`, `ghcr.io/nokia/srlinux:24.10.1`,
`ghcr.io/alexei-led/pumba:latest`.

### 2. Deploy / inspect / destroy

```bash
make -C sim up         # sudo containerlab deploy -t sim/topology.clab.yml
make -C sim inspect    # node list + mgmt IPs
make -C sim graph      # render the topology graph
make -C sim down       # sudo containerlab destroy --cleanup
```

### 3. (Optional) re-render configs from templates

```bash
python sim/configs/render_configs.py --write    # PE configs from templates/*.j2
```

---

## Fault injection (labeled — the supervised ground truth)

Every driver writes a contract-valid `ScenarioLabel` **before** it injects (the
window is fixed up front), guaranteeing the label bounds the impairment even if a
run is interrupted (`research/01` §4.3). Each is **dry-run by default** — it
prints the exact commands and writes the label with **no live lab needed** — and
takes `--run` to execute against a deployed lab.

```bash
# Inspect what every scenario would do + emit all four labels (no lab needed):
python sim/faults/run_all.py --labels sim/labels/run.jsonl

# Run one scenario live against the deployed lab:
make -C sim scenario-A         # progressive congestion on pe-hub:eth3 (tc/netem)
make -C sim scenario-B         # BGP route flap rr-dc<->pe-dc1 (clab bounce / ExaBGP)
make -C sim scenario-C         # intermittent core-link loss -> tunnel degrade (Pumba)
make -C sim scenario-D         # controller misconfig: wrong RT import (vtysh/NAPALM)

# Or all four in sequence, live:
make -C sim scenarios
```

| Scenario (`ScenarioId`) | Injector | Target entity (ground-truth root cause) | Predicted `IssueType` |
|---|---|---|---|
| `A_congestion` | `tc`/`netem` rate-step down | `hub:pe-hub:PE:eth3` | `interface_congestion` |
| `B_bgp_flap` | ExaBGP announce/withdraw **or** clab session bounce | `dc:rr-dc:RR:peer-pe-dc1` | `bgp_route_flap` |
| `C_tunnel_degradation` | **Pumba** netem loss bursts (`ghcr.io/alexei-led/pumba:latest`) | `br1:ce-br1:CE:tunnel-hub` | `tunnel_degradation` |
| `D_policy_drift` | vtysh / NAPALM config push (wrong RT import) | `dc:sdwan-ctl:controller` | `policy_drift` |

The emitted `labels/run.jsonl` is the sim-side equivalent of
`netra.datagen`'s `SyntheticSource.labels()` — the same `ScenarioLabel` shape the
predictive ensemble (Phase 3) trains on and the Phase-6 scoring measures lead
time against.

---

## How the live telemetry reaches NETRA

```
FRR / SR Linux nodes
  ─ gnmic (gNMI on-change + sample) + Telegraf (SNMP/syslog/NetFlow) [Phase 2]
  ─> NATS JetStream  telemetry.>
  ─> netra.datagen.ContainerlabSource (SIM)  ─ decodes each message into the
     matching netra.contracts record ─> the identical downstream pipeline.
```

`ContainerlabSource` (in [`../netra/datagen/source.py`](../netra/datagen/source.py))
is the documented adapter; it reads this live pipeline in a full deployment and
raises a clear error in the CPU-only container (use `SyntheticSource` there).

---

## Determinism & reproducibility

- All images **version-pinned**; fault drivers take a `--seed`; labels use
  **absolute** UTC timestamps and are written **before** injection.
- `containerlab destroy && deploy` gives sub-minute tear-up/down to repeat the
  four scenarios and regenerate datasets.
- Device names/roles/sites are identical across `netlab/topology.yml`,
  `topology.clab.yml`, and `netra/datagen/topology.py`, so the `SIM` and
  `SYNTHETIC` sources describe the same graph and emit aligned entity-ids.

## Contracts

Produces `TelemetryRecord`, `SyslogEvent`, `RoutingEvent`, `FlowRecord`,
`TunnelStat`, `ScenarioLabel`. See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md)
WS1 and [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §3 (Phase 1).
