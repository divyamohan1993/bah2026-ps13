# Phase 1 — Simulated SD-WAN/MPLS Network Environment, Fault Injection & Traffic Generation

**Project:** Air-Gapped Predictive Copilot for Secure MPLS Operations (Problem Statement 13)
**Scope of this report:** Objective 1 / Phase 1 — a reproducible, fully offline, free/open-source, multi-site simulated SD-WAN-over-MPLS network with CE/PE/P roles, MPLS forwarding, L3VPN segmentation, IPSec overlay, dynamic routing, QoS, realistic traffic, and labeled fault injection that feeds the telemetry pipeline (Phase 2) and the predictive ML engine (Phase 3).
**Date:** 2026-06-20

---

## 0. Executive Recommendation (TL;DR)

Build the entire environment as **topology-as-code on Containerlab**, with **FRRouting (FRR) 10.x** as the primary network OS for CE/PE/P routers, optionally mixing in **Nokia SR Linux** (free, container-native, best-in-class streaming telemetry) on a few PE/P nodes to get realistic gNMI telemetry. Use **netlab (v26.06)** as a higher-level generator that emits Containerlab topologies + device configs from a single declarative YAML — this is the single biggest accelerator for reproducibility and for correctly configuring MPLS/LDP/SR-MPLS/VRF/MP-BGP-VPNv4 without hand-writing every line.

| Decision area | Recommendation | Fallback |
|---|---|---|
| **Simulation platform** | **Containerlab** (YAML IaC, container-native, 200+ nodes/host, offline-friendly, free, Nokia-maintained) | GNS3 (if a hard-VM-only NOS is ever required) |
| **Topology generator** | **netlab 26.06** (declarative → Containerlab + auto-config of MPLS/VPN/BGP/IGP) | Hand-written Containerlab YAML + Jinja config templates |
| **Core/router NOS** | **FRRouting 10.3** (free, built-in, OSPF/IS-IS/LDP/SR-MPLS/MP-BGP VPNv4/VRF) | **Nokia SR Linux** (free image) for richer telemetry nodes |
| **SD-WAN overlay** | **strongSwan** (route-based VPN / VTI) + **GRE-over-IPSec** running a routing protocol on top | libreswan (RHEL-family default) |
| **Traffic generation** | **iperf3** (baseline/diurnal congestion) + **Cisco TRex ASTF** (realistic stateful app mix, latency/jitter) + **Scapy** (crafted/adversarial) | D-ITG / Ostinato |
| **Fault injection** | **Linux `tc`/`netem`** (delay/jitter/loss/rate) + **Pumba** (container-targeted chaos) + **ExaBGP/GoBGP** (route flap injection) | Chaos Mesh (only if the stack moves to Kubernetes) |
| **Ground-truth labels** | Orchestrator script writes timestamped `(t_start, t_end, scenario, params, target)` JSONL alongside every fault | — |

Everything below is **free/open-source and air-gap-deployable** (images pre-pulled to a local registry or `docker save`/`load` tarballs). Cisco cEOS and CML are intentionally avoided as primaries because they need licensing/BYOI; FRR + SR Linux cover all required protocol fidelity at zero cost.

---

## 1. Simulation Platform Choice

### 1.1 Comparison matrix

| Criterion | **Containerlab** | GNS3 | EVE-NG | vrnetlab | Kathará |
|---|---|---|---|---|---|
| **Reproducibility / IaC** | ★★★★★ Declarative YAML, Git-friendly, "labs as infrastructure-as-code" | ★★☆ GUI, manual build, no structured code | ★★★ Lab import/export, partial IaC | ★★★★ (wraps VMs into containers; used *by* clab) | ★★★★ Lab dir + `lab.conf`, scriptable |
| **Automation / CI/CD** | ★★★★★ Native Ansible/Terraform/GitHub Actions, spin-up in seconds | ★★ Limited add-ons | ★★ Limited add-ons | ★★★★ CLI-driven | ★★★★ CLI/`Makefile` driven |
| **Air-gapped / offline** | ★★★★★ Self-hosted; pre-pull or `docker load` tarballs | ★★★★ Self-hosted but needs local images | ★★★★ Self-hosted | ★★★★ Self-hosted | ★★★★ Self-hosted |
| **Resource footprint** | ★★★★★ Containers; **200+ nodes on one host**, sub-second boot | ★★ ~32 GB RAM recommended; "scales poorly past ~30 nodes" | ★★ Min 16 GB RAM / 8 vCPU for a useful server | ★★★ VM-per-node (heavy) | ★★★★ Lightweight containers |
| **License / cost** | **Free, OSS (Nokia)** | **Free, OSS** | Free Community / ~€150 Pro | Free, OSS | Free, OSS (academic, Roma Tre) |
| **CE/PE/P + MPLS modeling** | ★★★★★ FRR/SR Linux/cEOS/cRPD/IOL/XRd kinds, real MPLS dataplane via Linux kernel | ★★★★ Full protocol fidelity if you supply images | ★★★★ Same, image-dependent | ★★★★ Real vendor VMs (XRv, vMX) | ★★★ FRR/Quagga containers |
| **Telemetry-friendliness** | ★★★★★ gNMI/SNMP/syslog straight out of FRR & SR Linux containers | ★★★ | ★★★ | ★★★ | ★★★ |
| **Maturity for SP/MPLS labs** | ★★★★★ Large public catalog of MPLS/L3VPN/SR labs | ★★★★ | ★★★★ | ★★★ | ★★★ |

Sources: NetPilot lab comparison (https://www.netpilot.io/blog/gns3-vs-eve-ng-vs-containerlab-2026), Pi Stack (https://www.pistack.xyz/posts/2026-04-18-gns3-vs-eve-ng-vs-containerlab-self-hosted-network-simulation-2026/), Containerlab kinds (https://containerlab.dev/manual/kinds/), Containerlab image mgmt (https://containerlab.dev/manual/images/).

### 1.2 Recommendation & rationale

**Primary: Containerlab.** It is purpose-built for exactly the constraints of this problem statement:

- **Reproducibility** — the whole topology is one version-controlled `*.clab.yml`; `containerlab deploy`/`destroy` gives deterministic tear-up/down in seconds, which is essential for repeating the 4 validation scenarios and regenerating training datasets. ("YAML-based topology … version-controllable, shareable, repeatable — treats network labs like infrastructure-as-code.")
- **Air-gap** — fully self-hosted; images are pre-pulled (Section 5.3). No cloud dependency at any point, satisfying the 20% "Security & Offline Compliance" evaluation weight.
- **Footprint** — containers, not VMs. A multi-site MPLS topology (15–25 nodes) runs comfortably on a single laptop/server; clab can do 200+ nodes/host. This lets the ML team run many parallel/replayed experiments cheaply.
- **Cost** — free OSS, maintained by Nokia (`srl-labs/containerlab`).
- **Role fidelity** — first-class kinds for `linux`/FRR (CE/PE/P), `nokia_srlinux`, `arista_ceos`, `juniper_crpd`, `cisco_iol`, `cisco_xrd`. Real MPLS dataplane is provided by the host Linux kernel MPLS modules (Section 2.1), so PE/P label switching is genuine, not faked.
- **Telemetry** — FRR and SR Linux containers expose SNMP, syslog, and (SR Linux) gNMI/OpenConfig natively, which directly feeds Phase 2.

**Use netlab (26.06) on top of Containerlab.** netlab (https://netlab.tools, formerly netsim-tools) is a topology-as-code *generator*: you describe nodes, links, and protocol *modules* (`ospf`, `isis`, `bgp`, `mpls`, `vrf`, `sr`, `evpn`) in one YAML; netlab renders the Containerlab file **and** the per-device configuration for FRR/SR Linux/Arista/Cisco. It deliberately abstracts hard MPLS details (e.g., "netlab assumes you want to run LDP when using the MPLS module" and auto-handles SRGB/labels), which removes the most error-prone hand-config and makes labs perfectly reproducible across NOS vendors. Latest release **26.06 (2026-06-07)**, min Python **3.10**. (Sources: https://netlab.tools/release/, https://blog.ipspace.net/2026/03/netlab-sr-mpls-l3vpn/.)

> Practical posture: prototype/iterate with **netlab** (fast, vendor-neutral, correct), and **`netlab create`** can emit the raw Containerlab + config artifacts so the final, frozen, air-gapped lab is a self-contained Containerlab bundle that runs even without netlab installed.

**Fallback: GNS3.** Only needed if a scenario ever demands a NOS that exists exclusively as a full VM image and cannot run containerized. GNS3 has full protocol fidelity ("BGP, MPLS, EVPN … behaves as expected") but poor IaC and a heavy footprint, so it is a contingency, not the workhorse. (vrnetlab is the bridge: it packages such VMs into containers so they can still run *inside* Containerlab — keeping the IaC workflow intact.)

---

## 2. Routing / Data-Plane Stack

### 2.1 MPLS underlay (LDP or SR-MPLS) on FRR

FRRouting is the recommended core because it is free, built into Containerlab, and supports the full SP stack: **OSPF, IS-IS, LDP, BGP, MP-BGP VPNv4, SR-MPLS, EVPN, SRv6** (FRR 10.3 specifically fixed SR-MPLS output labels and isisd P2P handling). (Source: https://frrouting.org/release/10.3/.)

Real MPLS forwarding requires the **host** kernel MPLS modules (clab containers share the host kernel):

```bash
# On the Containerlab host (one-time, also persist via /etc/modules-load.d/)
modprobe mpls_router mpls_gso mpls_iptunnel
echo -e "mpls_router\nmpls_gso\nmpls_iptunnel" | sudo tee /etc/modules-load.d/mpls.conf
```

Per-router (inside each FRR node) MPLS must be enabled on `lo` and core-facing interfaces:

```bash
sysctl -w net.mpls.conf.lo.input=1
sysctl -w net.mpls.conf.eth1.input=1
sysctl -w net.mpls.platform_labels=1048575
```

…and in `frr.conf`:

```
interface eth1
 mpls enable
```

Two underlay options:

- **LDP** (classic): OSPF/IS-IS as IGP + `ldpd` for label distribution. Matches "traditional MPLS" mental model and the PS wording.
- **SR-MPLS** (modern, recommended): IS-IS (or OSPF) carries Segment IDs (node SIDs) directly — **no LDP control plane needed**, fewer moving parts, and it exercises traffic-engineering constructs the PS asks for. netlab disables LDP automatically when you select the `sr` module. Label stacking is genuine: a VPN label (e.g., `116384`) sits under a transport/node-SID label (e.g., `16006`), with vendor-specific SRGBs (FRR base 16000, Arista 900000). (Source: https://blog.ipspace.net/2026/03/netlab-sr-mpls-l3vpn/.)

> **Recommendation:** ship **SR-MPLS with IS-IS** as the default underlay (cleaner, TE-ready), and optionally keep an **LDP variant** as a second profile so the dataset covers both label-distribution styles. Both are one `module:` line apart in netlab.

### 2.2 L3VPN — VRFs + MP-BGP VPNv4

L3VPN gives the "VPN segmentation" the PS requires. The canonical FRR pattern (validated in `martimy/clab_mpls_frr` Lab3 and netlab's `mpls`/`vrf` modules):

```
! On each PE — define a customer VRF and export/import route-targets
router bgp 100 vrf blue
 address-family ipv4 unicast
  rd vpn export 100:1
  rt vpn both 100:1
  label vpn export auto
  export vpn
  import vpn
 exit-address-family
!
! PE-PE MP-BGP VPNv4 session (over the loopbacks / IGP)
router bgp 100
 neighbor 10.0.0.2 remote-as 100
 address-family ipv4 vpn
  neighbor 10.0.0.2 activate
  neighbor 10.0.0.2 send-community both   ! MANDATORY — else RT communities are stripped
 exit-address-family
```

Critical gotcha confirmed across sources: **`send-community both` is mandatory** on the VPNv4 session, otherwise the route-target extended communities that drive VRF import/export are dropped and L3VPN silently fails. (Sources: https://github.com/martimy/clab_mpls_frr, https://blog.ipspace.net/2022/04/netsim-mpls-vpn/.)

Use **≥2 VRFs** (e.g., `CORP` and `OT`/`GUEST`) to demonstrate isolation/segmentation — this also makes the "controller misconfiguration → policy drift" fault (Scenario D) concrete: drift = a wrong RT import that leaks routes between VRFs.

### 2.3 SD-WAN IPSec overlay

The "SD-WAN" character = encrypted overlay tunnels between sites with a routing protocol running *inside* the tunnels (so overlay paths can reroute), layered on the MPLS/IP underlay. Best free/containerizable approach:

- **strongSwan** (recommended) or **libreswan** (RHEL default) for IKEv2/IPSec. Both GPL, free, mature. strongSwan has the broader feature set (X.509/PKCS#11/TPM, route-based VPN/VTI); libreswan is simpler and is the default on RHEL-family. (Sources: https://en.wikipedia.org/wiki/StrongSwan, https://dohost.us/index.php/2025/10/13/installing-libreswan-on-your-linux-servers-alternative-to-strongswan/.)
- **Route-based VPN** is the right model for SD-WAN: strongSwan negotiates SAs, and you run a **GRE (or VTI/`XFRM`) interface** over the IPSec SA, then run OSPF/BGP across the GRE tunnel. This yields dynamic overlay routing and lets you generate the rekey/tunnel-health telemetry the PS calls out ("rekey anomalies", "tunnel health degradation"). (Sources: https://docs.strongswan.org/docs/latest/features/routeBasedVpn.html, https://techbloc.net/archives/2579 — open-source GRE-over-IPSec with strongSwan + FRR.)

Implementation in clab: run strongSwan **inside the CE/branch FRR container** (or a sidecar Linux container sharing the netns). The overlay tunnels (hub-spoke + a partial mesh) ride over the underlay; branch reachability to hub/DC is via the encrypted GRE-over-IPSec tunnels with BGP/OSPF on top.

> The IPSec overlay is the part with the least turnkey clab example, so budget integration time here. It is also the highest-value differentiator (real rekey/IKE telemetry, tunnel-flap faults), so it is worth doing properly rather than faking tunnels with plain GRE.

### 2.4 QoS / traffic engineering

- **QoS:** Linux `tc` HTB/`prio`/`fq_codel` qdiscs on PE/CE egress to enforce per-class bandwidth and priority (voice > business > bulk). This doubles as the *mechanism* for the congestion fault (Scenario A) and produces SNMP-visible queue/drop behavior.
- **TE:** SR-MPLS gives explicit-path/TE-policy capability natively (segment lists). FRR `pathd` supports SR-TE policies if explicit TE paths are wanted. (Source: https://docs.frrouting.org/en/latest/pathd.html.)

### 2.5 NOS image options — free vs BYOI

| NOS | clab kind | Free image? | RAM (approx) | MPLS/L3VPN | Telemetry strength | Use as |
|---|---|---|---|---|---|---|
| **FRRouting 10.3** | `linux` (frr) | **Yes, free/OSS, built-in** | very low (~tens of MB) | OSPF/IS-IS/LDP/SR-MPLS/MP-BGP VPNv4/VRF/EVPN/SRv6 | SNMP, syslog (gNMI via add-ons) | **Primary CE/PE/P** |
| **Nokia SR Linux** | `nokia_srlinux` | **Yes, free, freely downloadable** | low–moderate | MPLS/SR-MPLS, BGP, L3VPN | **Best-in-class** (100% YANG, native gNMI on-change/sample) | A few PE/P nodes for premium telemetry |
| Arista cEOS-lab | `arista_ceos` | BYOI (free account, must download) | moderate | MPLS/SR-MPLS + BGP | gNMI/SNMP | Optional multi-vendor realism |
| Juniper cRPD | `juniper_crpd` | BYOI (licensed image) | low | BGP/IS-IS/MPLS (control-plane) | gRPC | Optional |
| Cisco XRd / IOL | `cisco_xrd`/`cisco_iol` | BYOI (licensed) | higher | full | gNMI/SNMP | Avoid (licensing) |

Containerlab images that are not on a public registry are loaded via `docker load` (Section 5.3). (Sources: https://containerlab.dev/manual/kinds/, https://containerlab.dev/manual/images/, https://containerlab.dev/lab-examples/srl-ceos/, https://github.com/srl-labs/containerlab/discussions/807.)

> **Recommended mix for "free + realistic":** FRR everywhere (CE/PE/P) as the baseline; replace **2 PE and 1–2 P** nodes with **SR Linux** to get high-fidelity gNMI streaming telemetry into Phase 2 without any license cost. This also makes the dataset multi-vendor, which strengthens the ML generalization story.

---

## 3. Traffic Generation

Goal: produce SNMP-observable utilization, latency, jitter, congestion, and **realistic diurnal / application-aware** flows so the predictive engine has signal to forecast.

| Tool | Type | Strengths | Role in this project |
|---|---|---|---|
| **iperf3** | Stateless TCP/UDP, client/server | Simple, scriptable, precise bandwidth/UDP jitter+loss reporting, runs in any Linux container | **Baseline load + diurnal congestion ramps**; UDP mode reports jitter/loss for ground-truth correlation |
| **Cisco TRex (ASTF)** | Advanced **stateful** L7 | BSD-derived TCP/UDP stack, realistic app templates, **latency & jitter measurement**, scalable flows, pcap-template driven | **Realistic application mix** (HTTP/DNS/voice-like), congestion at scale, latency/jitter KPIs |
| **Scapy** | Python packet crafting | Arbitrary/adversarial packets, malformed flows, bursts | **Adversarial scenarios**, micro-bursts, crafted anomalies |
| **D-ITG** | Distributed flow gen | Per-flow IDT/PS distributions (exp, Poisson, normal), multi-protocol | Alternative for statistically-shaped diurnal flows |
| **Ostinato** | GUI + drone agents | Fine packet control, high throughput | Optional; controller/`drone` split is heavier to automate |
| **tcpreplay** | PCAP replay | Replay captured real traffic at controlled rate | Replay realistic captures for app-aware realism (offline-friendly) |

Sources: TRex ASTF (https://trex-tgn.cisco.com/trex/doc/trex_astf.html, https://deepwiki.com/cisco-system-traffic-generator/trex-core/4-stateful-traffic-generation-(astf)), traffic-gen comparisons (https://dl.acm.org/doi/10.1145/3488375, https://ostinato.org/guides/iperf-vs-ostinato-arch).

### 3.1 Recommended stack & how to drive realism

- **Primary:** `iperf3` for controllable baselines/ramps + **TRex ASTF** for realistic stateful application mix and built-in latency/jitter measurement. **Scapy** for adversarial/edge flows.
- **Diurnal pattern:** drive the generators from a scheduler (a Python `asyncio` orchestrator or cron-like loop) that modulates offered load by a **time-of-day profile** (e.g., sinusoid with business-hours peak, lunch dip, overnight trough), per-site weights (DC > hub > branch), and per-app class (voice constant low-rate, bulk bursty, web diurnal). This makes interface utilization, latency, and jitter exhibit learnable seasonality — exactly what LSTM/Prophet (Phase 3) consume.
- **Application-aware:** map flow classes to DSCP/QoS classes so traffic hits the `tc` QoS queues (Section 2.4); congestion then manifests as class-specific drops/latency, visible in SNMP and NetFlow/IPFIX.
- **Determinism:** seed every generator (TRex profile params, Scapy RNG, scheduler seed) so a given "day" is byte-for-byte reproducible across runs — critical for repeatable ML experiments.

Example diurnal driver sketch:

```python
# pseudo-driver: scale iperf3 offered load by a diurnal curve + jitter
import math, random, subprocess
random.seed(1337)  # determinism
def diurnal(hour):                      # 0..1 load multiplier
    base = 0.5 + 0.4*math.sin((hour-9)/24*2*math.pi)   # peak ~ 3pm
    return max(0.05, base + random.uniform(-0.05, 0.05))
for hour in range(24):
    mbps = int(diurnal(hour) * LINK_MBPS)
    subprocess.run(["docker","exec","clab-net-br1",
        "iperf3","-c","10.0.0.1","-u","-b",f"{mbps}M","-t","300"])
```

---

## 4. Fault Injection (with Ground-Truth Labels)

### 4.1 Tooling

| Tool | Mechanism | What it injects | Container-aware? |
|---|---|---|---|
| **Linux `tc`/`netem`** | qdisc on an interface | delay, jitter, loss (incl. correlation/Gilbert-Elliott), duplication, corruption, reorder, **rate** limit | Run inside target netns (`ip netns`/`docker exec`) |
| **`tc` HTB/TBF** | shaping qdisc | bandwidth ceilings → congestion | Inside target |
| **Pumba** | wraps `tc`/`netem` + iptables, targets containers by name/regex/label; can use a **sidekick `tc` image** if target lacks `tc` | delay/loss/rate/corrupt/duplicate + kill/pause/stop containers, on `--interval` | **Yes — Docker/containerd/Podman native** |
| **ExaBGP** | Python BGP speaker | programmatic route announce/withdraw → **route flap** | Runs as a container/peer |
| **GoBGP** | Go BGP daemon + CLI/gRPC | inject/withdraw routes, policy → flap & path change | Runs as a container/peer |
| **Link up/down scripts** | `ip link set ... down/up`, `containerlab tools` | underlay link failure/flap | Yes |
| **Chaos Mesh** | Kubernetes CRD `NetworkChaos` (partition/delay/loss/bandwidth/corrupt) | same netem effects, GitOps-style | **K8s only** (fallback if stack moves to K8s) |

Sources: Pumba (https://github.com/alexei-led/pumba, https://oneuptime.com/blog/post/2026-02-08-how-to-use-docker-for-chaos-engineering-with-pumba/view), ExaBGP (https://github.com/Exa-Networks/exabgp/wiki/use-cases-summary, https://labs.ripe.net/author/thomas_mangin/exabgp-a-new-tool-to-interact-with-bgp/), GoBGP (https://bizety.com/bgp-tools-quagga-exabgp-bird-openbgpd-and-gobgp/), Chaos Mesh NetworkChaos (https://chaos-mesh.org/docs/simulate-network-chaos-on-kubernetes/), netem (Linux `tc-netem`).

**Why Pumba as the chaos primary:** it is the only mature tool that targets **Docker containers directly** (clab's substrate) rather than K8s/cloud, it reuses kernel `tc`/`netem`, and its sidekick-`tc` model means even minimal FRR containers can be impaired without modifying images — ideal for air-gapped, reproducible runs.

### 4.2 Core `netem` cheat-sheet (the impairment engine)

```bash
# Inside a target container's netns (egress eth1):
# fixed delay + jitter (normal distribution), 25ms ±5ms
tc qdisc add dev eth1 root netem delay 25ms 5ms distribution normal
# 2% random loss with 25% correlation (bursty)
tc qdisc change dev eth1 root netem loss 2% 25%
# Gilbert-Elliott bursty loss model (good->bad transitions)
tc qdisc change dev eth1 root netem loss gemodel 1% 10% 70% 0.1%
# rate limit to 5 mbit (congestion ceiling)
tc qdisc change dev eth1 root netem rate 5mbit
# remove
tc qdisc del dev eth1 root
```

Pumba equivalents (container-targeted, time-boxed — easy to label):

```bash
# 120s of 200ms delay on the hub-spoke link container interface
pumba netem --duration 120s --interface eth1 delay --time 200 --jitter 30 clab-net-hub1
# 90s of 5% loss with correlation on a branch uplink
pumba netem --duration 90s --interface eth1 loss --percent 5 --correlation 25 clab-net-br3
# flap: kill a P-router every 30s for the window
pumba --interval 30s --random kill "re2:^clab-net-p[0-9]+$"
```

### 4.3 The 4 required validation scenarios — labeled fault-injection plan

For **every** fault the orchestrator records a ground-truth label so Phase 3 has supervised targets. Approach validated in the literature: containerized + `tc`/`netem` injection yields *perfect* ground truth because injection targets/timing are known and controlled. (Sources: https://arxiv.org/pdf/2410.18332, https://arxiv.org/pdf/2109.14276.)

#### Scenario A — Progressive congestion buildup on a hub-spoke link
- **Inject:** step a `tc rate`/HTB ceiling **downward** over time (e.g., 100→50→20→8 Mbit across 10 min) on the hub→spoke interface while diurnal+TRex load stays high; queue fills, latency/jitter climb, loss begins. Optionally add a slowly increasing `netem delay`.
- **Precursors the model should catch:** rising interface utilization slope, growing queue depth/`tc` drops, latency/jitter drift *before* loss threshold breach.
- **Telemetry touched:** SNMP ifHCInOctets/Out + ifOutDiscards, latency/jitter from TRex/iperf, NetFlow volume.

#### Scenario B — BGP route flap + downstream reroute cascade
- **Inject:** use **ExaBGP/GoBGP** (or `clab tools` to bounce a PE-CE/PE-PE session) to **announce/withdraw** a VPNv4 prefix repeatedly (e.g., 60s up / 20s down, N cycles); downstream PEs reconverge, best-path churns, traffic reroutes.
- **Precursors:** BGP UPDATE/withdraw rate spikes, adjacency flaps in syslog, path-attribute churn, transient path asymmetry — all *before* mass reachability loss.
- **Telemetry touched:** routing syslog (adjacency up/down), BGP UPDATE counters, NetFlow path shifts, latency change on rerouted flows.

#### Scenario C — Intermittent MPLS underlay failure with tunnel degradation
- **Inject:** intermittently fail a **core (P–P or P–PE) link** (`ip link set eth down`/Pumba kill, or `netem loss 100%` for short bursts) so LSPs/SR transport paths flap; the SD-WAN IPSec/GRE overlay riding on top sees rising loss/jitter and possible IKE rekey churn.
- **Precursors:** transport-label path changes, increasing tunnel packet-loss progression + jitter trend, IPSec rekey/SA-rebuild anomalies — the exact "tunnel health degradation scoring" signals the PS lists.
- **Telemetry touched:** MPLS/LFIB changes, overlay tunnel loss/jitter, strongSwan IKE/SA logs (rekeys), SNMP error counters.

#### Scenario D — Controller misconfiguration → policy drift
- **Inject:** push a **bad config delta** to a PE/CE (e.g., wrong RT import so VRF `CORP`↔`OT` leak; or wrong QoS class-map so a priority class is mis-marked; or a route-map that flips export). Apply via the same automation that "should" represent the SD-WAN controller, so it reads as controller-originated drift.
- **Precursors:** config-change event, gradually emerging anomalous reachability/flow patterns, QoS class behavior diverging from baseline — drift detectable before a hard SLA/security breach.
- **Telemetry touched:** config-change/syslog events, NetFlow showing leaked inter-VRF flows, QoS counters deviating, BGP table deltas.

#### Ground-truth label schema (emitted per fault)
```json
{"event_id":"f0007","scenario":"A_congestion","t_start":"2026-06-20T14:03:00Z",
 "t_end":"2026-06-20T14:13:00Z","target":"clab-net-hub1:eth1",
 "tool":"pumba+tc","params":{"rate_schedule_mbit":[100,50,20,8],"step_s":150},
 "expected_precursor_window_s":120,"severity":"high","seed":1337}
```
Store as **JSONL** alongside the telemetry; the `expected_precursor_window_s` defines the labeling window (pre-fault = "elevated risk" positive label) so Phase 3 can score **lead time**. Always log the RNG **seed** for byte-level reproducibility.

> **Determinism rule:** every scenario is driven by a single parameterized script with a fixed seed and absolute timestamps; fault windows are written *before* injection starts and closed on cleanup, guaranteeing labels exactly bound the impairment even if a run is interrupted.

---

## 5. Reproducibility & Topology-as-Code

### 5.1 Layered IaC model
1. **`topology.yml` (netlab)** — the single source of truth: sites, device roles, links, addressing, and `module:` selections (`isis`/`ospf`, `mpls`, `sr`, `bgp`, `vrf`). netlab renders…
2. **`*.clab.yml` + per-device configs** — frozen Containerlab artifacts (via `netlab create`) that run **without netlab installed** → ideal air-gapped bundle.
3. **`scenarios/*.py` + `traffic/*.py`** — seeded fault and traffic drivers.
4. **`labels/*.jsonl`** — ground-truth, regenerated deterministically with the topology.

Everything lives in Git; `make up` / `make down` / `make scenario-A` wrap the lifecycle.

### 5.2 Deterministic seeding & fast cycle
- Pin **all** versions (clab, netlab 26.06, FRR 10.3, SR Linux tag, TRex, Pumba, strongSwan) in a manifest.
- Fixed RNG seeds for traffic + faults; absolute (not relative) timestamps in labels.
- `containerlab deploy`/`destroy` for sub-minute tear-up/down → run the 4 scenarios repeatedly to grow the dataset and validate lead-time stability.

### 5.3 Air-gap image management (the hard offline requirement)
Containerlab pulls missing images at deploy time, so for air-gap you must **pre-stage every image**:
```bash
# On an internet-connected staging host:
for img in frrouting/frr:v10.3.0 ghcr.io/nokia/srlinux:latest \
           ghcr.io/alexei-led/pumba:latest strongswan/strongswan:latest; do
  docker pull "$img"; done
docker save frrouting/frr:v10.3.0 ghcr.io/nokia/srlinux:latest \
            ghcr.io/alexei-led/pumba:latest strongswan/strongswan:latest \
  | gzip > clab-images-airgap.tar.gz
# Transfer the tarball across the air-gap, then on the offline host:
gunzip -c clab-images-airgap.tar.gz | docker load
```
Optionally stand up a **local registry** (`registry:2`) inside the air-gap and retag images to it; clab then "pulls" locally with zero outbound traffic. This satisfies the PS "verifiably zero outbound dependency during runtime" requirement. (Sources: https://containerlab.dev/manual/images/, https://oneuptime.com/blog/post/2026-01-16-docker-export-import-images/view, https://cloudificationzone.com/2020/07/06/pull-and-push-docker-images-in-air-gapped-no-internet-environment/.)

---

## 6. Recommended Reference Topology

**Scale:** 5 sites, ~18–22 nodes (fits one host; expandable).

| Site | Role | Devices (kind) | Notes |
|---|---|---|---|
| **DC** (datacenter) | Hub of hubs | 2× PE (FRR or **SR Linux**), 2× CE (FRR), servers (Linux + iperf3/TRex server) | Dual-homed; primary VPNv4 RR candidate |
| **HQ/Hub** | Regional hub | 1× PE (SR Linux for telemetry), 1× CE (FRR), strongSwan overlay head-end | Hub-spoke aggregation (Scenario A target) |
| **Branch-1/2/3** | Spokes | 1× CE (FRR) each, strongSwan branch tunnels, traffic clients | Diurnal client load; overlay to Hub+DC |
| **MPLS core** | Provider core | 3–4× P routers (FRR + 1–2 **SR Linux**) | IS-IS + SR-MPLS; LSP/SR transport (Scenario C target) |
| **Route reflector** | Control | 1× RR (FRR) for VPNv4 | Scales MP-BGP; flap target (Scenario B) |

- **Underlay:** IS-IS + **SR-MPLS** (LDP variant as second profile).
- **L3VPN:** VRFs `CORP` and `OT` (≥2) across all PEs; RR-based VPNv4.
- **Overlay (SD-WAN):** strongSwan GRE-over-IPSec, hub-spoke + partial mesh, OSPF/BGP inside tunnels.
- **QoS:** `tc` HTB classes (voice/business/bulk) on CE/PE egress.

### 6.1 Addressing plan sketch
| Block | Purpose |
|---|---|
| `10.0.0.0/24` (loopbacks `/32`) | Router IDs / BGP peering / IS-IS node SIDs (SRGB base 16000) |
| `10.1.0.0/16` | MPLS core P-P / P-PE point-to-point links (`/30` or `/31`) |
| `10.2.0.0/16` | PE-CE links per VRF |
| `172.16.0.0/16` | Overlay/tunnel (GRE) addressing |
| `192.168.<site>.0/24` | Per-site customer LAN / hosts (VRF-scoped) |
| RD/RT | `CORP` = `100:1`, `OT` = `100:2` |

### 6.2 Skeleton netlab topology (illustrative)
```yaml
# topology.yml  (netlab 26.06 -> containerlab + FRR/SR Linux configs)
provider: clab
defaults.device: frr
module: [ isis, sr, mpls, bgp, vrf ]
vrfs:
  corp: { rd: "100:1", import: [corp], export: [corp] }
  ot:   { rd: "100:2", import: [ot],   export: [ot] }
nodes:
  p1: { device: frr }
  p2: { device: srlinux }      # SR Linux for premium telemetry
  pe-dc: { device: srlinux }
  pe-hub: { device: frr }
  ce-dc: { device: frr, module: [bgp, vrf] }   # CE at the DC site (VRF corp)
  ce-br1: { device: frr, module: [bgp, vrf] }
  rr: { device: frr }
links:
  # netlab link = two-endpoint list under `interfaces:` (plural). Short form `- [a, b]` also works.
  - interfaces: [ p1, p2 ]      # core link
    mpls.sr: True
  - interfaces: [ p1, pe-dc ]   # core link
  - interfaces: [ p2, pe-hub ]  # core link
  - interfaces: [ ce-dc, pe-dc ] # PE-CE link, placed in a VRF on the PE side
    vrf: corp
# (strongSwan overlay + tc QoS + traffic/fault drivers layered on the rendered clab lab)
```

---

## 7. Reference Repositories & Sources (pre-clone for air-gap)

- **Containerlab + FRR MPLS labs (static/LDP/L3VPN)** — `martimy/clab_mpls_frr` — https://github.com/martimy/clab_mpls_frr
- **netlab SR-MPLS L3VPN walkthrough** — https://blog.ipspace.net/2026/03/netlab-sr-mpls-l3vpn/ ; netlab MPLS/VPN — https://blog.ipspace.net/2022/04/netsim-mpls-vpn/
- **netlab** — https://netlab.tools (release 26.06: https://netlab.tools/release/) ; **bgplabs.net / isis.bgplabs.net** free hands-on SR/BGP/IS-IS labs (netlab-based)
- **Containerlab** — kinds https://containerlab.dev/manual/kinds/ ; images/offline https://containerlab.dev/manual/images/ ; SR Linux+cEOS example https://containerlab.dev/lab-examples/srl-ceos/
- **SR Linux streaming telemetry reference (gNMI+Prometheus+Grafana+Loki)** — https://github.com/srl-labs/srl-telemetry-lab ; Prometheus exporter https://learn.srlinux.dev/ndk/apps/srl-prom-exporter/
- **FRRouting 10.3** — https://frrouting.org/release/10.3/ ; pathd/SR-TE https://docs.frrouting.org/en/latest/pathd.html
- **Pumba** — https://github.com/alexei-led/pumba
- **ExaBGP** — https://github.com/Exa-Networks/exabgp ; **GoBGP** — https://bizety.com/bgp-tools-quagga-exabgp-bird-openbgpd-and-gobgp/
- **strongSwan** route-based VPN — https://docs.strongswan.org/docs/latest/features/routeBasedVpn.html ; GRE-over-IPSec + FRR https://techbloc.net/archives/2579
- **TRex ASTF** — https://trex-tgn.cisco.com/trex/doc/trex_astf.html
- **Chaos Mesh NetworkChaos** (K8s fallback) — https://chaos-mesh.org/docs/simulate-network-chaos-on-kubernetes/
- **Platform comparison** — https://www.netpilot.io/blog/gns3-vs-eve-ng-vs-containerlab-2026 ; https://www.pistack.xyz/posts/2026-04-18-gns3-vs-eve-ng-vs-containerlab-self-hosted-network-simulation-2026/
- **Ground-truth/fault-injection ML testbeds** — https://arxiv.org/pdf/2410.18332 ; https://arxiv.org/pdf/2109.14276

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| IPSec/SD-WAN overlay is the least turnkey piece | Start from strongSwan route-based VPN (VTI/GRE) + FRR; allocate explicit integration time; libreswan fallback |
| Host kernel MPLS modules absent/locked in target environment | Verify `modprobe mpls_router` early; document host prerequisite; SR-MPLS still needs kernel MPLS |
| SR Linux/cEOS image size/footprint on small hosts | Keep FRR as default; use SR Linux only on the few telemetry nodes; cap node count at ~20 |
| Air-gap image drift / missing image at deploy | Freeze a version manifest; pre-stage via `docker save`/local `registry:2`; CI check that all images load offline |
| L3VPN silently broken | Enforce `send-community both`; add an automated post-deploy reachability/route-target assertion test |
| Non-reproducible datasets | Seed all RNGs; absolute timestamps; write labels before injection; `clab destroy && deploy` between runs |

---

### Bottom line
**Containerlab + netlab + FRR (with SR Linux telemetry nodes) + strongSwan overlay + iperf3/TRex/Scapy traffic + tc/netem/Pumba/ExaBGP fault injection**, all version-pinned, seeded, and image-pre-staged, delivers a free, fully air-gapped, reproducible multi-site SD-WAN-over-MPLS environment with CE/PE/P roles, L3VPN segmentation, IPSec overlay, QoS/TE, realistic diurnal traffic, and the four required, precisely-labeled fault scenarios — directly feeding the telemetry pipeline and the predictive/LLM stack.
