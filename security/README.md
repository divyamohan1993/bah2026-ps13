# `security/` — Air-gap enforcement (Workstream 7)

Defense-in-depth, **layered** zero-egress controls for NETRA. **Any single layer
here blocks outbound traffic; together they are belt-and-suspenders.** This is
the enforcement half of the 20% "Security & Offline Compliance" score; the
*verification* half lives in [`../tests/airgap/`](../tests/airgap) (active
pytest proof) and the *offline delivery* half in [`../scripts/`](../scripts).

Maps to **NIST SP 800-53 SC-7** (boundary protection), **SC-7(21)** (component
isolation), **SI-4** (system monitoring). See
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §8 and
[`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS7.

---

## The layered egress model

Container egress in Docker does **not** traverse the host `OUTPUT` hook — it is
routed/NATed through the Docker bridge and therefore traverses the **`FORWARD`**
path. A host-`OUTPUT`-only firewall rule will **not** block a container reaching
the internet. NETRA therefore enforces egress at the layer each kind of traffic
actually traverses, six ways:

| # | Layer | File(s) | What it does | Traffic path it governs |
|---|---|---|---|---|
| **1** | **Container network isolation** | [`compose.security.yml`](compose.security.yml), [`networks.md`](networks.md) | `airgap_net` bridge with `internal: true` + **no gateway** → mesh has no route out by construction; pure-batch jobs use `network_mode: none`. | The Docker bridge itself (removes the route) |
| **2a** | **Host firewall — FORWARD (containers)** | [`nftables.conf`](nftables.conf) | native nftables `chain forward { hook forward priority 0; policy drop; }` — allow intra-lab, **LOG `EGRESS-DROP ` + counter, then DROP** the rest. **This (not `output`) is what container egress traverses.** | Forwarded (container) packets |
| **2b** | **Host firewall — DOCKER-USER (containers)** | [`docker-user.sh`](docker-user.sh) | iptables `-I DOCKER-USER`: allow intra-lab `RETURN`, then `LOG --log-prefix "EGRESS-DROP "`, then `DROP`. Docker consults `DOCKER-USER` **before** its own NAT/forward rules. | Forwarded (container) packets, iptables backend |
| **2c** | **Host firewall — OUTPUT (host)** | [`nftables.conf`](nftables.conf) | `chain output { policy drop; }` — defense-in-depth for **host-originated** traffic (the appliance itself). | Host-originated packets |
| **5** | **Process sandbox (the LLM)** | [`firejail/llama-server.profile`](firejail/llama-server.profile), [`firejail/llama-server-bwrap.sh`](firejail/llama-server-bwrap.sh), [`seccomp-llm.json`](seccomp-llm.json) | firejail `net none` / bubblewrap `--unshare-net` give the inference process an **empty network namespace**; the seccomp profile makes `socket`/`connect` return `EPERM`. The kernel kills/blocks any network syscall. | The LLM process's syscalls |
| **6** | **Telemetry-free, offline-by-design** | [`.env.example`](.env.example) | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DO_NOT_TRACK=1`, `GRADIO_ANALYTICS_ENABLED=False`, … switch off every phone-home/auto-update path. | Application-level network use |

> Layers numbered to match ARCHITECTURE.md §8. (DNS sinkhole — layer 3/4 there —
> is naturally satisfied: with `internal: true` + FORWARD-drop there is no path
> to an external resolver; point `/etc/resolv.conf` at a local-only address to
> make name resolution fail closed.)

The **always-on monitor** ([`falco-egress.yaml`](falco-egress.yaml)) sits across
all layers: Falco watches syscalls and fires **CRITICAL** the instant any NETRA
container attempts an external `connect()` — so even a *failed* attempt is
detected and logged, not just dropped.

---

## Files

| File | Purpose |
|---|---|
| [`nftables.conf`](nftables.conf) | Host firewall: default-DROP egress with the **FORWARD** chain governing container egress (the P1 correction) + an `output` backstop + `input` lockdown. `nft -c -f` clean. |
| [`docker-user.sh`](docker-user.sh) | iptables `DOCKER-USER` egress lockdown (allow intra-lab `RETURN` → `LOG EGRESS-DROP` → `DROP`). Idempotent; `--remove`/`--list`/`--help`. |
| [`compose.security.yml`](compose.security.yml) | Reusable compose fragment: `airgap_net` (`internal: true`, no gateway), per-service hardening anchors, and the `falco` monitor service. **The integrator layers this onto the top-level `docker-compose.yml`.** |
| [`networks.md`](networks.md) | The container-isolation model + exact **integrator wiring** instructions. |
| [`falco-egress.yaml`](falco-egress.yaml) | Falco CRITICAL rule on any external outbound connection from a NETRA container (always-on detection). |
| [`seccomp-llm.json`](seccomp-llm.json) | seccomp profile: default `ERRNO`, network-creating syscalls **omitted** → `socket`/`connect` denied. 134 compute/file/thread syscalls allow-listed. |
| [`firejail/llama-server.profile`](firejail/llama-server.profile) | firejail profile: `net none` + caps-drop + seccomp + read-only FS for the LLM. |
| [`firejail/llama-server-bwrap.sh`](firejail/llama-server-bwrap.sh) | bubblewrap launcher: `--unshare-net` empty netns for the LLM. |
| [`.env.example`](.env.example) | Offline env vars (layer 6). Copy to `.env` (git-ignored). |

---

## How each layer is verified

| Layer | Verification command / evidence |
|---|---|
| 1 (network) | `docker network inspect netra_airgap_net` → `"Internal": true`, no `Gateway`; `curl --max-time 3 https://1.1.1.1` from a mesh container fails. |
| 2a (nftables FORWARD) | `sudo nft -f security/nftables.conf` then `sudo nft list table inet netra_airgap` → see the `forward` chain `policy drop` + the `EGRESS-DROP` counter increment when something tries to egress. Syntax: `nft -c -f security/nftables.conf`. |
| 2b (DOCKER-USER) | `sudo security/docker-user.sh && sudo security/docker-user.sh --list` → RETURN/LOG/DROP rules present; `EGRESS-DROP` lines appear in `dmesg`/journal on a blocked attempt. |
| 5 (sandbox) | `firejail --profile=security/firejail/llama-server.profile curl https://1.1.1.1` → "Network is unreachable"; `python3 -c "import socket; socket.socket()"` under the seccomp profile → `PermissionError`. |
| 6 (offline env) | `env | grep -E 'OFFLINE|DO_NOT_TRACK'` inside the container shows the flags set. |
| **monitor** | `falco --validate security/falco-egress.yaml`; trigger a test connect with the intra-lab exclusion lifted → the CRITICAL line fires. |
| **active proof** | `pytest -q tests/airgap` (see [`../tests/airgap`](../tests/airgap)) — tries TCP/UDP/DNS/HTTPS egress, passes only if all blocked. One-command judge demo: `scripts/airgap_verify.sh`. |

**Apply order on the appliance (host controls need root):**

```bash
sudo nft -f security/nftables.conf          # layer 2a + 2c + input
sudo security/docker-user.sh                # layer 2b
cp security/.env.example .env               # layer 6
docker compose -f docker-compose.yml -f security/compose.security.yml up -d   # layers 1 + monitor
pytest -q tests/airgap                       # PROVE it (active conformance)
```
