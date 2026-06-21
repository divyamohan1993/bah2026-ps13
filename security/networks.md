# NETRA — Container network isolation (air-gap layer 1)

The **first and highest-leverage** air-gap control: put the copilot / LLM / RAG
/ analytics / API / UI mesh on a Docker **`internal: true`** bridge that has **no
gateway and no NAT to the host's external interface**, so the containers have *no
route to the internet by construction* — before any firewall rule is even
consulted (ARCHITECTURE.md §8 layer 1, research `07-noc-workflow-security.md`
§B6.1).

This document explains the model and tells the **integrator** exactly how to
wire the reusable fragment [`compose.security.yml`](compose.security.yml) into
the top-level `docker-compose.yml` (which the integrator owns).

---

## 1. The two isolation modes

| Mode | Directive | Use for | Egress possible? |
|---|---|---|---|
| **No network at all** | `network_mode: none` (`docker run --network none`) | pure-batch jobs that need zero networking (e.g. a one-shot analytics/SBOM job) | **No** — no stack at all |
| **Internal bridge** | a user-defined network with `internal: true` | the service mesh that must talk to *each other* but never outside | **No** — bridge has no gateway/NAT |

`internal: true` is the right mode for NETRA's mesh because the services must
reach each other (API→analytics, copilot→llama-server, copilot→qdrant,
grafana→victoriametrics) but must **never** reach the internet. Docker creates
the bridge **without** a gateway address and **without** installing the
MASQUERADE/NAT rules that normally let a bridge egress, so packets to any
off-bridge address have nowhere to go.

> Pure-batch components (no peer needs them on the network) can go one step
> further with `network_mode: none`.

---

## 2. The `airgap_net` network (defined in `compose.security.yml`)

```yaml
networks:
  airgap_net:
    name: netra_airgap_net
    driver: bridge
    internal: true                 # <-- no gateway, no NAT, no internet
    driver_opts:
      com.docker.network.bridge.name: br-airgap
    ipam:
      driver: default
      config:
        - subnet: 172.31.255.0/24  # inside 172.16/12 => allowed by nftables.conf
```

The subnet `172.31.255.0/24` sits inside `172.16.0.0/12`, which the host
firewall ([`nftables.conf`](nftables.conf)) and [`docker-user.sh`](docker-user.sh)
treat as intra-lab — so **intra-mesh traffic is allowed while everything else is
dropped**. Keeping the bridge subnet inside the allow-list means the
defense-in-depth layers agree with each other.

Network definitions **merge across `-f` compose files**, so simply layering this
fragment is enough to apply the network:

```bash
docker compose -f docker-compose.yml -f security/compose.security.yml up -d
```

---

## 3. Per-service hardening anchors

`compose.security.yml` defines two YAML anchors:

- `&netra-hardening` — `cap_drop: [ALL]`, `no-new-privileges`, `read_only: true`
  + `tmpfs: [/tmp]`, attach to `airgap_net`, offline env vars, log rotation.
- `&netra-llm-hardening` — the above **plus** `seccomp=./security/seccomp-llm.json`
  (denies `socket`/`connect`) for the `llama-server`.

> **Compose anchor caveat:** YAML anchors (`x-*`) are **file-local** — they do
> not cross `-f` boundaries. So to apply the *per-service* hardening, the
> integrator must either (a) define the services inside `compose.security.yml`,
> or (b) **copy the two anchors into the top-level compose** and reference them
> there. The **network** stanza and the **falco** service, by contrast, DO work
> purely by layering (they need no anchor from the main file).

### Integrator wiring (recommended: copy anchors into the main file)

In the top-level `docker-compose.yml`, paste the two `x-netra-*` anchors from
`compose.security.yml` at the top, then merge them into each service:

```yaml
services:
  netra-api:
    <<: *netra-hardening
    image: netra/api:pinned
    # API/UI/Grafana publish ONLY to loopback so they arrive via `lo`:
    ports: ["127.0.0.1:8000:8000"]

  netra-analytics:
    <<: *netra-hardening
    image: netra/analytics:pinned

  llama-server:
    <<: *netra-llm-hardening          # stricter: + seccomp-llm.json
    image: netra/llama-server:pinned
    volumes:
      - ./models:/models:ro           # GGUF mounted read-only
      - ./security/seccomp-llm.json:/seccomp-llm.json:ro

  qdrant:        { <<: *netra-hardening, image: qdrant/qdrant:pinned }
  grafana:       { <<: *netra-hardening, image: grafana/grafana:pinned, ports: ["127.0.0.1:3000:3000"] }
  victoriametrics: { <<: *netra-hardening, image: victoriametrics/victoria-metrics:pinned }
  nats:          { <<: *netra-hardening, image: nats:pinned }

networks:
  airgap_net:                          # merged from compose.security.yml
    external: false
```

Then bring it up with the security fragment layered on so the `falco` monitor
and the network options apply:

```bash
docker compose -f docker-compose.yml -f security/compose.security.yml up -d
```

### Pure-batch jobs

A one-shot analytics/SBOM/eval container that needs no peer traffic should use
the hardest isolation:

```yaml
  netra-batch:
    image: netra/analytics:pinned
    network_mode: none                 # no networking stack at all
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]
```

---

## 4. Publishing UI/API without breaking the air-gap

Bind published ports to **`127.0.0.1` only** (e.g. `127.0.0.1:8000:8000`).
Traffic then arrives over `lo`, which the host firewall's `input` chain accepts,
and no port is exposed on the external interface. The mesh stays internal; the
operator reaches the console at `http://127.0.0.1:8000`.

---

## 5. How this layer is verified

- `docker network inspect netra_airgap_net` → shows `"Internal": true` and **no
  `Gateway`** entry.
- From inside any mesh container: `curl --max-time 3 https://1.1.1.1` →
  fails/times out (no route).
- The active proof is the pytest suite in [`../tests/airgap/`](../tests/airgap),
  which runs the egress attempts *from inside the container* and passes only if
  they all fail.
- The passive proof is the [`falco-egress.yaml`](falco-egress.yaml) CRITICAL
  rule + the [`nftables.conf`](nftables.conf) `EGRESS-DROP` counter.

See [`README.md`](README.md) for the full layered model and the
verification matrix.
