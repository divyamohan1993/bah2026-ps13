# `tests/airgap/` — Air-gap conformance test (Workstream 7)

The runnable proof of zero egress (the "verifiably" in the 20% Security & Offline
Compliance criterion).

**What goes here:**
- `test_airgap_conformance.py` — pytest that inverts the success criterion of an
  egress tester: it tries TCP to well-known public IPs on common ports, external
  DNS resolution, an HTTPS fetch, and UDP/53 exfil, and **passes only if every
  attempt is blocked or times out**. Ships inside every container image and runs
  on first boot.

**Run:** `pytest -q tests/airgap` → all green = verifiable zero egress.

Pairs with the always-on monitors in [`../../security/`](../../security)
(nftables blocked-attempt counter + Falco CRITICAL outbound rule + `conntrack`).
See [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) §8.
