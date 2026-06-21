# `scripts/` — Offline packaging, install & verification (Workstream 7)

Hermetic, **hash-verified** packaging so the whole product installs and runs with
**no internet**, and so build/install time also has zero outbound dependency.
This is the offline-delivery third of the 20% "Security & Offline Compliance"
score (the enforcement third lives in [`../security/`](../security), the
active-proof third in [`../tests/airgap/`](../tests/airgap)).

| Script | Role |
|---|---|
| [`bundle.sh`](bundle.sh) | **Build host (connected), run once.** `docker save \| gzip` every image (+ per-image `.sha256`), vendor the Python wheel closure (`pip download`), generate an SBOM (CycloneDX if present, else the pure-python inventory), emit a **license report with copyleft flagged**, stage the air-gap controls + installer into the bundle, and write a single **`MANIFEST.sha256`** over everything. Degrades gracefully when a tool (docker/pip/cyclonedx) is absent. |
| [`install.sh`](install.sh) | **Air-gapped host.** Verify `MANIFEST.sha256` (**abort on any mismatch**) → `docker load` each image (per-file checksum) → `pip install --no-index --find-links wheels -r requirements.txt` (**no index, ever**) → run the air-gap conformance test as a **first-boot gate** (strict by default; abort if the host is not air-gapped). Makes no outbound request. |
| [`airgap_verify.sh`](airgap_verify.sh) | **Verify on demand.** Runs the active pytest conformance suite **and** shows the passive egress evidence (nftables `EGRESS-DROP` counters, `conntrack`/`ss` external-flow check, Falco rule armed, permissive-bundle license check). The `scripts/` entrypoint judges run. |
| [`license_inventory.py`](license_inventory.py) | **Offline license classifier.** Scans installed dists (stdlib `importlib.metadata`) and/or declared deps from a `requirements*.txt`, classifies each license, and **FLAGS copyleft (GPL/AGPL/LGPL/MPL…)**. Confirms the permissive CPU bundle is clean and flags `scikit-survival` (GPL-3.0) in the full tier. `--fail-on-copyleft` for CI gating; `--json` for the manifest. Pure-python, no deps. |

## Quickstart

```bash
# --- on the connected build host ---
NETRA_REQ=requirements-core.txt scripts/bundle.sh      # CPU permissive bundle
#   -> dist/netra-bundle/ : images/  wheels/  sbom/  security/  tests/  MANIFEST.sha256
#   (set NETRA_IMAGES="netra/api:pinned nats:2-alpine ..." to save images)

# --- transfer dist/netra-bundle/ to the air-gapped host, then ---
scripts/install.sh dist/netra-bundle                   # verified, offline, first-boot egress gate

# --- prove zero egress any time (judge demo) ---
NETRA_AIRGAP_STRICT=1 scripts/airgap_verify.sh         # appliance: enforce
scripts/airgap_verify.sh                               # dev box: conformance xfails on reach

# --- license / copyleft audit ---
scripts/license_inventory.py -r requirements-core.txt  # permissive bundle: CLEAN
scripts/license_inventory.py -r requirements.txt --fail-on-copyleft   # flags scikit-survival (GPL)
```

## Determinism / integrity notes

- `docker save` is piped through `gzip -n` (no mtime) and each image gets its own
  `.sha256`; pin images by **digest** in compose for full reproducibility.
- Wheels are vendored as a closure and installed with `--no-index` so pip never
  touches a network; add `--require-hashes` to a fully-pinned requirements file
  for tamper-evidence at install time.
- `MANIFEST.sha256` covers the **whole** bundle; `install.sh` refuses to proceed
  on any mismatch (verified: corrupting one staged file aborts the install).

See [`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS7 and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §8–§9. `cosign --offline` signing /
Rekor-proof verification is the documented production upgrade (research 07 §B8);
this bundle ships SHA-256 manifests as the baseline integrity gate.
