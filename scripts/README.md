# `scripts/` — Offline bundling & install (Workstream 7)

Hermetic, hash-verified packaging so the whole product installs and runs with
**no internet**, and so build/install time also has zero outbound dependency.

**What goes here:**
- `build_wheelhouse.sh` — resolve the full transitive dependency closure on a
  connected host (`pip download`) for offline `pip install --no-index
  --find-links=wheelhouse --require-hashes`.
- `build_bundle.sh` — `docker save | gzip` all images into one tarball + emit
  `*.sha256`; cosign-sign with bundled Rekor proof (`--tlog-upload=false`).
- `gen_sbom.sh` — CycloneDX/SPDX SBOM (`cyclonedx-py`/`syft`) for images + wheels.
- `install.sh` — verified offline install on the air-gapped host:
  `sha256sum -c` → `cosign verify --offline` → `docker load` → `pip --no-index
  --require-hashes` → `docker compose up -d` → **run the air-gap conformance test
  on first boot**; abort on any checksum/signature/conformance failure.
- `run_demo.sh` — bring up the CPU-only end-to-end demo (synthetic source →
  streaming → ensemble → risk → template/LLM copilot → API/UI).

**Contracts:** packages the built `netra/*` images + the wheelhouse. See
[`../docs/BUILD_PLAN.md`](../docs/BUILD_PLAN.md) WS7 and
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §8–§9.
