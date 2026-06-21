#!/usr/bin/env bash
#
# NETRA — offline bundle builder (deterministic, hash-verified).
# ==============================================================
# Run ONCE on a connected build host to produce a self-contained, air-gap-
# installable bundle (ARCHITECTURE.md §8 supply chain, research 07 §B7.3/§B9):
#   * `docker save | gzip` every image          -> images/*.tar.gz (+ .sha256)
#   * vendor the Python wheel closure            -> wheels/ (pip download)
#   * generate an SBOM                           -> sbom/ (cyclonedx | pip-licenses
#                                                   | pure-python inventory)
#   * a license report (copyleft flagged)        -> sbom/license-report.{txt,json}
#   * a single checksums manifest                -> MANIFEST.sha256
#
# The companion `install.sh` consumes this bundle on the air-gapped host with no
# network, verifying every checksum before loading anything.
#
# Usage:
#   scripts/bundle.sh                              # uses defaults below
#   NETRA_REQ=requirements-core.txt scripts/bundle.sh   # permissive CPU bundle
#   NETRA_IMAGES="netra/api:pinned nats:2-alpine" scripts/bundle.sh
#   NETRA_OUT=/srv/netra-bundle scripts/bundle.sh
#
# Every step degrades gracefully: a missing tool (docker / pip / cyclonedx)
# prints a clear "[skip]"/"[warn]" and the bundle still builds with whatever is
# available, so this is runnable in CI and on a dev box.
set -euo pipefail

# Resolve repo root from this script's location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

# ---- configuration (override via env) ----
OUT_DIR="${NETRA_OUT:-${REPO_ROOT}/dist/netra-bundle}"
# Which requirements file's closure to vendor. Default = the CPU permissive
# tier (the only one REQUIRED to run); set NETRA_REQ=requirements.txt for full.
REQ_FILE="${NETRA_REQ:-${REPO_ROOT}/requirements-core.txt}"
# Images to save. Default tries to read them from a top-level compose file if
# present; otherwise the caller supplies NETRA_IMAGES.
IMAGES="${NETRA_IMAGES:-}"
PY="${PYTHON:-python3}"

log()  { printf '[bundle] %s\n' "$*" >&2; }
warn() { printf '[bundle] WARN: %s\n' "$*" >&2; }
skip() { printf '[bundle] [skip] %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

mkdir -p "${OUT_DIR}/images" "${OUT_DIR}/wheels" "${OUT_DIR}/sbom"
log "output dir: ${OUT_DIR}"
log "requirements: ${REQ_FILE}"

# --------------------------------------------------------------------------- #
# 1. Save + compress docker images, each with its own sha256.
# --------------------------------------------------------------------------- #
discover_images() {
	# If NETRA_IMAGES is unset, try to extract `image:` refs from a compose file.
	[ -n "${IMAGES}" ] && { echo "${IMAGES}"; return; }
	local compose="${REPO_ROOT}/docker-compose.yml"
	if [ -f "${compose}" ] && have "${PY}"; then
		"${PY}" - "${compose}" <<'PY' 2>/dev/null || true
import sys, yaml
try:
    d = yaml.safe_load(open(sys.argv[1])) or {}
except Exception:
    sys.exit(0)
imgs = []
for name, svc in (d.get("services") or {}).items():
    img = (svc or {}).get("image")
    if img:
        imgs.append(img)
print(" ".join(imgs))
PY
	fi
}

save_images() {
	if ! have docker; then
		skip "docker not found — no images saved. Set NETRA_IMAGES + run on a host with docker."
		return
	fi
	local imgs; imgs="$(discover_images)"
	if [ -z "${imgs}" ]; then
		skip "no images to save (set NETRA_IMAGES or add a docker-compose.yml with image: refs)."
		return
	fi
	local img tar base
	for img in ${imgs}; do
		if ! docker image inspect "${img}" >/dev/null 2>&1; then
			warn "image not present locally, skipping: ${img} (pull/build it first)"
			continue
		fi
		base="$(printf '%s' "${img}" | tr '/:' '__')"
		tar="${OUT_DIR}/images/${base}.tar.gz"
		log "saving ${img} -> ${tar}"
		docker save "${img}" | gzip -n > "${tar}"   # -n = deterministic gzip (no mtime)
		( cd "${OUT_DIR}/images" && sha256sum "${base}.tar.gz" > "${base}.tar.gz.sha256" )
	done
	# Record the exact image list for install.sh.
	printf '%s\n' ${imgs} > "${OUT_DIR}/images/IMAGES.list"
}

# --------------------------------------------------------------------------- #
# 2. Vendor the wheel closure for offline `pip install --no-index`.
# --------------------------------------------------------------------------- #
vendor_wheels() {
	if [ ! -f "${REQ_FILE}" ]; then
		warn "requirements file not found: ${REQ_FILE} — skipping wheels."
		return
	fi
	if ! have "${PY}" || ! "${PY}" -m pip --version >/dev/null 2>&1; then
		skip "pip not available — no wheels vendored. Run on a connected build host."
		return
	fi
	log "downloading wheel closure for ${REQ_FILE} -> ${OUT_DIR}/wheels"
	# --only-binary=:all: prefers wheels; falls back leaves sdists which install
	# may need to build. We do NOT pin --platform here so the bundle matches the
	# build host's platform (document target platform in the bundle README).
	if ! "${PY}" -m pip download \
			-r "${REQ_FILE}" \
			-d "${OUT_DIR}/wheels" 2>"${OUT_DIR}/wheels/.pip-download.log"; then
		warn "pip download reported errors (see wheels/.pip-download.log); partial closure kept."
	fi
	( cd "${OUT_DIR}/wheels" && find . -maxdepth 1 -type f \
		\( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) -print \
		| sort > .wheels.list ) || true
	# Copy the requirements file alongside so install.sh installs the same set.
	cp "${REQ_FILE}" "${OUT_DIR}/requirements.txt"
}

# --------------------------------------------------------------------------- #
# 3. SBOM + license report (prefer cyclonedx; else pip-licenses; else our own
#    pure-python inventory — always produces SOMETHING).
# --------------------------------------------------------------------------- #
gen_sbom() {
	local sbom_dir="${OUT_DIR}/sbom"
	# 3a. CycloneDX SBOM if the tool is installed.
	if "${PY}" -m cyclonedx_py --help >/dev/null 2>&1; then
		log "generating CycloneDX SBOM (cyclonedx-py) ..."
		"${PY}" -m cyclonedx_py requirements "${REQ_FILE}" \
			-o "${sbom_dir}/sbom.cyclonedx.json" 2>/dev/null \
			|| "${PY}" -m cyclonedx_py environment \
				-o "${sbom_dir}/sbom.cyclonedx.json" 2>/dev/null \
			|| warn "cyclonedx-py invocation failed; relying on the pure-python inventory."
	elif have cyclonedx-bom; then
		log "generating CycloneDX SBOM (cyclonedx-bom) ..."
		cyclonedx-bom -r -i "${REQ_FILE}" -o "${sbom_dir}/sbom.cyclonedx.json" 2>/dev/null \
			|| warn "cyclonedx-bom failed; relying on the pure-python inventory."
	else
		skip "cyclonedx not installed; using the pure-python license inventory as the SBOM."
	fi

	# 3b. License report via OUR offline inventory (always runs; flags copyleft).
	log "generating license report (copyleft flagged) ..."
	"${PY}" "${HERE}/license_inventory.py" -r "${REQ_FILE}" --no-installed \
		> "${sbom_dir}/license-report.txt" 2>/dev/null || \
		warn "license-report.txt generation failed."
	"${PY}" "${HERE}/license_inventory.py" -r "${REQ_FILE}" --no-installed \
		--json "${sbom_dir}/license-report.json" >/dev/null 2>&1 || \
		warn "license-report.json generation failed."

	# 3c. pip-licenses (optional, richer per-package detail) if present.
	if have pip-licenses; then
		log "augmenting with pip-licenses inventory ..."
		pip-licenses --format=json --with-urls --with-license-file \
			> "${sbom_dir}/pip-licenses.json" 2>/dev/null || true
	fi
}

# --------------------------------------------------------------------------- #
# 4. Single checksums manifest over the WHOLE bundle (the integrity gate).
# --------------------------------------------------------------------------- #
write_manifest() {
	log "writing MANIFEST.sha256 over the bundle ..."
	(
		cd "${OUT_DIR}"
		# Hash every regular file except the manifest itself and pip logs.
		find . -type f \
			! -name 'MANIFEST.sha256' \
			! -name '.pip-download.log' \
			-print0 \
		| sort -z \
		| xargs -0 sha256sum > MANIFEST.sha256
	)
	log "manifest entries: $(wc -l < "${OUT_DIR}/MANIFEST.sha256")"
}

# --------------------------------------------------------------------------- #
# 5. Copy the runtime air-gap controls into the bundle so the appliance has
#    everything (firewall rules, falco rule, compose fragment, conformance test,
#    installer, verifier).
# --------------------------------------------------------------------------- #
stage_runtime_controls() {
	log "staging air-gap controls + installer into the bundle ..."
	mkdir -p "${OUT_DIR}/security" "${OUT_DIR}/tests/airgap" "${OUT_DIR}/scripts"
	cp -r "${REPO_ROOT}/security/." "${OUT_DIR}/security/" 2>/dev/null || true
	cp -r "${REPO_ROOT}/tests/airgap/." "${OUT_DIR}/tests/airgap/" 2>/dev/null || true
	cp "${HERE}/install.sh"        "${OUT_DIR}/scripts/" 2>/dev/null || true
	cp "${HERE}/airgap_verify.sh"  "${OUT_DIR}/scripts/" 2>/dev/null || true
	cp "${HERE}/license_inventory.py" "${OUT_DIR}/scripts/" 2>/dev/null || true
}

main() {
	save_images
	vendor_wheels
	stage_runtime_controls
	gen_sbom
	write_manifest
	log "DONE. Bundle ready at: ${OUT_DIR}"
	log "Transfer it to the air-gapped host, then run: scripts/install.sh ${OUT_DIR}"
}

main "$@"
