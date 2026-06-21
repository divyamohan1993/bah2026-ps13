#!/usr/bin/env bash
#
# NETRA — offline installer (deterministic, hash-verified, NO network).
# =====================================================================
# Installs the bundle produced by `bundle.sh` on an AIR-GAPPED host. It verifies
# every checksum BEFORE touching anything, then loads images and installs wheels
# entirely offline, and finally runs the air-gap conformance test as a first-
# boot gate (ARCHITECTURE.md §8/§9, research 07 §B9). It ABORTS on any checksum
# mismatch (and, in strict mode, any conformance failure).
#
# Crucially this script itself makes NO outbound request: `pip install
# --no-index --find-links` never contacts an index, and `docker load` never
# contacts a registry.
#
# Usage:
#   scripts/install.sh [BUNDLE_DIR]            # default: dir containing this script's bundle
#   NETRA_VENV=/opt/netra/venv scripts/install.sh /srv/netra-bundle
#   NETRA_SKIP_CONFORMANCE=1 scripts/install.sh   # skip the first-boot egress test
#   NETRA_AIRGAP_STRICT=1 scripts/install.sh      # treat a reachable egress as fatal
#
# Steps:
#   1. verify MANIFEST.sha256 over the whole bundle (integrity gate; ABORT on fail)
#   2. docker load every images/*.tar.gz  (after per-file sha256 check)
#   3. pip install --no-index --find-links wheels -r requirements.txt
#   4. run tests/airgap conformance (first-boot proof)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Bundle dir = arg 1, or the parent of this script (when run from inside a bundle).
BUNDLE_DIR="${1:-$(cd "${HERE}/.." && pwd)}"
VENV_DIR="${NETRA_VENV:-${BUNDLE_DIR}/.venv}"
PY="${PYTHON:-python3}"

log()  { printf '[install] %s\n' "$*" >&2; }
warn() { printf '[install] WARN: %s\n' "$*" >&2; }
skip() { printf '[install] [skip] %s\n' "$*" >&2; }
die()  { printf '[install] FATAL: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

[ -d "${BUNDLE_DIR}" ] || die "bundle dir not found: ${BUNDLE_DIR}"
log "bundle dir: ${BUNDLE_DIR}"

# --------------------------------------------------------------------------- #
# 1. Integrity gate — verify the whole bundle against MANIFEST.sha256.
# --------------------------------------------------------------------------- #
verify_manifest() {
	local manifest="${BUNDLE_DIR}/MANIFEST.sha256"
	[ -f "${manifest}" ] || die "MANIFEST.sha256 missing — refusing to install an unverifiable bundle."
	log "verifying bundle integrity against MANIFEST.sha256 ..."
	(
		cd "${BUNDLE_DIR}"
		# --quiet prints only failures; non-zero exit on any mismatch.
		if sha256sum --quiet -c MANIFEST.sha256; then
			:
		else
			die "checksum verification FAILED — bundle is corrupt or tampered. ABORTING."
		fi
	)
	log "integrity OK: all files match MANIFEST.sha256."
}

# --------------------------------------------------------------------------- #
# 2. Load docker images offline (each verified by its own .sha256 too).
# --------------------------------------------------------------------------- #
load_images() {
	local img_dir="${BUNDLE_DIR}/images"
	if [ ! -d "${img_dir}" ] || ! ls "${img_dir}"/*.tar.gz >/dev/null 2>&1; then
		skip "no images/*.tar.gz in bundle — nothing to load."
		return
	fi
	if ! have docker; then
		warn "docker not found — cannot load images. Install docker, then re-run."
		return
	fi
	local tar base
	for tar in "${img_dir}"/*.tar.gz; do
		base="$(basename "${tar}")"
		if [ -f "${img_dir}/${base}.sha256" ]; then
			( cd "${img_dir}" && sha256sum -c "${base}.sha256" >/dev/null ) \
				|| die "image checksum mismatch for ${base} — ABORTING."
		fi
		log "docker load < ${base}"
		gunzip -c "${tar}" | docker load >&2 || die "docker load failed for ${base}"
	done
	log "all images loaded (offline, no registry contacted)."
}

# --------------------------------------------------------------------------- #
# 3. Install wheels offline (no index, ever).
# --------------------------------------------------------------------------- #
install_wheels() {
	local wheels="${BUNDLE_DIR}/wheels"
	local req="${BUNDLE_DIR}/requirements.txt"
	if [ ! -d "${wheels}" ] || [ ! -f "${req}" ]; then
		skip "no wheels/ + requirements.txt in bundle — skipping python install."
		return
	fi
	have "${PY}" || die "python3 not found — cannot install wheels."
	log "creating venv at ${VENV_DIR} ..."
	"${PY}" -m venv "${VENV_DIR}" || die "venv creation failed."
	# Offline pip: --no-index guarantees NO network; --find-links uses our wheels.
	log "pip install --no-index --find-links wheels -r requirements.txt ..."
	"${VENV_DIR}/bin/python" -m pip install \
		--no-index \
		--find-links "${wheels}" \
		-r "${req}" >&2 \
		|| die "offline pip install failed (missing wheel in closure?). ABORTING."
	log "python deps installed offline into ${VENV_DIR}."
}

# --------------------------------------------------------------------------- #
# 4. First-boot air-gap conformance gate.
# --------------------------------------------------------------------------- #
run_conformance() {
	if [ "${NETRA_SKIP_CONFORMANCE:-0}" = "1" ]; then
		skip "NETRA_SKIP_CONFORMANCE=1 — not running the first-boot egress test."
		return
	fi
	local tests_dir="${BUNDLE_DIR}/tests/airgap"
	[ -d "${tests_dir}" ] || { skip "tests/airgap not in bundle — skipping conformance."; return; }

	# Prefer the freshly-built venv's pytest; else any pytest on PATH.
	local pybin="${PY}"
	[ -x "${VENV_DIR}/bin/python" ] && pybin="${VENV_DIR}/bin/python"

	if ! "${pybin}" -c "import pytest" >/dev/null 2>&1; then
		skip "pytest not available — cannot run conformance automatically."
		warn "Run manually on the appliance: NETRA_AIRGAP_STRICT=1 pytest -q ${tests_dir}"
		return
	fi

	log "running air-gap conformance (first-boot proof) ..."
	# On the appliance you WANT strict; default to strict here so install gates
	# on true zero-egress unless the operator opts out.
	local strict="${NETRA_AIRGAP_STRICT:-1}"
	if NETRA_AIRGAP_STRICT="${strict}" "${pybin}" -m pytest -q "${tests_dir}" >&2; then
		log "CONFORMANCE PASS: verifiable zero egress confirmed."
	else
		if [ "${strict}" = "1" ]; then
			die "CONFORMANCE FAILED in strict mode — host is NOT air-gapped. ABORTING install."
		else
			warn "conformance reported reachable egress (non-strict) — review before going live."
		fi
	fi
}

main() {
	verify_manifest
	load_images
	install_wheels
	run_conformance
	log "INSTALL COMPLETE."
	log "Next: cp security/.env.example .env && \\"
	log "      docker compose -f docker-compose.yml -f security/compose.security.yml up -d"
	log "      (apply host firewall first: sudo nft -f security/nftables.conf; sudo security/docker-user.sh)"
}

main "$@"
