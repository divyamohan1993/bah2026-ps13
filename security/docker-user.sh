#!/usr/bin/env bash
#
# NETRA — DOCKER-USER egress lockdown (air-gap layer 2, iptables data path).
# ==========================================================================
# Docker consults the `DOCKER-USER` chain BEFORE its own DOCKER/forward/NAT
# rules, so DOCKER-USER is the authoritative place to govern *container*
# egress when the host uses the iptables backend. This is the iptables
# counterpart to the native-nftables `forward` hook in `nftables.conf`
# (ARCHITECTURE.md §8, research 07 §B6.2). Run BOTH layers for belt-and-
# suspenders; this one specifically survives Docker re-writing its chains
# because DOCKER-USER is the chain Docker is contractually required to honour
# first and never to flush.
#
# Policy: allow intra-lab (Docker bridge + optional LAN telemetry) via RETURN,
# then LOG every other egress attempt with prefix "EGRESS-DROP ", then DROP.
# Idempotent: existing NETRA rules are removed before re-inserting.
#
# Usage:
#   sudo security/docker-user.sh            # apply rules
#   sudo security/docker-user.sh --remove   # remove NETRA rules only
#   sudo security/docker-user.sh --list     # show the DOCKER-USER chain
#   INTRA_LAB_CIDRS="172.16.0.0/12 10.0.0.0/8" sudo -E security/docker-user.sh
#
# Requires: iptables, CAP_NET_ADMIN (root). On nftables-only hosts use
# nftables.conf instead (iptables-nft shim also works if present).
set -euo pipefail

# ---------------------------------------------------------------------------
# Tunables. Override via the INTRA_LAB_CIDRS env var (space-separated).
#   172.16.0.0/12 covers the default Docker bridge pools (172.17/16 + the
#   172.18+/16 user-defined-bridge pools incl. airgap_net).
#   10.0.0.0/8 / 192.168.0.0/16 are OPTIONAL on-prem LAN telemetry ranges.
# ---------------------------------------------------------------------------
INTRA_LAB_CIDRS="${INTRA_LAB_CIDRS:-172.16.0.0/12 10.0.0.0/8 192.168.0.0/16}"
LOG_PREFIX="EGRESS-DROP "
CHAIN="DOCKER-USER"

# Marker comment so we can find/remove ONLY our rules (never touch Docker's).
MARK="netra-airgap"

log()  { printf '[docker-user] %s\n' "$*" >&2; }
die()  { printf '[docker-user] ERROR: %s\n' "$*" >&2; exit 1; }

command -v iptables >/dev/null 2>&1 || die "iptables not found in PATH"

ensure_chain() {
	# DOCKER-USER is created by the Docker daemon. If Docker has never run,
	# create the chain ourselves so the rules have somewhere to live (Docker
	# will adopt it). It must be referenced from FORWARD to take effect; we
	# add that jump if missing.
	if ! iptables -L "$CHAIN" -n >/dev/null 2>&1; then
		log "$CHAIN chain absent (is Docker installed/started?); creating it."
		iptables -N "$CHAIN" 2>/dev/null || true
	fi
	if ! iptables -C FORWARD -j "$CHAIN" 2>/dev/null; then
		log "Adding FORWARD -> $CHAIN jump."
		iptables -I FORWARD -j "$CHAIN"
	fi
}

remove_rules() {
	# Delete every rule in DOCKER-USER that carries our marker comment.
	# Loop because rule numbers shift as we delete.
	local removed=0
	while true; do
		# Find the first matching rule number (by our comment marker).
		local line
		line="$(iptables -L "$CHAIN" --line-numbers -n 2>/dev/null \
			| awk -v m="$MARK" '$0 ~ m {print $1; exit}')" || true
		[ -n "${line:-}" ] || break
		iptables -D "$CHAIN" "$line"
		removed=$((removed + 1))
	done
	log "Removed $removed existing NETRA rule(s) from $CHAIN."
}

apply_rules() {
	ensure_chain
	remove_rules   # idempotency: clear our old rules first

	# Build from the BOTTOM up using -I (insert at top) so final order is:
	#   1..N  RETURN intra-lab            (evaluated first)
	#   N+1   LOG    EGRESS-DROP
	#   N+2   DROP
	# Insert DROP first, then LOG above it, then the RETURNs above that.
	iptables -I "$CHAIN" -j DROP -m comment --comment "$MARK: default-deny container egress"

	iptables -I "$CHAIN" -m limit --limit 10/sec --limit-burst 20 \
		-j LOG --log-prefix "$LOG_PREFIX" --log-level warning \
		-m comment --comment "$MARK: log+count blocked egress"

	# Allow established/related replies (top-most so replies are cheap).
	# (Inserted last => ends up above the LOG/DROP and above the RETURNs is
	# fine; conntrack accept ordering among allows is immaterial.)
	local cidr
	for cidr in $INTRA_LAB_CIDRS; do
		iptables -I "$CHAIN" -d "$cidr" -j RETURN \
			-m comment --comment "$MARK: allow intra-lab $cidr"
	done
	iptables -I "$CHAIN" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN \
		-m comment --comment "$MARK: allow established/related"

	log "Applied egress lockdown to $CHAIN (allow: $INTRA_LAB_CIDRS; else LOG+DROP)."
	log "Verify with: sudo iptables -L $CHAIN -n -v --line-numbers"
}

list_chain() {
	iptables -L "$CHAIN" -n -v --line-numbers 2>/dev/null \
		|| die "$CHAIN chain not present"
}

main() {
	case "${1:-apply}" in
		apply|"")   apply_rules ;;
		--remove|remove) ensure_chain; remove_rules ;;
		--list|list)     list_chain ;;
		-h|--help)
			grep -E '^#( |$)' "$0" | sed -E 's/^# ?//'
			;;
		*) die "unknown argument: $1 (use apply | --remove | --list | --help)" ;;
	esac
}

main "$@"
