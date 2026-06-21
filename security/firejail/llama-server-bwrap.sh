#!/usr/bin/env bash
#
# NETRA — bubblewrap launcher for the offline LLM (air-gap layer 5, bwrap).
# ========================================================================
# bubblewrap is the lower-level, minimal-dependency alternative to firejail
# (research 07 §B6.5). `--unshare-net` puts the process in a BRAND-NEW network
# namespace with NO interfaces (not even loopback), so no socket can reach
# anything. Combined with the NETRA seccomp profile, the kernel refuses every
# network syscall. Use this for IN-PROCESS / no-HTTP inference; for an HTTP
# llama-server reachable by the mesh, run it on the Docker `internal: true`
# bridge instead (see security/networks.md).
#
# Usage:
#   security/firejail/llama-server-bwrap.sh \
#     /opt/llama.cpp/llama-server -m /models/model.gguf
#
# Requires: bwrap (bubblewrap). seccomp is applied via --seccomp <fd> using the
# compiled BPF; here we rely on the container/runtime seccomp (seccomp-llm.json)
# and the empty netns as the primary controls. bwrap's own seccomp needs a
# pre-compiled BPF program, which we keep out of scope (the netns alone makes
# networking impossible).
set -euo pipefail

MODELS_DIR="${NETRA_MODELS_DIR:-/models}"
CACHE_DIR="${NETRA_LLM_CACHE:-${HOME}/.llama-cache}"

[ "$#" -ge 1 ] || { echo "usage: $0 <llama-server-binary> [args...]" >&2; exit 2; }
command -v bwrap >/dev/null 2>&1 || { echo "ERROR: bwrap (bubblewrap) not found" >&2; exit 1; }

mkdir -p "$CACHE_DIR"

# --unshare-net      : new empty network namespace (no interfaces) -> NO egress.
# --unshare-all      : also isolate ipc/pid/uts/cgroup/user where permitted.
# --share-net is NOT passed, so networking stays unshared/empty.
# --die-with-parent  : kill the sandbox if the launcher dies (no orphan egress).
# Read-only system dirs; writable only the cache + a private /tmp.
exec bwrap \
  --unshare-all \
  --unshare-net \
  --die-with-parent \
  --new-session \
  --clearenv \
  --setenv HF_HUB_OFFLINE 1 \
  --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv DO_NOT_TRACK 1 \
  --setenv HOME "${HOME}" \
  --setenv PATH "/usr/local/bin:/usr/bin:/bin" \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind-try /lib64 /lib64 \
  --ro-bind-try /opt /opt \
  --ro-bind "${MODELS_DIR}" "${MODELS_DIR}" \
  --bind "${CACHE_DIR}" "${CACHE_DIR}" \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --tmpfs /run \
  "$@"
