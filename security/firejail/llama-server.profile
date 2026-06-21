# NETRA — firejail profile for the offline LLM (llama.cpp llama-server).
# ======================================================================
# Runs the inference process in its OWN empty network namespace (no
# interfaces at all) plus seccomp + capability drop + a read-only filesystem
# view, so the kernel makes outbound networking impossible at the process
# level (air-gap layer 5; ARCHITECTURE.md §8, research 07 §B6.5).
#
# Usage (host, no Docker):
#   firejail --profile=security/firejail/llama-server.profile \
#            /opt/llama.cpp/llama-server -m /models/qwen2.5-7b-instruct-q4_k_m.gguf \
#            --host 127.0.0.1 --port 8081
#
# NOTE on serving over HTTP: `net none` removes ALL interfaces, including
# loopback, so a TCP listener has nothing to bind to. Two supported modes:
#   (A) IN-PROCESS inference (llama-cpp-python embedded in netra-api) — use
#       this profile AS-IS: no socket is ever needed.
#   (B) HTTP llama-server reachable by other containers/processes — keep the
#       process on the Docker `internal: true` bridge instead (see
#       networks.md), and use `netns`/`net <bridge>` here so it has ONLY the
#       internal interface and no default route. The commented `net` line
#       below shows the bridge-restricted variant.

# ---- Network isolation (the core control) ----
net none
# Bridge-restricted variant (use INSTEAD of `net none` if serving HTTP to the
# internal mesh): attach only to the airgap bridge, no default route.
#   net airgap_net
#   netfilter
ignore net           # refuse any later attempt to re-add networking
nodbus

# ---- Capabilities & privilege ----
caps.drop all
nonewprivs
noroot
seccomp
# Explicitly block the socket-related syscalls even if a future change adds an
# interface (defense-in-depth on top of `net none`).
seccomp.drop socket,socketpair,connect,bind,listen,accept,accept4,sendto,sendmsg,recvfrom,recvmsg,getsockname,getpeername,getsockopt,setsockopt

# ---- Filesystem hardening ----
private-tmp
private-dev
read-only /opt
read-only /usr
# Whitelist ONLY the model directory (read-only) and a writable scratch dir.
# Adjust paths to your deployment.
read-only /models
mkdir ${HOME}/.llama-cache
whitelist ${HOME}/.llama-cache
disable-mnt
machine-id

# ---- Misc kernel-surface reduction ----
no3d
notv
nodvd
nogroups
novideo
nosound
shell none
