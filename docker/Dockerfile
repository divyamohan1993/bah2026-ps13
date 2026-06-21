# NETRA — netra-app image (FastAPI API + offline operator UI).
# =============================================================
# A slim Python image that runs the CORE tier only (requirements-core.txt): the
# end-to-end demo path — synthetic TelemetrySource -> streaming -> ensemble ->
# fusion/correlation/risk -> template-fallback copilot -> FastAPI/UI — needs no
# GPU, no LLM, and no network at runtime. Referenced by docker-compose.yml as the
# `netra-app` service (build.dockerfile = docker/Dockerfile).
#
# OFFLINE-FRIENDLY BUILD:
#   * The only build-time network access is pip resolving requirements-core.txt
#     from PyPI on a CONNECTED build host. For a fully air-gapped build, vendor
#     the wheel closure first (scripts/bundle.sh produces dist/netra-bundle/wheels)
#     and build with it, e.g.:
#
#       scripts/bundle.sh                       # on a connected host -> wheels/
#       docker build -f docker/Dockerfile \
#         --build-arg PIP_FIND_LINKS=/wheels --build-arg PIP_NO_INDEX=1 \
#         -t netra/app:latest .
#
#     (mount/copy the wheelhouse into the build context as /wheels). On the
#     air-gapped host the image itself is delivered via `docker save | docker load`
#     (scripts/bundle.sh / scripts/install.sh) — no registry pull.
#   * At RUNTIME the container makes zero outbound calls and runs on the
#     `internal: true` airgap_net bridge (docker-compose.yml).

FROM python:3.11-slim

# --- offline-by-design env (belt-and-suspenders vs phone-home) ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTHONPATH=/app

# Optional offline-build knobs: pass --build-arg PIP_NO_INDEX=1 and
# --build-arg PIP_FIND_LINKS=/wheels to install from a vendored wheelhouse with
# no index access. Defaults install from PyPI on a connected build host.
ARG PIP_NO_INDEX=""
ARG PIP_FIND_LINKS=""

WORKDIR /app

# Install the CORE dependency tier first so this layer caches across code edits.
# (statsmodels/scipy/numpy/etc. ship CPU wheels for slim, so no compiler toolchain
# is required for the core tier.)
COPY requirements-core.txt ./requirements-core.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        ${PIP_NO_INDEX:+--no-index} \
        ${PIP_FIND_LINKS:+--find-links=${PIP_FIND_LINKS}} \
        -r requirements-core.txt

# Copy the repo (the package + the UI assets the API serves from ../ui).
# .dockerignore keeps the context slim (no .git, venvs, caches, bundles).
COPY . .

# Non-root runtime user (no shell login needed); read-only rootfs is applied by
# compose, with a tmpfs for /tmp.
RUN useradd --create-home --uid 10001 netra \
    && chown -R netra:netra /app
USER netra

EXPOSE 8000

# Liveness probe target for orchestrators that honour HEALTHCHECK (compose also
# defines one). Uses stdlib only — no curl/wget dependency in the image.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3).status==200 else sys.exit(1)" || exit 1

# Bind 0.0.0.0 INSIDE the container; docker-compose publishes the port to
# 127.0.0.1 only, so nothing is exposed on the host's external interface.
CMD ["uvicorn", "netra.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
