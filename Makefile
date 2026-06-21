# NETRA — developer convenience targets (PS-13 air-gapped predictive NOC copilot).
# ================================================================================
# All targets are offline-friendly and use the REAL scripts/CLIs in this repo.
# The CPU-only demo/eval path needs only the CORE tier (requirements-core.txt) —
# no GPU, no LLM, no internet. Run `make help` for the menu.

# Use a project-local venv by default; override PY/VENV to use your own.
VENV    ?= .venv
PY      ?= python3
VENV_PY := $(VENV)/bin/python
PYTEST  ?= $(VENV_PY) -m pytest
# Compose: the core stack; layer the security fragment for the hardened appliance.
COMPOSE      ?= docker compose
COMPOSE_FILE ?= docker-compose.yml
SECURITY_FILE ?= security/compose.security.yml

.DEFAULT_GOAL := help

.PHONY: help setup demo demo-all test test-airgap up up-secure up-llm up-sim \
        down logs ps airgap-verify bundle install-offline license lint fmt \
        compose-config clean

help: ## Show this help.
	@echo "NETRA — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
setup: ## Create a venv and install the CORE tier (the only tier the demo needs).
	$(PY) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-core.txt
	@echo "core tier installed into $(VENV). For tests also: $(VENV_PY) -m pip install pytest pytest-asyncio httpx"

# --------------------------------------------------------------------------- #
# Demo + tests (CPU-only, offline)
# --------------------------------------------------------------------------- #
demo: ## Run the end-to-end 4-scenario demo (fast profile, offline, CPU-only).
	PYTHONPATH=. $(VENV_PY) scripts/demo.py --profile fast

demo-all: ## Run the demo with the full (heavy) ensemble profile.
	PYTHONPATH=. $(VENV_PY) scripts/demo.py --profile full

test: ## Run the full pytest suite (air-gap tests run LENIENT by default).
	PYTHONPATH=. $(PYTEST) tests/ -q

test-airgap: ## Run only the air-gap conformance suite.
	PYTHONPATH=. $(PYTEST) tests/airgap -q

# --------------------------------------------------------------------------- #
# Offline stack (docker compose)
# --------------------------------------------------------------------------- #
up: ## Bring up the core offline stack (NATS, VictoriaMetrics, Grafana, netra-app).
	$(COMPOSE) -f $(COMPOSE_FILE) up -d --build

up-secure: ## Bring up the stack hardened (+ Falco egress monitor + LLM seccomp).
	$(COMPOSE) -f $(COMPOSE_FILE) -f $(SECURITY_FILE) up -d --build

up-sim: ## Bring up the core stack + the live-sim collectors (gnmic/telegraf).
	$(COMPOSE) -f $(COMPOSE_FILE) --profile sim up -d --build

up-llm: ## Bring up the core stack + the local offline llama-server.
	$(COMPOSE) -f $(COMPOSE_FILE) --profile llm up -d --build

down: ## Stop and remove the stack (keeps named volumes).
	$(COMPOSE) -f $(COMPOSE_FILE) down

logs: ## Tail logs from the stack.
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=100

ps: ## Show stack service status.
	$(COMPOSE) -f $(COMPOSE_FILE) ps

compose-config: ## Validate the compose file (renders the merged config).
	$(COMPOSE) -f $(COMPOSE_FILE) config >/dev/null && echo COMPOSE_OK

# --------------------------------------------------------------------------- #
# Air-gap proof + offline packaging
# --------------------------------------------------------------------------- #
airgap-verify: ## Prove zero egress (active pytest conformance + passive evidence).
	scripts/airgap_verify.sh

bundle: ## Build the offline, hash-verified install bundle (scripts/bundle.sh).
	scripts/bundle.sh

install-offline: ## Install a bundle on an air-gapped host (scripts/install.sh).
	scripts/install.sh

license: ## Audit dependency licenses (permissive core bundle must be CLEAN).
	$(PY) scripts/license_inventory.py -r requirements-core.txt --no-installed --fail-on-copyleft

# --------------------------------------------------------------------------- #
# Lint / format (ruff if available)
# --------------------------------------------------------------------------- #
lint: ## Lint with ruff (no-op if ruff is not installed).
	@command -v ruff >/dev/null 2>&1 && ruff check . \
		|| echo "ruff not installed — skipping lint (pip install ruff)"

fmt: ## Auto-fix lint + format with ruff (no-op if ruff is not installed).
	@command -v ruff >/dev/null 2>&1 && { ruff check --fix . ; ruff format . ; } \
		|| echo "ruff not installed — skipping format (pip install ruff)"

# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
clean: ## Remove caches and build artifacts (keeps the venv).
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
