# NETRA sim/ — lifecycle wrapper (Workstream 1)
# =============================================
# Thin make targets around the Containerlab lab + the labeled fault drivers, so
# the whole Phase-1 lab is `make up` / `make scenario-A` / `make down`. Mirrors
# the layered IaC model in research/01 §5.1. All targets are offline; images must
# be pre-staged first (see `make images-save` / README "Air-gap").
#
#   make up            # deploy the Containerlab topology (needs sudo + clab)
#   make down          # destroy the lab
#   make scenario-A    # run scenario A fault driver (live) against the lab
#   make scenarios     # run all four, appending labels to LABELS
#   make dry-run       # print every fault command + write labels, no lab needed
#   make render        # (re)render FRR configs from templates
#   make images-save   # bundle all lab images to a tarball (staging host)
#   make images-load   # load the tarball on the air-gapped host

CLAB        ?= sudo containerlab
TOPO        ?= topology.clab.yml
PY          ?= python3
LABELS      ?= labels/run.jsonl
SEED        ?= 1337
IMG_TARBALL ?= clab-images-airgap.tar.gz

# Images this lab needs (keep in sync with topology.clab.yml + faults/).
IMAGES = frrouting/frr:v10.3.0 ghcr.io/nokia/srlinux:24.10.1 \
         ghcr.io/alexei-led/pumba:latest

.PHONY: up down inspect graph render dry-run scenarios scenario-A scenario-B \
        scenario-C scenario-D images-save images-load mpls-modules clean

up: mpls-modules
	$(CLAB) deploy -t $(TOPO)

down:
	$(CLAB) destroy -t $(TOPO) --cleanup

inspect:
	$(CLAB) inspect -t $(TOPO)

graph:
	$(CLAB) graph -t $(TOPO)

# Host kernel MPLS modules are a prerequisite for real SR-MPLS/LDP forwarding.
mpls-modules:
	sudo modprobe mpls_router mpls_gso mpls_iptunnel || \
	  echo "WARN: could not load MPLS modules (needed for the MPLS dataplane)"

render:
	$(PY) configs/render_configs.py --write

dry-run:
	$(PY) faults/run_all.py --labels $(LABELS) --seed $(SEED)

scenarios:
	$(PY) faults/run_all.py --labels $(LABELS) --seed $(SEED) --run

scenario-A:
	$(PY) faults/a_congestion.py --labels $(LABELS) --seed $(SEED) --run
scenario-B:
	$(PY) faults/b_bgp_flap.py --labels $(LABELS) --seed $(SEED) --run
scenario-C:
	$(PY) faults/c_tunnel.py --labels $(LABELS) --seed $(SEED) --run
scenario-D:
	$(PY) faults/d_drift.py --labels $(LABELS) --seed $(SEED) --run

# --- air-gap image staging (research/01 §5.3) ---
images-save:
	docker pull $(IMAGES); \
	docker save $(IMAGES) | gzip > $(IMG_TARBALL); \
	echo "saved -> $(IMG_TARBALL)  (sha256:)"; sha256sum $(IMG_TARBALL)

images-load:
	gunzip -c $(IMG_TARBALL) | docker load

clean:
	rm -f $(LABELS) exabgp.flap.conf drift.napalm.cfg
