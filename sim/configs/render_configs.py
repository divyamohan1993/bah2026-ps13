"""Render per-device FRR configs from node specs (Workstream 1).

A tiny, deterministic config renderer so the PE/CE/P/RR configs are reproducible
topology-as-code rather than hand-maintained drift. It mirrors what
``netlab create`` does (render device configs from one model) but stays
dependency-light and offline.

The canonical, fully-commented reference configs are checked in under
``sim/configs/frr/<node>/frr.conf`` (e.g. ``pe-dc2``, ``rr-dc``, ``p1``,
``ce-br1``). This script demonstrates how the remaining same-shaped nodes are
generated from the shared Jinja templates in ``sim/configs/templates/`` so the
whole set is one ``python render_configs.py`` away and never diverges.

Jinja2 is imported lazily; if it is unavailable the script prints the node specs
as JSON (so the model is still inspectable) and exits cleanly. Run::

    python sim/configs/render_configs.py            # render all PEs to stdout
    python sim/configs/render_configs.py --write    # write configs/frr/<n>/frr.conf

This file is part of the sim/ subtree (Workstream 1) and is NOT on the runtime
import path of the Python product; it is an offline lab tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Node specs for the PE routers (the route-reflector, P and CE shapes have their
# own templates/handwritten refs). Keep router-ids/SIDs aligned with
# topology.clab.yml and netra/datagen/topology.py.
PE_NODES: list[dict] = [
    {
        "name": "pe-dc2",
        "router_id": "10.0.0.6",
        "isis_net": "49.0001.0000.0000.0006.00",
        "sid_index": 6,
        "rr_loopback": "10.0.0.8",
        "core_iface": {"name": "eth1", "desc": "uplink-to-p2-core", "ip": "10.1.102.1/30"},
        "vrfs": [
            {
                "name": "OT",
                "rd": "100:2",
                "rt": "100:2",
                "ce_iface": {"name": "eth2", "desc": "pe-ce-br3-OT", "ip": "10.2.30.1/30"},
            }
        ],
    },
]

TEMPLATE_DIR = Path(__file__).parent / "templates"
OUT_DIR = Path(__file__).parent / "frr"


def _render(node: dict) -> str | None:
    try:
        from jinja2 import Environment, FileSystemLoader  # noqa: PLC0415
    except Exception:
        return None
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("frr_pe.conf.j2")
    return tmpl.render(node=node)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render FRR PE configs from specs.")
    ap.add_argument("--write", action="store_true", help="Write to configs/frr/<node>/.")
    args = ap.parse_args(argv)

    any_rendered = False
    for node in PE_NODES:
        text = _render(node)
        if text is None:
            print(
                "[render_configs] jinja2 not installed; emitting node specs as JSON "
                "(install jinja2 to render, or use the checked-in reference configs)",
                file=sys.stderr,
            )
            print(json.dumps(PE_NODES, indent=2))
            return 0
        any_rendered = True
        if args.write:
            dest = OUT_DIR / node["name"] / "frr.conf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            print(f"[render_configs] wrote {dest}")
        else:
            print(f"# ===== {node['name']} =====")
            print(text)
    return 0 if any_rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
