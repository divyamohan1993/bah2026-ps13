"""Pytest configuration for the NETRA air-gap conformance suite.

Registers the ``airgap`` marker and exposes the strict/lenient mode switch the
test module uses to decide whether a *reachable* egress path is a hard failure
(locked appliance) or an expected-failure / skip (connected dev box).

Modes
-----
* **STRICT** (``NETRA_AIRGAP_STRICT=1``): this box claims to be air-gapped, so
  ANY successful egress is a real air-gap breach and **fails** the test. Run
  this on the locked appliance and in the install.sh first-boot gate.
* **LENIENT / dev** (default): on a normal connected dev box egress obviously
  works; the suite stays *runnable* by reporting a reachable path as
  ``xfail``/``skip`` with a clear "NOT air-gapped (dev mode)" message instead
  of a red failure. Flip ``NETRA_AIRGAP_STRICT=1`` to enforce.
"""

from __future__ import annotations

import os


def pytest_configure(config):
    """Register the ``airgap`` marker so ``-m airgap`` works and there is no
    'unknown marker' warning under strict-markers configurations."""
    config.addinivalue_line(
        "markers",
        "airgap: active air-gap egress conformance checks (TCP/UDP/DNS/HTTPS).",
    )


def is_strict() -> bool:
    """True when STRICT enforcement is requested via ``NETRA_AIRGAP_STRICT``.

    Accepts ``1/true/yes/on`` (case-insensitive). Default is lenient/dev mode.
    """
    return os.environ.get("NETRA_AIRGAP_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
