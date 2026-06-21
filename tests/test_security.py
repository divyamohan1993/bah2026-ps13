"""Unit tests for the WS7 security/packaging deliverables.

Covers two things, both runnable with LIGHT deps (pytest + stdlib only — no
docker, no nft, no network):

  1. ``scripts/license_inventory.py``'s license **classifier** — the load-
     bearing logic for the air-gap supply-chain check: it must label
     ``scikit-survival`` as GPL/copyleft (FLAGGED), and ``lifelines`` / ``numpy``
     as permissive (clean). Plus the category rules and the curated-map
     resolution and the ``--fail-on-copyleft`` gate behaviour.

  2. A **smoke test** that every WS7 shell script is syntactically valid
     (``bash -n``) so a broken script can never ship in the bundle.

These tests intentionally avoid importing any heavy dependency so they pass on
the CPU-only / no-internet tier.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Locate the repo + dynamically import scripts/license_inventory.py without
# requiring `scripts` to be a package (it is a flat scripts dir).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SECURITY_DIR = REPO_ROOT / "security"
LICENSE_INV_PATH = SCRIPTS_DIR / "license_inventory.py"


def _load_license_inventory():
    """Import license_inventory.py as a module from its file path.

    The module is registered in ``sys.modules`` BEFORE ``exec_module`` because
    ``@dataclass`` with ``field(init=False)`` resolves ``cls.__module__`` via
    ``sys.modules`` during class creation; without registration that lookup
    returns ``None`` and dataclass construction raises.
    """
    import sys

    name = "netra_license_inventory"
    spec = importlib.util.spec_from_file_location(name, LICENSE_INV_PATH)
    assert spec and spec.loader, f"cannot load {LICENSE_INV_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


li = _load_license_inventory()


# --------------------------------------------------------------------------- #
# 1. Classifier — the explicitly-required cases.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "license_str, expected_category",
    [
        # The three the task names explicitly.
        ("GPL-3.0-or-later", li.CAT_COPYLEFT),  # scikit-survival's license
        ("GPL-3.0", li.CAT_COPYLEFT),
        ("MIT", li.CAT_PERMISSIVE),  # lifelines' license
        ("BSD-3-Clause", li.CAT_PERMISSIVE),  # numpy's license
        # Strong copyleft family.
        ("AGPL-3.0", li.CAT_COPYLEFT),
        ("GNU Affero General Public License v3", li.CAT_COPYLEFT),
        ("GNU General Public License v2 (GPLv2)", li.CAT_COPYLEFT),
        # Weak / library copyleft family.
        ("LGPL-2.1-only", li.CAT_WEAK_COPYLEFT),
        ("GNU Lesser General Public License v3 (LGPLv3)", li.CAT_WEAK_COPYLEFT),
        ("MPL-2.0", li.CAT_WEAK_COPYLEFT),
        ("Mozilla Public License 2.0 (MPL 2.0)", li.CAT_WEAK_COPYLEFT),
        ("EPL-2.0", li.CAT_WEAK_COPYLEFT),
        # Permissive family.
        ("Apache-2.0", li.CAT_PERMISSIVE),
        ("Apache Software License", li.CAT_PERMISSIVE),
        ("ISC", li.CAT_PERMISSIVE),
        ("BSD-2-Clause", li.CAT_PERMISSIVE),
        ("Python Software Foundation License", li.CAT_PERMISSIVE),
        # Public domain / ultra-permissive.
        ("The Unlicense (Unlicense)", li.CAT_PUBLIC_DOMAIN),
        ("CC0-1.0", li.CAT_PUBLIC_DOMAIN),
        ("0BSD", li.CAT_PUBLIC_DOMAIN),
        # Unknown / unparseable.
        ("", li.CAT_UNKNOWN),
        (None, li.CAT_UNKNOWN),
        ("UNKNOWN", li.CAT_UNKNOWN),
        ("Proprietary", li.CAT_UNKNOWN),
    ],
)
def test_classify_license(license_str, expected_category):
    assert li.classify_license(license_str) == expected_category


def test_compound_expression_gpl_dominates():
    """In a compound expression, any GPL/AGPL term must dominate (conservative
    for a redistributable air-gapped appliance)."""
    assert li.classify_license("GPL-2.0 OR MIT") == li.CAT_COPYLEFT
    assert li.classify_license("MIT OR Apache-2.0") == li.CAT_PERMISSIVE
    assert li.classify_license("(MIT OR GPL-3.0-only)") == li.CAT_COPYLEFT


def test_flagged_categories_membership():
    """Copyleft, weak-copyleft and unknown are flagged; permissive/public are not."""
    assert li.CAT_COPYLEFT in li.FLAGGED_CATEGORIES
    assert li.CAT_WEAK_COPYLEFT in li.FLAGGED_CATEGORIES
    assert li.CAT_UNKNOWN in li.FLAGGED_CATEGORIES
    assert li.CAT_PERMISSIVE not in li.FLAGGED_CATEGORIES
    assert li.CAT_PUBLIC_DOMAIN not in li.FLAGGED_CATEGORIES


# --------------------------------------------------------------------------- #
# 1b. Resolution via the curated SPDX map (the named packages).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pkg, expected_category, expected_flagged",
    [
        ("scikit-survival", li.CAT_COPYLEFT, True),  # GPL-3.0 -> FLAGGED
        ("lifelines", li.CAT_PERMISSIVE, False),  # MIT -> clean
        ("numpy", li.CAT_PERMISSIVE, False),  # BSD -> clean
        ("pydantic", li.CAT_PERMISSIVE, False),
        ("networkx", li.CAT_PERMISSIVE, False),
    ],
)
def test_resolve_license_curated(pkg, expected_category, expected_flagged):
    """resolve_license (curated map, no installed metadata) classifies the
    named NETRA deps as documented."""
    dep = li.resolve_license(pkg, prefer_installed=False)
    assert dep.category == expected_category, f"{pkg}: {dep.license} -> {dep.category}"
    assert dep.flagged is expected_flagged


def test_scikit_survival_is_flagged_copyleft():
    """The headline requirement, asserted directly: scikit-survival is copyleft
    and flagged, while lifelines (the permissive substitute) is not."""
    surv = li.resolve_license("scikit-survival", prefer_installed=False)
    life = li.resolve_license("lifelines", prefer_installed=False)
    assert "GPL" in (surv.license or "")
    assert surv.category == li.CAT_COPYLEFT and surv.flagged is True
    assert life.category == li.CAT_PERMISSIVE and life.flagged is False


def test_name_normalization():
    """PEP 503 normalisation so 'scikit_survival'/'Scikit-Survival' all resolve."""
    assert li.normalize_name("Scikit_Survival") == "scikit-survival"
    assert li.normalize_name("scikit.survival") == "scikit-survival"
    assert li.normalize_name("NumPy") == "numpy"


# --------------------------------------------------------------------------- #
# 1c. Requirements parsing + inventory + the fail-on-copyleft gate.
# --------------------------------------------------------------------------- #
def test_parse_requirements(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(
        "# a comment\n"
        "\n"
        "scikit-survival>=0.22,<0.24  # inline comment\n"
        "lifelines>=0.27,<0.31\n"
        "uvicorn[standard]>=0.29,<1\n"
        "-r other.txt\n"
        "--find-links ./wheels\n"
        "git+https://example.com/pkg.git#egg=pkg\n"
    )
    names = li.parse_requirements(str(req))
    assert "scikit-survival" in names
    assert "lifelines" in names
    assert "uvicorn" in names  # extras stripped
    assert "other.txt" not in names  # -r include skipped
    assert not any("http" in n for n in names)  # URL line skipped


def test_build_inventory_and_summary():
    inv = li.build_inventory(["scikit-survival", "lifelines", "numpy"], prefer_installed=False)
    by_name = {d.name: d for d in inv}
    assert by_name["scikit-survival"].flagged is True
    assert by_name["lifelines"].flagged is False
    counts = li.summarize(inv)
    assert counts.get(li.CAT_COPYLEFT, 0) >= 1
    assert counts.get(li.CAT_PERMISSIVE, 0) >= 2
    # Flagged items sort first.
    assert inv[0].flagged is True


def test_main_fail_on_copyleft(tmp_path, capsys):
    """The CLI gate: --fail-on-copyleft exits 2 when a copyleft dep is present,
    and 0 when the set is clean."""
    full = tmp_path / "full.txt"
    full.write_text("scikit-survival>=0.22\nlifelines>=0.27\nnumpy>=1.26\n")
    rc = li.main(["-r", str(full), "--no-installed", "--fail-on-copyleft"])
    assert rc == 2

    clean = tmp_path / "clean.txt"
    clean.write_text("lifelines>=0.27\nnumpy>=1.26\npydantic>=2.6\n")
    rc = li.main(["-r", str(clean), "--no-installed", "--fail-on-copyleft"])
    assert rc == 0


def test_render_json_is_valid(tmp_path):
    import json

    out = tmp_path / "report.json"
    li.main(["-r", str(_write(tmp_path, "r.txt", "scikit-survival>=0.22\nnumpy>=1.26\n")),
             "--no-installed", "--json", str(out)])
    data = json.loads(out.read_text())
    assert "summary" in data and "dependencies" in data and "flagged" in data
    flagged_names = {d["name"] for d in data["flagged"]}
    assert "scikit-survival" in flagged_names


def _write(base: Path, name: str, content: str) -> str:
    p = base / name
    p.write_text(content)
    return str(p)


# --------------------------------------------------------------------------- #
# 2. Shell-script syntax smoke test (bash -n on every WS7 script).
# --------------------------------------------------------------------------- #
WS7_SHELL_SCRIPTS = [
    SECURITY_DIR / "docker-user.sh",
    SECURITY_DIR / "firejail" / "llama-server-bwrap.sh",
    SCRIPTS_DIR / "bundle.sh",
    SCRIPTS_DIR / "install.sh",
    SCRIPTS_DIR / "airgap_verify.sh",
    REPO_ROOT / "tests" / "airgap" / "demo_airgap.sh",
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("script", WS7_SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_syntax_valid(script: Path):
    """Every WS7 shell script must pass `bash -n` (parse without executing)."""
    assert script.exists(), f"missing script: {script}"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n failed for {script.name}:\n{result.stderr}"
    )


def test_license_inventory_script_compiles():
    """license_inventory.py must be importable/compilable (it ships in bundles)."""
    import py_compile

    py_compile.compile(str(LICENSE_INV_PATH), doraise=True)


def test_ws7_files_present():
    """Guard against an incomplete WS7 deliverable: the key files must exist."""
    expected = [
        SECURITY_DIR / "nftables.conf",
        SECURITY_DIR / "docker-user.sh",
        SECURITY_DIR / "falco-egress.yaml",
        SECURITY_DIR / "seccomp-llm.json",
        SECURITY_DIR / "networks.md",
        SECURITY_DIR / "compose.security.yml",
        SECURITY_DIR / "README.md",
        REPO_ROOT / "tests" / "airgap" / "test_airgap_conformance.py",
        REPO_ROOT / "tests" / "airgap" / "conftest.py",
        REPO_ROOT / "tests" / "airgap" / "demo_airgap.sh",
        SCRIPTS_DIR / "bundle.sh",
        SCRIPTS_DIR / "install.sh",
        SCRIPTS_DIR / "airgap_verify.sh",
        SCRIPTS_DIR / "license_inventory.py",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in expected if not p.exists()]
    assert not missing, f"missing WS7 deliverables: {missing}"
