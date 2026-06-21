"""Ingest + chunk + index the ``corpus/`` artifacts for grounded retrieval.

Turns the internal NOC artifacts under ``corpus/`` into citable
:class:`~netra.copilot.rag.store.Chunk` objects with NOC metadata, then builds a
:class:`~netra.copilot.rag.retrieve.HybridRetriever`. Per-artifact, structure-
aware chunking (research 06 §5) — markdown by header section, JSON by record/
field — so each chunk is a semantic unit and its ``chunk_id`` is stable enough
to cite (``CopilotResponse.citations``).

No heavy deps: pure stdlib parsing (``json``, light YAML front-matter parsing,
markdown header splitting). The corpus is internal-only; nothing here ever
reaches the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .embed import Embedder
from .rerank import Reranker
from .retrieve import HybridRetriever
from .store import Chunk

#: Default corpus location: ``<repo>/corpus`` (three levels up from this file).
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[3] / "corpus"

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Extract a minimal YAML-ish front-matter block; return (meta, body).

    Supports the simple ``key: value`` and ``key: [a, b]`` forms used in the
    runbooks (avoids a PyYAML dependency for the light tier).
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
            meta[key] = items
        elif val:
            meta[key] = val
    return meta, text[m.end():]


def _chunk_markdown(path: Path) -> list[Chunk]:
    """Split a markdown runbook into header-section chunks with shared metadata."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    base_id = str(meta.get("runbook_id") or path.stem)
    common_meta = {
        "artifact_type": "runbook",
        "document_id": base_id,  # doc-level id (playbook runbook_ref points here)
        "source_path": str(path),
        "issue_type": meta.get("issue_type"),
        "scenario_id": meta.get("scenario_id"),
        "title": meta.get("title"),
        "sites": meta.get("sites"),
        "devices": meta.get("devices"),
    }

    # Split on headers, keeping each section (header + following text) together.
    headers = list(_HEADER_RE.finditer(body))
    chunks: list[Chunk] = []
    if not headers:
        chunks.append(Chunk(chunk_id=f"{base_id}#0", text=body.strip(), metadata=dict(common_meta)))
        return chunks

    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        section = body[start:end].strip()
        if not section:
            continue
        section_title = h.group(2).strip()
        cmeta = dict(common_meta)
        cmeta["section"] = section_title
        # Prefix the document identity into the text (Anthropic Contextual
        # Retrieval, computed deterministically here) to sharpen retrieval.
        prefix = f"[{common_meta.get('title') or base_id} — {section_title}] "
        chunks.append(
            Chunk(chunk_id=f"{base_id}#{i}", text=prefix + section, metadata=cmeta)
        )
    return chunks


def _chunk_incident(path: Path) -> list[Chunk]:
    """One past-incident record -> one chunk (symptom/RCA/resolution serialised)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    iid = str(data.get("incident_id") or path.stem)
    parts = [
        f"Past incident {iid}: {data.get('title', '')}.",
        f"Issue type: {data.get('issue_type', '')}.",
        f"Site/device: {data.get('site', '')}/{data.get('device', '')}.",
        f"Symptom: {data.get('symptom', '')}",
        f"Root cause: {data.get('root_cause', '')}",
        f"Resolution: {data.get('resolution', '')}",
    ]
    signals = data.get("contributing_signals") or []
    if signals:
        parts.append("Contributing signals: " + "; ".join(signals) + ".")
    text = "\n".join(p for p in parts if p.strip())
    meta = {
        "artifact_type": "incident",
        "document_id": iid,
        "source_path": str(path),
        "issue_type": data.get("issue_type"),
        "scenario_id": data.get("scenario_id"),
        "site": data.get("site"),
        "device": data.get("device"),
        "playbook_used": data.get("playbook_used"),
        "title": data.get("title"),
    }
    return [Chunk(chunk_id=iid, text=text, metadata=meta)]


def _chunk_playbook(path: Path) -> list[Chunk]:
    """One CACAO-style playbook -> one chunk (id + title + ordered steps)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pid = str(data.get("playbook_id") or path.stem)
    steps = data.get("actions") or []
    step_lines = [
        f"  Step {s.get('step')}: {s.get('description')} "
        f"[{s.get('command_or_guidance', '')}] (urgency={s.get('urgency')}, "
        f"approval={'yes' if s.get('requires_approval') else 'no'})"
        for s in steps
    ]
    text = "\n".join(
        [
            f"Playbook {pid}: {data.get('title', '')}.",
            f"Issue type: {data.get('issue_type', '')}.",
            f"Trigger: {data.get('trigger_signature', '')}",
            "Actions:",
            *step_lines,
        ]
    )
    meta = {
        "artifact_type": "playbook",
        "document_id": pid,
        "source_path": str(path),
        "issue_type": data.get("issue_type"),
        "title": data.get("title"),
        "source_ref": data.get("source_ref"),
    }
    return [Chunk(chunk_id=pid, text=text, metadata=meta)]


def _chunk_topology(path: Path) -> list[Chunk]:
    """Topology JSON -> one overview chunk + one chunk per device (for citation)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tid = str(data.get("topology_id") or path.stem)
    chunks: list[Chunk] = []

    overview = [
        f"Topology {tid}: {data.get('title', '')}.",
        data.get("description", ""),
        "Sites: "
        + ", ".join(
            f"{s.get('site')}({s.get('site_type')})" for s in data.get("sites", [])
        )
        + ".",
        "VPNs: "
        + ", ".join(
            f"{v.get('name')} RD/RT {v.get('rd')}/{v.get('rt')}"
            for v in data.get("vpns", [])
        )
        + ".",
    ]
    chunks.append(
        Chunk(
            chunk_id=f"{tid}#overview",
            text="\n".join(p for p in overview if p),
            metadata={"artifact_type": "topology", "source_path": str(path), "title": data.get("title")},
        )
    )

    for dev in data.get("devices", []):
        dtext = (
            f"Device {dev.get('device')} (role {dev.get('role')}) at site "
            f"{dev.get('site')}, entity_id {dev.get('entity_id')}"
            + (f", VRFs {', '.join(dev.get('vrfs', []))}" if dev.get("vrfs") else "")
            + "."
        )
        chunks.append(
            Chunk(
                chunk_id=f"{tid}#dev-{dev.get('device')}",
                text=dtext,
                metadata={
                    "artifact_type": "topology",
                    "source_path": str(path),
                    "site": dev.get("site"),
                    "device": dev.get("device"),
                    "role": dev.get("role"),
                    "entity_id": dev.get("entity_id"),
                },
            )
        )
    return chunks


def load_corpus_chunks(corpus_dir: str | Path | None = None) -> list[Chunk]:
    """Parse every artifact under ``corpus_dir`` into citable chunks.

    Recognised layouts: ``runbooks/*.md``, ``incidents/*.json``,
    ``playbooks/*.json``, ``topology/*.json``. Unknown files are ignored. Returns
    an empty list if the directory does not exist (the copilot then abstains).
    """
    root = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
    if not root.exists():
        return []

    chunks: list[Chunk] = []
    for path in sorted((root / "runbooks").glob("*.md")) if (root / "runbooks").exists() else []:
        chunks.extend(_chunk_markdown(path))
    for path in sorted((root / "incidents").glob("*.json")) if (root / "incidents").exists() else []:
        chunks.extend(_chunk_incident(path))
    for path in sorted((root / "playbooks").glob("*.json")) if (root / "playbooks").exists() else []:
        chunks.extend(_chunk_playbook(path))
    for path in sorted((root / "topology").glob("*.json")) if (root / "topology").exists() else []:
        chunks.extend(_chunk_topology(path))
    return chunks


def document_ids(chunks: Iterable[Chunk]) -> list[str]:
    """Return the unique doc-level ids of ``chunks`` (e.g. ``RB-CONGESTION-001``).

    A playbook step's ``runbook_ref`` cites the *document* (``RB-CONGESTION-001``),
    while retrieval returns *chunk* ids (``RB-CONGESTION-001#2``). Including these
    doc-level ids in the citation universe lets the closed-set citation check
    recognise a legitimately-grounded ``runbook_ref`` without weakening the check
    (the document is genuinely in the retrieved context).
    """
    out: list[str] = []
    for c in chunks:
        doc = c.metadata.get("document_id")
        if doc:
            out.append(str(doc))
        # Also expose the chunk_id's base before any '#chunk' suffix.
        base = c.chunk_id.split("#", 1)[0]
        if base and base != c.chunk_id:
            out.append(base)
    return list(dict.fromkeys(out))


def build_retriever(
    corpus_dir: str | Path | None = None,
    *,
    chunks: Iterable[Chunk] | None = None,
    prefer_model: bool = False,
) -> HybridRetriever:
    """Build a :class:`HybridRetriever` indexed over the corpus.

    Parameters
    ----------
    corpus_dir:
        Where to read artifacts from (defaults to the repo ``corpus/``).
    chunks:
        Pre-built chunks to index instead of reading the corpus (used by tests).
    prefer_model:
        Pass through to the embedder/reranker to try the heavy bge models;
        defaults to False (CPU/offline light path).
    """
    all_chunks = list(chunks) if chunks is not None else load_corpus_chunks(corpus_dir)
    retriever = HybridRetriever(
        embedder=Embedder(prefer_model=prefer_model),
        reranker=Reranker(prefer_model=prefer_model),
    )
    retriever.index(all_chunks)
    return retriever


__all__ = [
    "load_corpus_chunks",
    "build_retriever",
    "document_ids",
    "DEFAULT_CORPUS_DIR",
]
