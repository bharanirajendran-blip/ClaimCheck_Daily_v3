"""
store.py — Persistent Evidence Store 

Responsibilities:
  1. Receive a ResearchResult from the Claude Researcher
  2. Chunk the findings text into overlapping section windows
  3. Save chunks to a daily JSON file  (outputs/evidence_YYYY-MM-DD.json)
  4. Merge them into a cumulative store (outputs/evidence_store.json)
     so future runs can retrieve evidence from past claims

Why persistent across runs?
  A single daily run yields 3 claims and ~15-30 chunks.
  After several runs, the store holds 100+ chunks spanning multiple
  topics and sources — retrieval becomes genuinely non-trivial and the
  system starts showing "context management" across time, not just
  within a single day (the key concept).

Chunking strategy:
  - Split findings by markdown section headings (## ...)
  - Hard-cap each chunk at 700 chars (keeps context tight)
  - Preserve section label so the Director can cite it precisely
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from .models import EvidenceChunk, ResearchResult

logger = logging.getLogger(__name__)

CUMULATIVE_STORE_FILE = "evidence_store.json"
MAX_CHUNK_CHARS = 700


# ── Public API ─────────────────────────────────────────────────────────────────

def store_research(
    research: ResearchResult,
    claim_text: str,
    outputs_dir: str | Path,
) -> list[EvidenceChunk]:
    """
    Chunk a ResearchResult and persist to both the daily and cumulative store.
    Returns the list of EvidenceChunk objects created.
    """
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)

    date_slug = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    chunks = _chunk_research(research, claim_text, date_slug)

    # Write / append to daily evidence file
    daily_path = outputs_path / f"evidence_{date_slug}.json"
    _upsert_json_list(daily_path, [c.model_dump() for c in chunks])

    # Merge into cumulative store (deduplicated by chunk_id)
    cumulative_path = outputs_path / CUMULATIVE_STORE_FILE
    _upsert_cumulative(cumulative_path, chunks)

    logger.info(
        "[store] Stored %d chunks for claim %s → %s",
        len(chunks), research.claim_id, cumulative_path,
    )
    return chunks


def load_all_chunks(outputs_dir: str | Path) -> list[EvidenceChunk]:
    """Load every chunk from the cumulative store. Returns [] if store is empty."""
    path = Path(outputs_dir) / CUMULATIVE_STORE_FILE
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvidenceChunk.model_validate(r) for r in raw]


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_research(
    research: ResearchResult,
    claim_text: str,
    date_slug: str,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []

    # Determine primary source URL (first source if any, else empty)
    primary_url = research.sources[0].get("url", "") if research.sources else ""

    # Store fetched source-page content first so retrieval can ground itself
    # in raw evidence, not only in the researcher's summary prose.
    for index, page in enumerate(research.fetched_pages, start=1):
        url = page.get("url", "")
        content = (page.get("content", "") or "").strip()
        if not content:
            continue
        for part_index, text in enumerate(_split_long_text(content), start=1):
            chunk_id = _make_id(research.claim_id, f"Fetched Source {index}.{part_index}", text)
            chunks.append(EvidenceChunk(
                chunk_id=chunk_id,
                claim_id=research.claim_id,
                claim_text=claim_text[:200],
                source_url=url,
                section=f"Fetched Source {index}.{part_index}",
                text=text,
                date_slug=date_slug,
                chunk_kind="raw_source",
            ))

    # Split findings into (section_heading, body) pairs
    sections = _split_sections(research.findings)

    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        # Hard-cap and create chunk
        text = body[:MAX_CHUNK_CHARS]
        chunk_id = _make_id(research.claim_id, heading, text)
        chunks.append(EvidenceChunk(
            chunk_id=chunk_id,
            claim_id=research.claim_id,
            claim_text=claim_text[:200],
            source_url=primary_url,
            section=heading,
            text=text,
            date_slug=date_slug,
            chunk_kind="summary",
        ))

    # Also store each cited source as a lightweight chunk (for graph edges)
    for src in research.sources:
        title = src.get("title", "")
        url   = src.get("url", "")
        rel   = src.get("reliability", "")
        if not title:
            continue
        text = f"Source: {title}. URL: {url}. Reliability: {rel}."
        chunk_id = _make_id(research.claim_id, "Key Sources", text)
        chunks.append(EvidenceChunk(
            chunk_id=chunk_id,
            claim_id=research.claim_id,
            claim_text=claim_text[:200],
            source_url=url,
            section="Key Sources",
            text=text,
            date_slug=date_slug,
            chunk_kind="source_metadata",
        ))

    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs at ## headings."""
    pieces: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_lines:
                pieces.append((current_heading, "\n".join(current_lines)))
            current_heading = line.lstrip("#").strip() or "Overview"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        pieces.append((current_heading, "\n".join(current_lines)))

    return [(h, b) for h, b in pieces if b.strip()]


def _split_long_text(text: str) -> list[str]:
    compact = text.strip()
    if not compact:
        return []
    return [
        compact[i:i + MAX_CHUNK_CHARS]
        for i in range(0, len(compact), MAX_CHUNK_CHARS)
    ]


def _make_id(claim_id: str, section: str, text: str) -> str:
    raw = f"{claim_id}::{section}::{text[:80]}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Persistence helpers ────────────────────────────────────────────────────────

# Fields with defaults that may be absent in chunks saved by older code versions.
_CHUNK_DEFAULTS: dict[str, object] = {
    "chunk_kind": "summary",
    "source_url":  "",
    "claim_text":  "",
    "date_slug":   "",
    "section":     "",
}


def _normalize_chunk_dict(item: dict) -> dict:
    """Backfill any fields that were added after the initial schema so every
    chunk in every artifact has a consistent, complete set of fields."""
    for field, default in _CHUNK_DEFAULTS.items():
        if field not in item or item[field] is None:
            item[field] = default
    return item


def _upsert_json_list(path: Path, new_items: list[dict]) -> None:
    """Write new_items to path, deduplicating by chunk_id and normalising
    any legacy chunks so the daily file stays consistent across runs."""
    existing: dict[str, dict] = {}
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            existing[item["chunk_id"]] = _normalize_chunk_dict(item)
    for item in new_items:
        existing[item["chunk_id"]] = _normalize_chunk_dict(item)
    path.write_text(json.dumps(list(existing.values()), indent=2, ensure_ascii=False), encoding="utf-8")


def _upsert_cumulative(path: Path, new_chunks: list[EvidenceChunk]) -> None:
    """Merge new chunks into the cumulative store, deduplicating by chunk_id
    and normalising any legacy chunks on the way through."""
    existing: dict[str, dict] = {}
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            existing[item["chunk_id"]] = _normalize_chunk_dict(item)

    for chunk in new_chunks:
        existing[chunk.chunk_id] = chunk.model_dump()

    path.write_text(
        json.dumps(list(existing.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
