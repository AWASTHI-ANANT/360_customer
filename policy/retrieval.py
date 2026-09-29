"""Semantic memory: load the synthetic policy markdown, chunk it, retrieve by state.

The corpus in policy/*.md is SYNTHETIC -- authored for this project, because the
dataset ships no policy documents. Each file says so in its header. This is a
known limitation: the retrieved chunks and the policy_chunk_ids cited in HITL
records refer to documents we wrote ourselves.

ASSUMPTIONS (spec was silent):
- Chunk ids are "<file_stem>#<section_slug>#<n>", n starting at 1 within the
  section. A section's own heading is chunk 0 and is retrievable too.
- retrieve(state, k) ranks: exact section match on the state name first (in
  document order), then keyword overlap between the state name and the chunk
  text. Ties break on chunk id, so retrieval is deterministic.
- A subtype id is the text before the first ":" in a catalogue bullet.
- No embeddings and no vector store: keyword matching only, per the project's
  file-based, in-process constraint.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from tracing import traced

log = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).resolve().parent

# "- subtype_id: description (eligibility: ..., max_value: N)"
_BULLET_RE = re.compile(r"^-\s+([a-z0-9_]+)\s*:\s*(.*)$", re.IGNORECASE)
_MAX_VALUE_RE = re.compile(r"max_value\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


@dataclass(slots=True)
class PolicyChunk:
    chunk_id: str
    file: str
    section: str
    text: str
    subtype_id: Optional[str] = None
    max_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section": self.section,
            "text": self.text,
            "subtype_id": self.subtype_id,
            "max_value": self.max_value,
        }


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", heading.strip().lower()).strip("_")


def _parse_file(path: Path) -> list[PolicyChunk]:
    """Chunk one markdown file by '##' section, then by bullet.

    A bullet may wrap over several indented lines; those continuations are folded
    into the bullet they belong to, so a trailing "max_value: 500" stays attached
    to its subtype instead of becoming an orphan chunk.
    """
    chunks: list[PolicyChunk] = []
    stem = path.stem
    section = "preamble"
    n = 0
    pending: list[str] = []          # the bullet (or prose block) being built
    pending_is_bullet = False

    def flush() -> None:
        nonlocal pending, pending_is_bullet, n
        if not pending:
            return
        text = " ".join(pending).strip()
        pending, was_bullet = [], pending_is_bullet
        pending_is_bullet = False
        if not text:
            return
        n += 1
        subtype = None
        max_value = None
        if was_bullet:
            m = _BULLET_RE.match(f"- {text}")
            subtype = m.group(1) if m else None
            mv = _MAX_VALUE_RE.search(text)
            max_value = float(mv.group(1)) if mv else None
        chunks.append(
            PolicyChunk(f"{stem}#{section}#{n}", stem, section, text, subtype, max_value)
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        indented = raw_line[:1].isspace()

        if not line or line.startswith("<!--") or line.startswith("-->"):
            flush()
            continue
        if line.startswith("## "):
            flush()
            section = _slug(line[3:])
            n = 0
            chunks.append(PolicyChunk(f"{stem}#{section}#0", stem, section, line[3:].strip()))
            continue
        if line.startswith("# "):
            flush()
            continue  # document title
        if line.startswith("- "):
            flush()
            pending = [line[2:].strip()]
            pending_is_bullet = True
            continue
        if indented and pending:
            pending.append(line)  # continuation of the current bullet
            continue
        # Prose inside a section: its own chunk, so it can still be cited.
        flush()
        pending = [line]
        pending_is_bullet = False
    flush()
    return chunks


_CACHE: Optional[list[PolicyChunk]] = None


def load(policy_dir: str | Path | None = None, refresh: bool = False) -> list[PolicyChunk]:
    """All chunks from every policy markdown file, cached after first load."""
    global _CACHE
    if _CACHE is not None and not refresh and policy_dir is None:
        return _CACHE
    directory = Path(policy_dir) if policy_dir else POLICY_DIR
    chunks: list[PolicyChunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(_parse_file(path))
    if not chunks:
        log.warning("no policy chunks found in %s", directory)
    if policy_dir is None:
        _CACHE = chunks
    return chunks


@traced("retrieval")
def retrieve(state: str, k: int = 6, policy_dir: str | Path | None = None) -> list[PolicyChunk]:
    """Policy chunks relevant to one inferred_state, best first.

    Exact section matches on the state name come first in document order, then
    keyword-overlap matches. Deterministic: ties break on chunk_id.
    """
    chunks = load(policy_dir)
    target = _slug(state or "")
    exact = [c for c in chunks if c.section == target]

    tokens = {t for t in target.split("_") if len(t) > 3}
    scored: list[tuple[int, str, PolicyChunk]] = []
    for c in chunks:
        if c.section == target:
            continue
        blob = f"{c.section} {c.text}".lower()
        overlap = sum(1 for t in tokens if t in blob)
        if overlap:
            scored.append((-overlap, c.chunk_id, c))
    scored.sort(key=lambda row: (row[0], row[1]))

    out = exact + [row[2] for row in scored]
    if not out:
        log.info("no policy chunks matched state %r", state)
    return out[:k]


def subtypes_for(state: str, policy_dir: str | Path | None = None) -> list[str]:
    """Allowed action_subtype ids for a state, from the catalogue section."""
    target = _slug(state or "")
    return [
        c.subtype_id
        for c in load(policy_dir)
        if c.file == "offer_catalog" and c.section == target and c.subtype_id
    ]


def chunk_ids(chunks: Iterable[PolicyChunk]) -> list[str]:
    return [c.chunk_id for c in chunks]


def cap_for_subtype(subtype: str, policy_dir: str | Path | None = None) -> Optional[float]:
    """The catalogue's advisory max_value for a subtype, if it states one."""
    for c in load(policy_dir):
        if c.subtype_id == subtype and c.max_value is not None:
            return c.max_value
    return None
