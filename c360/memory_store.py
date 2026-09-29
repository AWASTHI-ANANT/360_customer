"""Two-layer memory: what agents actually query to make decisions.

WorkingMemory  -- short-term live state. Cheap to overwrite. Snapshotted to
                  memory/{customer_id}_working.json on save().
EpisodicMemory -- long-term durable episodes. Append-only JSONL at
                  memory/{customer_id}_episodic.jsonl, with an in-memory index
                  rebuilt on load so query() never rescans the file.

All timestamps are *simulated* time. Pass now_fn=clock.now so memory records
when something was inferred in the customer's timeline, not on the wall clock.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .loader import iso, parse_ts

log = logging.getLogger(__name__)

CONFIDENCE_BANDS = ("low", "medium", "high")

# Finding keys the agents are expected to write. Not enforced -- new keys are
# allowed -- but a typo'd key gets a warning rather than silently vanishing.
KNOWN_FINDING_KEYS = {
    "engagement_trend",
    "sentiment_state",
    "outflow_pattern",
    "income_pattern",
    "spend_shift",
    "balance_trend",
    "risk_signals",
    "life_event_signals",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _check_band(band: str) -> str:
    if band not in CONFIDENCE_BANDS:
        log.warning("confidence_band %r not in %s -- storing as-is", band, CONFIDENCE_BANDS)
    return band


@dataclass(slots=True)
class Finding:
    """A single current assessment of one dimension of the customer."""

    key: str
    value: Any
    confidence_band: str
    source_agent: str
    evidence_event_ids: list[str] = field(default_factory=list)
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence_band": self.confidence_band,
            "source_agent": self.source_agent,
            "evidence_event_ids": list(self.evidence_event_ids),
            "updated_at": iso(self.updated_at) if self.updated_at else None,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Finding":
        return cls(
            key=raw["key"],
            value=raw.get("value"),
            confidence_band=raw.get("confidence_band", "low"),
            source_agent=raw.get("source_agent", "unknown"),
            evidence_event_ids=list(raw.get("evidence_event_ids") or []),
            updated_at=parse_ts(raw["updated_at"]) if raw.get("updated_at") else None,
            notes=raw.get("notes"),
        )


@dataclass(slots=True)
class Hypothesis:
    """The system's current best explanation of what's happening.

    `state` uses the fixed inferred_state enum from the dataset README.
    `first_inferred` is when this state was *first* reached -- that's what
    timeliness scoring cares about, so set_hypothesis preserves it across
    updates to the same state.
    """

    state: str
    score: float = 0.0
    confidence_band: str = "low"
    first_inferred: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    supporting_events: list[str] = field(default_factory=list)
    runner_up: Optional[dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "score": self.score,
            "confidence_band": self.confidence_band,
            "first_inferred": iso(self.first_inferred) if self.first_inferred else None,
            "last_updated": iso(self.last_updated) if self.last_updated else None,
            "supporting_events": list(self.supporting_events),
            "runner_up": self.runner_up,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Hypothesis":
        return cls(
            state=raw["state"],
            score=float(raw.get("score") or 0.0),
            confidence_band=raw.get("confidence_band", "low"),
            first_inferred=parse_ts(raw["first_inferred"]) if raw.get("first_inferred") else None,
            last_updated=parse_ts(raw["last_updated"]) if raw.get("last_updated") else None,
            supporting_events=list(raw.get("supporting_events") or []),
            runner_up=raw.get("runner_up"),
            notes=raw.get("notes"),
        )


@dataclass(slots=True)
class Episode:
    """One meaningful, durable chapter in the customer's story."""

    episode_id: str
    customer_id: str
    opened_at: Optional[datetime]
    type: str
    closed_at: Optional[datetime] = None
    evidence_event_ids: list[str] = field(default_factory=list)
    outcome: Optional[str] = None
    system_action: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "customer_id": self.customer_id,
            "opened_at": iso(self.opened_at) if self.opened_at else None,
            "closed_at": iso(self.closed_at) if self.closed_at else None,
            "type": self.type,
            "evidence_event_ids": list(self.evidence_event_ids),
            "outcome": self.outcome,
            "system_action": self.system_action,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Episode":
        return cls(
            episode_id=raw["episode_id"],
            customer_id=raw.get("customer_id", "unknown"),
            opened_at=parse_ts(raw["opened_at"]) if raw.get("opened_at") else None,
            closed_at=parse_ts(raw["closed_at"]) if raw.get("closed_at") else None,
            type=raw.get("type", "unknown"),
            evidence_event_ids=list(raw.get("evidence_event_ids") or []),
            outcome=raw.get("outcome"),
            system_action=raw.get("system_action"),
            notes=raw.get("notes"),
        )


class WorkingMemory:
    """Short-term current state. Overwriting is the normal operation."""

    def __init__(self, customer_id: str, now_fn: Callable[[], datetime] = _now_utc) -> None:
        self.customer_id = customer_id
        self.now_fn = now_fn
        self._findings: dict[str, Finding] = {}
        self._hypothesis: Optional[Hypothesis] = None

    # --- findings -----------------------------------------------------------

    def set_finding(
        self,
        key: str,
        value: Any,
        confidence: str,
        agent: str,
        evidence: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> Finding:
        """Replace the current finding for `key`."""
        if key not in KNOWN_FINDING_KEYS:
            log.debug("new finding key %r (not in KNOWN_FINDING_KEYS)", key)
        finding = Finding(
            key=key,
            value=value,
            confidence_band=_check_band(confidence),
            source_agent=agent,
            evidence_event_ids=list(evidence or []),
            updated_at=self.now_fn(),
            notes=notes,
        )
        self._findings[key] = finding
        return finding

    def get_finding(self, key: str) -> Optional[Finding]:
        return self._findings.get(key)

    def all_findings(self) -> dict[str, Finding]:
        return dict(self._findings)

    def clear_finding(self, key: str) -> None:
        self._findings.pop(key, None)

    # --- hypothesis ---------------------------------------------------------

    def set_hypothesis(
        self,
        state: str,
        score: float = 0.0,
        confidence_band: str = "low",
        supporting_events: Optional[Sequence[str]] = None,
        runner_up: Optional[dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Hypothesis:
        """Set the top hypothesis.

        If `state` is unchanged, first_inferred is preserved (we want the moment
        the system first landed on this state, not the latest refresh).
        """
        now = self.now_fn()
        prior = self._hypothesis
        first_inferred = (
            prior.first_inferred if prior is not None and prior.state == state else now
        )
        self._hypothesis = Hypothesis(
            state=state,
            score=float(score),
            confidence_band=_check_band(confidence_band),
            first_inferred=first_inferred,
            last_updated=now,
            supporting_events=list(supporting_events or []),
            runner_up=runner_up,
            notes=notes,
        )
        return self._hypothesis

    def get_hypothesis(self) -> Optional[Hypothesis]:
        return self._hypothesis

    # --- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "saved_at": iso(self.now_fn()),
            "findings": {k: f.to_dict() for k, f in self._findings.items()},
            "hypothesis": self._hypothesis.to_dict() if self._hypothesis else None,
        }

    def load_dict(self, raw: dict[str, Any]) -> None:
        self._findings = {
            k: Finding.from_dict(v) for k, v in (raw.get("findings") or {}).items()
        }
        hyp = raw.get("hypothesis")
        self._hypothesis = Hypothesis.from_dict(hyp) if hyp else None

    def __repr__(self) -> str:
        state = self._hypothesis.state if self._hypothesis else None
        return f"<WorkingMemory {self.customer_id} findings={len(self._findings)} hypothesis={state}>"


class EpisodicMemory:
    """Durable episode log: append-only file + in-memory index.

    update_outcome() does not rewrite history -- it appends an amendment record
    that is replayed onto the index at load time. The file stays append-only.
    """

    RECORD_EPISODE = "episode"
    RECORD_OUTCOME = "outcome_update"

    def __init__(
        self,
        customer_id: str,
        path: str | Path,
        now_fn: Callable[[], datetime] = _now_utc,
    ) -> None:
        self.customer_id = customer_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.now_fn = now_fn
        self._episodes: list[Episode] = []          # insertion order
        self._by_id: dict[str, Episode] = {}
        self._by_type: dict[str, list[Episode]] = {}
        self._counter = 0

    # --- writes -------------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        candidate = f"EP_{self.customer_id}_{self._counter:04d}"
        while candidate in self._by_id:  # survives a resumed run with gaps
            self._counter += 1
            candidate = f"EP_{self.customer_id}_{self._counter:04d}"
        return candidate

    def _append_line(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    def _index(self, ep: Episode) -> None:
        self._episodes.append(ep)
        self._by_id[ep.episode_id] = ep
        self._by_type.setdefault(ep.type, []).append(ep)

    def append(self, episode: Episode | dict[str, Any]) -> Episode:
        """Add a new episode, generating episode_id (and opened_at) if absent."""
        if isinstance(episode, dict):
            raw = dict(episode)
            raw.setdefault("episode_id", "")
            raw.setdefault("customer_id", self.customer_id)
            ep = Episode.from_dict(raw)
        else:
            ep = episode
        if not ep.episode_id:
            ep.episode_id = self._next_id()
        if ep.episode_id in self._by_id:
            log.warning("episode_id %s already exists -- reassigning", ep.episode_id)
            ep.episode_id = self._next_id()
        if ep.opened_at is None:
            ep.opened_at = self.now_fn()
        if not ep.customer_id or ep.customer_id == "unknown":
            ep.customer_id = self.customer_id

        self._index(ep)
        self._append_line({"record_type": self.RECORD_EPISODE, **ep.to_dict()})
        return ep

    def open_episode(
        self,
        type: str,
        evidence_event_ids: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
        system_action: Optional[str] = None,
    ) -> Episode:
        """Shorthand for append(Episode(...)) with the common fields."""
        return self.append(
            Episode(
                episode_id="",
                customer_id=self.customer_id,
                opened_at=self.now_fn(),
                type=type,
                evidence_event_ids=list(evidence_event_ids or []),
                notes=notes,
                system_action=system_action,
            )
        )

    def update_outcome(
        self,
        episode_id: str,
        outcome: str,
        closed_at: Optional[datetime] = None,
        system_action: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[Episode]:
        """Backfill an outcome once it's known. Appends an amendment record."""
        ep = self._by_id.get(episode_id)
        if ep is None:
            log.warning("update_outcome: unknown episode_id %s", episode_id)
            return None
        ep.outcome = outcome
        if closed_at is not None:
            ep.closed_at = closed_at
        if system_action is not None:
            ep.system_action = system_action
        if notes is not None:
            ep.notes = notes
        self._append_line(
            {
                "record_type": self.RECORD_OUTCOME,
                "episode_id": episode_id,
                "outcome": outcome,
                "closed_at": iso(ep.closed_at) if ep.closed_at else None,
                "system_action": ep.system_action,
                "notes": ep.notes,
                "amended_at": iso(self.now_fn()),
            }
        )
        return ep

    def close_episode(
        self, episode_id: str, outcome: str, closed_at: Optional[datetime] = None, **kw
    ) -> Optional[Episode]:
        return self.update_outcome(
            episode_id, outcome, closed_at=closed_at or self.now_fn(), **kw
        )

    # --- reads --------------------------------------------------------------

    def query(
        self,
        type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        open_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[Episode]:
        """Filtered retrieval, newest last. `limit` keeps the most recent N."""
        pool = self._by_type.get(type, []) if type is not None else self._episodes
        out = []
        for ep in pool:
            if since is not None and (ep.opened_at is None or ep.opened_at < since):
                continue
            if until is not None and (ep.opened_at is None or ep.opened_at > until):
                continue
            if open_only and ep.closed_at is not None:
                continue
            out.append(ep)
        if limit is not None and limit >= 0:
            out = out[-limit:]
        return out

    def get(self, episode_id: str) -> Optional[Episode]:
        return self._by_id.get(episode_id)

    def latest(self, type: Optional[str] = None) -> Optional[Episode]:
        found = self.query(type=type, limit=1)
        return found[0] if found else None

    def types(self) -> list[str]:
        return sorted(self._by_type)

    def __len__(self) -> int:
        return len(self._episodes)

    # --- persistence --------------------------------------------------------

    def load(self) -> int:
        """Rebuild the in-memory index from the JSONL file. Returns episode count."""
        self._episodes.clear()
        self._by_id.clear()
        self._by_type.clear()
        self._counter = 0
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("%s:%d unparseable -- skipped", self.path.name, lineno)
                    continue
                rtype = raw.get("record_type", self.RECORD_EPISODE)
                if rtype == self.RECORD_OUTCOME:
                    ep = self._by_id.get(raw.get("episode_id", ""))
                    if ep is None:
                        log.warning(
                            "%s:%d outcome for unknown episode %s",
                            self.path.name,
                            lineno,
                            raw.get("episode_id"),
                        )
                        continue
                    ep.outcome = raw.get("outcome", ep.outcome)
                    if raw.get("closed_at"):
                        ep.closed_at = parse_ts(raw["closed_at"])
                    ep.system_action = raw.get("system_action", ep.system_action)
                    ep.notes = raw.get("notes", ep.notes)
                    continue
                try:
                    ep = Episode.from_dict(raw)
                except KeyError:
                    log.warning("%s:%d missing episode_id -- skipped", self.path.name, lineno)
                    continue
                if ep.episode_id in self._by_id:
                    continue  # idempotent reload
                self._index(ep)
                suffix = ep.episode_id.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    self._counter = max(self._counter, int(suffix))
        # Keep chronological order even if the file was appended out of order.
        self._episodes.sort(key=lambda e: (e.opened_at or datetime.min.replace(tzinfo=timezone.utc)))
        for eps in self._by_type.values():
            eps.sort(key=lambda e: (e.opened_at or datetime.min.replace(tzinfo=timezone.utc)))
        log.info("loaded %d episodes from %s", len(self._episodes), self.path)
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"<EpisodicMemory {self.customer_id} episodes={len(self._episodes)} {self.path}>"


class MemoryStore:
    """Per-customer memory: working + episodic, persisted under memory_dir."""

    def __init__(
        self,
        customer_id: str,
        memory_dir: str | Path = "memory",
        now_fn: Optional[Callable[[], datetime]] = None,
        autoload: bool = False,
    ) -> None:
        self.customer_id = customer_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.now_fn = now_fn or _now_utc
        self.working_path = self.memory_dir / f"{customer_id}_working.json"
        self.episodic_path = self.memory_dir / f"{customer_id}_episodic.jsonl"
        self.working = WorkingMemory(customer_id, now_fn=self.now_fn)
        self.episodic = EpisodicMemory(customer_id, self.episodic_path, now_fn=self.now_fn)
        if autoload:
            self.load()

    def bind_clock(self, clock: Any) -> "MemoryStore":
        """Point every timestamp at simulated time: store.bind_clock(clock)."""
        self.now_fn = clock.now
        self.working.now_fn = clock.now
        self.episodic.now_fn = clock.now
        return self

    def save(self) -> Path:
        """Snapshot working memory (episodic persists on every append)."""
        tmp = self.working_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.working.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        tmp.replace(self.working_path)  # atomic, so a crash never truncates it
        return self.working_path

    def load(self) -> "MemoryStore":
        """Restore both layers if files exist; start empty otherwise."""
        if self.working_path.exists():
            try:
                self.working.load_dict(
                    json.loads(self.working_path.read_text(encoding="utf-8"))
                )
                log.info("restored working memory from %s", self.working_path)
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("cannot restore %s (%s) -- starting empty", self.working_path, exc)
        self.episodic.load()
        return self

    def reset(self) -> "MemoryStore":
        """Delete both files and start clean (use between scenario re-runs)."""
        self.working_path.unlink(missing_ok=True)
        self.episodic_path.unlink(missing_ok=True)
        self.working = WorkingMemory(self.customer_id, now_fn=self.now_fn)
        self.episodic = EpisodicMemory(
            self.customer_id, self.episodic_path, now_fn=self.now_fn
        )
        return self

    def __repr__(self) -> str:
        return f"<MemoryStore {self.customer_id} {self.working!r} {self.episodic!r}>"
