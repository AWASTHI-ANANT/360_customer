"""Agent contract.

An agent observes events and writes FINDINGS. It never decides an
inferred_state, never picks an action, and never reads another agent's findings
(at this stage the agents form a swarm, not a pipeline). Every finding carries
the event IDs that justify it -- a finding without evidence is a bug.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional, Sequence

from c360.loader import CustomerProfile, Event

log = logging.getLogger(__name__)

CONFIDENCE_BANDS = ("low", "medium", "high")


@dataclass(slots=True)
class Finding:
    """One agent's assessment of one dimension, with its evidence."""

    key: str
    value: dict[str, Any]
    confidence_band: str
    agent: str
    evidence_event_ids: list[str] = field(default_factory=list)
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.confidence_band not in CONFIDENCE_BANDS:
            log.warning("finding %s: band %r not in %s", self.key, self.confidence_band, CONFIDENCE_BANDS)
        if not self.evidence_event_ids:
            log.warning("finding %s from %s carries no evidence event IDs", self.key, self.agent)

    @property
    def band_rank(self) -> int:
        return CONFIDENCE_BANDS.index(self.confidence_band) if self.confidence_band in CONFIDENCE_BANDS else -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence_band": self.confidence_band,
            "agent": self.agent,
            "evidence_event_ids": list(self.evidence_event_ids),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "notes": self.notes,
        }


@dataclass(slots=True)
class AgentContext:
    """Everything an agent is allowed to touch."""

    memory: Any            # MemoryStore
    log: Any               # DailyLogWriter
    profile: CustomerProfile
    event_store: Any       # EventStore
    clock: Any             # SimulatedClock

    def now(self) -> datetime:
        return self.clock.now()


class Agent(ABC):
    """Base class. Subclasses implement handles() and on_event()."""

    name: str = "agent"

    def __init__(self) -> None:
        self.findings_emitted = 0
        # Last value/band per finding key, so we only re-emit on real change.
        self._last: dict[str, tuple[str, str]] = {}

    # --- contract -----------------------------------------------------------

    @abstractmethod
    def handles(self, event: Event) -> bool:
        """Does this agent react to this event at all?"""

    @abstractmethod
    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        """React to one event. Return the findings written."""

    def on_day_boundary(self, sim_date: date, ctx: AgentContext) -> list[Finding]:
        """Recompute windowed views. Default: nothing."""
        return []

    def on_start(self, ctx: AgentContext, history: Sequence[Event]) -> list[Finding]:
        """One-time setup from the history seed. Default: nothing."""
        return []

    # --- helpers for subclasses --------------------------------------------

    def emit(
        self,
        ctx: AgentContext,
        key: str,
        value: dict[str, Any],
        band: str,
        evidence: Sequence[str],
        notes: Optional[str] = None,
        event_id: Optional[str] = None,
        force: bool = False,
    ) -> Optional[Finding]:
        """Write a finding to working memory, log it, and open an episode if it matters.

        Returns None when the value and band are unchanged since last time --
        that keeps the daily log readable instead of one line per day per key.
        """
        import json as _json

        signature = (_json.dumps(value, sort_keys=True, default=str), band)
        if not force and self._last.get(key) == signature:
            return None
        self._last[key] = signature

        finding = Finding(
            key=key,
            value=value,
            confidence_band=band,
            agent=self.name,
            evidence_event_ids=list(evidence),
            updated_at=ctx.now(),
            notes=notes,
        )
        ctx.memory.working.set_finding(
            key, value, band, self.name, finding.evidence_event_ids, notes=notes
        )
        ctx.log.log_agent_action(
            self.name,
            event_id,
            f"{key}={_summarise(value)} [{band}]",
            finding.evidence_event_ids,
            as_of=ctx.now(),
            finding_key=key,
            confidence_band=band,
        )
        # A finding that first reaches medium or high is a durable episode.
        if finding.band_rank >= 1:
            already = ctx.memory.episodic.query(type=key)
            if not already:
                ctx.memory.episodic.open_episode(
                    type=key,
                    evidence_event_ids=finding.evidence_event_ids,
                    notes=notes or _summarise(value),
                )
        self.findings_emitted += 1
        return finding


def _summarise(value: Any, limit: int = 160) -> str:
    """Compact one-line rendering of a finding value for log lines."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            # Identity checks, not `v in (None, False, ...)`: that uses ==, and
            # 0 == False, which would drop the most interesting values of all
            # (logins=0, purchases_7d=0) from the log line.
            if v is None or v is False:
                continue
            if isinstance(v, (list, dict, str)) and not v:
                continue
            if isinstance(v, float):
                v = round(v, 3)
            parts.append(f"{k}={v}")
        text = " ".join(parts)
    else:
        text = str(value)
    return text[:limit]
