"""Per-customer event store, kept sorted by event_time.

Agents compute windowed statistics by QUERYING this at each day boundary rather
than by keeping running counters. That is the whole point: a late-arriving event
inserts at its correct event_time position (bisect, not append), so every window
recomputed afterwards is correct. A running counter could not be fixed up.
"""

from __future__ import annotations

import bisect
import logging
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Iterable, Optional

from .loader import Event

log = logging.getLogger(__name__)


class EventStore:
    def __init__(self, customer_id: str = "", events: Optional[Iterable[Event]] = None) -> None:
        self.customer_id = customer_id
        self._events: list[Event] = []
        # Parallel sort keys, so bisect works without a key= on older APIs and
        # stays cheap. Key is (event_time, event_id) for a stable total order.
        self._keys: list[tuple[datetime, str]] = []
        self._by_id: dict[str, Event] = {}
        self.late_insertions = 0
        for ev in events or ():
            self.add(ev)

    # --- writes -------------------------------------------------------------

    def add(self, event: Event) -> Event:
        """Insert at the correct event_time position, even if it arrives late."""
        if event.event_id in self._by_id:
            return self._by_id[event.event_id]  # idempotent re-ingest
        key = (event.event_time, event.event_id)
        pos = bisect.bisect_left(self._keys, key)
        if pos != len(self._keys):
            self.late_insertions += 1
        self._keys.insert(pos, key)
        self._events.insert(pos, event)
        self._by_id[event.event_id] = event
        return event

    # --- reads --------------------------------------------------------------

    def window(
        self,
        start: datetime | date,
        end: datetime | date,
        source_system: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[Event]:
        """Events with start <= event_time < end, optionally filtered.

        Half-open so consecutive windows never double-count an event.
        """
        lo = bisect.bisect_left(self._keys, (_as_dt(start), ""))
        hi = bisect.bisect_left(self._keys, (_as_dt(end), ""))
        found = self._events[lo:hi]
        if source_system is not None:
            found = [e for e in found if e.source_system == source_system]
        if event_type is not None:
            found = [e for e in found if e.event_type == event_type]
        return found

    def trailing(
        self,
        end: datetime | date,
        days: int,
        source_system: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[Event]:
        """The `days` days immediately before `end`."""
        end_dt = _as_dt(end)
        return self.window(
            end_dt - timedelta(days=days), end_dt, source_system, event_type
        )

    def last(
        self,
        event_type: Optional[str] = None,
        source_system: Optional[str] = None,
        before: Optional[datetime] = None,
        where=None,
    ) -> Optional[Event]:
        """The most recent matching event, or None.

        `where` is an optional predicate, which is how callers ask for things
        like "last salary credit" (a derived flag, not an event_type).
        """
        hi = len(self._events) if before is None else bisect.bisect_left(
            self._keys, (_as_dt(before), "")
        )
        for ev in reversed(self._events[:hi]):
            if event_type is not None and ev.event_type != event_type:
                continue
            if source_system is not None and ev.source_system != source_system:
                continue
            if where is not None and not where(ev):
                continue
            return ev
        return None

    def get(self, event_id: str) -> Optional[Event]:
        return self._by_id.get(event_id)

    def all(self) -> list[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __repr__(self) -> str:
        span = (
            f"{self._events[0].event_time:%Y-%m-%d}..{self._events[-1].event_time:%Y-%m-%d}"
            if self._events
            else "empty"
        )
        return f"<EventStore {self.customer_id} n={len(self._events)} {span} late={self.late_insertions}>"


def _as_dt(value: datetime | date) -> datetime:
    """Accept a date (meaning midnight UTC) or a datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.combine(value, dtime.min, tzinfo=timezone.utc)
