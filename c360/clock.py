"""Simulated clock that replays events and day boundaries in simulated-time order.

The whole pipeline is driven from here. Two guarantees matter:

1. Every simulated day between simulated_start and the end of the run gets
   exactly one on_day_boundary callback at 00:00 -- including days with no
   events, and including days after the last event up to simulated_end. Daily
   checkpoints must still be emitted during dead periods.
2. A day's boundary fires *before* any event on that day.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

from .loader import Event, ReplayConfig, iso, parse_ts

log = logging.getLogger(__name__)

EventCallback = Callable[[Event], Any]
DayCallback = Callable[[date], Any]

# Sort keys: at an identical timestamp, the day boundary comes first.
_KIND_BOUNDARY = 0
_KIND_EVENT = 1


def _midnight(d: date) -> datetime:
    return datetime.combine(d, dtime.min, tzinfo=timezone.utc)


class SimulatedClock:
    """Drives callbacks over a merged timeline of events and day boundaries."""

    def __init__(
        self,
        simulated_start: datetime | str,
        simulated_end: datetime | str,
        replay_speed_seconds_per_simulated_day: float = 1.0,
        scenario_id: str = "scenario",
        strict_callbacks: bool = True,
    ) -> None:
        self.simulated_start = parse_ts(simulated_start)
        self.simulated_end = parse_ts(simulated_end)
        self.replay_speed_seconds_per_simulated_day = float(
            replay_speed_seconds_per_simulated_day
        )
        self.scenario_id = scenario_id
        # When False, a raising callback is logged and the run continues.
        self.strict_callbacks = strict_callbacks

        self._event_callbacks: list[EventCallback] = []
        self._day_callbacks: list[DayCallback] = []

        self._now: datetime = self.simulated_start
        self._current_day: Optional[date] = None
        self._previous_day: Optional[date] = None
        self.event_counts: dict[date, int] = {}
        self.events_dispatched = 0
        self.days_dispatched = 0
        self.callback_errors: list[str] = []

    # --- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, config: ReplayConfig | dict, **kwargs) -> "SimulatedClock":
        """Build from a ReplayConfig or a raw replay_config.json dict."""
        if isinstance(config, ReplayConfig):
            return cls(
                simulated_start=config.simulated_start,
                simulated_end=config.simulated_end,
                replay_speed_seconds_per_simulated_day=(
                    config.replay_speed_seconds_per_simulated_day
                ),
                scenario_id=config.scenario_id,
                **kwargs,
            )
        return cls(
            simulated_start=config["simulated_start"],
            simulated_end=config["simulated_end"],
            replay_speed_seconds_per_simulated_day=config.get(
                "replay_speed_seconds_per_simulated_day", 1.0
            ),
            scenario_id=config.get("scenario_id", "scenario"),
            **kwargs,
        )

    # --- registration -------------------------------------------------------

    def on_event(self, callback: EventCallback) -> EventCallback:
        """Register a per-event callback. Returns it, so it works as a decorator."""
        self._event_callbacks.append(callback)
        return callback

    def on_day_boundary(self, callback: DayCallback) -> DayCallback:
        """Register a per-simulated-day callback, called with a `date` at 00:00."""
        self._day_callbacks.append(callback)
        return callback

    # --- state --------------------------------------------------------------

    def now(self) -> datetime:
        """Current simulated time: valid before, during and after a run."""
        return self._now

    def now_iso(self) -> str:
        return iso(self._now)

    @property
    def current_day(self) -> Optional[date]:
        """The simulated day currently being replayed."""
        return self._current_day

    @property
    def events_in_previous_day(self) -> int:
        """Events seen on the day that just ended.

        A boundary fires before that day's events exist, so this -- not the
        current day -- is the count worth logging at a boundary.
        """
        if self._previous_day is None:
            return 0
        return self.event_counts.get(self._previous_day, 0)

    def event_count_for(self, day: date) -> int:
        return self.event_counts.get(day, 0)

    # --- timeline construction ---------------------------------------------

    def day_boundaries(self, events: Sequence[Event] = ()) -> list[date]:
        """Every simulated day that must get a boundary callback.

        Runs from simulated_start's date to simulated_end's date inclusive, and
        is extended if the stream contains events past simulated_end (scenario_01
        does: its last event is ~21h after simulated_end).
        """
        start_day = self.simulated_start.date()
        end_day = self.simulated_end.date()
        if events:
            last_event_day = max(e.event_time.date() for e in events)
            end_day = max(end_day, last_event_day)
            first_event_day = min(e.event_time.date() for e in events)
            if first_event_day < start_day:
                log.warning(
                    "%s: events before simulated_start (%s < %s); extending boundaries back",
                    self.scenario_id,
                    first_event_day,
                    start_day,
                )
                start_day = first_event_day
        days: list[date] = []
        d = start_day
        while d <= end_day:
            days.append(d)
            d += timedelta(days=1)
        return days

    def _timeline(self, events: Sequence[Event]) -> list[tuple[datetime, int, Any]]:
        """Merged, ordered list of ('boundary', date) and ('event', Event) items."""
        items: list[tuple[datetime, int, Any]] = [
            (_midnight(d), _KIND_BOUNDARY, d) for d in self.day_boundaries(events)
        ]
        items += [(e.event_time, _KIND_EVENT, e) for e in events]
        # (timestamp, kind, tiebreak) -- boundaries before events at the same
        # instant; event_id keeps same-timestamp events deterministic.
        items.sort(key=lambda it: (it[0], it[1], "" if it[1] == _KIND_BOUNDARY else it[2].event_id))
        return items

    # --- dispatch -----------------------------------------------------------

    def _fire(self, callbacks: Sequence[Callable], arg: Any, label: str) -> None:
        for cb in callbacks:
            try:
                cb(arg)
            except Exception as exc:  # noqa: BLE001 - one bad agent shouldn't kill a run
                name = getattr(cb, "__name__", repr(cb))
                msg = f"{label} callback {name} raised {type(exc).__name__}: {exc}"
                self.callback_errors.append(msg)
                log.exception("%s", msg)
                if self.strict_callbacks:
                    raise

    def _dispatch_boundary(self, day: date) -> None:
        self._now = _midnight(day)
        self._previous_day = self._current_day
        self._current_day = day
        self.days_dispatched += 1
        self.event_counts.setdefault(day, 0)
        self._fire(self._day_callbacks, day, "day_boundary")

    def _dispatch_event(self, event: Event) -> None:
        self._now = event.event_time
        day = event.event_time.date()
        self.event_counts[day] = self.event_counts.get(day, 0) + 1
        self.events_dispatched += 1
        self._fire(self._event_callbacks, event, "event")

    # --- run modes ----------------------------------------------------------

    def fast_forward(self, events: Iterable[Event]) -> "SimulatedClock":
        """Replay the whole timeline with no real-time delay (scoring/dev runs)."""
        evs = list(events)
        timeline = self._timeline(evs)
        log.info(
            "%s fast_forward: %d events, %d day boundaries",
            self.scenario_id,
            len(evs),
            len(timeline) - len(evs),
        )
        for _ts, kind, item in timeline:
            if kind == _KIND_BOUNDARY:
                self._dispatch_boundary(item)
            else:
                self._dispatch_event(item)
        self._finish()
        return self

    def run_realtime(
        self,
        events: Iterable[Event],
        speed_seconds_per_day: Optional[float] = None,
        max_sleep_seconds: float = 10.0,
        skip: bool = False,
    ) -> "SimulatedClock":
        """Replay with real sleeps, for the demo.

        `speed_seconds_per_day` overrides replay_config (pass something small to
        compress a 74-day scenario). `skip=True` degrades to fast_forward, and
        Ctrl-C drops out of the sleeping and finishes the rest at full speed --
        so a demo can never trap you.
        """
        if skip:
            return self.fast_forward(events)

        per_day = (
            self.replay_speed_seconds_per_simulated_day
            if speed_seconds_per_day is None
            else float(speed_seconds_per_day)
        )
        seconds_per_sim_second = per_day / 86400.0
        evs = list(events)
        timeline = self._timeline(evs)
        log.info(
            "%s run_realtime: %d events over %d days at %.3gs/sim-day",
            self.scenario_id,
            len(evs),
            len(timeline) - len(evs),
            per_day,
        )

        prev_ts: Optional[datetime] = None
        interrupted = False
        for ts, kind, item in timeline:
            if not interrupted and prev_ts is not None and ts > prev_ts:
                delay = (ts - prev_ts).total_seconds() * seconds_per_sim_second
                delay = min(delay, max_sleep_seconds)
                if delay > 0:
                    try:
                        time.sleep(delay)
                    except KeyboardInterrupt:
                        log.warning(
                            "%s: interrupted -- finishing remaining timeline at full speed",
                            self.scenario_id,
                        )
                        interrupted = True
            prev_ts = ts
            if kind == _KIND_BOUNDARY:
                self._dispatch_boundary(item)
            else:
                self._dispatch_event(item)
        self._finish()
        return self

    def run(self, events: Iterable[Event], realtime: bool = False, **kwargs) -> "SimulatedClock":
        """Convenience dispatcher: realtime=False is the scoring path."""
        if realtime:
            return self.run_realtime(events, **kwargs)
        return self.fast_forward(events)

    def _finish(self) -> None:
        """Leave now() at the end of the simulated window once a run completes."""
        if self._now < self.simulated_end:
            self._now = self.simulated_end
        log.info(
            "%s finished: %d events, %d days, now=%s",
            self.scenario_id,
            self.events_dispatched,
            self.days_dispatched,
            self.now_iso(),
        )
