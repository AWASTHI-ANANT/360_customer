#!/usr/bin/env python3
"""Invariant checks for the loader/clock/log/memory plumbing.

Plain stdlib asserts -- run with `python test_skeleton.py`, no pytest needed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from c360 import (
    DailyLogWriter,
    Episode,
    MemoryStore,
    SimulatedClock,
    find_scenarios,
    load_scenario,
)

ROOT = Path(__file__).resolve().parent
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok  {name}")


def test_loader() -> None:
    print("loader")
    scn = load_scenario("scenario_01")
    check("history/live split", len(scn.history_events) == 375 and len(scn.live_events) == 116)
    check(
        "both lists sorted by event_time",
        all(
            a.event_time <= b.event_time
            for lst in (scn.history_events, scn.live_events)
            for a, b in zip(lst, lst[1:])
        ),
    )
    check("no duplicate event_ids", len(scn.by_id()) == 375 + 116)
    check("stream tagged", {e.stream for e in scn.history_events} == {"history"})
    check("timezone-aware UTC", scn.live_events[0].event_time.tzinfo is timezone.utc)
    # derive.py now runs inside the loader, so `derived` arrives populated;
    # derive_features=False is the escape hatch that restores the raw contract.
    check("derived is filled at load time", all(e.derived for e in scn.live_events))
    check("derived carries the documented keys",
          {"is_salary", "counterparty_is_self", "counterparty_is_external_bank",
           "is_income_like", "text_fields"} <= set(scn.live_events[0].derived))
    check("derive_features=False leaves derived empty",
          all(e.derived == {} for e in load_scenario("scenario_01", derive_features=False).live_events))
    check("profile flattened from nested 'profile' key", scn.profile.name == "Marcus Vance")
    check("profile age captured", scn.profile.age == 48)
    check("account_type lookup", scn.profile.account_type("ACC_CC_001") == "credit_card")
    check("no load issues on shipped data", scn.issues == [], str(scn.issues))

    # null account_id (customer-level events) survives as None
    nulls = [e for e in scn.live_events if e.account_id is None]
    check("null account_id preserved", bool(nulls) and nulls[0].source_system == "support_logs")

    # unknown source_system/event_type warns instead of crashing
    tmp = Path(tempfile.mkdtemp())
    try:
        src = scn.path
        dst = tmp / "scenario_x"
        shutil.copytree(src, dst)
        with (dst / "live_stream.jsonl").open("a") as fh:
            fh.write(json.dumps({
                "event_id": "EVT_WEIRD", "event_time": "2026-03-01T00:00:00Z",
                "ingestion_time": "2026-03-01T00:00:00Z", "customer_id": "CUST_00088",
                "account_id": None, "source_system": "telepathy", "event_type": "vibes",
                "schema_version": "9.9", "payload": {},
            }) + "\n")
            fh.write("{ not json\n")
        scn2 = load_scenario(dst)
        check("unknown source_system warns, does not crash", any("telepathy" in i for i in scn2.issues))
        check("bad JSON line skipped, not fatal", any("unparseable" in i for i in scn2.issues))
        check("unknown event still loaded", "EVT_WEIRD" in scn2.by_id())

        # duplicate event_id: fatal by default, survivable with strict=False
        shutil.copy(dst / "live_stream.jsonl", dst / "live_stream.bak")
        with (dst / "live_stream.jsonl").open("a") as fh:
            fh.write(json.dumps(scn.live_events[0].to_dict()) + "\n")
        try:
            load_scenario(dst)
            raise AssertionError("expected duplicate event_id to raise")
        except Exception as exc:
            check("duplicate event_id raises by default", "duplicate" in str(exc).lower())
        scn3 = load_scenario(dst, strict=False)
        check("strict=False survives duplicates", any("duplicate" in i for i in scn3.issues))
    finally:
        shutil.rmtree(tmp)


def test_clock() -> None:
    print("clock")
    scn = load_scenario("scenario_03")  # last event 2026-04-03, end 2026-04-15
    clock = SimulatedClock.from_config(scn.replay_config)
    check("now() before run == simulated_start", clock.now() == scn.replay_config.simulated_start)

    seen: list[tuple[str, object, datetime]] = []
    clock.on_event(lambda e: seen.append(("event", e.event_id, clock.now())))
    clock.on_day_boundary(lambda d: seen.append(("day", d, clock.now())))
    clock.fast_forward(scn.live_events)

    days = [x for x in seen if x[0] == "day"]
    events = [x for x in seen if x[0] == "event"]
    check("every event fired once", len(events) == len(scn.live_events))
    check("event order preserved", [x[1] for x in events] == [e.event_id for e in scn.live_events])

    # Feb 1 .. Apr 15 inclusive
    expected = (scn.replay_config.simulated_end.date() - scn.replay_config.simulated_start.date()).days + 1
    check(f"one boundary per simulated day ({expected})", len(days) == expected)
    check("boundaries are consecutive, no gaps or repeats",
          [x[1] for x in days] == [scn.replay_config.simulated_start.date() + timedelta(days=i) for i in range(expected)])
    check("boundaries fire at midnight", all(x[2].hour == 0 and x[2].minute == 0 for x in days))

    # the critical one: boundaries continue after the last event, to simulated_end
    last_event_pos = max(i for i, x in enumerate(seen) if x[0] == "event")
    trailing = [x for x in seen[last_event_pos + 1:] if x[0] == "day"]
    check("boundaries keep firing after last event", len(trailing) == 12, f"got {len(trailing)}")
    check("last boundary is simulated_end date", days[-1][1] == scn.replay_config.simulated_end.date())

    # dead days in the middle still get a boundary
    event_days = {e.event_time.date() for e in scn.live_events}
    check("dead mid-stream days covered", len([x for x in days if x[1] not in event_days]) == 37)

    # boundary for day D precedes events on day D
    ok = True
    current: date | None = None
    for kind, item, _ in seen:
        if kind == "day":
            current = item
        else:
            ok &= current == item_day(item, scn)
    check("day boundary precedes that day's events", ok)
    check("now() after run == simulated_end", clock.now() == scn.replay_config.simulated_end)

    # scenario_01 has events PAST simulated_end -- they must not be dropped
    scn1 = load_scenario("scenario_01")
    c1 = SimulatedClock.from_config(scn1.replay_config)
    fired: list[str] = []
    c1.on_event(lambda e: fired.append(e.event_id))
    c1.fast_forward(scn1.live_events)
    past_end = [e for e in scn1.live_events if e.event_time > scn1.replay_config.simulated_end]
    check("events after simulated_end still fire", bool(past_end) and len(fired) == len(scn1.live_events))
    check("boundaries extend to cover them", c1.days_dispatched == 74)

    # realtime mode: same output, just slower
    c2 = SimulatedClock.from_config(scn.replay_config)
    rt: list[str] = []
    c2.on_event(lambda e: rt.append(e.event_id))
    c2.run_realtime(scn.live_events, speed_seconds_per_day=0.0005, max_sleep_seconds=0.01)
    check("run_realtime delivers identical event stream", rt == [e.event_id for e in scn.live_events])
    c3 = SimulatedClock.from_config(scn.replay_config)
    c3.on_event(lambda e: None)
    c3.run_realtime(scn.live_events, skip=True)
    check("run_realtime(skip=True) degrades to fast_forward", c3.events_dispatched == len(scn.live_events))


def item_day(event_id: str, scn) -> date:
    return scn.by_id()[event_id].event_time.date()


def test_log_writer(tmp: Path) -> None:
    print("daily_log_writer")
    scn = load_scenario("scenario_02")
    w = DailyLogWriter(scn.scenario_id, output_dir=tmp / "logs")
    w.log_event(scn.live_events[0])
    w.log_day_boundary(date(2026, 3, 1), 4)
    w.log_agent_action("test_agent", "EVT_000001", "did a thing", ["EVT_000001", "EVT_000002"])
    # append-only: reopening must not clobber
    w.close()
    w2 = DailyLogWriter(scn.scenario_id, output_dir=tmp / "logs")
    w2.log_event(scn.live_events[1])
    w2.close()

    lines = [json.loads(l) for l in w.path.read_text().splitlines() if l.strip()]
    check("append-only across reopen", len(lines) == 4)
    check("kinds present", [l["kind"] for l in lines] == ["event", "day_boundary", "agent_action", "event"])
    check("event line carries account_id", "account_id" in lines[0])
    check("event line has human summary", isinstance(lines[0]["summary"], str) and lines[0]["summary"])
    check("day marker carries count", lines[1]["event_count_that_day"] == 4)
    check("agent action carries evidence_ids", lines[2]["evidence_ids"] == ["EVT_000001", "EVT_000002"])
    check("one file per scenario", w.path.name == f"{scn.scenario_id}_daily.jsonl")

    # every event kind in the dataset renders a non-empty summary
    from c360 import summarize_event
    kinds = {}
    for s in find_scenarios():
        for e in load_scenario(s).all_events:
            kinds.setdefault(e.kind, e)
    check(f"all {len(kinds)} event kinds summarise", all(summarize_event(e) for e in kinds.values()))


def test_memory(tmp: Path) -> None:
    print("memory_store")
    scn = load_scenario("scenario_01")
    clock = SimulatedClock.from_config(scn.replay_config)
    mem = MemoryStore(scn.customer_id, memory_dir=tmp / "memory").bind_clock(clock)

    # timestamps must come from the simulated clock, not the wall clock
    clock._now = datetime(2026, 3, 12, tzinfo=timezone.utc)
    f = mem.working.set_finding("outflow_pattern", {"trend": "up"}, "medium", "outflow_agent", ["EVT_000392"])
    check("finding uses simulated time", f.updated_at == datetime(2026, 3, 12, tzinfo=timezone.utc))
    check("get_finding round-trips", mem.working.get_finding("outflow_pattern").value == {"trend": "up"})
    check("all_findings", set(mem.working.all_findings()) == {"outflow_pattern"})

    mem.working.set_hypothesis("medical_hardship", 0.4, "low", ["EVT_000392"])
    first = mem.working.get_hypothesis().first_inferred
    clock._now = datetime(2026, 3, 26, tzinfo=timezone.utc)
    h = mem.working.set_hypothesis("medical_hardship", 0.9, "high", ["EVT_000392", "EVT_000462"])
    check("first_inferred preserved while state is unchanged", h.first_inferred == first)
    check("last_updated advances", h.last_updated == datetime(2026, 3, 26, tzinfo=timezone.utc))
    h2 = mem.working.set_hypothesis("churn_risk", 0.5, "low")
    check("first_inferred resets on state change", h2.first_inferred == datetime(2026, 3, 26, tzinfo=timezone.utc))
    # re-entering a state after leaving it correctly restamps first_inferred
    reentered = mem.working.set_hypothesis("medical_hardship", 0.9, "high", ["EVT_000392"])
    check("first_inferred restamps on re-entry", reentered.first_inferred == datetime(2026, 3, 26, tzinfo=timezone.utc))

    ep1 = mem.episodic.append(Episode(
        episode_id="", customer_id=scn.customer_id, opened_at=None,
        type="medical_hardship", evidence_event_ids=["EVT_000392"], notes="ER visit",
    ))
    ep2 = mem.episodic.open_episode("engagement_dip", ["EVT_000420"])
    check("episode_id generated", ep1.episode_id == f"EP_{scn.customer_id}_0001")
    check("ids are unique", ep1.episode_id != ep2.episode_id)
    check("opened_at defaults to simulated now", ep1.opened_at == datetime(2026, 3, 26, tzinfo=timezone.utc))

    mem.episodic.update_outcome(ep1.episode_id, "payment_plan_offered", system_action="support_intervention")
    check("update_outcome applies", mem.episodic.get(ep1.episode_id).outcome == "payment_plan_offered")
    check("query by type", [e.episode_id for e in mem.episodic.query(type="medical_hardship")] == [ep1.episode_id])
    check("query since filters", mem.episodic.query(since=datetime(2026, 4, 1, tzinfo=timezone.utc)) == [])
    check("query limit keeps most recent", [e.episode_id for e in mem.episodic.query(limit=1)] == [ep2.episode_id])
    check("query open_only", [e.episode_id for e in mem.episodic.query(open_only=True)] == [ep1.episode_id, ep2.episode_id])

    mem.save()
    check("working snapshot written", mem.working_path.exists())
    check("episodic jsonl written", mem.episodic_path.exists())
    raw_lines = mem.episodic_path.read_text().splitlines()
    check("episodic file is append-only (2 episodes + 1 amendment)", len(raw_lines) == 3)
    check("amendment recorded, episode line untouched",
          json.loads(raw_lines[2])["record_type"] == "outcome_update"
          and json.loads(raw_lines[0])["outcome"] is None)

    # --- restart: a fresh store must recover the same state ---------------
    clock2 = SimulatedClock.from_config(scn.replay_config)
    mem2 = MemoryStore(scn.customer_id, memory_dir=tmp / "memory").bind_clock(clock2)
    mem2.load()
    check("working memory survives restart",
          mem2.working.get_finding("outflow_pattern").value == {"trend": "up"})
    check("finding timestamp survives restart",
          mem2.working.get_finding("outflow_pattern").updated_at == datetime(2026, 3, 12, tzinfo=timezone.utc))
    check("hypothesis survives restart", mem2.working.get_hypothesis().state == "medical_hardship")
    check("first_inferred survives restart",
          mem2.working.get_hypothesis().first_inferred == reentered.first_inferred)
    check("episodes survive restart", len(mem2.episodic) == 2)
    check("index rebuilt by type", [e.episode_id for e in mem2.episodic.query(type="engagement_dip")] == [ep2.episode_id])
    check("backfilled outcome replayed onto index",
          mem2.episodic.get(ep1.episode_id).outcome == "payment_plan_offered")
    check("id counter continues after restart",
          mem2.episodic.open_episode("x").episode_id == f"EP_{scn.customer_id}_0003")

    # empty start
    mem3 = MemoryStore("CUST_NOBODY", memory_dir=tmp / "memory").load()
    check("missing files start empty", mem3.working.all_findings() == {} and len(mem3.episodic) == 0)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c360_test_"))
    try:
        test_loader()
        test_clock()
        test_log_writer(tmp)
        test_memory(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
