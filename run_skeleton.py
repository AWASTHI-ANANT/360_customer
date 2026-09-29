#!/usr/bin/env python3
"""Replay a scenario through the agent swarm.

    python run_skeleton.py                     # all scenarios, fast-forward
    python run_skeleton.py -s scenario_03      # one scenario
    python run_skeleton.py -s scenario_03 --findings   # print every finding
    python run_skeleton.py -s scenario_03 --realtime --speed 0.05   # demo pacing

Wiring, per event:   derive (at load) -> event_store.add -> log_event -> every
                     agent whose handles() is true
Wiring, per day:     agent.on_day_boundary(...) for each agent -> memory.save()

SignalAgent and SupportAgent run independently and never read each other's
findings. This is the swarm stage: no agent decides an inferred_state or action.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hitl
from agents import ActionAgent, SignalAgent, SupportAgent, SynthesisAgent
from agents.base import AgentContext, Finding
from output import CheckpointWriter
from c360 import (
    DailyLogWriter,
    EventStore,
    MemoryStore,
    SimulatedClock,
    find_scenarios,
    iso,
    load_scenario,
)
from c360 import llm_client

ROOT = Path(__file__).resolve().parent


def run_one(
    scenario_id: str,
    data_root: Path | None,
    out_dir: Path,
    realtime: bool = False,
    speed: float | None = None,
    fresh: bool = True,
    show_findings: bool = False,
    mode: str = "auto",
    inject: Path | None = None,
) -> dict:
    scn = load_scenario(scenario_id, data_root=data_root)
    print(f"\n=== {scn.scenario_id} | {scn.profile.name} ({scn.customer_id}) ===")
    print(
        f"    {scn.profile.occupation}, age={scn.profile.age}, "
        f"tier={scn.profile.customer_value_tier}, tenure={scn.profile.tenure_months}mo, "
        f"dependents={scn.profile.dependents}"
    )
    print(f"    history={len(scn.history_events)} live={len(scn.live_events)} events")
    for issue in scn.issues:
        print(f"    ! {issue}")

    clock = SimulatedClock.from_config(scn.replay_config)
    memory = MemoryStore(scn.customer_id, memory_dir=out_dir / "memory").bind_clock(clock)
    if fresh:
        memory.reset()
    memory.load()

    writer = DailyLogWriter(scn.scenario_id, output_dir=out_dir / "logs")
    if fresh:
        writer.truncate_and_restart()

    # --inject splices synthetic events into a COPY of the live stream, in
    # event_time order; the scenario folder on disk is never modified.
    live_events = list(scn.live_events)
    if inject is not None:
        live_events = _inject(live_events, inject, scn)
        print(f"    injected {len(live_events) - len(scn.live_events)} events from {inject}")

    # History seeds the event store, so a 14-day window on day 1 of the live
    # stream still sees the customer's ordinary behaviour behind it.
    store = EventStore(scn.customer_id, scn.history_events)

    llm_client.configure(
        scn.scenario_id, cache_dir=out_dir / "llm_cache", trace_dir=out_dir / "logs"
    )
    hitl.configure(scn.scenario_id, log_dir=out_dir / "logs")
    checkpoints = CheckpointWriter(scn.scenario_id, output_dir=ROOT / "output")
    reason = llm_client.unavailable_reason()
    print(f"    llm: {'offline -- ' + reason if reason else 'available (' + llm_client.get_client().model + ')'}")

    ctx = AgentContext(
        memory=memory, log=writer, profile=scn.profile, event_store=store, clock=clock
    )
    # Order matters at the day boundary: signal -> synthesis -> action.
    signal, support = SignalAgent(), SupportAgent()
    synthesis, action = SynthesisAgent(), ActionAgent(mode=mode)
    agents = [signal, support, synthesis, action]

    # Every finding emitted during the run, for the eyeball list and the tests.
    trace: list[Finding] = []

    for agent in agents:
        trace.extend(agent.on_start(ctx, scn.history_events))

    @clock.on_event
    def _on_event(event):
        store.add(event)
        writer.log_event(event)
        for agent in agents:
            if agent.handles(event):
                trace.extend(agent.on_event(event, ctx))

    @clock.on_day_boundary
    def _on_day(sim_date):
        writer.log_day_boundary(sim_date, clock.events_in_previous_day)
        # signal -> synthesis -> action, then the checkpoint for the day.
        for agent in agents:
            trace.extend(agent.on_day_boundary(sim_date, ctx))
        checkpoints.write(sim_date, memory)
        memory.save()

    with writer:
        clock.run(
            live_events,
            realtime=realtime,
            **({"speed_seconds_per_day": speed} if realtime and speed else {}),
        )
        # One final checkpoint at simulated_end so the last state is always scored.
        checkpoints.write(scn.replay_config.simulated_end, memory)
        memory.save()

    empty_days = sum(1 for n in clock.event_counts.values() if n == 0)
    print(
        f"    replayed {clock.events_dispatched}/{len(scn.live_events)} events across "
        f"{clock.days_dispatched} days ({empty_days} dead)"
    )
    for agent in agents:
        extra = ""
        if isinstance(agent, SupportAgent):
            extra = (
                f" (llm={agent.llm_calls} fallback={agent.fallback_calls}"
                f" skipped_unconsented={agent.skipped_unconsented})"
            )
        print(f"    {agent.name}: {agent.findings_emitted} findings{extra}")
    hypothesis = memory.working.get_hypothesis()
    print(
        f"    findings total {len(trace)}, episodes {len(memory.episodic)}, "
        f"checkpoints {len(checkpoints)}"
    )
    if hypothesis is not None:
        print(
            f"    final hypothesis: {hypothesis.state} [{hypothesis.confidence_band}] "
            f"score={hypothesis.score:.3f}"
            + (" (fallback)" if hypothesis.used_fallback else "")
        )
    final = checkpoints.latest()
    if final is not None:
        print(
            f"    final checkpoint : {final['inferred_state']} / {final['action']}"
            f" / {final['hitl_status']}"
        )
    print(
        f"    synthesis: {synthesis.debates} debates, {synthesis.reviews} reviews, "
        f"llm={synthesis.llm_calls} fallback={synthesis.fallback_calls}"
    )
    print(
        f"    action   : llm={action.llm_calls} fallback={action.fallback_calls} "
        f"revisions={action.revisions}"
    )
    print(f"    checkpoints -> {checkpoints.path}")
    print(f"    log    -> {writer.path}")
    print(f"    memory -> {memory.working_path}")
    if store.late_insertions:
        print(f"    late/out-of-order events reinserted in place: {store.late_insertions}")
    if clock.callback_errors:
        print(f"    !! callback errors: {clock.callback_errors}")

    if show_findings:
        print_findings(trace)

    return {
        "scenario_id": scn.scenario_id,
        "events": clock.events_dispatched,
        "expected_events": len(live_events),
        "checkpoints": checkpoints.checkpoints,
        "checkpoint_path": str(checkpoints.path),
        "hypothesis": hypothesis,
        "days": clock.days_dispatched,
        "findings": len(trace),
        "episodes": len(memory.episodic),
        "trace": trace,
    }


def _inject(live_events, fixture: Path, scn):
    """Merge a JSONL fixture into the live stream, in event_time order.

    The scenario folder is never touched; this builds a new in-memory list, so
    an injected run and a clean run can be compared directly.
    """
    import json as _json

    from c360.derive import derive
    from c360.loader import Event

    extra = []
    for lineno, line in enumerate(fixture.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            extra.append(derive(Event.from_json(_json.loads(line), stream="live"), scn.profile))
        except (ValueError, KeyError, _json.JSONDecodeError) as exc:
            print(f"    ! {fixture.name}:{lineno} skipped ({exc})")
    merged = list(live_events) + extra
    merged.sort(key=lambda e: (e.event_time, e.event_id))
    return merged


def print_findings(trace: list[Finding]) -> None:
    """Readable list of every finding, with date and evidence IDs."""
    print(f"\n    --- findings ({len(trace)}) ---")
    print(f"    {'date':<11} {'band':<7} {'key':<28} {'agent':<14} evidence / detail")
    for f in trace:
        when = f.updated_at.strftime("%Y-%m-%d") if f.updated_at else "?"
        ev = ",".join(f.evidence_event_ids[:3])
        if len(f.evidence_event_ids) > 3:
            ev += f" (+{len(f.evidence_event_ids) - 3})"
        detail = f.notes or ""
        print(f"    {when:<11} {f.confidence_band:<7} {f.key:<28} {f.agent:<14} {ev}")
        if detail:
            print(f"    {'':<11} {'':<7} {'':<28} {'':<14}   {detail[:110]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-s", "--scenario", action="append", help="scenario id or path (repeatable)")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--realtime", action="store_true", help="sleep between events (demo mode)")
    ap.add_argument("--speed", type=float, default=None, help="override seconds per simulated day")
    ap.add_argument("--keep", action="store_true", help="append to existing log/memory")
    ap.add_argument("--findings", action="store_true", help="print every finding emitted")
    ap.add_argument("--interactive", action="store_true",
                    help="ask a human to approve each action (default: --auto)")
    ap.add_argument("--auto", action="store_true", default=True,
                    help="never ask a human; actions stay 'escalated' (default)")
    ap.add_argument("--inject", type=Path, default=None,
                    help="splice a JSONL fixture into a copy of the live stream")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    scenarios = args.scenario or [p.name for p in find_scenarios(args.data_root)]
    if not scenarios:
        print("no scenarios found -- pass --data-root", file=sys.stderr)
        return 1

    results = [
        run_one(
            s, args.data_root, args.out_dir, args.realtime, args.speed,
            fresh=not args.keep, show_findings=args.findings,
            mode="interactive" if args.interactive else "auto",
            inject=args.inject,
        )
        for s in scenarios
    ]

    print("\n=== summary ===")
    ok = True
    for r in results:
        complete = r["events"] == r["expected_events"]
        ok &= complete
        print(
            f"  {r['scenario_id']:12} events {r['events']}/{r['expected_events']} "
            f"{'OK' if complete else 'MISMATCH'}  days={r['days']}  "
            f"findings={r['findings']}  episodes={r['episodes']}  "
            f"checkpoints={len(r['checkpoints'])}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
