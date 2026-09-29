#!/usr/bin/env python3
"""Acceptance tests for signal_agent and support_agent on scenario_03 (David Chen).

Runs the full pipeline once with fast_forward, then asserts over every finding
emitted during the run -- not just the surviving working-memory snapshot, since
working memory holds only the latest value per key by design.

Forces C360_LLM_OFFLINE so the run is deterministic and needs no API key: the
support agent takes its keyword_fallback path. Run with an API key and without
that flag to exercise the LLM path.

    python -m tests.test_agents
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Deterministic, no-network run. Must be set before c360.llm_client is imported.
os.environ["C360_LLM_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import Finding  # noqa: E402
from agents.signal_agent import compute_baseline  # noqa: E402
from agents.support_agent import (  # noqa: E402
    SupportValidationError,
    keyword_fallback,
    validate_support_output,
)
from c360 import EventStore, load_scenario, mask  # noqa: E402
from run_skeleton import print_findings, run_one  # noqa: E402

SCENARIO = "scenario_03"
PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok  {name}")


# --- helpers over the finding trace -----------------------------------------


def of_key(trace: list[Finding], key: str) -> list[Finding]:
    return [f for f in trace if f.key == key]


def citing(trace: list[Finding], key: str, event_id: str) -> list[Finding]:
    return [f for f in of_key(trace, key) if event_id in f.evidence_event_ids]


def first_reaching(trace: list[Finding], key: str, predicate) -> Finding | None:
    for f in trace:
        if f.key == key and predicate(f):
            return f
    return None


# --- unit-level checks that don't need a full run ---------------------------


def test_derive_and_mask() -> None:
    print("derive + masking")
    scn = load_scenario(SCENARIO)
    by_id = scn.by_id()

    sweep = by_id["EVT_000461"]  # "David Chen - Chase Bank"
    check("derive: counterparty_is_self", sweep.derived["counterparty_is_self"] is True)
    check("derive: counterparty_is_external_bank", sweep.derived["counterparty_is_external_bank"] is True)
    check("derive: is_outbound", sweep.derived["is_outbound"] is True)

    salary = next(e for e in scn.live_events if e.derived.get("is_salary"))
    check("derive: is_salary", salary.payload["transaction_type"] == "salary_credit")
    check("derive: is_income_like covers salary", salary.derived["is_income_like"] is True)
    refund = by_id["EVT_000447"]  # tax_refund
    check("derive: tax_refund is not income_like", refund.derived["is_income_like"] is False)
    check("derive: tax_refund is not salary", refund.derived["is_salary"] is False)

    ticket = by_id["EVT_000409"]
    check("derive: text_fields picked up raw_text", ticket.derived["text_fields"] != [])

    masked, mapping = mask("David Chen moved 22500 from ACC_SAV_003 to 4111111111111111", scn.profile)
    check("mask: customer name replaced", "David" not in masked and "[CUSTOMER]" in masked)
    check("mask: account id replaced", "ACC_SAV_003" not in masked and "[ACCOUNT_1]" in masked)
    check("mask: long digit run replaced", "4111111111111111" not in masked)
    check("mask: money amount survives", "22500" in masked, masked)
    check("mask: mapping is reversible", mapping["[ACCOUNT_1]"] == "ACC_SAV_003")


def test_baseline() -> None:
    print("baseline")
    scn = load_scenario(SCENARIO)
    b = compute_baseline(scn.history_events)
    check("salary amount 5000", b.salary_amount == 5000)
    check("salary cadence 14d", b.salary_cadence_days == 14)
    check("purchases per week ~15", 14 < b.purchases_per_week < 17, str(b.purchases_per_week))
    check("logins per week ~3.3", 3 < b.logins_per_week < 4, str(b.logins_per_week))
    check("p95 amount is small (24)", b.p95_amount == 24, str(b.p95_amount))
    check("median session length found", b.median_session_sec == 307, str(b.median_session_sec))
    types = {si["transaction_type"] for si in b.standing_instructions}
    check("standing instructions detected", types == {"rent_payment", "gym_membership", "utilities"}, str(types))
    check("standing instructions on day 1", all(si["day_of_month"] == 1 for si in b.standing_instructions))
    check("category shares sum to ~1", abs(sum(b.category_share.values()) - 1.0) < 1e-6)
    check("empty history does not crash", compute_baseline([]).salary_amount is None)


def test_event_store() -> None:
    print("event_store")
    scn = load_scenario(SCENARIO)
    store = EventStore(scn.customer_id, scn.history_events)
    n = len(store)

    late = scn.live_events[-1]
    store.add(late)
    check("add is idempotent", len(store) == n + 1 and store.add(late) is late and len(store) == n + 1)

    # A late event must land in event_time order, not at the end.
    early = scn.live_events[0]
    store.add(early)
    times = [e.event_time for e in store.all()]
    check("store stays sorted after out-of-order insert", times == sorted(times))
    check("late insertion counted", store.late_insertions >= 1)

    window = store.window(date(2026, 2, 1), date(2026, 2, 2))
    check("window is half-open on the right", all(e.event_time.date() == date(2026, 2, 1) for e in window))
    check("window filters by event_type", all(e.event_type == "login" for e in store.window(date(2025, 10, 1), date(2026, 1, 1), event_type="login")))
    last_sal = store.last(where=lambda e: e.derived.get("is_salary"))
    check("last() with predicate finds salary", last_sal is not None and last_sal.derived["is_salary"])
    # Derive the cutoff from the data rather than hard-coding a date.
    first_login = next(e for e in store.all() if e.event_type == "login")
    cutoff = first_login.event_time + timedelta(seconds=1)
    check("last() respects before=", store.last(event_type="login", before=cutoff) is first_login)
    check("last() before= excludes everything", store.last(event_type="login", before=store.all()[0].event_time) is None)
    check("last() returns None when nothing matches", store.last(event_type="nope") is None)


def test_support_validation() -> None:
    print("support output validation")
    good = keyword_fallback("I was charged a $35 fee. Please refund this.", _fake_event("ticket_created", category="dispute"))
    check("fallback output passes validation", validate_support_output(good)["intent"] == "fee_dispute")

    try:
        validate_support_output({"sentiment": "furious", "sentiment_score": 0, "intent": "other"})
        raise AssertionError("expected bad enum to raise")
    except SupportValidationError as exc:
        check("invalid sentiment enum rejected", "sentiment" in str(exc))

    try:
        validate_support_output({"sentiment": "negative", "intent": "other", "sentiment_score": "very"})
        raise AssertionError("expected bad score to raise")
    except SupportValidationError:
        check("non-numeric sentiment_score rejected", True)

    coerced = validate_support_output({
        "sentiment": "negative", "intent": "other", "sentiment_score": -5,
        "urgency": "catastrophic", "life_event_hints": ["made_up_label", "medical_hardship"],
    })
    check("sentiment_score clamped to -1", coerced["sentiment_score"] == -1.0)
    check("unknown urgency defaults to low", coerced["urgency"] == "low")
    check("unknown life_event_hint dropped", coerced["life_event_hints"] == ["medical_hardship"])

    # The fallback must read the structured resolution_status, not just prose.
    rej = keyword_fallback(
        "As per our schedule of fees, the international transaction fee is valid and cannot be waived.",
        _fake_event("ticket_resolved", category="dispute", resolution_status="human_rejected"),
    )
    check("fallback reads human_rejected as unfavorable", rej["resolution_outcome"] == "unfavorable")
    churn = keyword_fallback("I am closing my account and switching to another bank.", _fake_event("ticket_created"))
    check("fallback detects churn language", churn["churn_language"] and churn["intent"] == "cancellation_or_exit")
    legal = keyword_fallback("I will contact my lawyer about this.", _fake_event("call_transcript"))
    check("fallback detects legal language", legal["legal_threat_language"] is True)
    hardship = keyword_fallback("I was hospitalized and need a payment plan.", _fake_event("ticket_created"))
    check("fallback detects hardship + medical hint", hardship["intent"] == "hardship_request" and "medical_hardship" in hardship["life_event_hints"])


def _fake_event(event_type: str, source_system: str = "support_logs", event_id: str = "EVT_TEST", **payload):
    from c360.loader import Event

    return Event(
        event_id=event_id, event_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        ingestion_time=datetime(2026, 3, 1, tzinfo=timezone.utc), customer_id="CUST_TEST",
        account_id=None, source_system=source_system, event_type=event_type,
        schema_version="1.0", payload=dict(payload),
    )


def _minimal_ctx(tmp: Path, scenario_id: str = "consent_test"):
    """A throwaway AgentContext, for paths the shipped scenarios don't contain."""
    from agents.base import AgentContext
    from c360 import DailyLogWriter, MemoryStore, SimulatedClock
    from c360.derive import derive

    scn = load_scenario(SCENARIO)
    clock = SimulatedClock.from_config(scn.replay_config)
    clock._now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    memory = MemoryStore(scenario_id, memory_dir=tmp / "cmem").bind_clock(clock)
    writer = DailyLogWriter(scenario_id, output_dir=tmp / "clogs")
    ctx = AgentContext(
        memory=memory, log=writer, profile=scn.profile,
        event_store=EventStore(scn.customer_id), clock=clock,
    )
    return ctx, writer, derive


def test_si_absence_month_arithmetic(tmp: Path) -> None:
    """A standing instruction late in the month must be judged for ITS month.

    Regression guard: deriving the expected date from sim_date's own month skips
    a day_of_month=28 instruction entirely, because by the time its grace period
    elapses (Mar 3) sim_date.replace(day=28) already points at March.
    """
    print("standing-instruction month arithmetic")
    from agents.signal_agent import Baseline, SignalAgent
    from config import thresholds as T

    for dom, expect_on in ((1, date(2026, 2, 4)), (28, date(2026, 3, 3))):
        ctx, writer, _ = _minimal_ctx(tmp, f"si_dom{dom}")
        agent = SignalAgent()
        # Only the standing-instruction check should be live; zero out the rest.
        agent.baseline = Baseline(
            standing_instructions=[
                {"transaction_type": "rent_payment", "amount": 2500,
                 "day_of_month": dom, "occurrences": 3}
            ]
        )
        emitted: list[Finding] = []
        day = date(2026, 1, 20)
        while day <= date(2026, 4, 10):
            ctx.clock._now = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            emitted.extend(agent.on_day_boundary(day, ctx))
            day += timedelta(days=1)
        writer.close()

        feb = [f for f in emitted if f.value.get("month") == "2026-02"]
        check(f"dom={dom}: February absence reported", len(feb) == 1, str([f.value.get("month") for f in emitted]))
        check(
            f"dom={dom}: reported once grace elapsed ({expect_on})",
            feb[0].updated_at.date() == expect_on,
            f"got {feb[0].updated_at.date()}",
        )
        check(f"dom={dom}: reported once per month, not daily", len(emitted) == len({f.value["month"] for f in emitted}))
        check(f"dom={dom}: grace is {T.SI_GRACE_DAYS} days", feb[0].value["grace_days"] == T.SI_GRACE_DAYS)
        check(f"dom={dom}: absence cites the expectation, not nothing", feb[0].evidence_event_ids != [])


def test_summarise_keeps_zeros() -> None:
    """`v in (None, False, ...)` uses ==, and 0 == False -- a zero count must
    survive into the log line, since zero logins is the whole signal."""
    print("log summary keeps zeros")
    from agents.base import _summarise

    text = _summarise({"logins": 0, "ratio": 0.0, "stopped": False, "note": None, "empty": []})
    check("zero int survives", "logins=0" in text, text)
    check("zero float survives", "ratio=0.0" in text or "ratio=0" in text, text)
    check("False is omitted", "stopped" not in text, text)
    check("None is omitted", "note" not in text, text)
    check("empty list is omitted", "empty" not in text, text)


def test_consent_gate(tmp: Path) -> None:
    """No shipped scenario contains a social_signal_consented event, so the
    consent gate would otherwise never be exercised. It is a safety rule."""
    print("consent gate")
    from agents import SupportAgent

    ctx, writer, derive = _minimal_ctx(tmp)
    agent = SupportAgent()

    unconsented = derive(
        _fake_event(
            "life_event_mention", "social_signal_consented", "EVT_NOCONSENT",
            platform="x", raw_text="We just had a baby!", consent_flag=False,
        ),
        ctx.profile,
    )
    out = agent.on_event(unconsented, ctx)
    check("unconsented social signal writes no findings", out == [])
    check("unconsented social signal is counted", agent.skipped_unconsented == 1)

    consented = derive(
        _fake_event(
            "life_event_mention", "social_signal_consented", "EVT_CONSENT",
            platform="x", raw_text="We just had a baby, so happy!", consent_flag=True,
        ),
        ctx.profile,
    )
    out2 = agent.on_event(consented, ctx)
    keys = {f.key for f in out2}
    check("consented social signal is analysed", "sentiment_state" in keys, str(keys))
    check("consented social signal yields a life_event_hint", "life_event_hint" in keys, str(keys))
    hint = next(f for f in out2 if f.key == "life_event_hint")
    check("hint label is new_child_life_event", "new_child_life_event" in hint.value["labels"], str(hint.value))
    check("hint cites its event", hint.evidence_event_ids == ["EVT_CONSENT"])

    writer.close()
    logged = writer.path.read_text()
    check("the skip is logged for traceability", "consent_flag is not true" in logged)
    check("skipped event's text never reaches the log", "had a baby" not in logged.split("consent_flag is not true")[0])


# --- the acceptance run -----------------------------------------------------


def test_full_run(tmp: Path) -> list[Finding]:
    print(f"full run on {SCENARIO}")
    result = run_one(SCENARIO, None, tmp, realtime=False, fresh=True, show_findings=False)
    trace: list[Finding] = result["trace"]
    check("all live events replayed", result["events"] == result["expected_events"])
    check("findings were emitted", len(trace) > 0)
    check("every finding carries evidence", all(f.evidence_event_ids for f in trace))
    check("every finding has a simulated timestamp", all(f.updated_at is not None for f in trace))

    # 1. sentiment_state from EVT_000409: negative, fee_dispute
    s409 = citing(trace, "sentiment_state", "EVT_000409")
    check("sentiment_state exists for EVT_000409", len(s409) == 1, str(len(s409)))
    check("EVT_000409 sentiment is negative", s409[0].value["sentiment"] == "negative", str(s409[0].value))
    check("EVT_000409 intent is fee_dispute", s409[0].value["intent"] == "fee_dispute", str(s409[0].value))

    # 2. relationship_damage with evidence EVT_000412
    rd = citing(trace, "relationship_damage", "EVT_000412")
    check("relationship_damage cites EVT_000412", len(rd) == 1, str(len(rd)))
    check("relationship_damage outcome unfavorable", rd[0].value["resolution_outcome"] == "unfavorable")

    # 3. large_inflow with evidence EVT_000447, transaction_type tax_refund
    li = citing(trace, "large_inflow", "EVT_000447")
    check("large_inflow cites EVT_000447", len(li) == 1, str(len(li)))
    check("large_inflow transaction_type is tax_refund", li[0].value["transaction_type"] == "tax_refund", str(li[0].value))
    check("large_inflow does not interpret the credit", "windfall" not in str(li[0].value).lower())

    # 4. standing_instruction_change with evidence EVT_000457
    si = citing(trace, "standing_instruction_change", "EVT_000457")
    check("standing_instruction_change cites EVT_000457", len(si) == 1, str(len(si)))
    check("standing_instruction_change kind is cancel_page_used", si[0].value["kind"] == "cancel_page_used")

    # 5. outflow_pattern for EVT_000461 with both counterparty flags
    of = citing(trace, "outflow_pattern", "EVT_000461")
    check("outflow_pattern cites EVT_000461", len(of) >= 1, str(len(of)))
    large = [f for f in of if f.value.get("kind") == "large_outflow"]
    check("EVT_000461 recorded as large_outflow", len(large) == 1)
    check("EVT_000461 counterparty_is_self is true", large[0].value["counterparty_is_self"] is True)
    check("EVT_000461 counterparty_is_external_bank is true", large[0].value["counterparty_is_external_bank"] is True)

    # 6. a salary-sweep outflow_pattern citing EVT_000469
    sweeps = [f for f in of_key(trace, "outflow_pattern") if f.value.get("kind") == "salary_sweep"]
    sweep469 = [f for f in sweeps if "EVT_000469" in f.evidence_event_ids]
    check("salary_sweep outflow_pattern cites EVT_000469", len(sweep469) == 1, str([f.evidence_event_ids for f in sweeps]))
    check("sweep is within the 24h window", sweep469[0].value["hours_after_salary"] <= 24)
    check("sweep cites the salary credit too", "EVT_000468" in sweep469[0].evidence_event_ids)

    # 7. card_activity_trend reaches stopped on or before 2026-03-17
    stopped = first_reaching(trace, "card_activity_trend", lambda f: f.value.get("stopped") is True)
    check("card_activity_trend reaches stopped", stopped is not None)
    check(
        "card usage stopped on or before 2026-03-17",
        stopped.updated_at.date() <= date(2026, 3, 17),
        f"got {stopped.updated_at.date()}",
    )

    # 8. engagement_trend reaches medium or high before 2026-03-08
    eng = first_reaching(trace, "engagement_trend", lambda f: f.band_rank >= 1)
    check("engagement_trend reaches medium or high", eng is not None)
    check(
        "engagement drop detected before 2026-03-08",
        eng.updated_at.date() < date(2026, 3, 8),
        f"got {eng.updated_at.date()} band={eng.confidence_band}",
    )

    # --- discipline checks: agents must not decide states or actions --------
    from c360.memory_store import MemoryStore

    mem = MemoryStore(load_scenario(SCENARIO).customer_id, memory_dir=tmp / "memory").load()
    check("no agent set a hypothesis", mem.working.get_hypothesis() is None)
    forbidden = {"inferred_state", "action", "hitl_status"}
    check(
        "no finding value contains a decision field",
        all(forbidden.isdisjoint(f.value.keys()) for f in trace),
    )
    check("episodes were opened for medium/high findings", len(mem.episodic) > 0)
    check(
        "a support_ticket episode was closed with its outcome",
        any(ep.type == "support_ticket" and ep.outcome for ep in mem.episodic.query()),
    )
    llm_trace = tmp / "logs" / f"{SCENARIO}_llm_trace.jsonl"
    check("llm trace written even when offline", llm_trace.exists())
    check("no raw customer name in the llm trace", "David Chen" not in llm_trace.read_text())
    return trace


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c360_agents_test_"))
    try:
        test_derive_and_mask()
        test_baseline()
        test_event_store()
        test_support_validation()
        test_summarise_keeps_zeros()
        test_si_absence_month_arithmetic(tmp)
        test_consent_gate(tmp)
        trace = test_full_run(tmp)
        print_findings(trace)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
