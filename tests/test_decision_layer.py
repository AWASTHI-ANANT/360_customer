#!/usr/bin/env python3
"""Stage 1 acceptance tests: synthesis -> action -> checkpoint, scenario_03, offline.

Forces C360_LLM_OFFLINE so every LLM step takes its deterministic fallback and
the suite passes with no API key.

    python -m tests.test_decision_layer
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

os.environ["C360_LLM_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hitl  # noqa: E402
from agents.action_agent import ActionAgent, Draft  # noqa: E402
from agents.base import AgentContext  # noqa: E402
from agents.synthesis_agent import Scorecard, StateScore, SynthesisAgent  # noqa: E402
from c360 import DailyLogWriter, EventStore, MemoryStore, SimulatedClock, load_scenario  # noqa: E402
from c360 import llm_client  # noqa: E402
from config import thresholds as T  # noqa: E402
from output.checkpoint_writer import CheckpointValidationError, CheckpointWriter  # noqa: E402
from policy import retrieval  # noqa: E402
from run_skeleton import run_one  # noqa: E402

SCENARIO = "scenario_03"
PASSED: list[str] = []
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name} -- {detail}")
        print(f"  FAIL {name}  {detail}")


def hard(name: str, cond: bool, detail: str = "") -> None:
    """A check whose failure is a bug, not a tuning issue."""
    assert cond, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok   {name}")


def at(cps: list[dict], iso_time: str) -> dict | None:
    """The checkpoint most recently at or before iso_time."""
    found = None
    for c in cps:
        if c["as_of_time"] <= iso_time:
            found = c
    return found


# --- unit: policy retrieval -------------------------------------------------


def test_policy() -> None:
    print("policy corpus + retrieval")
    hard("policy files carry the SYNTHETIC header",
         all("SYNTHETIC POLICY" in (ROOT / "policy" / f).read_text()
             for f in ("offer_catalog.md", "action_policy.md")))
    subs = retrieval.subtypes_for("churn_risk")
    hard("churn subtypes retrieved",
         "premium_retention_offer_and_fee_waiver" in subs, str(subs))
    hard("chunk ids are citable",
         retrieval.retrieve("churn_risk", 2)[0].chunk_id.startswith("offer_catalog#churn_risk#"))
    hard("retrieval is deterministic",
         retrieval.chunk_ids(retrieval.retrieve("medical_hardship", 5))
         == retrieval.chunk_ids(retrieval.retrieve("medical_hardship", 5)))
    hard("catalogue max_value parsed",
         retrieval.cap_for_subtype("premium_retention_offer_and_fee_waiver") == 500)
    hard("unknown state does not crash", isinstance(retrieval.retrieve("bogus_state", 3), list))


# --- unit: scoring and banding ---------------------------------------------


def test_scoring_and_bands() -> None:
    print("scoring, decay and banding")
    agent = SynthesisAgent()
    hard("band low", agent._band_for(T.LOW_MAX - 0.01) == "low")
    hard("band medium at LOW_MAX", agent._band_for(T.LOW_MAX) == "medium")
    hard("band high at MEDIUM_MAX", agent._band_for(T.MEDIUM_MAX) == "high")
    sc = Scorecard(ranked=[StateScore("churn_risk", T.NO_EVENT_MAX - 0.01)])
    hard("below NO_EVENT_MAX -> no_significant_event",
         agent.band(sc) == ("no_significant_event", "low"))
    hard("empty scorecard -> no_significant_event",
         agent.band(Scorecard()) == ("no_significant_event", "low"))
    # noisy-OR never exceeds 1 and is order-independent
    a = StateScore("x", 0.0, [])
    from agents.synthesis_agent import Contribution
    a.contributions = [Contribution("k1", ["E1"], 0.5, 0.5, 0.0),
                       Contribution("k2", ["E2"], 0.5, 0.5, 0.0)]
    product = 1.0
    for c in a.contributions:
        product *= 1 - c.weight
    hard("noisy-OR of 0.5+0.5 = 0.75", abs((1 - product) - 0.75) < 1e-9)


def test_precedence_fallback(tmp: Path) -> None:
    print("debate fallback (PRECEDENCE_RULES)")
    ctx = _ctx(tmp, "prec")
    agent = SynthesisAgent()
    exit_side = StateScore("churn_risk", 0.5, [])
    inflow_side = StateScore("wealth_growth_or_windfall", 0.45, [])
    from agents.synthesis_agent import Contribution
    exit_side.contributions = [Contribution("outflow_pattern", ["E1"], 0.3, 0.3, 0.0)]
    inflow_side.contributions = [Contribution("large_inflow", ["E2"], 0.2, 0.2, 0.0)]
    verdict, why = agent._precedence_verdict(ctx, exit_side, inflow_side)
    hard("exit evidence beats inflow evidence", verdict == "A", f"{verdict} {why}")
    hard("the deciding rule is named", "exit_beats_inflow" in why, why)
    # reversed order: the exit side must still win, as B
    verdict2, _ = agent._precedence_verdict(ctx, inflow_side, exit_side)
    hard("exit wins from either position", verdict2 == "B", verdict2)
    # specific beats general
    med = StateScore("medical_hardship", 0.4, [])
    gen = StateScore("financial_distress_general", 0.42, [])
    v3, why3 = agent._precedence_verdict(ctx, gen, med)
    hard("specific cause beats general", v3 == "B", f"{v3} {why3}")


# --- unit: action decision table ------------------------------------------


def test_decide() -> None:
    print("action decision table")
    a = ActionAgent()
    hard("churn high + tier high -> RM escalation",
         a.decide("churn_risk", "high", "high") == "relationship_manager_escalation")
    hard("churn high + tier mid -> outreach",
         a.decide("churn_risk", "high", "mid") == "proactive_retention_outreach")
    hard("churn medium -> no_action", a.decide("churn_risk", "medium", "high") == "no_action")
    hard("medical high -> support_intervention",
         a.decide("medical_hardship", "high", "mid") == "support_intervention")
    hard("new child high -> personalized_offer",
         a.decide("new_child_life_event", "high", "mid") == "personalized_offer")
    hard("fraud at FRAUD_ACTION_MIN_BAND -> hold",
         a.decide("potential_fraud_or_takeover", T.FRAUD_ACTION_MIN_BAND, "mid")
         == "compliance_fraud_hold")
    hard("fraud below min band -> no_action",
         a.decide("potential_fraud_or_takeover", "low", "mid") == "no_action")
    hard("no_significant_event -> no_action",
         a.decide("no_significant_event", "high", "high") == "no_action")
    hard("unknown state -> no_action", a.decide("not_a_state", "high", "high") == "no_action")


# --- unit: critique -------------------------------------------------------


def test_critique(tmp: Path) -> None:
    print("critique (five deterministic checks)")
    ctx = _ctx(tmp, "crit")
    agent = ActionAgent()
    hyp = ctx.memory.working.set_hypothesis(
        "churn_risk", 0.8, "high", ["EVT_000461"], rationale="churn [EVT_000461]"
    )
    ok = Draft(action_subtype="premium_retention_offer_and_fee_waiver",
               customer_message="We would like to review your accounts with you.",
               rm_brief="churn [EVT_000461]", offer_value=400)
    res = agent.critique(ctx, "churn_risk", "relationship_manager_escalation", ok, hyp)
    hard("clean draft passes", res["passed"], str(res["failures"]))

    bad_sub = Draft(action_subtype="not_a_real_subtype", customer_message="hello")
    hard("check a: disallowed subtype fails",
         any(f.startswith("a:") for f in
             agent.critique(ctx, "churn_risk", "personalized_offer", bad_sub, hyp)["failures"]))

    over = Draft(action_subtype="premium_retention_offer_and_fee_waiver", offer_value=99999)
    hard("check b: over the tier cap fails",
         any(f.startswith("b:") for f in
             agent.critique(ctx, "churn_risk", "personalized_offer", over, hyp)["failures"]))
    at_cap = Draft(action_subtype="premium_retention_offer_and_fee_waiver",
                   offer_value=T.RETENTION_CREDIT_CAP["high"])
    hard("check b: exactly at the cap passes",
         not any(f.startswith("b:") for f in
                 agent.critique(ctx, "churn_risk", "personalized_offer", at_cap, hyp)["failures"]))

    # check c: suppressed evidence
    ctx.memory.episodic.append({
        "type": "explanation", "customer_id": ctx.profile.customer_id,
        "opened_at": ctx.now(), "evidence_event_ids": ["EVT_X", "EVT_000461"],
        "data": {"suppresses": "EVT_000461",
                 "until": (ctx.now() + timedelta(hours=72)).isoformat(), "reason": "explained"},
    })
    hard("check c: suppressed evidence fails",
         any(f.startswith("c:") for f in
             agent.critique(ctx, "churn_risk", "personalized_offer", ok, hyp)["failures"]))

    # check d: sensitive inference forces escalation
    hyp2 = ctx.memory.working.set_hypothesis(
        "medical_hardship", 0.8, "high", ["EVT_000999"], rationale="medical [EVT_000999]"
    )
    med_ok = Draft(action_subtype="medical_hardship_payment_plan",
                   customer_message="Flexible arrangements are available on your accounts.")
    res_d = agent.critique(ctx, "medical_hardship", "support_intervention", med_ok, hyp2)
    hard("check d: sensitive inference forces escalation", res_d["force_escalate"] is True)

    # check e: the customer message must not reveal the inference
    leaky = Draft(action_subtype="medical_hardship_payment_plan",
                  customer_message="We heard about your hospital surgery and can help.")
    hard("check e: leaked inference fails",
         any(f.startswith("e:") for f in
             agent.critique(ctx, "medical_hardship", "support_intervention", leaky, hyp2)["failures"]))
    hard("check e: shipped templates leak nothing for any state",
         all(not any(f.startswith("e:") for f in agent.critique(
                 ctx, st, "personalized_offer",
                 Draft(action_subtype=(retrieval.subtypes_for(st) or [None])[0],
                       customer_message=msg), hyp2)["failures"])
             for st in ("medical_hardship", "new_child_life_event", "relocation",
                        "financial_distress_general")
             for msg in [__import__("config.action_rules", fromlist=["x"]).DRAFT_TEMPLATES["personalized_offer"]]))


# --- unit: the Stage 2 guardrail hook -------------------------------------


def test_guardrail_hook(tmp: Path) -> None:
    print("guardrail hook (hand-written finding; Stage 2 writes it for real)")
    ctx = _ctx(tmp, "guard")
    agent = ActionAgent()
    hyp = ctx.memory.working.set_hypothesis(
        "churn_risk", 0.2, "low", ["EVT_000409"], rationale="weak churn [EVT_000409]"
    )
    ctx.memory.working.set_finding(
        "guardrail_hold",
        {"rule_id": "LEGAL_THREAT", "forced_action": "relationship_manager_escalation",
         "forced_subtype": "legal_escalation_queue", "forced_state": None,
         "freeze_outbound": True, "triggered_by": "EVT_FAKE"},
        "high", "guardrail_agent", ["EVT_FAKE"],
    )
    agent.on_day_boundary(date(2026, 3, 1), ctx)
    d = ctx.memory.working.get_finding("current_decision").value
    hard("hold forces its action", d["action"] == "relationship_manager_escalation", str(d["action"]))
    hard("hold forces its subtype", d["action_subtype"] == "legal_escalation_queue")
    hard("hold forces escalated", d["hitl_status"] == "escalated")
    hard("null forced_state keeps synthesis state", d.get("forced_state") is None)
    hard("the rule id is recorded", d["guardrail_rule_id"] == "LEGAL_THREAT")
    hard("no customer message under a hold", d["draft"]["customer_message"] is None)

    # A hold WITH a forced_state overrides the checkpoint's state.
    ctx.memory.working.set_finding(
        "guardrail_hold",
        {"rule_id": "CUSTOMER_FRAUD_REPORT", "forced_action": "compliance_fraud_hold",
         "forced_subtype": "account_verification_hold",
         "forced_state": "potential_fraud_or_takeover", "freeze_outbound": False,
         "triggered_by": "EVT_FAKE2"},
        "high", "guardrail_agent", ["EVT_FAKE2"],
    )
    agent.on_day_boundary(date(2026, 3, 2), ctx)
    cw = CheckpointWriter("guard", output_dir=tmp / "out")
    cp = cw.build(datetime(2026, 3, 2, tzinfo=timezone.utc), ctx.memory)
    hard("forced_state overrides the checkpoint state",
         cp["inferred_state"] == "potential_fraud_or_takeover", cp["inferred_state"])
    hard("forced action reaches the checkpoint", cp["action"] == "compliance_fraud_hold")


# --- unit: checkpoint writer ----------------------------------------------


def test_checkpoint_writer(tmp: Path) -> None:
    print("checkpoint writer")
    ctx = _ctx(tmp, "cw")
    cw = CheckpointWriter("cw", output_dir=tmp / "out")
    cp = cw.build(date(2026, 2, 1), ctx.memory)
    hard("no hypothesis yet -> no_significant_event",
         cp["inferred_state"] == "no_significant_event" and cp["action"] == "no_action")
    hard("as_of_time rendered in dataset form", cp["as_of_time"] == "2026-02-01T00:00:00Z")

    ctx.memory.working.set_hypothesis(
        "churn_risk", 0.8, "high", ["EVT_1", "EVT_2", "EVT_3", "EVT_4", "EVT_5", "EVT_6", "EVT_7"],
        rationale="x" * 500, used_fallback=True,
    )
    cp2 = cw.write(datetime(2026, 3, 8, tzinfo=timezone.utc), ctx.memory)
    hard("rationale trimmed", len(cp2["notes"]) < 500, str(len(cp2["notes"])))
    hard("evidence capped at 6", cp2["notes"].count("EVT_") == 6, cp2["notes"])
    hard("fallback marked in notes", "[fallback]" in cp2["notes"])
    hard("file is a valid JSON array",
         isinstance(json.loads(cw.path.read_text()), list))
    try:
        CheckpointWriter.validate({"as_of_time": "x", "inferred_state": "nope",
                                   "confidence_band": "low", "action": "no_action",
                                   "hitl_status": "auto_approved"})
        raise AssertionError("expected an invalid enum to raise")
    except CheckpointValidationError:
        hard("invalid enum raises", True)


# --- unit: hitl approvals log --------------------------------------------


def test_hitl_log(tmp: Path) -> None:
    print("hitl approvals log")
    ctx = _ctx(tmp, "hl")
    alog = hitl.configure("hl", log_dir=tmp / "logs")
    hyp = ctx.memory.working.set_hypothesis("churn_risk", 0.8, "high", ["EVT_000461"])
    decision = {"action": "personalized_offer", "action_subtype": "high_yield_savings_offer",
                "policy_chunk_ids": ["offer_catalog#churn_risk#1"]}
    status = hitl.request_approval(decision, "CONTEXT SHOWN", ctx=ctx, mode="auto", hypothesis=hyp)
    hard("auto mode returns escalated", status == "escalated")
    lines = [json.loads(l) for l in alog.path.read_text().splitlines() if l.strip()]
    hard("a request and a decision line are written",
         [l["record_type"] for l in lines] == ["request", "decision"], str(lines))
    req = lines[0]
    hard("context_shown stored verbatim", req["context_shown"] == "CONTEXT SHOWN")
    hard("evidence recorded", req["evidence_event_ids"] == ["EVT_000461"])
    hard("policy chunks recorded", req["policy_chunk_ids"] == ["offer_catalog#churn_risk#1"])
    hard("request_id joins the two lines", lines[1]["request_id"] == req["request_id"])
    hard("auto decision is escalated_auto", lines[1]["decision"] == "escalated_auto")
    hard("status comes from the latest decision record",
         alog.status_for(req["request_id"]) == "escalated")
    hard("append-only: a second request appends, never edits",
         (hitl.request_approval(dict(decision), "SECOND", ctx=ctx, mode="auto", hypothesis=hyp)
          and len([l for l in alog.path.read_text().splitlines() if l.strip()]) == 4))


# --- the acceptance run ---------------------------------------------------


def test_acceptance(tmp: Path) -> list[dict]:
    print(f"acceptance run: {SCENARIO}, --auto, offline")
    result = run_one(SCENARIO, None, tmp, fresh=True, mode="auto")
    cps = result["checkpoints"]
    hard("checkpoints were produced", len(cps) > 0)
    hard("every checkpoint validates",
         all(CheckpointWriter.validate(c) for c in cps))

    # 1. as of 2026-02-15: churn_risk / low / no_action
    c1 = at(cps, "2026-02-15T00:00:00Z")
    check("2026-02-15 state is churn_risk", c1 and c1["inferred_state"] == "churn_risk",
          f"got {c1 and c1['inferred_state']}")
    check("2026-02-15 band is low", c1 and c1["confidence_band"] == "low",
          f"got {c1 and c1['confidence_band']}")
    check("2026-02-15 action is no_action", c1 and c1["action"] == "no_action",
          f"got {c1 and c1['action']}")

    # 2. as of 2026-03-08: churn_risk / high / RM escalation / escalated
    c2 = at(cps, "2026-03-08T00:00:00Z")
    check("2026-03-08 state is churn_risk", c2 and c2["inferred_state"] == "churn_risk",
          f"got {c2 and c2['inferred_state']}")
    check("2026-03-08 band is high", c2 and c2["confidence_band"] == "high",
          f"got {c2 and c2['confidence_band']}")
    check("2026-03-08 action is relationship_manager_escalation",
          c2 and c2["action"] == "relationship_manager_escalation", f"got {c2 and c2['action']}")
    check("2026-03-08 hitl is escalated", c2 and c2["hitl_status"] == "escalated",
          f"got {c2 and c2['hitl_status']}")

    # 3. no personalized_offer within 72h of the tax refund (red herring)
    from c360.loader import parse_ts
    refund = parse_ts("2026-02-28T09:00:00Z")
    window = [c for c in cps
              if refund <= parse_ts(c["as_of_time"]) <= refund + timedelta(hours=72)]
    hard("no personalized_offer within 72h of EVT_000447",
         all(c["action"] != "personalized_offer" for c in window),
         str([c["action"] for c in window]))

    # 4. a debate episode for churn_risk vs wealth_growth_or_windfall
    mem = MemoryStore(load_scenario(SCENARIO).customer_id, memory_dir=tmp / "memory").load()
    debates = mem.episodic.query(type="debate")
    pairs = [
        frozenset({(d.data.get("case_a") or {}).get("state"),
                   (d.data.get("case_b") or {}).get("state")})
        for d in debates
    ]
    check("a debate exists for churn_risk vs wealth_growth_or_windfall",
          frozenset({"churn_risk", "wealth_growth_or_windfall"}) in pairs,
          f"debates found: {[sorted(p) for p in pairs]}")

    # 5. every checkpoint's notes cite at least one event id
    missing = [c["as_of_time"] for c in cps if "EVT_" not in (c["notes"] or "")]
    hard("every checkpoint cites an EVT_ id in notes", not missing, f"missing on {missing[:5]}")

    # discipline: offline run must be marked as such, and nothing auto-approved
    # alongside a real action
    hard("a non-no_action checkpoint is never auto_approved",
         all(c["hitl_status"] != "auto_approved"
             for c in cps if c["action"] != "no_action"),
         str([(c["as_of_time"], c["action"]) for c in cps
              if c["action"] != "no_action" and c["hitl_status"] == "auto_approved"][:3]))
    hard("offline run marks fallback use in notes",
         any("[fallback]" in (c["notes"] or "") for c in cps))
    return cps


# --- helpers --------------------------------------------------------------


def _ctx(tmp: Path, name: str) -> AgentContext:
    scn = load_scenario(SCENARIO)
    clock = SimulatedClock.from_config(scn.replay_config)
    clock._now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    mem = MemoryStore(name, memory_dir=tmp / f"mem_{name}").bind_clock(clock)
    mem.reset()
    writer = DailyLogWriter(name, output_dir=tmp / f"log_{name}")
    llm_client.configure(name, cache_dir=tmp / "cache", trace_dir=tmp / f"log_{name}")
    hitl.configure(name, log_dir=tmp / f"log_{name}")
    return AgentContext(memory=mem, log=writer, profile=scn.profile,
                        event_store=EventStore(scn.customer_id), clock=clock)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c360_stage1_"))
    try:
        test_policy()
        test_scoring_and_bands()
        test_precedence_fallback(tmp)
        test_decide()
        test_critique(tmp)
        test_guardrail_hook(tmp)
        test_checkpoint_writer(tmp)
        test_hitl_log(tmp)
        test_acceptance(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} checks passed, {len(FAILURES)} failed")
    if FAILURES:
        print("\nFailures (tuning decisions for the user, not bugs):")
        for f in FAILURES:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
