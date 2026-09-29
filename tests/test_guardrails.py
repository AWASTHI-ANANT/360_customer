#!/usr/bin/env python3
"""Guardrails, output veto and tracing. Offline, no API key.

    python -m tests.test_guardrails

Regression: on the three real scenarios the guardrail never fires, and the
checkpoints are byte-identical with GUARDRAILS_ENABLED on and off. That stays
true after the evidence weights are retuned, unlike a comparison with a
recorded baseline.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ["C360_LLM_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.guardrail_rules import match_rules  # noqa: E402
from output.checkpoint_writer import CheckpointWriter  # noqa: E402
from run_skeleton import run_one  # noqa: E402
from tracing import trace_view  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
SCENARIOS = ("scenario_01", "scenario_02", "scenario_03")
NAMES = ("Marcus Vance", "Priya Sharma", "David Chen")
PASSED: list[str] = []
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name} -- {detail}")
        print(f"  FAIL {name}  {detail}")


def _run(tmp: Path, tag: str, scenario: str, **kw) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return run_one(scenario, None, tmp / tag, fresh=True,
                       checkpoint_dir=tmp / tag / "output", **kw)


def test_rules() -> None:
    print("rules")
    fires = lambda t: [r["rule_id"] for r in match_rules(t)]  # noqa: E731
    for text in ("I will sue you", "my lawyer will call", "we are taking legal action",
                 "expect a lawsuit", "My Attorney has the letter"):
        check(f"LEGAL_THREAT fires: {text!r}", "LEGAL_THREAT" in fires(text), str(fires(text)))
    for text in ("I have power of attorney for my mother",
                 "Adding a Power of Attorney to the account", "there is an issue with my fee",
                 "please pursue a refund"):
        check(f"LEGAL_THREAT silent: {text!r}", "LEGAL_THREAT" not in fires(text), str(fires(text)))
    for text in ("I didn't make this purchase", "an unauthorized charge", "my card was stolen",
                 "stolen credit card", "my account was hacked", "I did not authorize this"):
        check(f"CUSTOMER_FRAUD_REPORT fires: {text!r}",
              "CUSTOMER_FRAUD_REPORT" in fires(text), str(fires(text)))
    for text in ("I made this purchase myself", "my account was not hacked, I just forgot",
                 "Just confirming it's me, I bought a new video baby monitor online."):
        check(f"CUSTOMER_FRAUD_REPORT silent: {text!r}",
              "CUSTOMER_FRAUD_REPORT" not in fires(text), str(fires(text)))
    check("large amounts alone are not a guardrail", not fires("I just spent $9,000 on a TV"))


def test_regression(tmp: Path) -> None:
    print("regression: real scenarios, GUARDRAILS_ENABLED on vs off")
    for s in SCENARIOS:
        on = _run(tmp, f"on_{s}", s, guardrails=True)
        off = _run(tmp, f"off_{s}", s, guardrails=False)
        check(f"{s}: zero guardrail fires", not on["guardrail_fires"], str(on["guardrail_fires"]))
        check(f"{s}: zero vetoes", not on["vetoes"], str(on["vetoes"]))
        a = (tmp / f"on_{s}" / "output" / f"{s}_checkpoints.json").read_bytes()
        b = (tmp / f"off_{s}" / "output" / f"{s}_checkpoints.json").read_bytes()
        check(f"{s}: checkpoints byte-identical on vs off", a == b)


def test_injected(tmp: Path) -> None:
    print("injected events (scenario_03)")
    legal = _run(tmp, "legal", "scenario_03", inject=FIXTURES / "guardrail_legal.jsonl",
                 guardrails=True)
    fired = [f["rule_id"] for f in legal["guardrail_fires"]]
    check("legal threat fires LEGAL_THREAT", fired == ["LEGAL_THREAT"], str(fired))
    cps = legal["checkpoints"]
    now = next((c for c in cps if c["as_of_time"] == "2026-03-10T10:30:00Z"), None)
    check("immediate checkpoint at the event's time", now is not None,
          str([c["as_of_time"] for c in cps if c["as_of_time"].startswith("2026-03-10")]))
    if now:
        check("legal: relationship_manager_escalation",
              now["action"] == "relationship_manager_escalation", now["action"])
        check("legal: subtype legal_escalation_queue",
              now["action_subtype"] == "legal_escalation_queue", str(now["action_subtype"]))
        check("legal: hitl escalated", now["hitl_status"] == "escalated", now["hitl_status"])
    later = [c for c in cps if c["as_of_time"] > "2026-03-10T10:30:00Z"]
    check("legal: no offer/outreach after the hold",
          all(c["action"] not in ("personalized_offer", "proactive_retention_outreach") for c in later),
          str(sorted({c["action"] for c in later})))
    times = [c["as_of_time"] for c in cps]
    check("as_of_time still strictly increasing", all(x < y for x, y in zip(times, times[1:])))
    mem = json.loads((tmp / "legal" / "memory" / "CUST_00184_working.json").read_text())
    decision = mem["findings"]["current_decision"]["value"]
    check("legal: no customer message drafted", not (decision.get("draft") or {}).get("customer_message"))
    eps = [json.loads(l) for l in (tmp / "legal" / "memory" / "CUST_00184_episodic.jsonl").open()]
    check("legal: guardrail_hold episode written",
          any(e.get("type") == "guardrail_hold" for e in eps))

    poa = _run(tmp, "poa", "scenario_03", inject=FIXTURES / "guardrail_poa.jsonl", guardrails=True)
    check("power of attorney does not fire", not poa["guardrail_fires"], str(poa["guardrail_fires"]))

    fraud = _run(tmp, "fraud", "scenario_03", inject=FIXTURES / "guardrail_fraud.jsonl",
                 guardrails=True)
    fired = [f["rule_id"] for f in fraud["guardrail_fires"]]
    check("fraud report fires CUSTOMER_FRAUD_REPORT", fired == ["CUSTOMER_FRAUD_REPORT"], str(fired))
    now = next((c for c in fraud["checkpoints"] if c["as_of_time"] == "2026-03-10T10:30:00Z"), None)
    check("fraud: compliance_fraud_hold now", bool(now) and now["action"] == "compliance_fraud_hold",
          str(now and now["action"]))
    check("fraud: state potential_fraud_or_takeover",
          bool(now) and now["inferred_state"] == "potential_fraud_or_takeover",
          str(now and now["inferred_state"]))

    off = _run(tmp, "legal_off", "scenario_03", inject=FIXTURES / "guardrail_legal.jsonl",
               guardrails=False)
    check("GUARDRAILS_ENABLED=False: legal threat does not fire", not off["guardrail_fires"])


def test_veto() -> None:
    print("output veto")
    hold = SimpleNamespace(value={"freeze_outbound": True, "forced_action":
                                  "relationship_manager_escalation",
                                  "forced_subtype": "legal_escalation_queue"})
    cp = {"action": "personalized_offer", "action_subtype": "x", "hitl_status": "escalated"}
    decision = {"draft": {"customer_message": "Hi there, a special offer..."}}
    reasons = CheckpointWriter.veto(cp, decision, hold)
    check("frozen: offer blocked", cp["action"] == "relationship_manager_escalation", cp["action"])
    check("frozen: customer message stripped", decision["draft"]["customer_message"] is None)
    check("frozen: reasons recorded", len(reasons) == 2, str(reasons))
    cp = {"action": "support_intervention", "action_subtype": None, "hitl_status": "auto_approved"}
    CheckpointWriter.veto(cp, {}, None)
    check("non-no_action never auto_approved", cp["hitl_status"] == "escalated", cp["hitl_status"])
    cp = {"action": "no_action", "action_subtype": None, "hitl_status": "auto_approved"}
    check("no_action untouched", CheckpointWriter.veto(cp, {}, None) == [])


def test_tracing(tmp: Path) -> None:
    print("tracing")
    path = tmp / "on_scenario_03" / "logs" / "scenario_03_trace.jsonl"
    check("trace file written", path.exists())
    if not path.exists():
        return
    spans = [json.loads(l) for l in path.open()]
    ids = {s["span_id"] for s in spans}
    check("every parent_span_id exists", all(s["parent_span_id"] in ids for s in spans
                                             if s["parent_span_id"]))
    fields = {"trace_id", "span_id", "parent_span_id", "name", "kind", "sim_time",
              "latency_ms", "inputs", "outputs", "status"}
    check("every span has the required fields", all(fields <= s.keys() for s in spans))
    kinds = {s["kind"] for s in spans}
    for k in ("day", "event", "agent", "checkpoint", "retrieval", "hitl"):
        check(f"span kind {k!r} present", k in kinds, str(sorted(kinds)))
    text = path.read_text()
    check("no customer name or ACC_ in the trace",
          not any(n.lower() in text.lower() for n in NAMES) and "ACC_" not in text)
    lines = trace_view.render(spans, "2026-03-07")
    check("trace_view renders 2026-03-07 with the action's retrieval + hitl",
          any("retrieve" in l for l in lines) and any("request_approval" in l for l in lines))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c360_guard_"))
    try:
        test_rules()
        test_veto()
        test_regression(tmp)
        test_injected(tmp)
        test_tracing(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} checks passed, {len(FAILURES)} failed")
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
