"""Human-in-the-loop approval. Basic version; Stage 2 extends the log format.

request_approval(decision, context_shown) -> hitl_status

--auto mode returns "escalated" without asking anyone: nothing reaches a
customer in an unattended run. --interactive prints the decision and reads a
keystroke.

Every request and every answer is appended to
logs/{scenario_id}_approvals.jsonl, append-only, never edited. `context_shown`
is stored verbatim (and already masked by the caller) so a reviewer's decision
can be audited against exactly what they saw.

ASSUMPTIONS (spec was silent):
- request_id is "{scenario_id}:{customer_id}:{n:04d}", n counting requests in
  this run. It is the join key between the request and decision lines.
- In auto mode a single "request" line and a single "decision" line with
  decision="escalated_auto" are both written, so the log shape is identical in
  both modes and metrics can count them uniformly.
- reviewer comes from env REVIEWER_NAME, else "cli_user".
- seconds_to_decide is wall-clock, 0.0 in auto mode.
- On EOF or KeyboardInterrupt at the prompt, the answer is treated as
  "escalated" rather than approving anything by accident.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

APPROVAL_DECISIONS = (
    "human_approved",
    "human_rejected",
    "human_modified",
    "escalated_auto",
    "hold_cleared",
)

# Maps a recorded decision to the README hitl_status enum.
_STATUS_FOR = {
    "human_approved": "human_approved",
    "human_rejected": "human_rejected",
    "human_modified": "human_modified",
    "escalated_auto": "escalated",
    "hold_cleared": "escalated",
}


class ApprovalLog:
    """Append-only approvals log for one scenario."""

    def __init__(self, scenario_id: str, log_dir: str | Path = "out/logs") -> None:
        self.scenario_id = scenario_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{scenario_id}_approvals.jsonl"
        self.requests = 0
        # request_id -> latest decision record, so a checkpoint can read status.
        self.latest: dict[str, dict[str, Any]] = {}

    def next_request_id(self, customer_id: str) -> str:
        self.requests += 1
        return f"{self.scenario_id}:{customer_id}:{self.requests:04d}"

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"scenario_id": self.scenario_id, **record}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
        return record

    def write_request(self, **kw) -> dict[str, Any]:
        return self._append({"record_type": "request", **kw})

    def write_decision(self, request_id: str, **kw) -> dict[str, Any]:
        rec = self._append({"record_type": "decision", "request_id": request_id, **kw})
        self.latest[request_id] = rec
        return rec

    def status_for(self, request_id: str) -> Optional[str]:
        """hitl_status from the LATEST decision record for that request."""
        rec = self.latest.get(request_id)
        if rec is None:
            return None
        return _STATUS_FOR.get(rec.get("decision"), "escalated")

    def load(self) -> int:
        """Replay the file so `latest` reflects every decision, newest wins."""
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                if rec.get("record_type") == "decision" and rec.get("request_id"):
                    self.latest[rec["request_id"]] = rec
        return n


_log: Optional[ApprovalLog] = None


def configure(scenario_id: str, log_dir: str | Path = "out/logs") -> ApprovalLog:
    global _log
    _log = ApprovalLog(scenario_id, log_dir)
    return _log


def get_log() -> ApprovalLog:
    global _log
    if _log is None:
        _log = ApprovalLog("scenario")
    return _log


def _reviewer() -> str:
    return os.environ.get("REVIEWER_NAME") or "cli_user"


def request_approval(
    decision: dict[str, Any],
    context_shown: str,
    ctx=None,
    mode: str = "auto",
    hypothesis=None,
) -> str:
    """Request approval for a non-no_action decision. Returns a hitl_status."""
    alog = get_log()
    customer_id = getattr(getattr(ctx, "profile", None), "customer_id", "unknown")
    request_id = alog.next_request_id(customer_id)
    sim_time = ctx.now().isoformat() if ctx is not None else None

    alog.write_request(
        request_id=request_id,
        sim_time=sim_time,
        wall_time=datetime.now(timezone.utc).isoformat(),
        customer_id=customer_id,
        proposed_action=decision.get("action"),
        proposed_subtype=decision.get("action_subtype"),
        context_shown=context_shown,
        evidence_event_ids=list(getattr(hypothesis, "supporting_events", []) or []),
        policy_chunk_ids=list(decision.get("policy_chunk_ids") or []),
        mode=mode,
    )
    decision["request_id"] = request_id

    if mode != "interactive":
        alog.write_decision(
            request_id,
            decision="escalated_auto",
            modifications=None,
            why_queries=[],
            reviewer="auto",
            seconds_to_decide=0.0,
        )
        if ctx is not None:
            ctx.log.log_agent_action(
                "hitl", None,
                f"auto mode: {decision.get('action')} escalated, not executed",
                list(getattr(hypothesis, "supporting_events", []) or [])[:5],
                as_of=ctx.now(), request_id=request_id, mode="auto",
            )
        return "escalated"

    return _interactive(alog, request_id, decision, context_shown, ctx, hypothesis)


def _interactive(alog, request_id, decision, context_shown, ctx, hypothesis) -> str:
    started = time.monotonic()
    why_queries: list[str] = []
    modifications: Optional[dict[str, dict[str, Any]]] = None

    print("\n" + "=" * 72)
    print(f"APPROVAL REQUEST {request_id}")
    print("=" * 72)
    print(context_shown)
    print("-" * 72)

    while True:
        try:
            answer = input("[a]pprove  [r]eject  [m]odify  [w]hy  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(no input -- escalating)")
            answer = ""

        if answer.startswith("a"):
            recorded = "human_approved"
            break
        if answer.startswith("r"):
            recorded = "human_rejected"
            break
        if answer.startswith("m"):
            modifications = {}
            new_subtype = input("  new action_subtype (blank to keep) > ").strip()
            if new_subtype:
                modifications["action_subtype"] = {
                    "from": decision.get("action_subtype"), "to": new_subtype
                }
                decision["action_subtype"] = new_subtype
                (decision.setdefault("draft", {}))["action_subtype"] = new_subtype
            new_msg = input("  new customer_message (blank to keep) > ").strip()
            if new_msg:
                draft = decision.setdefault("draft", {})
                modifications["customer_message"] = {
                    "from": draft.get("customer_message"), "to": new_msg
                }
                draft["customer_message"] = new_msg
            recorded = "human_modified" if modifications else "human_approved"
            break
        if answer.startswith("w"):
            why_queries.append("scorecard_and_debate")
            _print_why(ctx, hypothesis)
            continue
        print("  (unrecognised -- escalating)")
        recorded = "escalated_auto"
        break

    alog.write_decision(
        request_id,
        decision=recorded,
        modifications=modifications or None,
        why_queries=why_queries,
        reviewer=_reviewer(),
        seconds_to_decide=round(time.monotonic() - started, 2),
    )
    if ctx is not None:
        ctx.log.log_agent_action(
            "hitl", None, f"reviewer {recorded} for {decision.get('action')}",
            list(getattr(hypothesis, "supporting_events", []) or [])[:5],
            as_of=ctx.now(), request_id=request_id, decision=recorded,
        )
    return _STATUS_FOR.get(recorded, "escalated")


def _print_why(ctx, hypothesis) -> None:
    """The full scorecard and any debate record: the reviewer's 'why'."""
    print("-" * 72)
    if hypothesis is not None:
        print(f"state={hypothesis.state} band={hypothesis.confidence_band} "
              f"score={hypothesis.score:.3f} needs_human={hypothesis.needs_human}")
        print(f"rationale: {hypothesis.rationale}")
        print(f"runner_up: {hypothesis.runner_up}")
    if ctx is None:
        return
    for ep in ctx.memory.episodic.query(type="debate", limit=2):
        data = ep.data or {}
        print(f"\ndebate {ep.episode_id}: verdict={data.get('verdict')} "
              f"fallback={data.get('used_fallback')}")
        print(f"  reasoning: {data.get('reasoning')}")
        for side in ("case_a", "case_b"):
            case = data.get(side) or {}
            print(f"  {side}: {case.get('state')} score={case.get('score')}")
    for key in ("engagement_trend", "card_activity_trend", "outflow_pattern"):
        f = ctx.memory.working.get_finding(key)
        if f is not None:
            print(f"  {key}: {f.value} [{f.confidence_band}] {f.evidence_event_ids}")
    print("-" * 72)


def clear_hold(ctx, rule_id: str, request_id: Optional[str] = None) -> str:
    """Stage 2 uses this to let a human clear a guardrail hold."""
    alog = get_log()
    rid = request_id or alog.next_request_id(
        getattr(getattr(ctx, "profile", None), "customer_id", "unknown")
    )
    alog.write_decision(
        rid, decision="hold_cleared", modifications={"rule_id": {"from": rule_id, "to": None}},
        why_queries=[], reviewer=_reviewer(), seconds_to_decide=0.0,
    )
    return "escalated"
