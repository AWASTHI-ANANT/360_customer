"""
================================================================================
AGENT_NAME : guardrail_agent
TYPE       : deterministic (regex only, no LLM)

TRIGGERS
  on_event, FIRST among the agents, for support_logs events
  (config.guardrail_rules.GUARDRAIL_SOURCES).

INPUT
  The event's free text (derive.py text_fields, else raw_text).

OUTPUT
  memory.working "guardrail_hold" {rule_id, rule_ids, forced_action,
    forced_subtype, forced_state, freeze_outbound, triggered_by}
  Episode "guardrail_hold".
  self.pending: set when a rule fired on this event, so run_skeleton writes an
  immediate checkpoint at the event's time.

TOOLS
  config.guardrail_rules.match_rules, ctx.memory, ctx.log.
================================================================================

ASSUMPTIONS (spec was silent):
- When several rules fire on one event, the first in GUARDRAIL_RULES order sets
  the action/subtype/state, freeze_outbound is OR-ed across all of them, and
  every rule id is recorded.
- A later fire replaces the earlier hold (newest wins); both episodes remain.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from c360.loader import Event
from config import guardrail_rules as GR
from tracing import traced

from .base import Agent, AgentContext, Finding

log = logging.getLogger(__name__)


class GuardrailAgent(Agent):
    name = "guardrail_agent"

    def __init__(self) -> None:
        super().__init__()
        self.fires: list[dict] = []
        self.pending: Optional[Event] = None

    def handles(self, event: Event) -> bool:
        return event.source_system in GR.GUARDRAIL_SOURCES

    @staticmethod
    def _text(event: Event) -> str:
        texts = (event.derived or {}).get("text_fields") or []
        if not texts and event.payload.get("raw_text"):
            texts = [str(event.payload["raw_text"])]
        return " ".join(texts)

    @traced("agent")
    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        fired = GR.match_rules(self._text(event))
        if not fired:
            return []
        primary = fired[0]
        value = {
            "rule_id": primary["rule_id"],
            "rule_ids": [r["rule_id"] for r in fired],
            "forced_action": primary["forced_action"],
            "forced_subtype": primary["forced_subtype"],
            "forced_state": primary["forced_state"],
            "freeze_outbound": any(r["freeze_outbound"] for r in fired),
            "triggered_by": event.event_id,
        }
        # Written directly, not via emit(): emit opens an episode only the first
        # time a key appears, and every fire needs its own.
        notes = f"guardrail {'+'.join(value['rule_ids'])} on {event.event_id}"
        ctx.memory.working.set_finding(
            "guardrail_hold", value, "high", self.name, [event.event_id], notes=notes
        )
        finding = Finding(key="guardrail_hold", value=value, confidence_band="high",
                          agent=self.name, evidence_event_ids=[event.event_id],
                          updated_at=ctx.now(), notes=notes)
        ctx.log.log_agent_action(
            self.name, event.event_id,
            f"guardrail {value['rule_id']} -> {value['forced_action']}"
            + (" (outbound frozen)" if value["freeze_outbound"] else ""),
            [event.event_id], as_of=ctx.now(), rule_id=value["rule_id"],
        )
        self.findings_emitted += 1
        ctx.memory.episodic.open_episode(
            type="guardrail_hold",
            evidence_event_ids=[event.event_id],
            notes=f"{value['rule_id']} -> {value['forced_action']}"
            + (" (outbound frozen)" if value["freeze_outbound"] else ""),
            system_action=value["forced_action"],
            data=value,
        )
        self.fires.append({**value, "event_time": event.event_time})
        self.pending = event
        return [finding]

    def on_day_boundary(self, sim_date: date, ctx: AgentContext) -> list[Finding]:
        return []
