"""
================================================================================
AGENT_NAME : action_agent
TYPE       : derived (one LLM draft + at most one LLM revision, both with
             deterministic fallbacks; critique is pure code)

TRIGGERS
  on_day_boundary, immediately after synthesis_agent.

INPUT
  memory.working.get_hypothesis() and the "guardrail_hold" finding if present.
  policy.retrieval chunks for the hypothesis state. profile.customer_value_tier.
  memory.episodic for open "intervention" episodes (cooldown and follow-up).
  Never reads raw events.

OUTPUT
  memory.working.set_finding("current_decision", {action, action_subtype,
    hitl_status, customer_message, rm_brief, offer_value, policy_chunk_ids,
    critique, used_fallback, carried_forward, notes})
  Episode "intervention" (outcome null) for each NEW action.

TOOLS
  config.action_rules (ACTION_TABLE, SENSITIVE_*, DRAFT_TEMPLATES),
  config.thresholds (caps, cooldowns), policy.retrieval, hitl.request_approval,
  c360.llm_client.complete_json, c360.masking, ctx.memory, ctx.log.

PROMPT
  ACTION_PROMPT below, stored verbatim.
================================================================================

WHICH action to take is decided deterministically by ACTION_TABLE. The LLM only
picks a subtype from the retrieved policy list and writes the text.

ASSUMPTIONS (spec was silent):
- "A NEW non-no_action decision" means the (state, action) pair differs from the
  open intervention episode, so a change of action within the same state still
  re-drafts.
- Cooldown carries forward the action AND the drafted text verbatim from the
  open episode's data, so checkpoints stay stable without re-calling the LLM.
- The escalation follow-up fires once: after it fires, the episode is marked so
  it cannot fire again. This is OUR READING of an ambiguous ground-truth
  checkpoint, per the spec's instruction to say so.
- critique check (b) scans the drafted customer_message, rm_brief and
  offer_value for any number, and compares the largest against the tier cap.
- critique check (e) applies SENSITIVE_WORDS for the hypothesis state only, to
  the customer_message only; the rm_brief is allowed to state the inference.
- hitl_status for no_action is "auto_approved" and no approval is requested.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

import hitl
from c360 import masking
from c360.llm_client import LLMInvalidJSON, LLMUnavailable, get_client
from c360.loader import Event
from config import action_rules as AR
from config import llm as llm_config
from config import thresholds as T
from config.enums import VALID_ACTIONS, VALID_HITL
from policy import retrieval

from .base import Agent, AgentContext, Finding
from tracing import traced

log = logging.getLogger(__name__)


ACTION_PROMPT = """
You draft a bank's intervention for one customer. WHICH action to take
has already been decided; you only choose the best subtype from the
allowed list and write the text. Names are masked.

Rules: pick action_subtype ONLY from the provided policy entries. The
customer message must be warm and helpful and must NOT mention or hint
at what the bank inferred about the customer's personal life (no
references to health, family, relationships, moving, or money
troubles). The relationship-manager brief CAN state the inference and
must cite event_ids and policy chunk ids.

Return ONLY JSON:
{
  "action_subtype": "<one of the allowed ids>",
  "customer_message": "string under 90 words, or null for
     relationship_manager_escalation and compliance_fraud_hold",
  "rm_brief": "3-5 sentences for the human reviewer, citing [EVT_...]
     and [policy chunk ids]",
  "offer_value": number or null
}
"""

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(slots=True)
class Draft:
    action_subtype: Optional[str] = None
    customer_message: Optional[str] = None
    rm_brief: Optional[str] = None
    offer_value: Optional[float] = None
    policy_chunk_ids: list[str] = field(default_factory=list)
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_subtype": self.action_subtype,
            "customer_message": self.customer_message,
            "rm_brief": self.rm_brief,
            "offer_value": self.offer_value,
            "policy_chunk_ids": list(self.policy_chunk_ids),
            "used_fallback": self.used_fallback,
        }


class ActionAgent(Agent):
    name = "action_agent"

    def __init__(self, llm=None, mode: str = "auto") -> None:
        super().__init__()
        self.llm = llm
        self.mode = mode  # "auto" | "interactive"
        self.llm_calls = 0
        self.fallback_calls = 0
        self.revisions = 0

    def _client(self):
        return self.llm if self.llm is not None else get_client()

    def handles(self, event: Event) -> bool:
        return False  # derived

    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        return []

    # --- main pass ----------------------------------------------------------

    @traced("agent")
    def on_day_boundary(self, sim_date: date, ctx: AgentContext) -> list[Finding]:
        hypothesis = ctx.memory.working.get_hypothesis()
        if hypothesis is None:
            return []

        # --- step 1: guardrail hook ----------------------------------------
        hold = ctx.memory.working.get_finding("guardrail_hold")
        if hold is not None and (hold.value or {}).get("forced_action"):
            return self._apply_guardrail_hold(ctx, hold, hypothesis)

        state, band = hypothesis.state, hypothesis.confidence_band

        # --- step 3a: carry an open intervention forward -------------------
        carried = self._carry_forward(ctx, state)
        if carried is not None:
            return self._write(ctx, carried, hypothesis, carried_forward=True)

        # --- step 2: decide -------------------------------------------------
        action = self.decide(state, band, ctx.profile.customer_value_tier)

        # --- step 3b: escalation follow-up ---------------------------------
        followup = self._escalation_followup(ctx, state, hypothesis)
        if followup is not None:
            action = followup

        if action == "no_action":
            decision = {
                "action": "no_action",
                "action_subtype": None,
                "hitl_status": "auto_approved",
                "draft": Draft().to_dict(),
                "policy_chunk_ids": [],
                "critique": {"passed": True, "failures": []},
                "used_fallback": bool(hypothesis.used_fallback),
                "notes": None,
            }
            return self._write(ctx, decision, hypothesis)

        # --- step 4: draft --------------------------------------------------
        draft = self.draft(ctx, state, action, hypothesis)

        # --- step 5: critique ----------------------------------------------
        critique = self.critique(ctx, state, action, draft, hypothesis)
        if not critique["passed"]:
            draft, critique = self._revise(ctx, state, action, draft, hypothesis, critique)

        hitl_status = "escalated" if critique.get("force_escalate") else None

        decision = {
            "action": action,
            "action_subtype": draft.action_subtype,
            "hitl_status": hitl_status,
            "draft": draft.to_dict(),
            "policy_chunk_ids": list(draft.policy_chunk_ids),
            "critique": critique,
            "used_fallback": bool(draft.used_fallback or hypothesis.used_fallback),
            "notes": "; ".join(critique.get("failures") or []) or None,
        }

        # --- step 6: HITL ---------------------------------------------------
        context_shown = self._render_context(ctx, decision, hypothesis)
        status = hitl.request_approval(
            decision, context_shown, ctx=ctx, mode=self.mode, hypothesis=hypothesis
        )
        decision["hitl_status"] = status
        if status not in VALID_HITL:
            log.warning("hitl returned %r, forcing 'escalated'", status)
            decision["hitl_status"] = "escalated"

        return self._write(ctx, decision, hypothesis, new_action=True)

    # --- step 1 -------------------------------------------------------------

    def _apply_guardrail_hold(self, ctx: AgentContext, hold: Finding, hypothesis) -> list[Finding]:
        """A guardrail hold overrides everything. Stage 2 writes the finding."""
        v = hold.value or {}
        forced_state = v.get("forced_state")
        decision = {
            "action": v.get("forced_action"),
            "action_subtype": v.get("forced_subtype"),
            "hitl_status": "escalated",
            "draft": Draft(
                action_subtype=v.get("forced_subtype"),
                customer_message=None,
                rm_brief=(
                    f"Guardrail {v.get('rule_id')} fired on {v.get('triggered_by')}. "
                    f"{hypothesis.rationale or ''}"
                ).strip(),
            ).to_dict(),
            "policy_chunk_ids": [],
            "critique": {"passed": True, "failures": [], "skipped": "guardrail_hold"},
            "used_fallback": False,
            "forced_state": forced_state,
            "guardrail_rule_id": v.get("rule_id"),
            "notes": f"guardrail {v.get('rule_id')} forced this action",
        }
        ctx.log.log_agent_action(
            self.name, v.get("triggered_by"),
            f"guardrail hold {v.get('rule_id')} -> {decision['action']} (escalated)",
            hold.evidence_event_ids, as_of=ctx.now(),
            rule_id=v.get("rule_id"), forced=True,
        )
        return self._write(ctx, decision, hypothesis)

    @traced("agent")
    def apply_guardrail_now(self, ctx: AgentContext) -> list[Finding]:
        """Apply a just-fired hold mid-day, for the guardrail's immediate checkpoint.

        Before the first synthesis pass there is no hypothesis yet; the hold
        still applies, citing only the triggering event.
        """
        hold = ctx.memory.working.get_finding("guardrail_hold")
        if hold is None or not (hold.value or {}).get("forced_action"):
            return []
        hypothesis = ctx.memory.working.get_hypothesis()
        if hypothesis is None:
            from types import SimpleNamespace

            hypothesis = SimpleNamespace(
                state=None, confidence_band="high", rationale="",
                supporting_events=list(hold.evidence_event_ids),
            )
        return self._apply_guardrail_hold(ctx, hold, hypothesis)

    # --- step 2 -------------------------------------------------------------

    def decide(self, state: str, band: str, tier: str) -> str:
        entry = AR.ACTION_TABLE.get(state)
        if entry is None:
            return "no_action"
        min_band, action = entry
        if min_band == "__fraud_min__":
            min_band = T.FRAUD_ACTION_MIN_BAND
        if not AR.band_at_least(band, min_band):
            return "no_action"
        if action == "__tier_branch__":
            return AR.CHURN_ACTION_BY_TIER.get(tier, AR.CHURN_ACTION_DEFAULT)
        return action

    # --- step 3 -------------------------------------------------------------

    def _open_intervention(self, ctx: AgentContext, state: str):
        for ep in reversed(ctx.memory.episodic.query(type="intervention")):
            if ep.outcome is None and (ep.data or {}).get("state") == state:
                return ep
        return None

    def _carry_forward(self, ctx: AgentContext, state: str) -> Optional[dict]:
        """Keep reporting an open action verbatim while inside the cooldown."""
        ep = self._open_intervention(ctx, state)
        if ep is None or ep.opened_at is None:
            return None
        age_days = (ctx.now() - ep.opened_at).total_seconds() / 86400
        if age_days >= T.ACTION_COOLDOWN_DAYS:
            return None
        prior = dict((ep.data or {}).get("decision") or {})
        if not prior.get("action"):
            return None
        prior["carried_forward"] = True
        prior["notes"] = (
            f"carrying forward the open {prior['action']} opened "
            f"{ep.opened_at:%Y-%m-%d} ({age_days:.0f}d ago)"
        )
        ctx.log.log_agent_action(
            self.name, None,
            f"carry forward {prior['action']}/{prior.get('action_subtype')} "
            f"({age_days:.0f}d into {T.ACTION_COOLDOWN_DAYS}d cooldown)",
            ep.evidence_event_ids, as_of=ctx.now(), carried_forward=True,
        )
        return prior

    def _escalation_followup(self, ctx: AgentContext, state: str, hypothesis) -> Optional[str]:
        """An RM escalation left open too long, with new evidence, gets outreach.

        ASSUMPTION: this is our reading of an ambiguous ground-truth checkpoint,
        which shows a second, softer action after an escalation. It fires once.
        """
        for ep in reversed(ctx.memory.episodic.query(type="intervention")):
            data = ep.data or {}
            if data.get("state") != state:
                continue
            if (data.get("decision") or {}).get("action") != "relationship_manager_escalation":
                continue
            if data.get("followup_fired") or ep.opened_at is None:
                continue
            age_days = (ctx.now() - ep.opened_at).total_seconds() / 86400
            if age_days < T.ESCALATION_FOLLOWUP_DAYS:
                continue
            new_evidence = [
                eid for eid in (hypothesis.supporting_events or [])
                if eid not in set(ep.evidence_event_ids)
            ]
            if not new_evidence:
                continue
            data["followup_fired"] = True
            ep.data = data
            ctx.memory.episodic.update_outcome(
                ep.episode_id, outcome=ep.outcome or "followup_emitted",
            )
            ctx.log.log_agent_action(
                self.name, None,
                f"escalation open {age_days:.0f}d with new evidence -> follow-up outreach",
                new_evidence[:5], as_of=ctx.now(), followup=True,
            )
            return "proactive_retention_outreach"
        return None

    # --- step 4 -------------------------------------------------------------

    def draft(self, ctx: AgentContext, state: str, action: str, hypothesis) -> Draft:
        chunks = retrieval.retrieve(state)
        chunk_ids = retrieval.chunk_ids(chunks)
        allowed = retrieval.subtypes_for(state) or [
            c.subtype_id for c in chunks if c.subtype_id
        ]
        prompt = self._draft_prompt(ctx, state, action, hypothesis, chunks, allowed)
        client = self._client()
        attempts = 1 + max(0, llm_config.MAX_RETRIES)
        last_error: Optional[str] = None

        for _ in range(attempts):
            try:
                raw = client.complete_json(ACTION_PROMPT, prompt, "action_draft")
                self.llm_calls += 1
                subtype = raw.get("action_subtype")
                if subtype not in allowed:
                    raise LLMInvalidJSON(f"action_subtype {subtype!r} not in {allowed}")
                message = raw.get("customer_message")
                if action in AR.NO_CUSTOMER_MESSAGE_ACTIONS:
                    message = None
                value = raw.get("offer_value")
                return Draft(
                    action_subtype=subtype,
                    customer_message=str(message) if message else None,
                    rm_brief=str(raw.get("rm_brief") or "").strip() or None,
                    offer_value=float(value) if isinstance(value, (int, float)) else None,
                    policy_chunk_ids=chunk_ids,
                    used_fallback=False,
                )
            except LLMUnavailable as exc:
                last_error = str(exc)
                break
            except (LLMInvalidJSON, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("%s draft: %s", self.name, last_error)

        self.fallback_calls += 1
        ctx.log.log_agent_action(
            self.name, None, f"draft [fallback: {last_error}]", [], as_of=ctx.now(), fallback=True,
        )
        return self._template_draft(state, action, hypothesis, chunks, chunk_ids, allowed)

    def _template_draft(
        self, state: str, action: str, hypothesis, chunks, chunk_ids: list[str], allowed: list[str]
    ) -> Draft:
        """Deterministic draft: first retrieved subtype + a neutral template."""
        message = AR.DRAFT_TEMPLATES.get(action)
        if action in AR.NO_CUSTOMER_MESSAGE_ACTIONS:
            message = None
        brief = (hypothesis.rationale or f"{state} inferred.").strip()
        if chunk_ids:
            brief += " Policy: " + ", ".join(f"[{cid}]" for cid in chunk_ids[:3])
        return Draft(
            action_subtype=(allowed[0] if allowed else None),
            customer_message=message,
            rm_brief=brief,
            offer_value=None,
            policy_chunk_ids=chunk_ids,
            used_fallback=True,
        )

    def _draft_prompt(self, ctx, state, action, hypothesis, chunks, allowed) -> str:
        lines = [
            f"decided_action: {action}",
            f"inferred_state: {state} (confidence {hypothesis.confidence_band})",
            f"customer_value_tier: {ctx.profile.customer_value_tier}",
            f"retention_credit_cap: {T.RETENTION_CREDIT_CAP.get(ctx.profile.customer_value_tier)}",
            f"evidence_event_ids: {', '.join(hypothesis.supporting_events[:8])}",
            "",
            f"assessment: {hypothesis.rationale or '(none)'}",
            "",
            "allowed action_subtype ids: " + ", ".join(allowed),
            "",
            "policy entries:",
        ]
        for c in chunks:
            lines.append(f"  [{c.chunk_id}] {c.text}")
        masked, _ = masking.mask("\n".join(lines), ctx.profile)
        return masked

    # --- step 5: five deterministic checks ---------------------------------

    def critique(self, ctx: AgentContext, state: str, action: str, draft: Draft, hypothesis) -> dict:
        failures: list[str] = []
        tier = ctx.profile.customer_value_tier
        cap = T.RETENTION_CREDIT_CAP.get(tier)

        # (a) the action's subtype is allowed for this state in the catalogue
        allowed = retrieval.subtypes_for(state)
        if allowed and draft.action_subtype not in allowed:
            failures.append(f"a: subtype {draft.action_subtype!r} not allowed for {state}")

        # (b) no money amount above the tier cap
        if cap is not None:
            amounts: list[float] = []
            if draft.offer_value is not None:
                amounts.append(float(draft.offer_value))
            for text in (draft.customer_message, draft.rm_brief):
                if text:
                    amounts.extend(float(m) for m in _NUMBER_RE.findall(text))
            # Event ids and dates produce large spurious numbers, so only values
            # that look like money (<= 1e6) are considered.
            money = [a for a in amounts if a <= 1_000_000]
            if money and max(money) > cap:
                failures.append(f"b: amount {max(money):.0f} exceeds {tier} cap {cap}")

        # (c) no evidence id of the hypothesis is currently suppressed
        suppressed = self._suppressed(ctx)
        hit = [e for e in (hypothesis.supporting_events or []) if e in suppressed]
        if hit:
            failures.append(f"c: evidence {','.join(hit)} is suppressed by an explanation")

        # (d) a sensitive inference always goes to a human
        force_escalate = state in AR.SENSITIVE_INFERENCES

        # (e) the customer message must not reveal the inference
        words = AR.SENSITIVE_WORDS.get(state, ())
        if draft.customer_message and words:
            low = draft.customer_message.lower()
            leaked = [w for w in words if w in low]
            if leaked:
                failures.append(f"e: customer_message leaks {', '.join(leaked)}")

        return {
            "passed": not failures,
            "failures": failures,
            "force_escalate": force_escalate,
            "checks": {"a": "subtype_allowed", "b": "cost_cap", "c": "not_suppressed",
                       "d": "sensitive_inference", "e": "no_inference_leak"},
        }

    def _suppressed(self, ctx: AgentContext) -> set[str]:
        from c360.loader import parse_ts

        out: set[str] = set()
        for ep in ctx.memory.episodic.query(type="explanation"):
            data = ep.data or {}
            target = data.get("suppresses")
            if not target:
                continue
            until = data.get("until")
            if until:
                try:
                    if parse_ts(until) <= ctx.now():
                        continue
                except (ValueError, TypeError):
                    pass
            out.add(target)
        return out

    def _revise(self, ctx, state, action, draft: Draft, hypothesis, critique: dict):
        """At most ONE LLM revision. Offline: drop to the template draft."""
        self.revisions += 1
        chunks = retrieval.retrieve(state)
        allowed = retrieval.subtypes_for(state) or [c.subtype_id for c in chunks if c.subtype_id]
        prompt = (
            self._draft_prompt(ctx, state, action, hypothesis, chunks, allowed)
            + "\n\nYour previous draft failed these checks; fix every one:\n"
            + "\n".join(f"  - {f}" for f in critique["failures"])
        )
        try:
            raw = self._client().complete_json(ACTION_PROMPT, prompt, "action_revision")
            self.llm_calls += 1
            subtype = raw.get("action_subtype")
            if subtype in allowed:
                message = raw.get("customer_message")
                if action in AR.NO_CUSTOMER_MESSAGE_ACTIONS:
                    message = None
                value = raw.get("offer_value")
                revised = Draft(
                    action_subtype=subtype,
                    customer_message=str(message) if message else None,
                    rm_brief=str(raw.get("rm_brief") or "").strip() or None,
                    offer_value=float(value) if isinstance(value, (int, float)) else None,
                    policy_chunk_ids=retrieval.chunk_ids(chunks),
                    used_fallback=False,
                )
                recheck = self.critique(ctx, state, action, revised, hypothesis)
                ctx.log.log_agent_action(
                    self.name, None,
                    f"revision {'fixed' if recheck['passed'] else 'still failing'}: "
                    + "; ".join(critique["failures"]),
                    [], as_of=ctx.now(), revised=True,
                )
                if recheck["passed"]:
                    return revised, recheck
                recheck["force_escalate"] = True
                return revised, recheck
        except (LLMUnavailable, LLMInvalidJSON, ValueError) as exc:
            log.info("%s revision unavailable (%s) -- using template", self.name, exc)

        fallback = self._template_draft(
            state, action, hypothesis, chunks, retrieval.chunk_ids(chunks), allowed
        )
        recheck = self.critique(ctx, state, action, fallback, hypothesis)
        recheck["force_escalate"] = True
        recheck["failures"] = list(recheck["failures"]) + [
            f"original draft failed: {'; '.join(critique['failures'])}"
        ]
        self.fallback_calls += 1
        ctx.log.log_agent_action(
            self.name, None,
            "revision [fallback: template] after " + "; ".join(critique["failures"]),
            [], as_of=ctx.now(), fallback=True,
        )
        return fallback, recheck

    # --- step 6 helpers -----------------------------------------------------

    def _render_context(self, ctx: AgentContext, decision: dict, hypothesis) -> str:
        """The exact text a reviewer is shown. Masked, stored verbatim in the log."""
        d = decision.get("draft") or {}
        lines = [
            f"Customer tier      : {ctx.profile.customer_value_tier}",
            f"Inferred state     : {hypothesis.state} ({hypothesis.confidence_band})",
            f"Score              : {hypothesis.score:.3f}",
            f"Proposed action    : {decision['action']} / {decision.get('action_subtype')}",
            f"Evidence           : {', '.join(hypothesis.supporting_events[:8])}",
            f"Policy chunks      : {', '.join(decision.get('policy_chunk_ids') or [])}",
            f"Rationale          : {hypothesis.rationale or '(none)'}",
        ]
        if hypothesis.needs_human:
            lines.append("Flagged            : synthesis could not resolve a conflict")
        if decision.get("critique", {}).get("failures"):
            lines.append("Critique failures  : " + "; ".join(decision["critique"]["failures"]))
        if d.get("customer_message"):
            lines.append(f"Customer message   : {d['customer_message']}")
        if d.get("rm_brief"):
            lines.append(f"RM brief           : {d['rm_brief']}")
        if d.get("offer_value") is not None:
            lines.append(f"Offer value        : {d['offer_value']}")
        masked, _ = masking.mask("\n".join(lines), ctx.profile)
        return masked

    # --- step 7 -------------------------------------------------------------

    def _write(
        self, ctx: AgentContext, decision: dict, hypothesis,
        new_action: bool = False, carried_forward: bool = False,
    ) -> list[Finding]:
        decision.setdefault("carried_forward", carried_forward)
        if decision["action"] not in VALID_ACTIONS:
            log.warning("action %r not a README enum; forcing no_action", decision["action"])
            decision["action"] = "no_action"

        finding = self.emit(
            ctx,
            "current_decision",
            decision,
            hypothesis.confidence_band,
            hypothesis.supporting_events or ["baseline"],
            notes=decision.get("notes"),
            force=True,  # the checkpoint writer reads this every day
        )
        if new_action and decision["action"] != "no_action":
            ctx.memory.episodic.open_episode(
                type="intervention",
                evidence_event_ids=list(hypothesis.supporting_events or []),
                notes=f"{decision['action']} / {decision.get('action_subtype')} "
                      f"({decision['hitl_status']})",
                data={
                    "state": hypothesis.state,
                    "decision": decision,
                    "opened_band": hypothesis.confidence_band,
                },
            )
        return [finding] if finding else []
