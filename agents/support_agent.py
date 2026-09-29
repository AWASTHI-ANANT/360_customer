"""
================================================================================
AGENT_NAME : support_agent
TYPE       : LLM (claude-opus-5), with a deterministic keyword fallback

TRIGGERS
  support_logs           : ticket_created | ticket_resolved | call_transcript
  web_app_events         : search_query
  social_signal_consented: life_event_mention -- ONLY when
                           payload.consent_flag is true (a skip is logged)

INPUT
  One event's free text (raw_text / search_text), MASKED via c360.masking,
  plus channel, category, resolution_status, and a short masked summary of this
  customer's open support episodes from the last 30 days (memory.episodic).

OUTPUT (findings written to memory.working)
  sentiment_state      {sentiment, score, urgency, intent, source}
  relationship_damage  {resolution_outcome, category, channel, quote, source}
                       -- only on ticket_resolved with an unfavorable outcome
  intent_signal        {intent, life_event_hints, query, source}
                       -- search_query events
  life_event_hint      {labels, evidence_quote, source}
                       -- only when the model reports hints

  Plus, in CODE (never from the model): an episodic "explanation" episode
  {suppresses, until, reason} when the customer explains a transaction that
  already landed. This is what stops a red herring being read as alarm.

TOOLS
  c360.masking.mask (before every call), c360.llm_client.complete_json,
  ctx.event_store (to match an explained transaction), ctx.memory, ctx.log,
  config.thresholds.

PROMPT
  SUPPORT_PROMPT below, stored verbatim. Sent as the system prompt.
================================================================================

This agent does NOT decide life events or actions. `life_event_hints` are
hints, weighed later alongside the deterministic signals.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, Optional, Sequence

from c360 import masking
from c360.llm_client import LLMInvalidJSON, LLMUnavailable, get_client
from c360.loader import Event
from config import llm as llm_config
from config import thresholds as T

from .base import Agent, AgentContext, Finding

log = logging.getLogger(__name__)


SUPPORT_PROMPT = """
You analyse customer-written text for a retail bank's monitoring system.
You receive one support ticket, call transcript, app search query, or
consented social post. Names and account numbers are masked.

Report only what the text supports. Do not guess at life events that
are not mentioned or strongly implied. Quote short evidence phrases
(under 12 words) copied from the text.

Return ONLY a JSON object, no prose, with exactly these fields:
{
  "sentiment": "negative" | "neutral" | "positive",
  "sentiment_score": number from -1.0 to 1.0,
  "urgency": "low" | "medium" | "high",
  "intent": one of "fee_dispute", "hardship_request",
     "cancellation_or_exit", "purchase_explanation", "product_inquiry",
     "complaint_other", "information_request", "other",
  "resolution_outcome": "favorable" | "unfavorable" | "pending" |
     "not_applicable",
  "churn_language": true | false,
  "legal_threat_language": true | false,
  "life_event_hints": list of zero or more of
     "new_child_life_event", "marriage_or_relationship_change",
     "job_change_or_promotion", "job_loss_or_income_disruption",
     "medical_hardship", "financial_distress_general", "relocation",
     "retirement_transition", "wealth_growth_or_windfall",
     "elder_vulnerability_or_scam_risk", "small_business_cashflow_event",
  "evidence_quotes": list of short quotes supporting the above,
  "explains_transaction": {
     "mentioned": true | false,
     "description": string or null,
     "amount_mentioned": number or null,
     "merchant_or_item": string or null
  }
}

resolution_outcome refers to how the bank resolved the customer's
request, judged from the bank's reply if present.
legal_threat_language is advisory only; a separate deterministic
guardrail handles legal escalation.
"""


# --- output schema ----------------------------------------------------------

SENTIMENTS = ("negative", "neutral", "positive")
URGENCIES = ("low", "medium", "high")
INTENTS = (
    "fee_dispute",
    "hardship_request",
    "cancellation_or_exit",
    "purchase_explanation",
    "product_inquiry",
    "complaint_other",
    "information_request",
    "other",
)
RESOLUTION_OUTCOMES = ("favorable", "unfavorable", "pending", "not_applicable")
LIFE_EVENT_HINTS = (
    "new_child_life_event",
    "marriage_or_relationship_change",
    "job_change_or_promotion",
    "job_loss_or_income_disruption",
    "medical_hardship",
    "financial_distress_general",
    "relocation",
    "retirement_transition",
    "wealth_growth_or_windfall",
    "elder_vulnerability_or_scam_risk",
    "small_business_cashflow_event",
)

# resolution_status values meaning the customer did not get what they asked for.
UNFAVORABLE_STATUSES = {
    "human_rejected", "rejected", "denied", "declined", "closed_unresolved",
}
FAVORABLE_STATUSES = {
    "resolved", "approved", "refunded", "waived", "human_approved", "closed_resolved",
}


class SupportValidationError(ValueError):
    """The model's JSON did not match the declared schema."""


def validate_support_output(raw: dict) -> dict:
    """Coerce and check the model's object. Raises SupportValidationError."""
    if not isinstance(raw, dict):
        raise SupportValidationError(f"expected object, got {type(raw).__name__}")
    out: dict[str, Any] = {}

    def _enum(field: str, allowed: Sequence[str], default: Optional[str] = None) -> str:
        value = raw.get(field)
        if isinstance(value, str) and value in allowed:
            return value
        if default is not None:
            log.debug("support output: %s=%r invalid, defaulting to %r", field, value, default)
            return default
        raise SupportValidationError(f"{field}={value!r} not in {allowed}")

    out["sentiment"] = _enum("sentiment", SENTIMENTS)
    out["urgency"] = _enum("urgency", URGENCIES, "low")
    out["intent"] = _enum("intent", INTENTS)
    out["resolution_outcome"] = _enum("resolution_outcome", RESOLUTION_OUTCOMES, "not_applicable")

    score = raw.get("sentiment_score")
    if not isinstance(score, (int, float)):
        raise SupportValidationError(f"sentiment_score={score!r} is not a number")
    out["sentiment_score"] = max(-1.0, min(1.0, float(score)))

    out["churn_language"] = bool(raw.get("churn_language"))
    out["legal_threat_language"] = bool(raw.get("legal_threat_language"))

    hints = raw.get("life_event_hints") or []
    if not isinstance(hints, list):
        raise SupportValidationError("life_event_hints must be a list")
    unknown = [h for h in hints if h not in LIFE_EVENT_HINTS]
    if unknown:
        log.warning("support output: dropping unknown life_event_hints %s", unknown)
    out["life_event_hints"] = [h for h in hints if h in LIFE_EVENT_HINTS]

    quotes = raw.get("evidence_quotes") or []
    out["evidence_quotes"] = (
        [str(q) for q in quotes if q][:5] if isinstance(quotes, list) else []
    )

    ex = raw.get("explains_transaction") or {}
    if not isinstance(ex, dict):
        ex = {}
    amount = ex.get("amount_mentioned")
    out["explains_transaction"] = {
        "mentioned": bool(ex.get("mentioned")),
        "description": (str(ex["description"]) if ex.get("description") else None),
        "amount_mentioned": float(amount) if isinstance(amount, (int, float)) else None,
        "merchant_or_item": (str(ex["merchant_or_item"]) if ex.get("merchant_or_item") else None),
    }
    return out


# --- deterministic fallback -------------------------------------------------

_NEGATIVE_WORDS = (
    "not disclosed", "wasn't disclosed", "undisclosed", "unacceptable", "frustrat",
    "angry", "disappointed", "unhappy", "terrible", "awful", "worst", "poor service",
    "unfair", "ridiculous", "complain", "dispute", "refuse", "denied", "cannot be",
    "can't afford", "cannot afford", "struggling", "worried", "upset", "wrong",
    "error", "failed", "still waiting", "no response", "overcharg",
)
_POSITIVE_WORDS = (
    "thank", "great", "excellent", "appreciate", "helpful", "resolved quickly", "happy",
)
_CHURN_WORDS = (
    "close my account", "closing my account", "cancel my account", "switch to",
    "switching to", "another bank", "leave the bank", "leaving", "take my business",
    "competitor", "better rate elsewhere",
)
_LEGAL_WORDS = (
    "lawyer", "attorney", "legal action", "sue", "lawsuit", "ombudsman",
    "regulator", "complaint to",
)
_HARDSHIP_WORDS = (
    "payment plan", "hardship", "cannot afford", "can't afford", "defer", "forbearance",
    "reduce my payment", "financial difficulty", "lost my job", "medical bill",
    "hospital bill", "out of work",
)
_FEE_WORDS = ("fee", "charge", "charged", "refund", "waive", "overcharge", "billed")
_EXPLAIN_WORDS = (
    "that was me", "this was me", "i made that", "i authorized", "it was for",
    "i paid", "i bought", "i transferred",
)
_INQUIRY_WORDS = ("how do i", "how can i", "interest rate", "apply for", "eligible", "what is the")
_INFO_WORDS = ("statement", "balance", "when will", "confirm", "status of")

_LIFE_EVENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "medical_hardship": (
        "hospital", "surgery", "medical", "diagnosis", "illness", "treatment",
        "er visit", "disability",
    ),
    "new_child_life_event": (
        "baby", "newborn", "pregnan", "maternity", "paternity", "child care",
        "childcare", "nursery",
    ),
    "marriage_or_relationship_change": (
        "married", "marriage", "wedding", "divorce", "separated", "engaged",
    ),
    "job_loss_or_income_disruption": (
        "lost my job", "laid off", "redundan", "unemploy", "out of work", "terminated",
    ),
    "job_change_or_promotion": ("new job", "promotion", "raise", "new role", "started at"),
    "relocation": ("moving", "relocat", "new address", "new city"),
    "retirement_transition": ("retire", "pension"),
    "financial_distress_general": (
        "struggling", "behind on", "overdraft", "debt", "cannot afford", "can't afford",
    ),
    "wealth_growth_or_windfall": ("inheritance", "windfall", "bonus", "sold my"),
    "elder_vulnerability_or_scam_risk": ("scam", "fraud", "someone called me", "gift card"),
    "small_business_cashflow_event": ("my business", "invoice", "payroll", "clients"),
}

_AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")


def keyword_fallback(text: str, event: Event) -> dict:
    """Deterministic substitute when the LLM is unavailable or unusable.

    Deliberately conservative: it reads resolution_status and category first --
    those are structured fields, not prose -- and only then the text.
    """
    low = (text or "").lower()
    category = str(event.payload.get("category") or "").lower()
    status = str(event.payload.get("resolution_status") or "").lower()

    neg = sum(1 for w in _NEGATIVE_WORDS if w in low)
    pos = sum(1 for w in _POSITIVE_WORDS if w in low)
    if category in ("dispute", "complaint") or neg > pos:
        sentiment, score = "negative", -min(1.0, 0.4 + 0.15 * max(neg, 1))
    elif pos > neg:
        sentiment, score = "positive", min(1.0, 0.3 + 0.2 * pos)
    else:
        sentiment, score = "neutral", 0.0

    churn = any(w in low for w in _CHURN_WORDS)
    legal = any(w in low for w in _LEGAL_WORDS)

    # Most specific intent first.
    if churn:
        intent = "cancellation_or_exit"
    elif any(w in low for w in _HARDSHIP_WORDS):
        intent = "hardship_request"
    elif any(w in low for w in _FEE_WORDS) or category == "dispute":
        intent = "fee_dispute"
    elif any(w in low for w in _EXPLAIN_WORDS):
        intent = "purchase_explanation"
    elif any(w in low for w in _INQUIRY_WORDS):
        intent = "product_inquiry"
    elif any(w in low for w in _INFO_WORDS):
        intent = "information_request"
    elif sentiment == "negative":
        intent = "complaint_other"
    else:
        intent = "other"

    if event.event_type == "ticket_resolved":
        if status in UNFAVORABLE_STATUSES or "cannot be waived" in low or "is valid" in low:
            outcome = "unfavorable"
        elif status in FAVORABLE_STATUSES:
            outcome = "favorable"
        else:
            outcome = "pending"
    elif status in ("open", "pending", "in_progress"):
        outcome = "pending"
    else:
        outcome = "not_applicable"

    hints = [
        label for label, words in _LIFE_EVENT_KEYWORDS.items() if any(w in low for w in words)
    ]

    urgency = (
        "high"
        if (churn or legal or intent == "hardship_request")
        else ("medium" if sentiment == "negative" else "low")
    )

    amounts = _AMOUNT_RE.findall(text or "")
    explains = any(w in low for w in _EXPLAIN_WORDS)
    return {
        "sentiment": sentiment,
        "sentiment_score": round(score, 2),
        "urgency": urgency,
        "intent": intent,
        "resolution_outcome": outcome,
        "churn_language": churn,
        "legal_threat_language": legal,
        "life_event_hints": hints,
        "evidence_quotes": [text[:80]] if text else [],
        "explains_transaction": {
            "mentioned": explains,
            "description": text[:120] if explains else None,
            "amount_mentioned": float(amounts[0].replace(",", "")) if amounts and explains else None,
            "merchant_or_item": None,
        },
    }


class SupportAgent(Agent):
    name = "support_agent"

    def __init__(self, llm=None) -> None:
        super().__init__()
        self.llm = llm  # resolved lazily so llm_client.configure() can run first
        self.llm_calls = 0
        self.fallback_calls = 0
        self.skipped_unconsented = 0

    def _client(self):
        return self.llm if self.llm is not None else get_client()

    # --- routing ------------------------------------------------------------

    def handles(self, event: Event) -> bool:
        if event.source_system == "support_logs":
            return event.event_type in ("ticket_created", "ticket_resolved", "call_transcript")
        if event.source_system == "web_app_events":
            return event.event_type == "search_query"
        if event.source_system == "social_signal_consented":
            return event.event_type == "life_event_mention"
        return False

    # --- per event ----------------------------------------------------------

    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        # Consent gate: an unconsented social signal is never analysed.
        if event.source_system == "social_signal_consented" and not event.payload.get("consent_flag"):
            self.skipped_unconsented += 1
            ctx.log.log_agent_action(
                self.name,
                event.event_id,
                "skipped social signal: consent_flag is not true",
                [event.event_id],
                as_of=ctx.now(),
                skipped=True,
            )
            return []

        masked_text, _mapping = masking.mask_event_text(event, ctx.profile)
        if not masked_text.strip():
            return []

        analysis, source = self._analyse(event, masked_text, ctx)
        findings: list[Optional[Finding]] = []

        # 1. sentiment_state -- always
        findings.append(
            self.emit(
                ctx,
                "sentiment_state",
                {
                    "sentiment": analysis["sentiment"],
                    "score": analysis["sentiment_score"],
                    "urgency": analysis["urgency"],
                    "intent": analysis["intent"],
                    "churn_language": analysis["churn_language"],
                    "legal_threat_language": analysis["legal_threat_language"],
                    "event_type": event.event_type,
                    "channel": event.payload.get("channel"),
                    "category": event.payload.get("category"),
                    "source": source,
                },
                self._sentiment_band(analysis),
                [event.event_id],
                notes=f"{analysis['sentiment']} / {analysis['intent']} ({source})",
                event_id=event.event_id,
                force=True,  # each message is its own datapoint
            )
        )

        # 2. relationship_damage -- resolved against the customer
        if event.event_type == "ticket_resolved" and analysis["resolution_outcome"] == "unfavorable":
            findings.append(
                self.emit(
                    ctx,
                    "relationship_damage",
                    {
                        "resolution_outcome": "unfavorable",
                        "category": event.payload.get("category"),
                        "channel": event.payload.get("channel"),
                        "resolution_status": event.payload.get("resolution_status"),
                        "quote": (analysis["evidence_quotes"] or [None])[0],
                        "source": source,
                    },
                    "medium",
                    [event.event_id],
                    notes="ticket resolved unfavorably for the customer",
                    event_id=event.event_id,
                    force=True,
                )
            )

        # 3. intent_signal -- app searches
        if event.event_type == "search_query":
            findings.append(
                self.emit(
                    ctx,
                    "intent_signal",
                    {
                        "intent": analysis["intent"],
                        "life_event_hints": analysis["life_event_hints"],
                        "query": masked_text[:200],
                        "urgency": analysis["urgency"],
                        "source": source,
                    },
                    "medium" if analysis["life_event_hints"] or analysis["intent"] != "other" else "low",
                    [event.event_id],
                    notes=f"search intent {analysis['intent']}",
                    event_id=event.event_id,
                    force=True,
                )
            )

        # 4. life_event_hint -- only when the text actually supports one
        if analysis["life_event_hints"]:
            findings.append(
                self.emit(
                    ctx,
                    "life_event_hint",
                    {
                        "labels": analysis["life_event_hints"],
                        "evidence_quote": (analysis["evidence_quotes"] or [None])[0],
                        "urgency": analysis["urgency"],
                        "source": source,
                    },
                    "medium",
                    [event.event_id],
                    notes=", ".join(analysis["life_event_hints"]),
                    event_id=event.event_id,
                    force=True,
                )
            )

        # 5. Track the ticket as an episode, so "open tickets" is answerable.
        self._track_ticket(event, ctx, analysis)

        # 6. Suppression -- decided in CODE, never by the model.
        if analysis["explains_transaction"]["mentioned"]:
            self._suppress_explained(event, ctx, analysis["explains_transaction"])

        return [f for f in findings if f is not None]

    # --- the LLM call -------------------------------------------------------

    def _analyse(self, event: Event, masked_text: str, ctx: AgentContext) -> tuple[dict, str]:
        """Returns (validated_analysis, source) where source is 'llm' or 'fallback'."""
        user_prompt = self._build_prompt(event, masked_text, ctx)
        client = self._client()

        attempts = 1 + max(0, llm_config.MAX_RETRIES)
        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                raw = client.complete_json(
                    SUPPORT_PROMPT, user_prompt, "support_analysis", event_id=event.event_id
                )
                self.llm_calls += 1
                return validate_support_output(raw), "llm"
            except LLMUnavailable as exc:
                # No credentials / offline / refusal: a retry will not help.
                last_error = str(exc)
                break
            except (LLMInvalidJSON, SupportValidationError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "%s attempt %d/%d on %s: %s",
                    self.name, attempt + 1, attempts, event.event_id, last_error,
                )

        self.fallback_calls += 1
        ctx.log.log_agent_action(
            self.name,
            event.event_id,
            f"LLM unusable ({last_error}) -- using keyword_fallback",
            [event.event_id],
            as_of=ctx.now(),
            fallback=True,
        )
        return validate_support_output(keyword_fallback(masked_text, event)), "fallback"

    def _build_prompt(self, event: Event, masked_text: str, ctx: AgentContext) -> str:
        p = event.payload
        open_tickets = self._open_ticket_summary(ctx)
        lines = [
            f"event_type: {event.event_type}",
            f"source_system: {event.source_system}",
            f"channel: {p.get('channel') or 'n/a'}",
            f"category: {p.get('category') or 'n/a'}",
            f"resolution_status: {p.get('resolution_status') or 'n/a'}",
            f"occurred_at: {event.event_time:%Y-%m-%d}",
        ]
        if event.source_system == "social_signal_consented":
            lines.append(f"platform: {p.get('platform') or 'n/a'} (consented)")
        lines.append("")
        lines.append("TEXT:")
        lines.append(masked_text)
        if open_tickets:
            lines.append("")
            lines.append(
                f"This customer's open support episodes in the last {T.SUPPORT_HISTORY_DAYS} days:"
            )
            lines.extend(f"  - {t}" for t in open_tickets)
        return "\n".join(lines)

    def _open_ticket_summary(self, ctx: AgentContext) -> list[str]:
        """Masked one-liners for still-open support episodes."""
        since = ctx.now() - timedelta(days=T.SUPPORT_HISTORY_DAYS)
        episodes = ctx.memory.episodic.query(type="support_ticket", since=since, open_only=True)
        out = []
        for ep in episodes:
            note, _ = masking.mask(ep.notes or "", ctx.profile)
            opened = ep.opened_at.strftime("%Y-%m-%d") if ep.opened_at else "?"
            out.append(f"opened {opened}: {note[:120]}")
        return out[-5:]

    # --- bookkeeping --------------------------------------------------------

    def _sentiment_band(self, analysis: dict) -> str:
        if analysis["urgency"] == "high" or analysis["churn_language"]:
            return "high"
        if analysis["sentiment"] == "negative":
            return "medium"
        return "low"

    def _track_ticket(self, event: Event, ctx: AgentContext, analysis: dict) -> None:
        """Open an episode on ticket_created; close it on ticket_resolved."""
        if event.event_type == "ticket_created":
            ctx.memory.episodic.open_episode(
                type="support_ticket",
                evidence_event_ids=[event.event_id],
                notes=(
                    f"{event.payload.get('category')} via {event.payload.get('channel')}: "
                    f"{analysis['intent']}"
                ),
            )
        elif event.event_type == "ticket_resolved":
            still_open = ctx.memory.episodic.query(type="support_ticket", open_only=True)
            if still_open:
                ctx.memory.episodic.close_episode(
                    still_open[-1].episode_id,
                    outcome=analysis["resolution_outcome"],
                    closed_at=event.event_time,
                )

    def _suppress_explained(self, event: Event, ctx: AgentContext, ex: dict) -> None:
        """Find the transaction the customer just explained and suppress alarm.

        Matching happens here, in code, against the event store -- the model only
        reports what the customer said; it never picks an event.
        """
        amount = ex.get("amount_mentioned")
        merchant = (ex.get("merchant_or_item") or "").strip().lower()
        if amount is None and not merchant:
            return

        lookback = ctx.event_store.trailing(
            event.event_time, T.SUPPRESSION_LOOKBACK_HOURS / 24
        )
        candidates = [
            e
            for e in lookback
            if e.source_system in ("card_payments", "instant_payments", "ach_wire")
            and e.event_id != event.event_id
        ]
        matches: list[tuple[Event, str]] = []
        for cand in candidates:
            if amount is not None and cand.amount:
                if abs(cand.amount - amount) <= T.SUPPRESSION_AMOUNT_TOLERANCE * amount:
                    matches.append((cand, f"amount {cand.amount:.2f} ~ stated {amount:.2f}"))
                    continue
            if merchant:
                haystack = " ".join(
                    str(cand.payload.get(f) or "")
                    for f in ("merchant_name", "counterparty_name")
                ).lower().strip()
                if haystack and (merchant in haystack or (len(merchant) > 4 and haystack in merchant)):
                    named = cand.payload.get("merchant_name") or cand.payload.get("counterparty_name")
                    matches.append((cand, f"merchant {named!r} matches stated {merchant!r}"))

        until = event.event_time + timedelta(hours=T.SUPPRESSION_DURATION_HOURS)
        for cand, reason in matches:
            ctx.memory.episodic.append(
                {
                    "type": "explanation",
                    "customer_id": ctx.profile.customer_id,
                    "opened_at": ctx.now(),
                    "evidence_event_ids": [event.event_id, cand.event_id],
                    # The suppression envelope a later agent must honour.
                    "notes": (
                        f"suppresses={cand.event_id} until={until.isoformat()} reason={reason}"
                    ),
                }
            )
            ctx.log.log_agent_action(
                self.name,
                event.event_id,
                f"explanation suppresses {cand.event_id} until {until:%Y-%m-%dT%H:%M:%SZ}: {reason}",
                [event.event_id, cand.event_id],
                as_of=ctx.now(),
                suppresses=cand.event_id,
                until=until.isoformat(),
                reason=reason,
            )
        if not matches:
            ctx.log.log_agent_action(
                self.name,
                event.event_id,
                f"customer explained a transaction but nothing matched in the last "
                f"{T.SUPPRESSION_LOOKBACK_HOURS}h",
                [event.event_id],
                as_of=ctx.now(),
            )
