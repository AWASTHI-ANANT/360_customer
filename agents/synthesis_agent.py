"""
================================================================================
AGENT_NAME : synthesis_agent
TYPE       : derived (one LLM adjudication + one LLM review, both with
             deterministic fallbacks)

TRIGGERS
  on_day_boundary only, and only when some finding's updated_at is newer than
  the last synthesis run. Otherwise it logs "carried forward" and returns
  without touching the hypothesis.

INPUT
  memory.working.all_findings() -- the other agents' findings, never raw events.
  memory.episodic for "explanation" suppression windows and for the episodes
  shown to the review step. config/evidence_rules.py for all weights.

OUTPUT
  memory.working.set_hypothesis({state, score, confidence_band, first_inferred,
    last_updated, supporting_events, runner_up, rationale, needs_human,
    debate_ref, used_fallback})
  Episodes: "debate" (both cases, verdict, reasoning, fallback flag) and
  "hypothesis_change" (when state or band moved).

TOOLS
  config.evidence_rules (EVIDENCE_RULES, LABEL_RULES, INCOMPATIBLE_PAIRS,
  PRECEDENCE_RULES), c360.llm_client.complete_json, c360.masking, ctx.memory,
  ctx.log, config.thresholds.

PROMPT
  DEBATE_PROMPT and SYNTHESIS_PROMPT below, both stored verbatim.
================================================================================

This agent picks a state; it does NOT pick an action. The LLM here can only
agree, disagree (which LOWERS the band by one) or adjudicate between two
candidates the scorer already produced -- it can never raise a band, invent a
state, or change the state label.

ASSUMPTIONS (spec was silent):
- "some finding is newer than the last run" compares max(finding.updated_at)
  against the timestamp of the previous synthesis pass. The first pass always
  runs, and a pass also runs whenever a guardrail_hold finding is present.
- Working memory holds one finding per key, so each key contributes at most
  once per state per pass; repeated emissions over the run do not stack.
- Combination is noisy-OR per state: score = 1 - prod(1 - w_i), which keeps
  every score in [0, 1) without needing a cap.
- Age for decay is measured in fractional days from finding.updated_at to
  clock.now(); a negative age (clock skew) is clamped to 0.
- A label rule's label must be a valid inferred_state; unknown labels are
  logged once and ignored.
- The review step is given at most 5 episodes and the top 3 states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from c360 import masking
from c360.llm_client import LLMInvalidJSON, LLMUnavailable, get_client
from c360.loader import Event, iso
from config import llm as llm_config
from config import thresholds as T
from config.enums import BAND_ORDER, NO_EVENT_STATE, VALID_STATES, lower_band
from config.evidence_rules import (
    EVIDENCE_RULES,
    FRAUD_EXPLANATION_WINDOW_HOURS,
    INCOMPATIBLE_PAIRS,
    LABEL_RULES,
    PRECEDENCE_RULES,
    EXIT_FINDING_KEYS,
    INFLOW_FINDING_KEYS,
    matches,
)

from .base import Agent, AgentContext, Finding
from tracing import traced

log = logging.getLogger(__name__)


DEBATE_PROMPT = """
You are the arbiter between two competing readings of one bank
customer's situation. Each reading comes with a score and the evidence
behind it. Names and account numbers are masked.

Decide which reading better explains ALL of the evidence, taking into
account timing, direction of money movement, and whether any evidence
has a benign explanation. If both are genuinely true at once, say so.
If the evidence cannot separate them, say it is unresolved rather than
guessing.

Return ONLY JSON:
{
  "verdict": "A" | "B" | "both" | "unresolved",
  "reasoning": "2-4 sentences citing event_ids",
  "key_event_ids": [event ids that decided it]
}
"""

SYNTHESIS_PROMPT = """
You review a scoring system's assessment of a bank customer. You get
the ranked candidate states with their evidence, the previous
assessment, and relevant history. Names and account numbers are masked.

Your job is to check the top state, not to invent a new one. Agree if
the evidence supports it. Disagree if the evidence is thin,
contradictory, or better explained by the runner-up, and say why.

Return ONLY JSON:
{
  "agrees": true | false,
  "rationale": "2-4 sentences explaining the customer's situation,
     citing event_ids in brackets like [EVT_000461]",
  "missing_evidence": "what would raise confidence, or null"
}
"""

VERDICTS = ("A", "B", "both", "unresolved")


# --- scorecard --------------------------------------------------------------


@dataclass(slots=True)
class Contribution:
    finding_key: str
    event_ids: list[str]
    weight: float          # after decay
    raw_weight: float
    age_days: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_key": self.finding_key,
            "event_ids": list(self.event_ids),
            "weight": round(self.weight, 4),
            "raw_weight": self.raw_weight,
            "age_days": round(self.age_days, 2),
        }


@dataclass(slots=True)
class StateScore:
    state: str
    score: float
    contributions: list[Contribution] = field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        """Contributing event ids, de-duplicated, strongest contribution first."""
        out: list[str] = []
        for c in sorted(self.contributions, key=lambda c: -c.weight):
            for eid in c.event_ids:
                if eid not in out:
                    out.append(eid)
        return out

    @property
    def finding_keys(self) -> list[str]:
        return [c.finding_key for c in sorted(self.contributions, key=lambda c: -c.weight)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "score": round(self.score, 4),
            "contributions": [c.to_dict() for c in self.contributions],
        }


@dataclass(slots=True)
class Scorecard:
    ranked: list[StateScore] = field(default_factory=list)
    suppressed_events: list[str] = field(default_factory=list)
    generated_at: Optional[datetime] = None

    @property
    def top(self) -> Optional[StateScore]:
        return self.ranked[0] if self.ranked else None

    @property
    def runner_up(self) -> Optional[StateScore]:
        return self.ranked[1] if len(self.ranked) > 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": iso(self.generated_at) if self.generated_at else None,
            "suppressed_events": list(self.suppressed_events),
            "ranked": [s.to_dict() for s in self.ranked],
        }


class SynthesisAgent(Agent):
    name = "synthesis_agent"

    def __init__(self, llm=None) -> None:
        super().__init__()
        self.llm = llm
        self.last_run_at: Optional[datetime] = None
        self.debates = 0
        self.reviews = 0
        self.llm_calls = 0
        self.fallback_calls = 0
        self._unknown_labels: set[str] = set()
        self.last_scorecard: Optional[Scorecard] = None

    def _client(self):
        return self.llm if self.llm is not None else get_client()

    # --- routing ------------------------------------------------------------

    def handles(self, event: Event) -> bool:
        return False  # derived: never event-driven

    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        return []

    # --- main pass ----------------------------------------------------------

    @traced("agent")
    def on_day_boundary(self, sim_date: date, ctx: AgentContext) -> list[Finding]:
        findings = ctx.memory.working.all_findings()
        newest = max(
            (f.updated_at for f in findings.values() if f.updated_at), default=None
        )
        has_hold = "guardrail_hold" in findings
        if (
            self.last_run_at is not None
            and newest is not None
            and newest <= self.last_run_at
            and not has_hold
        ):
            ctx.log.log_agent_action(
                self.name, None, "carried forward: no finding newer than last synthesis",
                [], as_of=ctx.now(), carried_forward=True,
            )
            return []

        self.last_run_at = ctx.now()
        scorecard = self.score(ctx, findings)
        self.last_scorecard = scorecard
        state, band = self.band(scorecard)

        prior = ctx.memory.working.get_hypothesis()
        needs_human = False
        debate_ref: Optional[str] = None
        used_fallback = False

        # --- step 3: debate, only on a conflict ----------------------------
        conflict = self._conflict(ctx, scorecard)
        if conflict is not None:
            self.debates += 1
            verdict, reasoning, fb = self.debate(ctx, scorecard, conflict)
            used_fallback |= fb
            a, b = conflict
            if verdict == "B":
                state = b.state
                band = self._band_for(b.score)
            elif verdict == "unresolved":
                needs_human = True
            ep = ctx.memory.episodic.open_episode(
                type="debate",
                evidence_event_ids=(a.evidence_ids + b.evidence_ids)[:10],
                notes=f"{a.state} vs {b.state} -> {verdict}"
                + (" [fallback]" if fb else ""),
                data={
                    "case_a": a.to_dict(),
                    "case_b": b.to_dict(),
                    "verdict": verdict,
                    "reasoning": reasoning,
                    "used_fallback": fb,
                },
            )
            debate_ref = ep.episode_id

        # --- step 4: review, only on change or after a debate --------------
        changed = (
            prior is None
            or prior.state != state
            or prior.confidence_band != band
            or conflict is not None
        )
        rationale: Optional[str] = None
        if changed and scorecard.top is not None:
            self.reviews += 1
            rationale, band, fb = self.review(ctx, scorecard, state, band, prior)
            used_fallback |= fb
        elif prior is not None:
            rationale = prior.rationale
        if not rationale:
            # No rule has matched yet (the first days of the live stream). Cite
            # the baseline, so every checkpoint's notes still carry an event id:
            # "nothing has deviated from THIS measured behaviour".
            rationale = self._baseline_rationale(ctx, state, band)

        # --- step 5: write -------------------------------------------------
        top = next((s for s in scorecard.ranked if s.state == state), scorecard.top)
        score = top.score if top else 0.0
        runner = scorecard.runner_up
        hypothesis = ctx.memory.working.set_hypothesis(
            state=state,
            score=score,
            confidence_band=band,
            supporting_events=(top.evidence_ids[:10] if top else []),
            runner_up=(
                {"state": runner.state, "score": round(runner.score, 4)} if runner else None
            ),
            notes=None,
            rationale=rationale,
            needs_human=needs_human,
            debate_ref=debate_ref,
            used_fallback=used_fallback,
        )
        ctx.log.log_agent_action(
            self.name,
            None,
            f"hypothesis {state} [{band}] score={score:.3f}"
            + (f" runner_up={runner.state}:{runner.score:.3f}" if runner else "")
            + (" [fallback]" if used_fallback else ""),
            hypothesis.supporting_events,
            as_of=ctx.now(),
            state=state,
            confidence_band=band,
            score=round(score, 4),
            needs_human=needs_human,
            debate_ref=debate_ref,
            used_fallback=used_fallback,
        )
        if prior is None or prior.state != state or prior.confidence_band != band:
            ctx.memory.episodic.open_episode(
                type="hypothesis_change",
                evidence_event_ids=hypothesis.supporting_events,
                notes=(
                    f"{prior.state if prior else 'none'}/"
                    f"{prior.confidence_band if prior else '-'} -> {state}/{band}"
                ),
                data={
                    "from": {"state": prior.state, "band": prior.confidence_band} if prior else None,
                    "to": {"state": state, "band": band, "score": round(score, 4)},
                },
            )
        return []  # writes a hypothesis, not findings

    # --- step 1: score ------------------------------------------------------

    def score(self, ctx: AgentContext, findings: dict[str, Finding]) -> Scorecard:
        now = ctx.now()
        suppressed = self._suppressed_events(ctx)
        buckets: dict[str, list[Contribution]] = {}

        for key, finding in findings.items():
            if finding.evidence_event_ids and any(
                eid in suppressed for eid in finding.evidence_event_ids
            ):
                # Spec: skip the contribution if ANY evidence id is suppressed.
                continue
            age_days = 0.0
            if finding.updated_at is not None:
                age_days = max(0.0, (now - finding.updated_at).total_seconds() / 86400)
            decay = 0.5 ** (age_days / T.EVIDENCE_HALF_LIFE_DAYS)
            value = finding.value or {}
            band = finding.confidence_band

            for rule in EVIDENCE_RULES:
                if rule["finding_key"] != key or not matches(rule, value, band):
                    continue
                buckets.setdefault(rule["state"], []).append(
                    Contribution(
                        finding_key=key,
                        event_ids=list(finding.evidence_event_ids),
                        weight=rule["weight"] * decay,
                        raw_weight=rule["weight"],
                        age_days=age_days,
                    )
                )

            for rule in LABEL_RULES:
                if rule["finding_key"] != key or not matches(rule, value, band):
                    continue
                for label in value.get(rule["labels_from"]) or []:
                    if label not in VALID_STATES:
                        if label not in self._unknown_labels:
                            self._unknown_labels.add(label)
                            log.info("unknown life-event label %r ignored", label)
                        continue
                    buckets.setdefault(label, []).append(
                        Contribution(
                            finding_key=key,
                            event_ids=list(finding.evidence_event_ids),
                            weight=rule["weight"] * decay,
                            raw_weight=rule["weight"],
                            age_days=age_days,
                        )
                    )

        ranked: list[StateScore] = []
        for state, contribs in buckets.items():
            product = 1.0
            for c in contribs:
                product *= 1.0 - min(1.0, max(0.0, c.weight))
            ranked.append(StateScore(state, 1.0 - product, contribs))
        # Deterministic: score desc, then state name.
        ranked.sort(key=lambda s: (-s.score, s.state))
        return Scorecard(ranked=ranked, suppressed_events=sorted(suppressed), generated_at=now)

    def _suppressed_events(self, ctx: AgentContext) -> set[str]:
        """Event ids currently covered by an 'explanation' episode."""
        now = ctx.now()
        out: set[str] = set()
        for ep in ctx.memory.episodic.query(type="explanation"):
            data = ep.data or {}
            target = data.get("suppresses")
            until = data.get("until")
            if not target:
                continue
            if until:
                try:
                    from c360.loader import parse_ts

                    if parse_ts(until) <= now:
                        continue
                except (ValueError, TypeError):
                    pass
            out.add(target)
        return out

    # --- step 2: band -------------------------------------------------------

    def _band_for(self, score: float) -> str:
        if score < T.LOW_MAX:
            return "low"
        if score < T.MEDIUM_MAX:
            return "medium"
        return "high"

    def band(self, scorecard: Scorecard) -> tuple[str, str]:
        top = scorecard.top
        if top is None or top.score < T.NO_EVENT_MAX:
            return NO_EVENT_STATE, "low"
        return top.state, self._band_for(top.score)

    # --- step 3: debate -----------------------------------------------------

    def _conflict(
        self, ctx: AgentContext, scorecard: Scorecard
    ) -> Optional[tuple[StateScore, StateScore]]:
        a, b = scorecard.top, scorecard.runner_up
        if a is None or b is None:
            return None
        if a.score < T.CONFLICT_MIN_SCORE or b.score < T.CONFLICT_MIN_SCORE:
            return None
        if a.score - b.score < T.CONFLICT_MARGIN:
            return (a, b)
        if frozenset({a.state, b.state}) in INCOMPATIBLE_PAIRS:
            return (a, b)
        # Fraud conflicts with anything when its own evidence was explained.
        for fraud, other in ((a, b), (b, a)):
            if fraud.state != "potential_fraud_or_takeover":
                continue
            if self._recently_explained(ctx, fraud.evidence_ids):
                return (a, b)
        return None

    def _recently_explained(self, ctx: AgentContext, event_ids: list[str]) -> bool:
        window = timedelta(hours=FRAUD_EXPLANATION_WINDOW_HOURS)
        now = ctx.now()
        for ep in ctx.memory.episodic.query(type="explanation"):
            if (ep.data or {}).get("suppresses") in event_ids:
                if ep.opened_at is None or now - ep.opened_at <= window:
                    return True
        return False

    def debate(
        self, ctx: AgentContext, scorecard: Scorecard, conflict: tuple[StateScore, StateScore]
    ) -> tuple[str, str, bool]:
        """Returns (verdict, reasoning, used_fallback)."""
        a, b = conflict
        prompt = self._debate_prompt(ctx, a, b)
        client = self._client()
        attempts = 1 + max(0, llm_config.MAX_RETRIES)
        last_error: Optional[str] = None
        for _ in range(attempts):
            try:
                raw = client.complete_json(DEBATE_PROMPT, prompt, "debate")
                self.llm_calls += 1
                verdict = raw.get("verdict")
                if verdict not in VERDICTS:
                    raise LLMInvalidJSON(f"verdict={verdict!r} not in {VERDICTS}")
                reasoning = str(raw.get("reasoning") or "").strip()
                ctx.log.log_agent_action(
                    self.name, None, f"debate {a.state} vs {b.state} -> {verdict}",
                    (raw.get("key_event_ids") or [])[:6], as_of=ctx.now(), verdict=verdict,
                )
                return verdict, reasoning, False
            except LLMUnavailable as exc:
                last_error = str(exc)
                break
            except (LLMInvalidJSON, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("%s debate: %s", self.name, last_error)

        verdict, reasoning = self._precedence_verdict(ctx, a, b)
        self.fallback_calls += 1
        ctx.log.log_agent_action(
            self.name, None,
            f"debate {a.state} vs {b.state} -> {verdict} [fallback: {last_error}]",
            [], as_of=ctx.now(), verdict=verdict, fallback=True,
        )
        return verdict, reasoning, True

    def _precedence_verdict(
        self, ctx: AgentContext, a: StateScore, b: StateScore
    ) -> tuple[str, str]:
        """Deterministic PRECEDENCE_RULES adjudication. A wins -> 'A'."""
        for rule in PRECEDENCE_RULES:
            spec = rule.get("beats") or {}
            for winner, loser, verdict in ((a, b, "A"), (b, a, "B")):
                if not spec:
                    # Terminal rule: higher score wins (a is already ranked first).
                    return "A", f"{rule['why']} ({a.state} {a.score:.2f} vs {b.state} {b.score:.2f})"
                ok = True
                if "winner_has_any" in spec:
                    ok &= any(k in winner.finding_keys for k in spec["winner_has_any"])
                if "loser_has_any" in spec:
                    ok &= any(k in loser.finding_keys for k in spec["loser_has_any"])
                if "winner_state" in spec:
                    ok &= winner.state == spec["winner_state"]
                if "loser_state" in spec:
                    ok &= loser.state == spec["loser_state"]
                if spec.get("loser_explained"):
                    ok &= self._recently_explained(ctx, loser.evidence_ids)
                if ok:
                    return verdict, (
                        f"{rule['why']} [{rule['rule_id']}] "
                        f"{winner.state} over {loser.state}."
                    )
        return "A", "No precedence rule applied; higher score stands."

    def _debate_prompt(self, ctx: AgentContext, a: StateScore, b: StateScore) -> str:
        lines = ["Two readings of the same customer.", ""]
        for label, side in (("A", a), ("B", b)):
            lines.append(f"READING {label}: {side.state} (score {side.score:.2f})")
            for c in sorted(side.contributions, key=lambda c: -c.weight)[:6]:
                desc = self._describe(ctx, c)
                lines.append(
                    f"  - {c.finding_key} (weight {c.weight:.2f}, {c.age_days:.0f}d old) "
                    f"{desc} evidence {','.join(c.event_ids[:4])}"
                )
            lines.append("")
        if a.evidence_ids or b.evidence_ids:
            explained = sorted(self._suppressed_events(ctx))
            if explained:
                lines.append(f"Transactions the customer already explained: {', '.join(explained)}")
        masked, _ = masking.mask("\n".join(lines), ctx.profile)
        return masked

    def _describe(self, ctx: AgentContext, c: Contribution) -> str:
        finding = ctx.memory.working.get_finding(c.finding_key)
        if finding is None:
            return ""
        from .base import _summarise

        text, _ = masking.mask(_summarise(finding.value, 120), ctx.profile)
        return text

    # --- step 4: review -----------------------------------------------------

    def review(
        self,
        ctx: AgentContext,
        scorecard: Scorecard,
        state: str,
        band: str,
        prior,
    ) -> tuple[str, str, bool]:
        """Returns (rationale, band, used_fallback). Band can only go DOWN."""
        top = next((s for s in scorecard.ranked if s.state == state), scorecard.top)
        valid_ids = {eid for s in scorecard.ranked for eid in s.evidence_ids}
        prompt = self._review_prompt(ctx, scorecard, state, band, prior)
        client = self._client()
        attempts = 1 + max(0, llm_config.MAX_RETRIES)
        last_error: Optional[str] = None

        for _ in range(attempts):
            try:
                raw = client.complete_json(SYNTHESIS_PROMPT, prompt, "synthesis_review")
                self.llm_calls += 1
                rationale = str(raw.get("rationale") or "").strip()
                if not rationale:
                    raise LLMInvalidJSON("empty rationale")
                # The rationale must cite an event id the scorecard actually has.
                if not any(eid in rationale for eid in valid_ids):
                    raise LLMInvalidJSON("rationale cites no scorecard event id")
                agrees = bool(raw.get("agrees"))
                out_band = band
                if not agrees:
                    out_band = lower_band(band)
                    rationale += f" (reviewer disagreed; confidence lowered to {out_band})"
                ctx.log.log_agent_action(
                    self.name, None,
                    f"review {state}: agrees={agrees} band {band}->{out_band}",
                    sorted(valid_ids)[:6], as_of=ctx.now(), agrees=agrees,
                    missing_evidence=raw.get("missing_evidence"),
                )
                return rationale, out_band, False
            except LLMUnavailable as exc:
                last_error = str(exc)
                break
            except (LLMInvalidJSON, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning("%s review: %s", self.name, last_error)

        self.fallback_calls += 1
        rationale = self._fallback_rationale(state, band, top)
        ctx.log.log_agent_action(
            self.name, None, f"review {state} [fallback: {last_error}]",
            (top.evidence_ids[:6] if top else []), as_of=ctx.now(), fallback=True,
        )
        return rationale, band, True

    def _baseline_rationale(self, ctx: AgentContext, state: str, band: str) -> str:
        """Rationale for a day with no matching evidence, citing the baseline."""
        baseline = ctx.memory.working.get_finding("baseline")
        ids = (baseline.evidence_event_ids[:2] if baseline else []) or ["baseline"]
        return (
            f"{state} ({band}): activity remains consistent with the measured "
            f"baseline [{','.join(ids)}]; no rule has matched yet."
        )

    def _fallback_rationale(self, state: str, band: str, top: Optional[StateScore]) -> str:
        """Deterministic rationale: state, band, top 3 findings with event ids."""
        if top is None:
            return f"{state} ({band}): no contributing evidence."
        parts = []
        for c in sorted(top.contributions, key=lambda c: -c.weight)[:3]:
            ids = ",".join(c.event_ids[:2]) or "baseline"
            parts.append(f"{c.finding_key} [{ids}]")
        return f"{state} ({band}): " + "; ".join(parts) + "."

    def _review_prompt(
        self, ctx: AgentContext, scorecard: Scorecard, state: str, band: str, prior
    ) -> str:
        lines = [f"Proposed assessment: {state} (confidence {band})", "", "Ranked candidates:"]
        for s in scorecard.ranked[:3]:
            lines.append(f"  {s.state}: score {s.score:.2f}")
            for c in sorted(s.contributions, key=lambda c: -c.weight)[:4]:
                lines.append(
                    f"    - {c.finding_key} (weight {c.weight:.2f}) "
                    f"{self._describe(ctx, c)} [{','.join(c.event_ids[:3])}]"
                )
        lines.append("")
        if prior is not None:
            lines.append(
                f"Previous assessment: {prior.state} ({prior.confidence_band}) "
                f"first inferred {iso(prior.first_inferred) if prior.first_inferred else '?'}"
            )
        top = next((s for s in scorecard.ranked if s.state == state), None)
        types = {c.finding_key for c in (top.contributions if top else [])}
        episodes = []
        for t in list(types) + ["intervention"]:
            episodes.extend(ctx.memory.episodic.query(type=t, limit=2))
        if episodes:
            lines.append("")
            lines.append("Relevant history:")
            for ep in episodes[:5]:
                when = ep.opened_at.strftime("%Y-%m-%d") if ep.opened_at else "?"
                lines.append(f"  - {when} {ep.type}: {ep.notes or ''}")
        masked, _ = masking.mask("\n".join(lines), ctx.profile)
        return masked
