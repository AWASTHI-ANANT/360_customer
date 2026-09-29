"""
================================================================================
AGENT_NAME : signal_agent
TYPE       : deterministic (NO LLM)

TRIGGERS
  on_start        : computes a Baseline from history_seed, stored as the
                    "baseline" finding.
  per event       : card_payments | core_banking_ledger | instant_payments |
                    ach_wire | loan_kyc | web_app_events(feature_used)
  per day boundary: always (windowed recomputation, including dead days)

INPUT
  event.payload + event.derived (is_salary, is_income_like,
  counterparty_is_self, counterparty_is_external_bank), and
  ctx.event_store windows. Never reads another agent's findings.

OUTPUT (findings written to memory.working)
  baseline                    {purchases_per_week, spend_per_week, p95_amount,
                               max_amount, logins_per_week, median_session_sec,
                               salary_amount, salary_cadence_days,
                               standing_instructions[], category_share{}}
  large_purchase              {amount, merchant, mcc_category, p95_baseline, ratio}
  large_inflow                {amount, transaction_type, salary_amount, ratio}
  outflow_pattern             {kind: large_outflow|salary_sweep, amount,
                               counterparty_is_self, counterparty_is_external_bank,
                               hours_after_salary?, fraction_of_salary?}
  standing_instruction_change {kind: cancel_page_used|expected_missing, ...}
  kyc_change                  {event_subtype, old_value, new_value}
  income_pattern              {kind: new_income_source|salary_amount_change|
                               salary_missing, ...}
  engagement_trend            {login_ratio, session_ratio, logins, expected_logins}
  card_activity_trend         {purchases_7d, expected_7d, ratio,
                               days_since_last_purchase, stopped}
  merchant_shift              {risen: {...}, new: {...}}

TOOLS
  ctx.event_store (windowed queries), ctx.memory (write findings + episodes),
  ctx.log (traceability), config.thresholds (all numbers).

PROMPT
  None -- this agent is fully deterministic and must stay that way, so that
  numeric signals are reproducible and independently auditable.
================================================================================

This agent does NOT decide life events or actions. It reports what the numbers
did. Interpretation belongs to a later reasoning stage.
"""

from __future__ import annotations

import logging
import statistics as stats
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional, Sequence

from c360.loader import Event
from config import thresholds as T
from config.categories import category_group, normalise

from .base import Agent, AgentContext, Finding

log = logging.getLogger(__name__)

# mcc_category values with no life-event group. Anything that is neither
# grouped (config/categories.py) nor listed here is logged (the brief asks for
# it) but still scored -- we never drop an unknown category. Compared after
# normalise(), so "groceries"/"grocery" and "baby_supplies"/"baby_products"
# are both recognised.
KNOWN_MCC_CATEGORIES = {
    normalise(c) for c in (
        "dining", "gas", "general_retail", "utilities", "entertainment", "travel",
        "transport", "transit", "home_improvement", "education", "insurance",
        "subscriptions", "clothing", "electronics", "hardware", "lodging",
    )
}


def is_known_category(name: str) -> bool:
    return category_group(name) is not None or normalise(name) in KNOWN_MCC_CATEGORIES

HANDLED_SOURCES = {
    "card_payments",
    "core_banking_ledger",
    "instant_payments",
    "ach_wire",
    "loan_kyc",
}


@dataclass(slots=True)
class Baseline:
    """The customer's ordinary behaviour, measured once from history_seed."""

    history_events: int = 0
    history_days: float = 0.0
    purchases_per_week: float = 0.0
    spend_per_week: float = 0.0
    p95_amount: float = 0.0
    max_amount: float = 0.0
    logins_per_week: float = 0.0
    median_session_sec: Optional[float] = None
    salary_amount: Optional[float] = None
    salary_cadence_days: Optional[float] = None
    last_salary_at: Optional[datetime] = None
    standing_instructions: list[dict[str, Any]] = field(default_factory=list)
    category_share: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["last_salary_at"] = self.last_salary_at.isoformat() if self.last_salary_at else None
        return d


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. Plain stdlib, no numpy."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def compute_baseline(history_events: Sequence[Event]) -> Baseline:
    """Measure ordinary behaviour from the history seed."""
    b = Baseline(history_events=len(history_events))
    if not history_events:
        return b

    span_days = max(
        1.0, (history_events[-1].event_time - history_events[0].event_time).total_seconds() / 86400
    )
    b.history_days = span_days
    weeks = max(span_days / 7.0, 1e-9)

    purchases = [
        e for e in history_events
        if e.source_system == "card_payments" and e.event_type == "purchase"
    ]
    amounts = sorted(e.amount for e in purchases if e.amount is not None)
    b.purchases_per_week = len(purchases) / weeks
    b.spend_per_week = sum(amounts) / weeks
    b.p95_amount = _percentile(amounts, 0.95)
    b.max_amount = amounts[-1] if amounts else 0.0

    logins = [e for e in history_events if e.event_type == "login"]
    b.logins_per_week = len(logins) / weeks

    sessions = [
        e.payload.get("session_length_sec")
        for e in history_events
        if e.event_type == "session_duration" and isinstance(e.payload.get("session_length_sec"), (int, float))
    ]
    b.median_session_sec = float(stats.median(sessions)) if sessions else None

    salaries = [e for e in history_events if e.derived.get("is_salary")]
    if salaries:
        amts = [e.amount for e in salaries if e.amount is not None]
        # Median, so one odd bonus-month salary doesn't move the baseline.
        b.salary_amount = float(stats.median(amts)) if amts else None
        b.last_salary_at = salaries[-1].event_time
        if len(salaries) > 1:
            gaps = [
                (b_.event_time - a.event_time).total_seconds() / 86400
                for a, b_ in zip(salaries, salaries[1:])
            ]
            b.salary_cadence_days = float(stats.median(gaps))

    # Standing instructions: one entry per (type, amount, day-of-month) seen.
    si = [e for e in history_events if e.event_type == "standing_instruction"]
    seen: dict[tuple, dict[str, Any]] = {}
    for e in si:
        key = (e.payload.get("transaction_type"), e.amount, e.event_time.day)
        seen.setdefault(
            key,
            {
                "transaction_type": e.payload.get("transaction_type"),
                "amount": e.amount,
                "day_of_month": e.event_time.day,
                "occurrences": 0,
            },
        )
        seen[key]["occurrences"] += 1
    b.standing_instructions = sorted(seen.values(), key=lambda d: (-d["occurrences"], str(d["transaction_type"])))

    total_spend = sum(amounts) or 1.0
    cat = Counter()
    for e in purchases:
        if e.amount is not None:
            cat[e.payload.get("mcc_category") or "unknown"] += e.amount
    b.category_share = {k: v / total_spend for k, v in cat.most_common()}
    return b


class SignalAgent(Agent):
    name = "signal_agent"

    def __init__(self) -> None:
        super().__init__()
        self.baseline = Baseline()
        # Remembers which (year, month) standing-instruction absences we've
        # already reported, so one missing instruction isn't reported daily.
        self._si_absence_reported: set[tuple[int, int, str]] = set()
        self._unknown_categories: set[str] = set()

    # --- startup ------------------------------------------------------------

    def on_start(self, ctx: AgentContext, history: Sequence[Event]) -> list[Finding]:
        self.baseline = compute_baseline(history)
        evidence = [history[0].event_id, history[-1].event_id] if history else []
        finding = self.emit(
            ctx,
            "baseline",
            self.baseline.to_dict(),
            "high",
            evidence,
            notes=f"measured from {len(history)} history events over {self.baseline.history_days:.0f} days",
            force=True,
        )
        return [finding] if finding else []

    # --- routing ------------------------------------------------------------

    def handles(self, event: Event) -> bool:
        if event.source_system in HANDLED_SOURCES:
            return True
        return event.source_system == "web_app_events" and event.event_type == "feature_used"

    # --- per event ----------------------------------------------------------

    def on_event(self, event: Event, ctx: AgentContext) -> list[Finding]:
        out: list[Optional[Finding]] = []
        ss, et = event.source_system, event.event_type
        amt = event.amount

        if ss == "card_payments" and et == "purchase" and amt is not None:
            out.append(self._large_purchase(event, ctx, amt))
        if ss == "card_payments" and et == "refund" and amt is not None:
            out.append(self._large_inflow(event, ctx, amt, "refund"))
        if ss == "core_banking_ledger" and et == "deposit" and amt is not None:
            out.append(self._large_inflow(event, ctx, amt, event.derived.get("transaction_type")))
            out.append(self._income_event(event, ctx, amt))
        if ss in ("instant_payments", "ach_wire") and event.derived.get("is_outbound") and amt is not None:
            out.extend(self._outflow(event, ctx, amt))
        if ss == "web_app_events" and et == "feature_used":
            out.append(self._si_cancel_page(event, ctx))
        if ss == "loan_kyc":
            out.append(self._kyc_change(event, ctx))

        return [f for f in out if f is not None]

    def _large_purchase(self, event: Event, ctx: AgentContext, amt: float) -> Optional[Finding]:
        p95 = self.baseline.p95_amount
        if p95 <= 0:
            return None
        threshold = T.LARGE_PURCHASE_MULT * p95
        if amt <= threshold:
            return None
        return self.emit(
            ctx,
            "large_purchase",
            {
                "amount": amt,
                "merchant": event.payload.get("merchant_name"),
                "mcc_category": event.payload.get("mcc_category"),
                "p95_baseline": p95,
                "ratio": amt / p95,
            },
            "medium" if amt < 2 * threshold else "high",
            [event.event_id],
            notes=f"{amt:.0f} vs baseline p95 {p95:.0f}",
            event_id=event.event_id,
        )

    def _large_inflow(self, event: Event, ctx: AgentContext, amt: float, txn_type) -> Optional[Finding]:
        """A big credit that is not salary. Recorded, NOT interpreted."""
        if event.derived.get("is_salary"):
            return None
        salary = self.baseline.salary_amount
        if not salary:
            return None
        if amt <= T.LARGE_INFLOW_SALARY_FRACTION * salary:
            return None
        return self.emit(
            ctx,
            "large_inflow",
            {
                # transaction_type is reported verbatim; deciding whether a
                # tax_refund is a windfall is not this agent's job.
                "transaction_type": txn_type,
                "amount": amt,
                "salary_amount": salary,
                "ratio": amt / salary,
                "source_system": event.source_system,
                "event_type": event.event_type,
            },
            "medium",
            [event.event_id],
            notes=f"{txn_type} {amt:.0f} = {amt / salary:.2f}x salary",
            event_id=event.event_id,
        )

    def _outflow(self, event: Event, ctx: AgentContext, amt: float) -> list[Optional[Finding]]:
        salary = self.baseline.salary_amount
        found: list[Optional[Finding]] = []
        if not salary:
            return found

        is_large = amt > T.LARGE_OUTFLOW_SALARY_MULT * salary

        # Salary sweep: money leaves right after it arrives.
        last_salary = ctx.event_store.last(
            before=event.event_time, where=lambda e: e.derived.get("is_salary")
        )
        sweep = None
        if last_salary is not None:
            hours = (event.event_time - last_salary.event_time).total_seconds() / 3600
            sal_amt = last_salary.amount or salary
            if hours <= T.SWEEP_WINDOW_HOURS and amt >= T.SWEEP_FRACTION * sal_amt:
                sweep = (hours, sal_amt, last_salary.event_id)

        if is_large:
            found.append(
                self.emit(
                    ctx,
                    "outflow_pattern",
                    {
                        "kind": "large_outflow",
                        "amount": amt,
                        "salary_multiple": amt / salary,
                        "counterparty_is_self": bool(event.derived.get("counterparty_is_self")),
                        "counterparty_is_external_bank": bool(
                            event.derived.get("counterparty_is_external_bank")
                        ),
                        "counterparty_country": event.payload.get("counterparty_country"),
                        "transfer_type": event.payload.get("transfer_type"),
                        "is_also_salary_sweep": sweep is not None,
                    },
                    "medium",
                    [event.event_id],
                    notes=f"outbound {amt:.0f} = {amt / salary:.1f}x salary",
                    event_id=event.event_id,
                )
            )
        if sweep is not None:
            hours, sal_amt, sal_id = sweep
            found.append(
                self.emit(
                    ctx,
                    "outflow_pattern",
                    {
                        "kind": "salary_sweep",
                        "amount": amt,
                        "hours_after_salary": round(hours, 2),
                        "salary_amount": sal_amt,
                        "fraction_of_salary": amt / sal_amt if sal_amt else None,
                        "counterparty_is_self": bool(event.derived.get("counterparty_is_self")),
                        "counterparty_is_external_bank": bool(
                            event.derived.get("counterparty_is_external_bank")
                        ),
                    },
                    "medium",
                    # Both the transfer and the salary it followed are evidence.
                    [event.event_id, sal_id],
                    notes=f"{amt:.0f} swept {hours:.1f}h after salary {sal_amt:.0f}",
                    event_id=event.event_id,
                )
            )
        return found

    def _si_cancel_page(self, event: Event, ctx: AgentContext) -> Optional[Finding]:
        page = str(event.payload.get("feature_or_page") or "").lower()
        if "standing_instruction" not in page or "cancel" not in page:
            return None
        return self.emit(
            ctx,
            "standing_instruction_change",
            {
                "kind": "cancel_page_used",
                "feature_or_page": event.payload.get("feature_or_page"),
                "device_type": event.payload.get("device_type"),
            },
            "medium",
            [event.event_id],
            notes="customer opened the standing-instruction cancellation page",
            event_id=event.event_id,
        )

    def _kyc_change(self, event: Event, ctx: AgentContext) -> Optional[Finding]:
        return self.emit(
            ctx,
            "kyc_change",
            {
                "event_subtype": event.payload.get("event_subtype") or event.event_type,
                "old_value": event.payload.get("old_value"),
                "new_value": event.payload.get("new_value"),
                "event_type": event.event_type,
            },
            "high",  # a declared KYC change is a fact, not an inference
            [event.event_id],
            notes=f"{event.event_type}: {event.payload.get('old_value')} -> {event.payload.get('new_value')}",
            event_id=event.event_id,
        )

    def _income_event(self, event: Event, ctx: AgentContext, amt: float) -> Optional[Finding]:
        """A new income-like source, or a salary whose amount moved."""
        derived = event.derived
        baseline_salary = self.baseline.salary_amount

        if derived.get("is_salary"):
            if not baseline_salary:
                return None
            delta = abs(amt - baseline_salary) / baseline_salary
            if delta <= T.INCOME_CHANGE_FRACTION:
                return None
            return self.emit(
                ctx,
                "income_pattern",
                {
                    "kind": "salary_amount_change",
                    "amount": amt,
                    "baseline_salary": baseline_salary,
                    "change_fraction": (amt - baseline_salary) / baseline_salary,
                    "direction": "up" if amt > baseline_salary else "down",
                },
                "high",
                [event.event_id],
                notes=f"salary {amt:.0f} vs baseline {baseline_salary:.0f}",
                event_id=event.event_id,
            )

        if derived.get("is_income_like"):
            return self.emit(
                ctx,
                "income_pattern",
                {
                    "kind": "new_income_source",
                    "transaction_type": derived.get("transaction_type"),
                    "amount": amt,
                    "baseline_salary": baseline_salary,
                    "fraction_of_salary": (amt / baseline_salary) if baseline_salary else None,
                },
                "high",
                [event.event_id],
                notes=f"income-like credit {derived.get('transaction_type')} {amt:.0f}",
                event_id=event.event_id,
            )
        return None

    # --- per day boundary ---------------------------------------------------

    def on_day_boundary(self, sim_date: date, ctx: AgentContext) -> list[Finding]:
        out: list[Optional[Finding]] = [
            self._engagement_trend(sim_date, ctx),
            self._card_activity_trend(sim_date, ctx),
            self._salary_absence(sim_date, ctx),
            self._merchant_shift(sim_date, ctx),
        ]
        out.extend(self._si_absence(sim_date, ctx))
        return [f for f in out if f is not None]

    def _engagement_trend(self, sim_date: date, ctx: AgentContext) -> Optional[Finding]:
        if self.baseline.logins_per_week <= 0:
            return None
        window = T.ENGAGEMENT_WINDOW_DAYS
        logins = ctx.event_store.trailing(sim_date, window, event_type="login")
        expected = self.baseline.logins_per_week * (window / 7.0)
        login_ratio = len(logins) / expected if expected else None

        sessions = [
            e.payload.get("session_length_sec")
            for e in ctx.event_store.trailing(sim_date, window, event_type="session_duration")
            if isinstance(e.payload.get("session_length_sec"), (int, float))
        ]
        med = float(stats.median(sessions)) if sessions else None
        base_med = self.baseline.median_session_sec
        session_ratio = (med / base_med) if (med is not None and base_med) else None

        if login_ratio is None:
            return None
        # Band on the login ratio; session length is reported as context.
        if login_ratio < T.ENGAGEMENT_DROP_HIGH:
            band = "high"
        elif login_ratio < T.ENGAGEMENT_DROP_MED:
            band = "medium"
        else:
            band = "low"

        return self.emit(
            ctx,
            "engagement_trend",
            {
                "window_days": window,
                "logins": len(logins),
                "expected_logins": round(expected, 2),
                "login_ratio": round(login_ratio, 3),
                "median_session_sec": med,
                "baseline_median_session_sec": base_med,
                "session_ratio": round(session_ratio, 3) if session_ratio is not None else None,
                "direction": "down" if login_ratio < 1 else "up",
            },
            band,
            [e.event_id for e in logins] or ["baseline"],
            notes=f"{len(logins)} logins in {window}d vs {expected:.1f} expected",
        )

    def _card_activity_trend(self, sim_date: date, ctx: AgentContext) -> Optional[Finding]:
        base_weekly = self.baseline.purchases_per_week
        if base_weekly <= 0:
            return None
        window = T.CARD_WINDOW_DAYS
        purchases = ctx.event_store.trailing(
            sim_date, window, source_system="card_payments", event_type="purchase"
        )
        expected = base_weekly * (window / 7.0)
        ratio = len(purchases) / expected if expected else None

        last = ctx.event_store.last(
            event_type="purchase", source_system="card_payments", before=_midnight(sim_date)
        )
        days_since = (
            (_midnight(sim_date) - last.event_time).total_seconds() / 86400 if last else None
        )
        stopped = bool(
            days_since is not None
            and days_since >= T.CARD_SILENCE_DAYS
            and base_weekly > T.MIN_BASELINE_WEEKLY
        )

        if ratio is None:
            return None
        if stopped:
            band = "high"
        elif ratio < T.ENGAGEMENT_DROP_HIGH:
            band = "high"
        elif ratio < T.ENGAGEMENT_DROP_MED:
            band = "medium"
        else:
            band = "low"

        return self.emit(
            ctx,
            "card_activity_trend",
            {
                "window_days": window,
                "purchases_7d": len(purchases),
                "expected_7d": round(expected, 2),
                "ratio": round(ratio, 3),
                "days_since_last_purchase": round(days_since, 2) if days_since is not None else None,
                "stopped": stopped,
                "baseline_weekly": round(base_weekly, 2),
            },
            band,
            [e.event_id for e in purchases] or ([last.event_id] if last else ["baseline"]),
            notes=(
                f"card usage stopped: {days_since:.0f}d since last purchase"
                if stopped
                else f"{len(purchases)} purchases in {window}d vs {expected:.1f} expected"
            ),
            # days_since_last_purchase climbs every day while the finding means
            # exactly the same thing, so it must not drive change detection --
            # otherwise this one key floods the log with a line per dead day.
            dedupe_on={"stopped": stopped, "purchases_7d": len(purchases)},
        )

    def _salary_absence(self, sim_date: date, ctx: AgentContext) -> Optional[Finding]:
        cadence = self.baseline.salary_cadence_days
        if not cadence:
            return None
        last = ctx.event_store.last(
            before=_midnight(sim_date), where=lambda e: e.derived.get("is_salary")
        )
        if last is None:
            return None
        due = last.event_time + timedelta(days=cadence + T.SALARY_GRACE_DAYS)
        if _midnight(sim_date) < due:
            return None
        overdue = (_midnight(sim_date) - due).total_seconds() / 86400
        return self.emit(
            ctx,
            "income_pattern",
            {
                "kind": "salary_missing",
                "last_salary_at": last.event_time.isoformat(),
                "last_salary_amount": last.amount,
                "cadence_days": cadence,
                "grace_days": T.SALARY_GRACE_DAYS,
                "days_overdue": round(overdue, 1),
            },
            "high",
            [last.event_id],
            notes=f"no salary since {last.event_time:%Y-%m-%d} ({overdue:.0f}d past due)",
        )

    def _si_absence(self, sim_date: date, ctx: AgentContext) -> list[Optional[Finding]]:
        """An expected standing instruction that never posted this month."""
        found: list[Optional[Finding]] = []
        for si in self.baseline.standing_instructions:
            dom = si.get("day_of_month")
            txn = si.get("transaction_type")
            if not dom or not txn:
                continue
            # Judge the month whose grace period has just elapsed -- NOT
            # sim_date's own month. Deriving the expected date from sim_date
            # would skip a late-month instruction entirely: with dom=28 and 3
            # grace days, the check first runs on Mar 3, by which point
            # sim_date.replace(day=28) points at March and February is never
            # evaluated at all.
            anchor = sim_date - timedelta(days=T.SI_GRACE_DAYS)
            try:
                expected_date = date(anchor.year, anchor.month, int(dom))
            except ValueError:
                continue  # e.g. day 31 in a 30-day month
            if anchor < expected_date:
                continue  # grace has not elapsed for this month's occurrence
            marker = (anchor.year, anchor.month, str(txn))
            if marker in self._si_absence_reported:
                continue
            # Look for it anywhere in the month being judged.
            month_start = date(anchor.year, anchor.month, 1)
            next_month = (
                date(anchor.year + 1, 1, 1)
                if anchor.month == 12
                else date(anchor.year, anchor.month + 1, 1)
            )
            window_end = min(_midnight(sim_date), _midnight(next_month))
            posted = [
                e
                for e in ctx.event_store.window(
                    month_start, window_end, event_type="standing_instruction"
                )
                if e.payload.get("transaction_type") == txn
            ]
            self._si_absence_reported.add(marker)
            if posted:
                continue
            found.append(
                self.emit(
                    ctx,
                    "standing_instruction_change",
                    {
                        "kind": "expected_missing",
                        "transaction_type": txn,
                        "expected_amount": si.get("amount"),
                        "expected_day_of_month": dom,
                        "month": f"{anchor.year}-{anchor.month:02d}",
                        "grace_days": T.SI_GRACE_DAYS,
                    },
                    "medium",
                    # Evidence for an absence is the prior occurrences that set
                    # the expectation.
                    self._si_evidence(ctx, txn),
                    notes=f"{txn} did not post in {anchor.year}-{anchor.month:02d}",
                )
            )
        return found

    def _si_evidence(self, ctx: AgentContext, txn: str) -> list[str]:
        prior = [
            e.event_id
            for e in ctx.event_store.all()
            if e.event_type == "standing_instruction" and e.payload.get("transaction_type") == txn
        ]
        return prior[-3:] or ["baseline"]

    def _merchant_shift(self, sim_date: date, ctx: AgentContext) -> Optional[Finding]:
        base = self.baseline.category_share
        if not base:
            return None
        window = T.MERCHANT_WINDOW_DAYS
        purchases = [
            e
            for e in ctx.event_store.trailing(
                sim_date, window, source_system="card_payments", event_type="purchase"
            )
            if e.amount is not None
        ]
        total = sum(e.amount for e in purchases)
        if total <= 0:
            return None
        recent = Counter()
        for e in purchases:
            recent[e.payload.get("mcc_category") or "unknown"] += e.amount

        risen: dict[str, dict[str, float]] = {}
        new: dict[str, dict[str, float]] = {}
        evidence: list[str] = []
        for cat, spend in recent.items():
            share = spend / total
            if not is_known_category(cat) and cat not in self._unknown_categories:
                self._unknown_categories.add(cat)
                log.warning("unrecognised mcc_category %r (first seen %s)", cat, sim_date)
            prior = base.get(cat)
            cat_events = [e.event_id for e in purchases if (e.payload.get("mcc_category") or "unknown") == cat]
            if prior is None:
                new[cat] = {"share": round(share, 4), "spend": round(spend, 2)}
                evidence.extend(cat_events[:3])
            elif share - prior > T.MERCHANT_SHIFT_DELTA:
                risen[cat] = {
                    "share": round(share, 4),
                    "baseline_share": round(prior, 4),
                    "delta": round(share - prior, 4),
                }
                evidence.extend(cat_events[:3])
        if not risen and not new:
            return None
        return self.emit(
            ctx,
            "merchant_shift",
            {
                "window_days": window,
                "risen": risen,
                "new": new,
                "unknown_categories": sorted(self._unknown_categories) or None,
            },
            "medium" if (risen or new) else "low",
            evidence or ["baseline"],
            notes="; ".join(
                [f"{c} +{v['delta']:.0%}" for c, v in risen.items()]
                + [f"new {c} {v['share']:.0%}" for c, v in new.items()]
            ),
        )


def _midnight(d: date) -> datetime:
    from datetime import time as dtime, timezone

    return datetime.combine(d, dtime.min, tzinfo=timezone.utc)
