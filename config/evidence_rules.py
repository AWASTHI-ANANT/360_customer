"""Evidence -> state weights, incompatible pairs, and the debate fallback.

ASSUMPTIONS (spec was silent):
- A rule's `condition` is a callable (value: dict, band: str) -> bool. Returning
  a truthy value counts as a match; any exception inside a condition is caught,
  logged and treated as "no match", so a malformed finding can never crash a run.
- Working memory holds exactly one finding per key (the latest), so each
  finding_key contributes AT MOST ONCE per state per scoring pass. Repeated
  emissions of the same key over the run do not stack.
- `merchant_shift` categories are matched against the finding's `risen` and
  `new` category names by normalised group (config/categories.py), not by
  exact name.
- An unrecognised mcc_category or life-event label is logged once and ignored;
  it never raises (per spec: "Unknown category names are logged, never crash").
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from config.categories import category_group

log = logging.getLogger(__name__)


# transaction_type fragments that mean income was replaced by a benefit.
BENEFIT_TOKENS = ("benefit", "disability")


def _has(value: dict[str, Any], group: str) -> bool:
    """A merchant_shift finding's risen or new categories include `group`."""
    names = list((value.get("risen") or {}).keys()) + list((value.get("new") or {}).keys())
    return any(category_group(n) == group for n in names)


def _kind(value: dict[str, Any]) -> str:
    return str(value.get("kind") or "")


def _txn(value: dict[str, Any]) -> str:
    return str(value.get("transaction_type") or "").lower()


# Each rule: finding_key, condition(value, band) -> bool, state, weight.
# One finding may match several rules and so vote for several states at once;
# that is intended, and is what makes CONFLICT_MARGIN meaningful.
EVIDENCE_RULES: list[dict[str, Any]] = [
    # --- churn -------------------------------------------------------------
    {"finding_key": "relationship_damage", "condition": lambda v, b: True,
     "state": "churn_risk", "weight": 0.25},
    {"finding_key": "sentiment_state",
     "condition": lambda v, b: v.get("sentiment") == "negative" and v.get("intent") == "fee_dispute",
     "state": "churn_risk", "weight": 0.10},
    {"finding_key": "engagement_trend", "condition": lambda v, b: b == "medium",
     "state": "churn_risk", "weight": 0.10},
    {"finding_key": "engagement_trend", "condition": lambda v, b: b == "high",
     "state": "churn_risk", "weight": 0.20},
    {"finding_key": "card_activity_trend", "condition": lambda v, b: bool(v.get("stopped")),
     "state": "churn_risk", "weight": 0.25},
    {"finding_key": "standing_instruction_change", "condition": lambda v, b: True,
     "state": "churn_risk", "weight": 0.30},
    {"finding_key": "outflow_pattern",
     "condition": lambda v, b: _kind(v) == "large_outflow"
     and bool(v.get("counterparty_is_self")) and bool(v.get("counterparty_is_external_bank")),
     "state": "churn_risk", "weight": 0.35},
    {"finding_key": "outflow_pattern", "condition": lambda v, b: _kind(v) == "salary_sweep",
     "state": "churn_risk", "weight": 0.30},

    # --- fraud -------------------------------------------------------------
    {"finding_key": "outflow_pattern",
     "condition": lambda v, b: _kind(v) == "large_outflow" and not v.get("counterparty_is_self"),
     "state": "potential_fraud_or_takeover", "weight": 0.15},
    # "not suppressed" is enforced centrally in synthesis_agent.score(), which
    # drops any contribution whose evidence is under an explanation episode.
    {"finding_key": "large_purchase", "condition": lambda v, b: True,
     "state": "potential_fraud_or_takeover", "weight": 0.25},

    # --- windfall ----------------------------------------------------------
    {"finding_key": "large_inflow", "condition": lambda v, b: True,
     "state": "wealth_growth_or_windfall", "weight": 0.20},

    # --- income ------------------------------------------------------------
    {"finding_key": "income_pattern",
     "condition": lambda v, b: _kind(v) == "new_income_source"
     and any(t in _txn(v) for t in BENEFIT_TOKENS),
     "state": "medical_hardship", "weight": 0.25},
    {"finding_key": "income_pattern",
     "condition": lambda v, b: _kind(v) == "new_income_source"
     and any(t in _txn(v) for t in BENEFIT_TOKENS),
     "state": "job_loss_or_income_disruption", "weight": 0.20},
    {"finding_key": "income_pattern",
     "condition": lambda v, b: _kind(v) == "salary_amount_change" and v.get("direction") == "down",
     "state": "job_loss_or_income_disruption", "weight": 0.15},
    {"finding_key": "income_pattern",
     "condition": lambda v, b: _kind(v) == "salary_amount_change" and v.get("direction") == "down",
     "state": "new_child_life_event", "weight": 0.10},
    {"finding_key": "income_pattern",
     "condition": lambda v, b: _kind(v) == "salary_amount_change" and v.get("direction") == "down",
     "state": "medical_hardship", "weight": 0.10},
    {"finding_key": "income_pattern", "condition": lambda v, b: _kind(v) == "salary_missing",
     "state": "job_loss_or_income_disruption", "weight": 0.30},

    # --- spend mix ---------------------------------------------------------
    {"finding_key": "merchant_shift",
     "condition": lambda v, b: _has(v, "health"),
     # 0.20 -> 0.35 (docs/tuning_log.md): the only direct medical-cost signal,
     # weighted like the other direct signals (SI cancel 0.30, exit transfer 0.35).
     "state": "medical_hardship", "weight": 0.35},
    {"finding_key": "merchant_shift",
     "condition": lambda v, b: _has(v, "baby"),
     "state": "new_child_life_event", "weight": 0.20},

    # --- declared facts (KYC) ---------------------------------------------
    {"finding_key": "kyc_change",
     "condition": lambda v, b: "dependents" in str(v.get("event_subtype") or v.get("event_type") or "")
     and _increased(v),
     # 0.45 -> 0.60 (docs/tuning_log.md): a customer-declared dependents
     # increase is the strongest single fact for this state.
     "state": "new_child_life_event", "weight": 0.60},
    {"finding_key": "kyc_change",
     "condition": lambda v, b: "marital" in str(v.get("event_subtype") or v.get("event_type") or ""),
     "state": "marriage_or_relationship_change", "weight": 0.45},
    {"finding_key": "kyc_change",
     "condition": lambda v, b: "address" in str(v.get("event_subtype") or v.get("event_type") or ""),
     "state": "relocation", "weight": 0.35},

    # --- stated intent -----------------------------------------------------
    {"finding_key": "sentiment_state",
     "condition": lambda v, b: v.get("intent") == "hardship_request",
     "state": "financial_distress_general", "weight": 0.20},
]

# Rules whose target state comes from the finding itself (a label list) rather
# than being fixed. Handled separately in synthesis_agent.score().
LABEL_RULES: list[dict[str, Any]] = [
    {"finding_key": "life_event_hint", "labels_from": "labels", "weight": 0.20},
    {"finding_key": "intent_signal", "labels_from": "life_event_hints", "weight": 0.20},
    # A hardship request also lends weight to whatever label it hinted at.
    {"finding_key": "sentiment_state", "labels_from": "life_event_hints", "weight": 0.20,
     "condition": lambda v, b: v.get("intent") == "hardship_request"},
]


def _increased(value: dict[str, Any]) -> bool:
    """dependents_change went up. Tolerates non-numeric values."""
    try:
        return float(value.get("new_value")) > float(value.get("old_value"))
    except (TypeError, ValueError):
        # A non-numeric dependents change is treated as an increase, since the
        # dataset only ever records additions; logged so it is visible.
        log.info("dependents_change with non-numeric values: %r -> %r",
                 value.get("old_value"), value.get("new_value"))
        return True


def matches(rule: dict[str, Any], value: dict[str, Any], band: str) -> bool:
    """Evaluate a rule condition defensively; a bad condition is never fatal."""
    cond: Callable[[dict, str], Any] | None = rule.get("condition")
    if cond is None:
        return True
    try:
        return bool(cond(value or {}, band))
    except Exception as exc:  # noqa: BLE001 - a malformed finding must not crash a run
        log.warning("evidence rule %s/%s condition raised %s: %s",
                    rule.get("finding_key"), rule.get("state"), type(exc).__name__, exc)
        return False


# --- conflict definition ----------------------------------------------------
# Pairs that cannot both be true, so a near-tie between them must be adjudicated
# even when the score gap is wide. Stored as frozensets for order-free lookup.
INCOMPATIBLE_PAIRS: list[frozenset[str]] = [
    frozenset({"churn_risk", "wealth_growth_or_windfall"}),
    frozenset({"medical_hardship", "financial_distress_general"}),
]

# potential_fraud_or_takeover conflicts with ANY other state when its evidence
# has a customer explanation within this window. Handled in code.
FRAUD_EXPLANATION_WINDOW_HOURS = 72


# --- debate fallback --------------------------------------------------------
# Applied in order; the first rule that fires decides. Used whenever the LLM is
# offline, refuses, or returns output that fails validation twice.
EXIT_EVIDENCE_KINDS = ("large_outflow", "salary_sweep")
EXIT_FINDING_KEYS = ("standing_instruction_change", "outflow_pattern", "card_activity_trend")
INFLOW_FINDING_KEYS = ("large_inflow",)

PRECEDENCE_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "exit_beats_inflow",
        "why": "Money leaving for an external bank, a cancelled standing "
               "instruction or a salary sweep is exit behaviour; a one-off "
               "credit is not evidence against it.",
        # Decided in synthesis_agent._precedence_verdict against the scorecard.
        "beats": {"winner_has_any": EXIT_FINDING_KEYS, "loser_has_any": INFLOW_FINDING_KEYS},
    },
    {
        "rule_id": "explanation_beats_fraud",
        "why": "The customer explained the transaction that raised suspicion.",
        "beats": {"loser_state": "potential_fraud_or_takeover", "loser_explained": True},
    },
    {
        "rule_id": "specific_beats_general",
        "why": "A specific cause explains the evidence better than a general one.",
        "beats": {"winner_state": "medical_hardship", "loser_state": "financial_distress_general"},
    },
    {
        "rule_id": "higher_score_wins",
        "why": "No precedence rule applied; the higher-scoring reading stands.",
        "beats": {},  # always matches, so it must stay last
    },
]
