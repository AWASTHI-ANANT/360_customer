"""Action selection table, sensitive-inference list, and offline draft templates.

ASSUMPTIONS (spec was silent):
- ACTION_TABLE entries are (minimum_band, action). A state absent from the table
  never produces an action. "band >= X" uses the low<medium<high ordering.
- churn_risk branches on profile.customer_value_tier: "high" tier gets a
  relationship_manager_escalation, every other tier gets outreach.
- SENSITIVE_WORDS is checked case-insensitively as substrings against the
  drafted customer_message only (never the rm_brief, which is allowed to state
  the inference). Word fragments are used deliberately so "hospitalised" and
  "pregnancy" are caught as well as "hospital" and "pregnant".
- Offline templates mention no inference at all, so they pass check (e) for
  every state by construction.
"""

from __future__ import annotations

BAND_ORDER = {"low": 0, "medium": 1, "high": 2}


def band_at_least(band: str, minimum: str) -> bool:
    return BAND_ORDER.get(band, -1) >= BAND_ORDER.get(minimum, 99)


# state -> (minimum band, action). churn_risk is resolved by tier at runtime.
ACTION_TABLE: dict[str, tuple[str, str]] = {
    "churn_risk": ("high", "__tier_branch__"),
    "medical_hardship": ("high", "support_intervention"),
    "job_loss_or_income_disruption": ("high", "support_intervention"),
    "financial_distress_general": ("high", "support_intervention"),
    "new_child_life_event": ("high", "personalized_offer"),
    "marriage_or_relationship_change": ("high", "personalized_offer"),
    "job_change_or_promotion": ("high", "personalized_offer"),
    "relocation": ("high", "personalized_offer"),
    "retirement_transition": ("high", "personalized_offer"),
    "wealth_growth_or_windfall": ("high", "personalized_offer"),
    # FRAUD_ACTION_MIN_BAND is substituted for these two at runtime.
    "potential_fraud_or_takeover": ("__fraud_min__", "compliance_fraud_hold"),
    "elder_vulnerability_or_scam_risk": ("__fraud_min__", "compliance_fraud_hold"),
}

# Tier branch for churn_risk.
CHURN_ACTION_BY_TIER = {"high": "relationship_manager_escalation"}
CHURN_ACTION_DEFAULT = "proactive_retention_outreach"

# States whose inference is private. An offer based on one of these always goes
# to a human, and the customer-facing text must not reveal the inference.
# Mirrors policy/action_policy.md -> ## sensitive_inferences.
SENSITIVE_INFERENCES = {
    "new_child_life_event",
    "medical_hardship",
    "relocation",
    "marriage_or_relationship_change",
    "financial_distress_general",
    "job_loss_or_income_disruption",
}

# Words a customer message must never contain, per state. Checked as
# case-insensitive substrings so inflected forms are caught too.
SENSITIVE_WORDS: dict[str, tuple[str, ...]] = {
    "medical_hardship": (
        "hospital", "medical", "surgery", "surgeon", "illness", "ill health",
        "diagnos", "treatment", "disability", "clinic", "prescription",
    ),
    "new_child_life_event": (
        "baby", "babies", "pregnan", "newborn", "maternity", "paternity",
        "childcare", "nursery", "expecting",
    ),
    "marriage_or_relationship_change": (
        "divorce", "separation", "married", "marriage", "wedding", "partner",
    ),
    "relocation": ("moving", "relocat", "new home", "new address"),
    "financial_distress_general": (
        "debt", "struggling", "hardship", "arrears", "overdue", "distress",
        "cannot afford", "can't afford",
    ),
    "job_loss_or_income_disruption": (
        "job loss", "lost your job", "unemploy", "redundan", "laid off",
        "out of work",
    ),
}

# Fallback customer_message per action. Deliberately generic: they state no
# inference, so critique check (e) passes for every state by construction.
# None means the action has no customer-facing message at all.
DRAFT_TEMPLATES: dict[str, str | None] = {
    "proactive_retention_outreach": (
        "We would like to make sure your accounts are still working well for "
        "you. If it would help to talk through your options with someone, just "
        "reply and we will arrange a convenient time. We are glad to have you "
        "with us."
    ),
    "personalized_offer": (
        "We have reviewed your accounts and there are a few products you may "
        "now be eligible for. If you would like a short summary of the options "
        "available to you, reply to this message and we will send one over."
    ),
    "support_intervention": (
        "We want to make sure you have everything you need from us. Flexible "
        "arrangements are available on your accounts, and our support team can "
        "walk you through them whenever suits you. Reply and we will be in touch."
    ),
    # These reach a human, never the customer.
    "relationship_manager_escalation": None,
    "compliance_fraud_hold": None,
    "no_action": None,
}

# Actions whose customer_message must be null (spec: ACTION_PROMPT).
NO_CUSTOMER_MESSAGE_ACTIONS = {"relationship_manager_escalation", "compliance_fraud_hold", "no_action"}
