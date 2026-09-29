"""The fixed enums from README_dataset_schema.md.

The scorer in eval/harness.py keeps its own independent copy on purpose: a
grader that imports the thing it grades cannot catch a drifted enum.
"""

from __future__ import annotations

VALID_STATES = (
    "no_significant_event",
    "new_child_life_event",
    "marriage_or_relationship_change",
    "job_change_or_promotion",
    "job_loss_or_income_disruption",
    "medical_hardship",
    "financial_distress_general",
    "relocation",
    "retirement_transition",
    "wealth_growth_or_windfall",
    "potential_fraud_or_takeover",
    "elder_vulnerability_or_scam_risk",
    "churn_risk",
    "small_business_cashflow_event",
)

VALID_ACTIONS = (
    "no_action",
    "proactive_retention_outreach",
    "relationship_manager_escalation",
    "personalized_offer",
    "support_intervention",
    "compliance_fraud_hold",
)

VALID_HITL = (
    "auto_approved",
    "escalated",
    "human_approved",
    "human_rejected",
    "human_modified",
)

BANDS = ("low", "medium", "high")
BAND_ORDER = {b: i for i, b in enumerate(BANDS)}

NO_EVENT_STATE = "no_significant_event"


def lower_band(band: str) -> str:
    """One level down, never below 'low'."""
    return BANDS[max(0, BAND_ORDER.get(band, 0) - 1)]
