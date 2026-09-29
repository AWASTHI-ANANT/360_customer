"""Deterministic guardrail rules: regex on customer-authored text.

A rule fires when any `patterns` regex matches the event text AFTER every
`exceptions` span has been removed. That is how "power of attorney" is kept out
of LEGAL_THREAT without also hiding "my attorney will call you".

On fire the guardrail agent writes a "guardrail_hold" finding whose fields
(forced_action, forced_subtype, forced_state, freeze_outbound) are what
ActionAgent._apply_guardrail_hold and the checkpoint writer already consume.

Large purchases and large inflows are deliberately NOT guardrail rules: they are
ambiguous behaviour (see the red herrings), so they go through synthesis.

ASSUMPTIONS (spec was silent):
- Only support_logs text is scanned: that is the customer talking to the bank.
  Web searches and consented social posts are not addressed to the bank, so
  "how to sue a landlord" in a search box must not freeze an account.
- Matching is case-insensitive on the raw (unmasked) text. It runs locally and
  nothing leaves the process; the hold records only the rule id and event id.
- A hold stays in force for the rest of the run: clearing one is a human act
  (hitl.clear_hold), and the replay has no human.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Off -> Stage 1 behaviour exactly: no guardrail agent, no output veto.
GUARDRAILS_ENABLED = os.environ.get("C360_GUARDRAILS", "1") not in ("0", "false", "False")

# Only these source systems are scanned (see ASSUMPTIONS).
GUARDRAIL_SOURCES = ("support_logs",)

GUARDRAIL_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "LEGAL_THREAT",
        "patterns": [
            r"\blaw\s?suits?\b",
            r"\bsu(?:e|ing)\b",
            r"\blegal action\b",
            r"\bmy (?:attorney|lawyer|solicitor)s?\b",
        ],
        "exceptions": [r"\bpowers? of attorney\b"],
        "forced_action": "relationship_manager_escalation",
        "forced_subtype": "legal_escalation_queue",
        "forced_state": None,  # keep synthesis's state; the threat is not a life event
        "freeze_outbound": True,
    },
    {
        "rule_id": "CUSTOMER_FRAUD_REPORT",
        "patterns": [
            r"\b(?:didn'?t|did not|never) (?:make|authori[sz]e)\b",
            r"\bunauthori[sz]ed\b",
            r"\bstolen (?:card|debit card|credit card)\b",
            r"\bcard (?:was|got|has been) stolen\b",
            r"\bhacked\b",
        ],
        "exceptions": [
            r"\b(?:not|wasn'?t|was not|hasn'?t been|has not been) hacked\b",
            r"\bnot (?:an? )?unauthori[sz]ed\b",
        ],
        "forced_action": "compliance_fraud_hold",
        "forced_subtype": "account_verification_hold",
        "forced_state": "potential_fraud_or_takeover",
        "freeze_outbound": False,
    },
]

_COMPILED = [
    (
        rule,
        [re.compile(p, re.IGNORECASE) for p in rule["patterns"]],
        [re.compile(p, re.IGNORECASE) for p in rule["exceptions"]],
    )
    for rule in GUARDRAIL_RULES
]


def match_rules(text: str) -> list[dict[str, Any]]:
    """Every rule that fires on `text`, in GUARDRAIL_RULES order."""
    fired = []
    for rule, patterns, exceptions in _COMPILED:
        stripped = text or ""
        for exc in exceptions:
            stripped = exc.sub(" ", stripped)
        if any(p.search(stripped) for p in patterns):
            fired.append(rule)
    return fired
