"""Feature derivation, run by the loader on every event before anything sees it.

Derivation happens BEFORE masking, so signals that depend on the real name
(counterparty_is_self) survive. Masking is applied later -- right before an LLM
call and before any log write carrying raw text.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from .loader import CustomerProfile, Event

log = logging.getLogger(__name__)

# transaction_type substrings that mean "this is income".
INCOME_TOKENS = ("salary", "benefit", "disability", "pension", "payroll")

# Tokens that mark a counterparty as some bank. "bank" catches "Ext Bank",
# "Chase Bank"; the named ones catch banks that don't say "Bank".
BANK_TOKENS = (
    "bank", "chase", "wells fargo", "citibank", "citi ", "hsbc", "barclays",
    "santander", "bofa", "capital one", "pnc", "usaa", "ally", "schwab",
    "fidelity", "credit union", "sofi", "revolut", "monzo", "n26",
)

# Our own institution. The dataset never names it, so nothing matches by
# default; set this if a scenario starts naming the incumbent bank.
OUR_BANK_TOKENS: tuple[str, ...] = ()

# Free-text payload fields, in the order we prefer to read them.
TEXT_PAYLOAD_FIELDS = ("raw_text", "search_text")


def _name_tokens(profile: Optional[CustomerProfile]) -> list[str]:
    """Full name plus individual parts, lowercased, for self-transfer matching."""
    if profile is None or not profile.name:
        return []
    full = profile.name.strip().lower()
    parts = [p for p in full.split() if len(p) > 2]
    return [full, *parts]


def derive(event: Event, profile: Optional[CustomerProfile] = None) -> Event:
    """Fill event.derived in place and return the event."""
    p = event.payload or {}
    d = event.derived

    txn_type = str(p.get("transaction_type") or "").lower()
    d["transaction_type"] = txn_type or None
    d["is_salary"] = txn_type == "salary_credit"
    d["is_income_like"] = any(tok in txn_type for tok in INCOME_TOKENS)

    counterparty = str(p.get("counterparty_name") or "")
    cp_low = counterparty.lower()
    names = _name_tokens(profile)
    # Full name match is decisive; a single shared surname is not enough on its
    # own, so require the full name or at least two name parts.
    full_hit = bool(names) and names[0] in cp_low
    part_hits = sum(1 for tok in names[1:] if tok in cp_low)
    d["counterparty_is_self"] = bool(counterparty) and (full_hit or part_hits >= 2)

    is_bank = any(tok in cp_low for tok in BANK_TOKENS)
    is_ours = any(tok in cp_low for tok in OUR_BANK_TOKENS) if OUR_BANK_TOKENS else False
    d["counterparty_is_external_bank"] = bool(counterparty) and is_bank and not is_ours

    texts = [str(p[f]) for f in TEXT_PAYLOAD_FIELDS if p.get(f)]
    d["text_fields"] = texts
    d["has_text"] = bool(texts)

    # Small conveniences the agents would otherwise recompute constantly.
    d["amount"] = event.amount
    d["mcc_category"] = p.get("mcc_category")
    d["direction"] = p.get("direction")
    d["is_outbound"] = str(p.get("direction") or "").lower().startswith("out") or (
        event.event_type == "outbound_transfer"
    )
    d["is_inbound"] = str(p.get("direction") or "").lower().startswith("in") or (
        event.event_type == "inbound_transfer"
    )
    return event


def derive_all(events: Iterable[Event], profile: Optional[CustomerProfile] = None) -> list[Event]:
    return [derive(e, profile) for e in events]


def explain(event: Event) -> dict[str, Any]:
    """Only the derived flags that are set, for compact logging.

    Identity checks rather than `v not in (None, False, ...)`: that uses ==, and
    0 == False, so a legitimate zero amount would be dropped.
    """
    return {
        k: v
        for k, v in event.derived.items()
        if v is not None and v is not False and not (isinstance(v, (list, str)) and not v)
    }
