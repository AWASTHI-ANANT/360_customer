"""Normalised category vocabulary for mcc_category and transaction_type.

Exact-name lists broke on the shipped data ("grocery" vs "groceries",
"baby_products" vs "baby_supplies"). Names are normalised instead -- lowercase,
split on "_" / "-" / space, trailing plural "s" stripped per token -- and a name
belongs to a group when any token starts with one of the group's stems.

Distinct values in the three shipped scenarios (history + live):
  mcc_category     baby_products, clothing, dining, electronics, gas,
                   general_retail, grocery, healthcare, lodging, pharmacy, transit
  transaction_type ach_withdrawal, benefits_credit, daycare_payment,
                   gym_membership, internal_transfer_in, internal_transfer_out,
                   mortgage_payment, rent_payment, salary_credit, tax_refund,
                   utilities
Mapped: baby_products, daycare_payment -> baby; healthcare, pharmacy -> health;
grocery -> grocery. Income-like transaction types (salary_credit,
benefits_credit) are classified by c360/derive.py INCOME_TOKENS and
config/evidence_rules.py BENEFIT_TOKENS, not here. The rest carry no
life-event meaning on their own and stay ungrouped.
"""

from __future__ import annotations

import re
from typing import Optional

# Checked in order; the first group with a matching stem wins.
CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "health": ("health", "pharm", "hospital", "medical", "clinic", "doctor", "dental"),
    "grocery": ("grocer",),
    # "daycare" is here so the daycare_payment transaction_type maps; none of
    # the other stems covers it.
    "baby": ("baby", "infant", "child", "nursery", "kid", "daycare"),
}

_SPLIT_RE = re.compile(r"[_\-\s]+")


def normalise(name: object) -> str:
    """'Baby-Supplies' -> 'baby_supplie'; plural 's' stripped per token."""
    tokens = [t for t in _SPLIT_RE.split(str(name or "").strip().lower()) if t]
    return "_".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens)


def category_group(name: object) -> Optional[str]:
    """The group a category or transaction_type belongs to, or None."""
    tokens = normalise(name).split("_")
    for group, stems in CATEGORY_GROUPS.items():
        if any(tok.startswith(stem) for tok in tokens for stem in stems):
            return group
    return None
