#!/usr/bin/env python3
"""Category normalisation and account pseudonyms.

    python -m tests.test_categories
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.signal_agent import is_known_category  # noqa: E402
from c360.masking import pseudonymize_account  # noqa: E402
from config.categories import category_group, normalise  # noqa: E402
from config.evidence_rules import _has  # noqa: E402

FAILURES: list[str] = []
PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}  {detail}")


def main() -> int:
    check("normalise strips plural per token", normalise("Baby-Supplies") == "baby_supplie")
    for name, group in [
        ("grocery", "grocery"), ("groceries", "grocery"), ("GROCERY_STORE", "grocery"),
        ("baby_products", "baby"), ("baby_supplies", "baby"), ("childcare", "baby"),
        ("kids_clothing", "baby"), ("daycare_payment", "baby"), ("infant_formula", "baby"),
        ("healthcare", "health"), ("pharmacy", "health"), ("hospital", "health"),
        ("dental_clinic", "health"), ("doctors", "health"),
        ("dining", None), ("gas", None), ("salary_credit", None), ("tax_refund", None),
    ]:
        got = category_group(name)
        check(f"{name} -> {group}", got == group, f"got {got}")
    check("data categories all recognised",
          all(is_known_category(c) for c in (
              "baby_products", "clothing", "dining", "electronics", "gas", "general_retail",
              "grocery", "healthcare", "lodging", "pharmacy", "transit")))
    check("made-up category is unknown", not is_known_category("zeppelin_rides"))
    check("merchant_shift baby rule matches baby_products",
          _has({"new": {"baby_products": {}}}, "baby"))
    check("merchant_shift health rule matches risen pharmacy",
          _has({"risen": {"pharmacy": {}}}, "health"))
    check("grocery shift is not health", not _has({"risen": {"grocery": {}}}, "health"))
    a, b = pseudonymize_account("ACC_CHK_003"), pseudonymize_account("ACC_SAV_003")
    check("account pseudonym hides ACC_", "ACC_" not in a and a.startswith("acct:chk:"), a)
    check("account pseudonym stable and distinct",
          a == pseudonymize_account("ACC_CHK_003") and a != b)
    check("null account stays null", pseudonymize_account(None) is None)
    print(f"\n{PASSED} checks passed, {len(FAILURES)} failed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
