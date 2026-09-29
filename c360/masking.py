"""PII masking, applied before every LLM call and every log write of raw text.

Derivation runs earlier (see derive.py), so masking never costs us a signal.
The returned mapping lets a reviewer unmask a trace when auditing.
"""

from __future__ import annotations

import re
from typing import Optional

from .loader import CustomerProfile

# ACC_CHK_001 and friends.
_ACCOUNT_RE = re.compile(r"\bACC_[A-Z0-9_]+\b")
# CUST_00184 and friends.
_CUSTOMER_RE = re.compile(r"\bCUST_\d+\b")
# Any run of 7+ digits (card/account numbers). Kept above 6 so money amounts
# and years survive -- "$35" and "2026" must stay readable to the model.
_LONG_DIGITS_RE = re.compile(r"\b\d{7,}\b")
# Emails and phone-like runs.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{8,}\d)\b")


def mask(text: Optional[str], profile: Optional[CustomerProfile] = None) -> tuple[str, dict[str, str]]:
    """Replace identifying strings with stable tokens.

    Returns (masked_text, mapping) where mapping is token -> original. Tokens are
    numbered per distinct value, so [ACCOUNT_1] is the same account throughout
    one call and the model can still reason about "the same account".
    """
    if not text:
        return "", {}
    mapping: dict[str, str] = {}
    out = str(text)

    def _sub(pattern: re.Pattern, label: str, value: str) -> None:
        nonlocal out
        counters: dict[str, str] = {}
        def repl(m: re.Match) -> str:
            original = m.group(0)
            if original not in counters:
                token = f"[{label}_{len(counters) + 1}]"
                counters[original] = token
                mapping[token] = original
            return counters[original]
        out = pattern.sub(repl, out)

    # Name first: it is the most specific and may contain no other pattern.
    if profile is not None and profile.name:
        full = profile.name.strip()
        if full:
            name_re = re.compile(re.escape(full), re.IGNORECASE)
            if name_re.search(out):
                mapping["[CUSTOMER]"] = full
                out = name_re.sub("[CUSTOMER]", out)
            # Individual parts, longest first so "David Chen" beats "Chen".
            for part in sorted((p for p in full.split() if len(p) > 2), key=len, reverse=True):
                part_re = re.compile(rf"\b{re.escape(part)}\b", re.IGNORECASE)
                if part_re.search(out):
                    mapping.setdefault("[CUSTOMER]", full)
                    out = part_re.sub("[CUSTOMER]", out)

    _sub(_ACCOUNT_RE, "ACCOUNT", "")
    _sub(_CUSTOMER_RE, "CUSTOMER_ID", "")
    _sub(_EMAIL_RE, "EMAIL", "")
    _sub(_PHONE_RE, "PHONE", "")
    _sub(_LONG_DIGITS_RE, "NUMBER", "")
    return out, mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore originals, for auditing a trace."""
    out = text
    for token, original in mapping.items():
        out = out.replace(token, original)
    return out


def mask_event_text(event, profile: Optional[CustomerProfile] = None) -> tuple[str, dict[str, str]]:
    """Mask the concatenated free-text fields derive.py found on an event."""
    texts = (event.derived or {}).get("text_fields") or []
    if not texts:
        for field in ("raw_text", "search_text"):
            if event.payload.get(field):
                texts = [str(event.payload[field])]
                break
    return mask(" ".join(texts), profile)
