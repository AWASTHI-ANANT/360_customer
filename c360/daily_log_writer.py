"""Append-only, human-readable trace of a replay run.

This is *not* memory. Nothing here is ever rewritten, summarised or compacted --
it exists so a judge (or you, at 3am) can read exactly what the system saw and
did, in order. Agents query memory_store for decisions; they write here only to
leave a trail.

One file per scenario: logs/{scenario_id}_daily.jsonl
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .loader import CustomerProfile, Event, iso
from .masking import mask

log = logging.getLogger(__name__)


def summarize_event(event: Event) -> str:
    """One short human-readable sentence describing an event.

    Falls back to a generic rendering for any source_system we don't know, so
    new scenarios still produce readable logs.
    """
    p = event.payload
    amt = event.amount
    money = f"{amt:,.2f}" if amt is not None else "?"
    cur = p.get("currency", "")
    ss, et = event.source_system, event.event_type

    if ss == "card_payments":
        bits = [f"{et} {money} {cur} at {p.get('merchant_name', 'unknown merchant')}"]
        if p.get("mcc_category"):
            bits.append(f"[{p['mcc_category']}]")
        if p.get("is_international"):
            bits.append("international")
        if not p.get("card_present", True):
            bits.append("card-not-present")
        if p.get("decline_reason"):
            bits.append(f"declined: {p['decline_reason']}")
        return " ".join(bits)

    if ss in ("instant_payments", "ach_wire"):
        direction = p.get("direction") or et
        return (
            f"{ss} {direction} {money} {cur} "
            f"{'to' if 'out' in str(direction) else 'from'} "
            f"{p.get('counterparty_name', 'unknown')} "
            f"({p.get('counterparty_country', '??')}, {p.get('transfer_type', 'n/a')}, "
            f"{p.get('status', 'n/a')})"
        )

    if ss == "core_banking_ledger":
        txn = p.get("transaction_type", et)
        bal = p.get("balance_after")
        tail = f", balance_after {bal:,.2f}" if isinstance(bal, (int, float)) else ""
        return f"ledger {et} {money} ({txn}){tail}"

    if ss == "trading_brokerage":
        pv = p.get("portfolio_value_after")
        tail = f", portfolio {pv:,.2f}" if isinstance(pv, (int, float)) else ""
        return (
            f"brokerage {et} {money} {p.get('instrument_type', 'n/a')} "
            f"risk={p.get('risk_category', 'n/a')}{tail}"
        )

    if ss == "loan_kyc":
        sub = p.get("event_subtype", et)
        return f"kyc {sub}: {p.get('old_value')!r} -> {p.get('new_value')!r}"

    if ss == "web_app_events":
        bits = [f"app {et} {p.get('feature_or_page', '')}".strip()]
        if p.get("search_text"):
            bits.append(f'searched "{p["search_text"]}"')
        if p.get("session_length_sec") is not None:
            bits.append(f"{p['session_length_sec']}s session")
        if p.get("device_type"):
            bits.append(f"on {p['device_type']}")
        return " | ".join(bits)

    if ss == "support_logs":
        text = str(p.get("raw_text", "") or "")
        if len(text) > 160:
            text = text[:157] + "..."
        return (
            f"support {et} [{p.get('category', 'n/a')} via {p.get('channel', 'n/a')}, "
            f"{p.get('resolution_status', 'n/a')}]: {text}"
        )

    if ss == "social_signal_consented":
        text = str(p.get("raw_text", "") or "")
        if len(text) > 160:
            text = text[:157] + "..."
        return f"social {p.get('platform', 'n/a')} (consent={p.get('consent_flag')}): {text}"

    return f"{ss}/{et} {json.dumps(p, default=str, sort_keys=True)[:200]}"


class DailyLogWriter:
    """Strictly append-only JSONL writer, flushed after every line.

    Opened in 'a' mode and flushed + fsynced per write, so an interrupted run
    keeps everything written up to that point.
    """

    def __init__(
        self,
        scenario_id: str,
        output_dir: str | Path = "logs",
        filename: Optional[str] = None,
        fsync: bool = True,
        profile: Optional[CustomerProfile] = None,
    ) -> None:
        self.scenario_id = scenario_id
        # Summaries quote free text (counterparty names, support transcripts),
        # so they go through the same masking as LLM prompts before writing.
        self.profile = profile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / (filename or f"{scenario_id}_daily.jsonl")
        self.fsync = fsync
        self._fh = self.path.open("a", encoding="utf-8")
        self.seq = 0
        self.counts: dict[str, int] = {"event": 0, "day_boundary": 0, "agent_action": 0}

    # --- plumbing -----------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        record = {"seq": self.seq, "scenario_id": self.scenario_id, **record}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        if self.fsync:
            import os

            os.fsync(self._fh.fileno())
        self.counts[record["kind"]] = self.counts.get(record["kind"], 0) + 1
        return record

    def truncate_and_restart(self) -> None:
        """Start a fresh log for this scenario.

        The log is append-only *during* a run; re-running a scenario from
        scratch is the one case where clearing is correct, and it's explicit.
        """
        self._fh.close()
        self._fh = self.path.open("w", encoding="utf-8")
        self.seq = 0
        self.counts = {"event": 0, "day_boundary": 0, "agent_action": 0}

    # --- log lines ----------------------------------------------------------

    def log_event(self, event: Event) -> dict[str, Any]:
        """One line per ingested event."""
        return self._write(
            {
                "kind": "event",
                "timestamp": iso(event.event_time),
                "ingestion_time": iso(event.ingestion_time),
                "event_id": event.event_id,
                "account_id": event.account_id,
                "source_system": event.source_system,
                "event_type": event.event_type,
                "summary": mask(summarize_event(event), self.profile)[0],
                "stream": event.stream,
                "late_arriving": event.is_late_arriving,
            }
        )

    def log_day_boundary(self, sim_date: date | datetime, event_count_that_day: int = 0) -> dict:
        """Day marker. Fired even on days with no events."""
        d = sim_date.date() if isinstance(sim_date, datetime) else sim_date
        return self._write(
            {
                "kind": "day_boundary",
                "timestamp": f"{d.isoformat()}T00:00:00Z",
                "sim_date": d.isoformat(),
                "event_count_that_day": int(event_count_that_day),
                "summary": f"--- {d.isoformat()} ({event_count_that_day} events) ---",
            }
        )

    def log_agent_action(
        self,
        agent_name: str,
        event_id_or_none: Optional[str],
        summary: str,
        evidence_ids: Optional[Sequence[str]] = None,
        as_of: Optional[datetime] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """One line whenever an agent does something, for traceability.

        `as_of` should be the simulated time (clock.now()); it defaults to wall
        clock only so this stays callable outside a replay.
        """
        return self._write(
            {
                "kind": "agent_action",
                "timestamp": iso(as_of or datetime.now(timezone.utc)),
                "agent_name": agent_name,
                "event_id": event_id_or_none,
                "summary": summary,
                "evidence_ids": list(evidence_ids or []),
                **extra,
            }
        )

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "DailyLogWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<DailyLogWriter {self.path} lines={self.seq}>"
