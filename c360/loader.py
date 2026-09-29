"""Scenario loading: entities.json + history_seed.jsonl + live_stream.jsonl -> typed objects.

Everything here is stdlib-only and file-based. Load is cheap enough (a few
thousand events) that we just read whole files into memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

# Repo-relative location the Prepathon package unzips to. Overridable everywhere.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Prepathon_ps" / "customer_360_dataset"

# From README_dataset_schema.md. Used only to warn on drift -- never to reject
# an event, so a new scenario with new event types still loads and runs.
KNOWN_SOURCE_SYSTEMS: dict[str, set[str]] = {
    "card_payments": {"purchase", "refund", "decline"},
    "instant_payments": {"inbound_transfer", "outbound_transfer"},
    "ach_wire": {"inbound_transfer", "outbound_transfer"},
    "core_banking_ledger": {
        "deposit",
        "withdrawal",
        "standing_instruction",
        "fee",
        "interest_credit",
    },
    "trading_brokerage": {"buy", "sell", "dividend", "deposit_to_brokerage"},
    "loan_kyc": {
        "loan_application",
        "loan_disbursed",
        "kyc_update",
        "address_change",
        "marital_status_change",
        "dependents_change",
    },
    "web_app_events": {"login", "search_query", "feature_used", "session_duration"},
    "support_logs": {"ticket_created", "ticket_resolved", "call_transcript"},
    "social_signal_consented": {"life_event_mention"},
}

# Payload fields that carry free text worth scanning for life-event language.
_TEXT_FIELDS = ("raw_text", "search_text", "merchant_name", "feature_or_page", "counterparty_name")


class ScenarioLoadError(Exception):
    """Raised when a scenario cannot be loaded at all (missing/unparseable files)."""


def parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, normalising to timezone-aware UTC.

    The dataset uses a trailing 'Z'. datetime.fromisoformat handles that from
    3.11 on, but we strip it explicitly so behaviour is identical everywhere.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.strip()
    if text.endswith(("z", "Z")):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def iso(dt: datetime) -> str:
    """Render a datetime back in the dataset's '...Z' form."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class Event:
    """One line of history_seed.jsonl / live_stream.jsonl.

    `payload` is kept exactly as it came off disk. `derived` is ours: feature
    extractors fill it in later (is_salary, counterparty_is_self,
    amount_vs_baseline_ratio, ...). Empty at load time for now.
    """

    event_id: str
    event_time: datetime
    ingestion_time: datetime
    customer_id: str
    account_id: Optional[str]
    source_system: str
    event_type: str
    schema_version: str
    payload: dict[str, Any]
    derived: dict[str, Any] = field(default_factory=dict)
    # "history" or "live" -- which file this came from.
    stream: str = "live"

    # --- convenience accessors used constantly downstream -------------------

    @property
    def sim_date(self) -> date:
        return self.event_time.date()

    @property
    def amount(self) -> Optional[float]:
        raw = self.payload.get("amount")
        return float(raw) if isinstance(raw, (int, float)) else None

    @property
    def is_late_arriving(self) -> bool:
        """True when the event reached us after it happened (out-of-order replay)."""
        return self.ingestion_time > self.event_time

    @property
    def kind(self) -> str:
        """'source_system/event_type', handy as a dict key or in log lines."""
        return f"{self.source_system}/{self.event_type}"

    def text_blob(self) -> str:
        """All free-text payload fields joined and lowercased, for keyword checks."""
        parts = [str(self.payload[f]) for f in _TEXT_FIELDS if self.payload.get(f)]
        return " ".join(parts).lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_time": iso(self.event_time),
            "ingestion_time": iso(self.ingestion_time),
            "customer_id": self.customer_id,
            "account_id": self.account_id,
            "source_system": self.source_system,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "payload": self.payload,
            "derived": self.derived,
            "stream": self.stream,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any], stream: str = "live") -> "Event":
        return cls(
            event_id=raw["event_id"],
            event_time=parse_ts(raw["event_time"]),
            # A few scenarios omit ingestion_time; fall back to event_time.
            ingestion_time=parse_ts(raw.get("ingestion_time") or raw["event_time"]),
            customer_id=raw["customer_id"],
            account_id=raw.get("account_id"),
            source_system=raw.get("source_system", "unknown"),
            event_type=raw.get("event_type", "unknown"),
            schema_version=str(raw.get("schema_version", "1.0")),
            payload=raw.get("payload") or {},
            derived={},
            stream=stream,
        )


@dataclass(slots=True)
class Account:
    account_id: str
    type: str
    opened_date: Optional[str] = None


@dataclass(slots=True)
class CustomerProfile:
    """entities.json, flattened (the file nests demographics under 'profile')."""

    customer_id: str
    name: str
    occupation: str
    customer_value_tier: str
    tenure_months: int
    marital_status: str
    dependents: int
    accounts: list[Account]
    # Not in the brief's field list but present in the data and needed for
    # age-sensitive states (e.g. elder_vulnerability_or_scam_risk).
    age: Optional[int] = None
    household_id: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def account_ids(self) -> list[str]:
        return [a.account_id for a in self.accounts]

    def account_type(self, account_id: Optional[str]) -> Optional[str]:
        for a in self.accounts:
            if a.account_id == account_id:
                return a.type
        return None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "CustomerProfile":
        # Tolerate both nested ({"profile": {...}}) and flat entity files.
        prof = raw.get("profile") or raw
        accounts = [
            Account(
                account_id=a["account_id"],
                type=a.get("type", "unknown"),
                opened_date=a.get("opened_date"),
            )
            for a in raw.get("accounts") or []
        ]
        return cls(
            customer_id=raw.get("customer_id") or prof.get("customer_id", "UNKNOWN"),
            name=prof.get("name", "Unknown"),
            occupation=prof.get("occupation", "unknown"),
            customer_value_tier=prof.get("customer_value_tier", "unknown"),
            tenure_months=int(prof.get("tenure_months") or 0),
            marital_status=prof.get("marital_status", "unknown"),
            dependents=int(prof.get("dependents") or 0),
            accounts=accounts,
            age=prof.get("age"),
            household_id=raw.get("household_id"),
            raw=raw,
        )


@dataclass(slots=True)
class ReplayConfig:
    scenario_id: str
    simulated_start: datetime
    simulated_end: datetime
    replay_speed_seconds_per_simulated_day: float
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: dict[str, Any], fallback_id: str = "scenario") -> "ReplayConfig":
        return cls(
            scenario_id=raw.get("scenario_id") or fallback_id,
            simulated_start=parse_ts(raw["simulated_start"]),
            simulated_end=parse_ts(raw["simulated_end"]),
            replay_speed_seconds_per_simulated_day=float(
                raw.get("replay_speed_seconds_per_simulated_day") or 1.0
            ),
            raw=raw,
        )


@dataclass(slots=True)
class Scenario:
    """Everything one scenario folder contains, loaded and sorted."""

    scenario_id: str
    path: Path
    profile: CustomerProfile
    replay_config: ReplayConfig
    history_events: list[Event]
    live_events: list[Event]
    # Non-fatal problems found while loading (unknown types, bad lines, ...).
    issues: list[str] = field(default_factory=list)

    @property
    def customer_id(self) -> str:
        return self.profile.customer_id

    @property
    def all_events(self) -> list[Event]:
        """History then live, in one event_time-ordered list."""
        return sorted(
            self.history_events + self.live_events, key=lambda e: (e.event_time, e.event_id)
        )

    def by_id(self) -> dict[str, Event]:
        return {e.event_id: e for e in self.all_events}

    def __repr__(self) -> str:  # keeps run output readable
        return (
            f"<Scenario {self.scenario_id} {self.customer_id} "
            f"history={len(self.history_events)} live={len(self.live_events)} "
            f"issues={len(self.issues)}>"
        )


def _read_jsonl(path: Path, stream: str, issues: list[str]) -> Iterator[Event]:
    """Yield Events from a JSONL file, skipping (and reporting) bad lines."""
    if not path.exists():
        issues.append(f"missing file: {path.name}")
        log.warning("missing file %s -- treating as empty", path)
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"{path.name}:{lineno} unparseable JSON ({exc.msg})")
                log.warning("%s:%d unparseable JSON: %s", path.name, lineno, exc.msg)
                continue
            try:
                yield Event.from_json(raw, stream=stream)
            except (KeyError, ValueError) as exc:
                issues.append(f"{path.name}:{lineno} bad event ({exc})")
                log.warning("%s:%d bad event: %s", path.name, lineno, exc)


def _check_vocabulary(events: list[Event], issues: list[str]) -> None:
    """Warn once per unrecognised source_system/event_type pair."""
    seen: set[str] = set()
    for ev in events:
        if ev.kind in seen:
            continue
        seen.add(ev.kind)
        known_types = KNOWN_SOURCE_SYSTEMS.get(ev.source_system)
        if known_types is None:
            msg = f"unknown source_system {ev.source_system!r} (first seen {ev.event_id})"
        elif ev.event_type not in known_types:
            msg = (
                f"unknown event_type {ev.event_type!r} for source_system "
                f"{ev.source_system!r} (first seen {ev.event_id})"
            )
        else:
            continue
        issues.append(msg)
        log.warning("%s", msg)


def _dedupe(events: list[Event], strict: bool, issues: list[str]) -> list[Event]:
    """Enforce event_id uniqueness across the whole scenario.

    strict=True (default) raises, because a duplicate id means the evidence
    trail is ambiguous. strict=False keeps the first occurrence and warns, so a
    malformed scenario can still be run.
    """
    seen: dict[str, Event] = {}
    dupes: list[str] = []
    kept: list[Event] = []
    for ev in events:
        prior = seen.get(ev.event_id)
        if prior is not None:
            dupes.append(f"{ev.event_id} ({prior.stream} and {ev.stream})")
            continue
        seen[ev.event_id] = ev
        kept.append(ev)
    if dupes:
        msg = f"duplicate event_ids in scenario: {', '.join(dupes[:10])}"
        if len(dupes) > 10:
            msg += f" (+{len(dupes) - 10} more)"
        if strict:
            raise ScenarioLoadError(msg)
        issues.append(msg)
        log.warning("%s -- keeping first occurrence of each", msg)
    return kept


def resolve_scenario_dir(scenario: str | Path, data_root: str | Path | None = None) -> Path:
    """Accept a scenario id ('scenario_01') or a direct path, return the folder."""
    p = Path(scenario)
    if p.is_dir() and (p / "entities.json").exists():
        return p
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    candidate = root / str(scenario)
    if candidate.is_dir():
        return candidate
    raise ScenarioLoadError(
        f"cannot find scenario {scenario!r} (looked at {p} and {candidate})"
    )


def find_scenarios(data_root: str | Path | None = None) -> list[Path]:
    """All scenario folders under data_root, sorted by name."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "entities.json").exists())


def load_scenario(
    scenario: str | Path,
    data_root: str | Path | None = None,
    strict: bool = True,
    derive_features: bool = True,
) -> Scenario:
    """Load one scenario folder into a Scenario.

    Returns history_events and live_events as separate lists, each sorted by
    event_time ascending (event_id breaks ties for determinism). Every event's
    `derived` dict is filled by derive.py unless derive_features is False.
    """
    path = resolve_scenario_dir(scenario, data_root)
    issues: list[str] = []

    try:
        entities_raw = json.loads((path / "entities.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioLoadError(f"cannot read {path / 'entities.json'}: {exc}") from exc
    profile = CustomerProfile.from_json(entities_raw)

    cfg_path = path / "replay_config.json"
    try:
        cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        replay_config = ReplayConfig.from_json(cfg_raw, fallback_id=path.name)
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ScenarioLoadError(f"cannot read {cfg_path}: {exc}") from exc

    history = list(_read_jsonl(path / "history_seed.jsonl", "history", issues))
    live = list(_read_jsonl(path / "live_stream.jsonl", "live", issues))

    # Feature derivation runs here, before anything else sees an event, and
    # before masking -- so signals that need the real name (counterparty_is_self)
    # survive. Imported at call time because derive imports from this module.
    if derive_features:
        from .derive import derive_all  # noqa: PLC0415 - avoids a circular import

        derive_all(history + live, profile)

    # Uniqueness is scenario-wide, so check the combined set before splitting.
    combined = _dedupe(history + live, strict=strict, issues=issues)
    history = [e for e in combined if e.stream == "history"]
    live = [e for e in combined if e.stream == "live"]

    _check_vocabulary(combined, issues)

    for ev in combined:
        if ev.customer_id != profile.customer_id:
            msg = (
                f"{ev.event_id} customer_id {ev.customer_id!r} != entities "
                f"{profile.customer_id!r}"
            )
            issues.append(msg)
            log.warning("%s", msg)
            break  # one warning is enough

    sort_key = lambda e: (e.event_time, e.event_id)  # noqa: E731
    history.sort(key=sort_key)
    live.sort(key=sort_key)

    scn = Scenario(
        scenario_id=replay_config.scenario_id,
        path=path,
        profile=profile,
        replay_config=replay_config,
        history_events=history,
        live_events=live,
        issues=issues,
    )
    log.info(
        "loaded %s: %d history + %d live events for %s (%d issues)",
        scn.scenario_id,
        len(history),
        len(live),
        profile.customer_id,
        len(issues),
    )
    return scn


def load_ground_truth(scenario: str | Path, data_root: str | Path | None = None) -> Optional[dict]:
    """ground_truth.json if present (dev/scoring only -- agents must not read it)."""
    path = resolve_scenario_dir(scenario, data_root) / "ground_truth.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
