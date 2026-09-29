"""Builds and persists the scored deliverable: one checkpoint per simulated day.

Called at the END of each on_day_boundary, after signal -> synthesis -> action.
Every enum is validated against the README lists before writing; an invalid
value raises rather than silently scoring as a miss.

The whole array is rewritten each day so output/{scenario_id}_checkpoints.json is
always a valid JSON document, even if the run is interrupted mid-replay.

ASSUMPTIONS (spec was silent):
- A day with no hypothesis yet still emits a checkpoint, with
  "no_significant_event" / "low" / "no_action" / "auto_approved". Emitting
  nothing would leave the scorer with no checkpoint at early ground-truth times.
- notes = rationale trimmed to ~300 chars + " Evidence: " + up to 6 event ids,
  with " [fallback]" appended when any fallback ran in that decision path.
- as_of_time is rendered in the dataset's "...Z" form.
- A guardrail-forced state (Stage 2) overrides the synthesis state when the hold
  carries forced_state; a null forced_state keeps synthesis's state.

Output veto (only with config.guardrail_rules.GUARDRAILS_ENABLED), applied to
every checkpoint after it is built and before it is validated and written:
- while a hold has freeze_outbound set, the decision's drafted customer_message
  is removed and an offer/outreach action is replaced by the hold's action;
- a non-no_action checkpoint is never auto_approved (forced to escalated);
- enums are then validated as before, raising on anything off the README list.
Each veto is appended to notes as "[veto: ...]" and recorded in self.vetoes.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from c360.loader import iso
from config import guardrail_rules as GR
from config.enums import BANDS, NO_EVENT_STATE, VALID_ACTIONS, VALID_HITL, VALID_STATES
from tracing import traced

log = logging.getLogger(__name__)

NOTES_RATIONALE_CHARS = 300
NOTES_MAX_EVIDENCE = 6

# Actions that contact the customer; blocked while outbound is frozen.
OUTBOUND_ACTIONS = ("personalized_offer", "proactive_retention_outreach")


class CheckpointValidationError(ValueError):
    """A checkpoint field is not one of the README enum values."""


class CheckpointWriter:
    def __init__(
        self, scenario_id: str, output_dir: str | Path = "output",
        guardrails: Optional[bool] = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.guardrails = GR.GUARDRAILS_ENABLED if guardrails is None else guardrails
        # (as_of_time, [reasons]) for every checkpoint the veto changed.
        self.vetoes: list[tuple[str, list[str]]] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{scenario_id}_checkpoints.json"
        self.checkpoints: list[dict[str, Any]] = []

    # --- build --------------------------------------------------------------

    def build(self, as_of_time: datetime | date, memory) -> dict[str, Any]:
        hypothesis = memory.working.get_hypothesis()
        decision_f = memory.working.get_finding("current_decision")
        decision = (decision_f.value if decision_f else None) or {}

        if hypothesis is None:
            state, band = NO_EVENT_STATE, "low"
            rationale, evidence, used_fallback = None, [], False
        else:
            state = hypothesis.state
            band = hypothesis.confidence_band
            rationale = hypothesis.rationale
            evidence = list(hypothesis.supporting_events or [])
            used_fallback = bool(hypothesis.used_fallback)

        # A guardrail hold may force the state (Stage 2 writes forced_state).
        forced_state = decision.get("forced_state")
        if forced_state:
            state = forced_state

        action = decision.get("action", "no_action") or "no_action"
        subtype = decision.get("action_subtype")
        hitl_status = decision.get("hitl_status") or "auto_approved"
        used_fallback = used_fallback or bool(decision.get("used_fallback"))

        checkpoint = {
            "as_of_time": _as_of(as_of_time),
            "inferred_state": state,
            "confidence_band": band,
            "action": action,
            "action_subtype": subtype,
            "hitl_status": hitl_status,
            "notes": self._notes(rationale, evidence, used_fallback, decision),
        }
        if self.guardrails:
            reasons = self.veto(checkpoint, decision, memory.working.get_finding("guardrail_hold"))
            if reasons:
                checkpoint["notes"] = (checkpoint["notes"] + f" [veto: {'; '.join(reasons)}]").strip()
                self.vetoes.append((checkpoint["as_of_time"], reasons))
        self.validate(checkpoint)
        return checkpoint

    @staticmethod
    def veto(checkpoint: dict[str, Any], decision: dict, hold) -> list[str]:
        """Enforce the output rules in place. Returns what was changed."""
        reasons: list[str] = []
        hold_v = (hold.value if hold is not None else None) or {}
        if hold_v.get("freeze_outbound"):
            draft = decision.get("draft") or {}
            if draft.get("customer_message"):
                draft["customer_message"] = None  # the executor reads this; nothing goes out
                reasons.append("customer_message stripped, outbound frozen")
            if checkpoint["action"] in OUTBOUND_ACTIONS:
                blocked = checkpoint["action"]
                checkpoint["action"] = hold_v.get("forced_action") or "relationship_manager_escalation"
                checkpoint["action_subtype"] = hold_v.get("forced_subtype")
                reasons.append(f"{blocked} blocked, outbound frozen")
        if checkpoint["action"] != "no_action" and checkpoint["hitl_status"] == "auto_approved":
            checkpoint["hitl_status"] = "escalated"
            reasons.append(f"{checkpoint['action']} cannot be auto_approved")
        return reasons

    def _notes(
        self, rationale: Optional[str], evidence: list[str],
        used_fallback: bool, decision: dict,
    ) -> str:
        text = (rationale or "").strip()
        if len(text) > NOTES_RATIONALE_CHARS:
            text = text[: NOTES_RATIONALE_CHARS - 3].rstrip() + "..."
        parts = [text] if text else []
        extra = (decision.get("notes") or "").strip()
        if extra:
            parts.append(extra)
        if evidence:
            parts.append("Evidence: " + ", ".join(evidence[:NOTES_MAX_EVIDENCE]))
        if used_fallback:
            parts.append("[fallback]")
        return " ".join(parts).strip()

    # --- validate -----------------------------------------------------------

    @staticmethod
    def validate(checkpoint: dict[str, Any]) -> dict[str, Any]:
        for field, allowed in (
            ("inferred_state", VALID_STATES),
            ("confidence_band", BANDS),
            ("action", VALID_ACTIONS),
            ("hitl_status", VALID_HITL),
        ):
            value = checkpoint.get(field)
            if value not in allowed:
                raise CheckpointValidationError(
                    f"{field}={value!r} is not one of {list(allowed)}"
                )
        if not checkpoint.get("as_of_time"):
            raise CheckpointValidationError("as_of_time is required")
        return checkpoint

    # --- write --------------------------------------------------------------

    @traced("checkpoint")
    def write(self, as_of_time: datetime | date, memory) -> dict[str, Any]:
        """Append one checkpoint and rewrite the whole file."""
        checkpoint = self.build(as_of_time, memory)
        self.checkpoints.append(checkpoint)
        self.flush()
        return checkpoint

    def flush(self) -> Path:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.checkpoints, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic: the file is never half-written
        return self.path

    def latest(self) -> Optional[dict[str, Any]]:
        return self.checkpoints[-1] if self.checkpoints else None

    def __len__(self) -> int:
        return len(self.checkpoints)

    def __repr__(self) -> str:
        return f"<CheckpointWriter {self.scenario_id} n={len(self.checkpoints)}>"


def _as_of(value: datetime | date) -> str:
    if isinstance(value, datetime):
        return iso(value)
    return f"{value.isoformat()}T00:00:00Z"
