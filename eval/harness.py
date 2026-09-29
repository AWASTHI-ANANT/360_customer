"""Scoring harness: my checkpoint output vs. the hackathon ground-truth format.

Scoring is deliberately blunt and readable -- every number in summary() can be
traced to a per-checkpoint line in score_checkpoints(). Retune WEIGHTS at the
top rather than editing the maths.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Sequence

from c360.loader import Event, iso, parse_ts

log = logging.getLogger(__name__)

# Retune here. Keys must match the component names in summary().
WEIGHTS: dict[str, float] = {
    "state_accuracy": 0.30,
    "band_accuracy": 0.20,
    "action_accuracy": 0.25,
    "timeliness": 0.15,
    "false_positive_pass_rate": 0.10,
}

# Ordinal positions for band distance.
BAND_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# From README_dataset_schema.md -- used only to flag typos in my own output,
# never to reject a ground-truth value.
VALID_STATES = {
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
}
VALID_ACTIONS = {
    "no_action",
    "proactive_retention_outreach",
    "relationship_manager_escalation",
    "personalized_offer",
    "support_intervention",
    "compliance_fraud_hold",
}
VALID_HITL = {
    "auto_approved",
    "escalated",
    "human_approved",
    "human_rejected",
    "human_modified",
}

TIMELINESS_VERDICTS = ("on_time", "early", "late", "never")


def _safe_fraction(hits: int, total: int) -> Optional[float]:
    """None when nothing was applicable -- distinct from a genuine 0.0."""
    return None if total == 0 else hits / total


class ScenarioScorer:
    """Scores one scenario's checkpoint output against its ground truth."""

    def __init__(
        self,
        ground_truth: dict,
        my_checkpoints: Optional[Sequence[dict]],
        events: Optional[Iterable[Event]] = None,
    ) -> None:
        self.ground_truth = ground_truth or {}
        self.scenario_id = self.ground_truth.get("scenario_id", "unknown")

        # Sorted once; every lookup below relies on this ordering.
        raw = list(my_checkpoints or [])
        self.my_checkpoints: list[dict] = sorted(
            (c for c in raw if self._as_of(c) is not None),
            key=lambda c: self._as_of(c),  # type: ignore[arg-type]
        )
        self.dropped_checkpoints = len(raw) - len(self.my_checkpoints)
        if self.dropped_checkpoints:
            log.warning(
                "%s: dropped %d checkpoint(s) with missing/unparseable as_of_time",
                self.scenario_id,
                self.dropped_checkpoints,
            )

        # event_id -> event_time, for false-positive windows. Events are passed
        # in already loaded; we never touch JSONL here.
        self.event_times: dict[str, datetime] = {}
        for ev in events or ():
            if isinstance(ev, Event):
                self.event_times[ev.event_id] = ev.event_time
            elif isinstance(ev, dict) and ev.get("event_id"):  # tolerate raw dicts
                self.event_times[ev["event_id"]] = parse_ts(ev["event_time"])

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _as_of(checkpoint: dict) -> Optional[datetime]:
        value = (checkpoint or {}).get("as_of_time")
        if not value:
            return None
        try:
            return parse_ts(value)
        except (ValueError, TypeError):
            return None

    def _checkpoint_as_of(self, as_of_time: datetime | str) -> Optional[dict]:
        """My most recent checkpoint at or before as_of_time, else None.

        None means I had said nothing yet at that moment, which scores as a
        miss on every field rather than an error.
        """
        target = parse_ts(as_of_time) if not isinstance(as_of_time, datetime) else as_of_time
        found = None
        for cp in self.my_checkpoints:  # sorted ascending
            ts = self._as_of(cp)
            if ts is not None and ts <= target:
                found = cp
            else:
                break
        return found

    def _invalid_enums(self) -> list[str]:
        """Enum typos in my own output -- reported, not scored."""
        problems = []
        for cp in self.my_checkpoints:
            at = cp.get("as_of_time")
            if cp.get("inferred_state") not in VALID_STATES:
                problems.append(f"{at}: inferred_state={cp.get('inferred_state')!r}")
            if cp.get("confidence_band") not in BAND_ORDER:
                problems.append(f"{at}: confidence_band={cp.get('confidence_band')!r}")
            if cp.get("action") not in VALID_ACTIONS:
                problems.append(f"{at}: action={cp.get('action')!r}")
            hitl = cp.get("hitl_status")
            if hitl is not None and hitl not in VALID_HITL:
                problems.append(f"{at}: hitl_status={hitl!r}")
        return problems

    # --- timeliness ---------------------------------------------------------

    def score_timeliness(self, gt_checkpoint: dict) -> str:
        """When did I *first* reach the expected state+band+action?

        on_time : first match inside [as_of - lead_days, as_of]
        early   : first match before that window (fired on weaker evidence)
        late    : first match after as_of
        never   : no checkpoint anywhere matches all three fields
        """
        as_of = parse_ts(gt_checkpoint["as_of_time"])
        lead_days = gt_checkpoint.get("ideal_action_lead_time_days")
        window_start = as_of - timedelta(days=lead_days or 0)

        first_match: Optional[datetime] = None
        for cp in self.my_checkpoints:
            if (
                cp.get("inferred_state") == gt_checkpoint.get("expected_inferred_state")
                and cp.get("confidence_band") == gt_checkpoint.get("expected_confidence_band")
                and cp.get("action") == gt_checkpoint.get("expected_action")
            ):
                first_match = self._as_of(cp)
                break

        if first_match is None:
            return "never"
        if first_match < window_start:
            return "early"
        if first_match > as_of:
            return "late"
        return "on_time"

    # --- checkpoints --------------------------------------------------------

    def score_checkpoints(self) -> list[dict]:
        """One result row per ground-truth checkpoint."""
        results: list[dict] = []
        for gt in self.ground_truth.get("checkpoints") or []:
            as_of = gt.get("as_of_time")
            mine = self._checkpoint_as_of(as_of) if as_of else None

            exp_state = gt.get("expected_inferred_state")
            exp_band = gt.get("expected_confidence_band")
            exp_action = gt.get("expected_action")
            exp_hitl = gt.get("expected_hitl_status")

            got_state = (mine or {}).get("inferred_state")
            got_band = (mine or {}).get("confidence_band")
            got_action = (mine or {}).get("action")
            got_hitl = (mine or {}).get("hitl_status")

            # Missing band scores the worst possible distance across the scale.
            if got_band in BAND_ORDER and exp_band in BAND_ORDER:
                band_distance = abs(BAND_ORDER[got_band] - BAND_ORDER[exp_band])
            else:
                band_distance = max(BAND_ORDER.values())

            # hitl is only compared when ground truth pins it down.
            hitl_match: Optional[bool] = None if exp_hitl is None else got_hitl == exp_hitl

            # Timeliness only applies to checkpoints that expect a real action
            # *and* specify a lead time.
            timeliness: Optional[str] = None
            if exp_action and exp_action != "no_action" and gt.get("ideal_action_lead_time_days") is not None:
                timeliness = self.score_timeliness(gt)

            results.append(
                {
                    "as_of_time": as_of,
                    "matched": mine is not None,
                    "state_match": got_state == exp_state,
                    "band_match": got_band == exp_band,
                    "band_distance": band_distance,
                    "action_match": got_action == exp_action,
                    "hitl_match": hitl_match,
                    "timeliness": timeliness,
                    # action_subtype is free text per the schema: reported for
                    # eyeballing, never scored.
                    "subtype_match": (
                        None
                        if gt.get("expected_action_subtype") is None
                        else (mine or {}).get("action_subtype")
                        == gt.get("expected_action_subtype")
                    ),
                    "expected": {
                        "inferred_state": exp_state,
                        "confidence_band": exp_band,
                        "action": exp_action,
                        "action_subtype": gt.get("expected_action_subtype"),
                        "hitl_status": exp_hitl,
                        "ideal_action_lead_time_days": gt.get("ideal_action_lead_time_days"),
                        "notes": gt.get("notes"),
                    },
                    "actual": {
                        "as_of_time": (mine or {}).get("as_of_time"),
                        "inferred_state": got_state,
                        "confidence_band": got_band,
                        "action": got_action,
                        "action_subtype": (mine or {}).get("action_subtype"),
                        "hitl_status": got_hitl,
                        "notes": (mine or {}).get("notes"),
                    },
                }
            )
        return results

    # --- false positives ----------------------------------------------------

    def score_false_positives(self) -> list[dict]:
        """Did I fire a forbidden action in the window after a red herring?"""
        results: list[dict] = []
        for check in self.ground_truth.get("false_positive_checks") or []:
            event_id = check.get("event_id")
            forbidden = set(check.get("must_not_trigger_action") or [])
            window_hours = check.get("window_hours") or 0
            event_time = self.event_times.get(event_id)

            if event_time is None:
                # Can't place the window; don't silently claim a pass.
                results.append(
                    {
                        "event_id": event_id,
                        "passed": True,
                        "resolved": False,
                        "window": None,
                        "must_not_trigger_action": sorted(forbidden),
                        "checkpoints_in_window": 0,
                        "violating_checkpoints": [],
                        "notes": check.get("notes"),
                        "reason": f"event_id {event_id} not found in the supplied events -- check excluded",
                    }
                )
                log.warning("%s: FP check event_id %s not in events", self.scenario_id, event_id)
                continue

            window_end = event_time + timedelta(hours=window_hours)
            in_window = [
                cp
                for cp in self.my_checkpoints
                if (ts := self._as_of(cp)) is not None and event_time <= ts <= window_end
            ]
            violations = [
                {
                    "as_of_time": cp.get("as_of_time"),
                    "action": cp.get("action"),
                    "inferred_state": cp.get("inferred_state"),
                    "notes": cp.get("notes"),
                }
                for cp in in_window
                if cp.get("action") in forbidden
            ]
            passed = not violations
            results.append(
                {
                    "event_id": event_id,
                    "passed": passed,
                    "resolved": True,
                    "window": [iso(event_time), iso(window_end)],
                    "must_not_trigger_action": sorted(forbidden),
                    "checkpoints_in_window": len(in_window),
                    # A pass with nothing in the window proves only silence,
                    # not restraint. Surfaced so it can't be mistaken for skill.
                    "vacuous": passed and not in_window,
                    "violating_checkpoints": violations,
                    "notes": check.get("notes"),
                    "reason": (
                        f"no forbidden action in {len(in_window)} checkpoint(s) in window"
                        if passed
                        else "fired "
                        + ", ".join(sorted({v["action"] for v in violations}))
                        + " inside window"
                    ),
                }
            )
        return results

    # --- aggregate ----------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate numbers plus the weighted overall_score.

        A component with nothing applicable (no timeliness checks, no FP checks)
        is reported as None and its weight is redistributed over the components
        that did apply, so an untestable dimension neither helps nor hurts.
        """
        cps = self.score_checkpoints()
        fps = self.score_false_positives()
        n = len(cps)

        state_accuracy = _safe_fraction(sum(c["state_match"] for c in cps), n)
        band_accuracy = _safe_fraction(sum(c["band_match"] for c in cps), n)
        action_accuracy = _safe_fraction(sum(c["action_match"] for c in cps), n)
        avg_band_distance = (
            None if n == 0 else sum(c["band_distance"] for c in cps) / n
        )

        hitl_scored = [c for c in cps if c["hitl_match"] is not None]
        hitl_accuracy = _safe_fraction(
            sum(c["hitl_match"] for c in hitl_scored), len(hitl_scored)
        )

        timed = [c for c in cps if c["timeliness"] is not None]
        breakdown = {v: sum(1 for c in timed if c["timeliness"] == v) for v in TIMELINESS_VERDICTS}
        timeliness = _safe_fraction(breakdown["on_time"], len(timed))

        resolved_fps = [f for f in fps if f["resolved"]]
        vacuous_fps = sum(1 for f in resolved_fps if f.get("vacuous"))
        fp_rate = _safe_fraction(sum(f["passed"] for f in resolved_fps), len(resolved_fps))

        components: dict[str, Optional[float]] = {
            "state_accuracy": state_accuracy,
            "band_accuracy": band_accuracy,
            "action_accuracy": action_accuracy,
            "timeliness": timeliness,
            "false_positive_pass_rate": fp_rate,
        }
        applicable = {k: v for k, v in components.items() if v is not None}
        weight_total = sum(WEIGHTS[k] for k in applicable)
        overall = (
            sum(WEIGHTS[k] * v for k, v in applicable.items()) / weight_total
            if weight_total
            else 0.0
        )

        return {
            "scenario_id": self.scenario_id,
            "checkpoints_expected": n,
            "checkpoints_submitted": len(self.my_checkpoints),
            "checkpoints_dropped": self.dropped_checkpoints,
            "state_accuracy": state_accuracy,
            "band_accuracy": band_accuracy,
            "avg_band_distance": avg_band_distance,
            "action_accuracy": action_accuracy,
            "hitl_accuracy": hitl_accuracy,
            "timeliness": timeliness,
            "timeliness_breakdown": breakdown,
            "false_positive_checks": len(fps),
            "false_positive_unresolved": len(fps) - len(resolved_fps),
            "false_positive_pass_rate": fp_rate,
            "false_positive_vacuous_passes": vacuous_fps,
            "overall_score": overall,
            "weights_applied": {k: WEIGHTS[k] for k in applicable},
            "components_skipped": sorted(set(components) - set(applicable)),
            "invalid_enum_values": self._invalid_enums(),
        }

    def report(self) -> dict:
        """Everything, ready to serialise to eval_results/{scenario_id}_report.json."""
        return {
            "scenario_id": self.scenario_id,
            "true_narrative": self.ground_truth.get("true_narrative"),
            "signal_events": self.ground_truth.get("signal_events") or [],
            "red_herring_events": self.ground_truth.get("red_herring_events") or [],
            "checkpoint_results": self.score_checkpoints(),
            "false_positive_results": self.score_false_positives(),
            "summary": self.summary(),
        }
