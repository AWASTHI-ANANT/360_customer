#!/usr/bin/env python3
"""Invariant checks for the scoring harness. Run: python -m eval.test_harness"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from c360.loader import load_scenario
from eval.harness import WEIGHTS, ScenarioScorer
from eval.run_eval import load_my_checkpoints, main as run_eval_main

PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    assert cond, f"FAILED: {name} {detail}"
    PASSED.append(name)
    print(f"  ok  {name}")


def cp(as_of, state="medical_hardship", band="high", action="support_intervention",
       hitl="escalated", subtype=None, notes="") -> dict:
    return {
        "as_of_time": as_of, "inferred_state": state, "confidence_band": band,
        "action": action, "action_subtype": subtype, "hitl_status": hitl, "notes": notes,
    }


GT = {
    "scenario_id": "test",
    "checkpoints": [
        {   # no action expected -> timeliness must be None
            "as_of_time": "2026-02-15T00:00:00Z",
            "expected_inferred_state": "medical_hardship",
            "expected_confidence_band": "low",
            "expected_action": "no_action",
            "expected_action_subtype": None,
            "expected_hitl_status": None,
            "ideal_action_lead_time_days": None,
            "notes": "early, weak",
        },
        {   # action + lead time -> timeliness applies
            "as_of_time": "2026-03-26T00:00:00Z",
            "expected_inferred_state": "medical_hardship",
            "expected_confidence_band": "high",
            "expected_action": "support_intervention",
            "expected_action_subtype": "medical_hardship_payment_plan",
            "expected_hitl_status": "escalated",
            "ideal_action_lead_time_days": 1,
            "notes": "act now",
        },
    ],
    "false_positive_checks": [],
}


def test_as_of_matching() -> None:
    print("_checkpoint_as_of")
    mine = [cp("2026-02-10T00:00:00Z"), cp("2026-03-01T00:00:00Z"), cp("2026-04-01T00:00:00Z")]
    s = ScenarioScorer(GT, mine, [])
    check("picks most recent <= target",
          s._checkpoint_as_of("2026-03-15T00:00:00Z")["as_of_time"] == "2026-03-01T00:00:00Z")
    check("exact boundary is inclusive",
          s._checkpoint_as_of("2026-03-01T00:00:00Z")["as_of_time"] == "2026-03-01T00:00:00Z")
    check("None before my first checkpoint", s._checkpoint_as_of("2026-01-01T00:00:00Z") is None)
    check("last checkpoint carries forward",
          s._checkpoint_as_of("2027-01-01T00:00:00Z")["as_of_time"] == "2026-04-01T00:00:00Z")
    check("input order does not matter",
          ScenarioScorer(GT, list(reversed(mine)), [])._checkpoint_as_of("2026-03-15T00:00:00Z")[
              "as_of_time"] == "2026-03-01T00:00:00Z")
    check("unparseable as_of_time dropped, not fatal",
          ScenarioScorer(GT, mine + [cp("not-a-date")], []).dropped_checkpoints == 1)


def test_timeliness() -> None:
    print("score_timeliness")
    gt = GT["checkpoints"][1]  # as_of 03-26, lead 1d -> window [03-25, 03-26]

    def verdict(*times, **kw):
        return ScenarioScorer(GT, [cp(t, **kw) for t in times], []).score_timeliness(gt)

    check("inside window -> on_time", verdict("2026-03-25T12:00:00Z") == "on_time")
    check("exactly at window_start -> on_time", verdict("2026-03-25T00:00:00Z") == "on_time")
    check("exactly at as_of -> on_time", verdict("2026-03-26T00:00:00Z") == "on_time")
    check("before window_start -> early", verdict("2026-03-01T00:00:00Z") == "early")
    check("after as_of -> late", verdict("2026-03-27T00:00:00Z") == "late")
    check("no full match anywhere -> never", verdict("2026-03-25T00:00:00Z", action="no_action") == "never")
    check("band mismatch alone -> never", verdict("2026-03-25T00:00:00Z", band="medium") == "never")
    check("state mismatch alone -> never", verdict("2026-03-25T00:00:00Z", state="churn_risk") == "never")
    check("no checkpoints at all -> never", ScenarioScorer(GT, [], []).score_timeliness(gt) == "never")
    check("uses FIRST match, so an early call is not rescued by a later on-time one",
          verdict("2026-03-01T00:00:00Z", ) == "early"
          and ScenarioScorer(GT, [cp("2026-03-01T00:00:00Z"), cp("2026-03-25T12:00:00Z")], []
                             ).score_timeliness(gt) == "early")


def test_checkpoints() -> None:
    print("score_checkpoints")
    mine = [
        cp("2026-02-15T00:00:00Z", band="low", action="no_action", hitl=None),
        cp("2026-03-26T00:00:00Z", subtype="medical_hardship_payment_plan"),
    ]
    rows = ScenarioScorer(GT, mine, []).score_checkpoints()
    check("one row per ground-truth checkpoint", len(rows) == 2)
    check("perfect first row", rows[0]["state_match"] and rows[0]["band_match"] and rows[0]["action_match"])
    check("band_distance 0 on exact match", rows[0]["band_distance"] == 0)
    check("hitl skipped when expected is null", rows[0]["hitl_match"] is None)
    check("timeliness None when expected_action is no_action", rows[0]["timeliness"] is None)
    check("timeliness applies on action row", rows[1]["timeliness"] == "on_time")
    check("hitl compared when expected is set", rows[1]["hitl_match"] is True)
    check("subtype reported when expected", rows[1]["subtype_match"] is True)
    check("expected/actual both echoed", rows[1]["expected"]["notes"] == "act now"
          and rows[1]["actual"]["as_of_time"] == "2026-03-26T00:00:00Z")

    # band ordinal distance
    dists = {
        b: ScenarioScorer(GT, [cp("2026-02-15T00:00:00Z", band=b, action="no_action")], []
                          ).score_checkpoints()[0]["band_distance"]
        for b in ("low", "medium", "high")
    }
    check("band distance is ordinal |mine-expected|", dists == {"low": 0, "medium": 1, "high": 2}, str(dists))

    # gt checkpoint before my first checkpoint -> clean miss
    late_only = ScenarioScorer(GT, [cp("2026-04-01T00:00:00Z")], []).score_checkpoints()
    check("gt before my first checkpoint is a miss, not an error",
          late_only[0]["matched"] is False and late_only[0]["state_match"] is False)
    check("missing band scores max distance", late_only[0]["band_distance"] == 2)


def test_false_positives() -> None:
    print("score_false_positives")
    scn = load_scenario("scenario_01")
    gt = json.loads((scn.path / "ground_truth.json").read_text())
    # EVT_000382 @ 2026-02-05T10:15Z, 72h window, must not fire compliance_fraud_hold
    violating = [cp("2026-02-06T00:00:00Z", action="compliance_fraud_hold")]
    res = ScenarioScorer(gt, violating, scn.all_events).score_false_positives()
    check("violation inside window caught", res[0]["passed"] is False)
    check("violating checkpoint listed", res[0]["violating_checkpoints"][0]["action"] == "compliance_fraud_hold")
    check("window resolved from event_time", res[0]["window"][0] == "2026-02-05T10:15:00Z")
    check("72h window end", res[0]["window"][1] == "2026-02-08T10:15:00Z")

    outside = [cp("2026-02-20T00:00:00Z", action="compliance_fraud_hold")]
    res2 = ScenarioScorer(gt, outside, scn.all_events).score_false_positives()
    check("same action outside window passes", res2[0]["passed"] is True)

    allowed = [cp("2026-02-06T00:00:00Z", action="support_intervention")]
    res3 = ScenarioScorer(gt, allowed, scn.all_events).score_false_positives()
    check("non-forbidden action in window passes", res3[0]["passed"] is True)
    check("non-vacuous pass flagged as such", res3[0]["vacuous"] is False)
    check("empty-window pass flagged vacuous",
          ScenarioScorer(gt, [], scn.all_events).score_false_positives()[0]["vacuous"] is True)

    # unresolvable event_id excluded rather than silently passed
    bad_gt = {**gt, "false_positive_checks": [
        {"event_id": "EVT_NOPE", "must_not_trigger_action": ["personalized_offer"],
         "window_hours": 72, "notes": ""}]}
    res4 = ScenarioScorer(bad_gt, violating, scn.all_events)
    check("unknown event_id marked unresolved", res4.score_false_positives()[0]["resolved"] is False)
    check("unresolved check excluded from pass rate",
          res4.summary()["false_positive_pass_rate"] is None
          and res4.summary()["false_positive_unresolved"] == 1)
    check("no events passed in -> unresolved, no crash",
          ScenarioScorer(gt, violating, []).score_false_positives()[0]["resolved"] is False)


def test_summary() -> None:
    print("summary")
    mine = [
        cp("2026-02-15T00:00:00Z", band="low", action="no_action", hitl=None),
        cp("2026-03-26T00:00:00Z"),
    ]
    s = ScenarioScorer(GT, mine, []).summary()
    check("state accuracy 1.0", s["state_accuracy"] == 1.0)
    check("band accuracy 1.0", s["band_accuracy"] == 1.0)
    check("avg_band_distance 0", s["avg_band_distance"] == 0.0)
    check("action accuracy 1.0", s["action_accuracy"] == 1.0)
    check("hitl accuracy ignores the null row", s["hitl_accuracy"] == 1.0)
    check("timeliness breakdown counts only applicable rows",
          s["timeliness_breakdown"] == {"on_time": 1, "early": 0, "late": 0, "never": 0})
    check("no FP checks -> rate None", s["false_positive_pass_rate"] is None)
    check("skipped component named", s["components_skipped"] == ["false_positive_pass_rate"])
    check("weights renormalised over applicable components", abs(s["overall_score"] - 1.0) < 1e-9,
          str(s["overall_score"]))

    # a known partial: state right, band one off, action wrong, timing never
    half = [cp("2026-02-15T00:00:00Z", band="low", action="no_action", hitl=None),
            cp("2026-03-26T00:00:00Z", band="medium", action="no_action", hitl=None)]
    s2 = ScenarioScorer(GT, half, []).summary()
    applied = WEIGHTS["state_accuracy"] + WEIGHTS["band_accuracy"] + WEIGHTS["action_accuracy"] + WEIGHTS["timeliness"]
    want = (WEIGHTS["state_accuracy"] * 1.0 + WEIGHTS["band_accuracy"] * 0.5
            + WEIGHTS["action_accuracy"] * 0.5 + WEIGHTS["timeliness"] * 0.0) / applied
    check("overall_score matches the weighted formula", abs(s2["overall_score"] - want) < 1e-9,
          f"{s2['overall_score']} vs {want}")
    check("avg_band_distance averages", s2["avg_band_distance"] == 0.5)
    check("hitl accuracy None when nothing comparable", s2["hitl_accuracy"] == 0.0 or s2["hitl_accuracy"] is None)

    check("empty ground truth does not divide by zero",
          ScenarioScorer({}, mine, []).summary()["overall_score"] == 0.0)
    check("invalid enum values reported",
          ScenarioScorer(GT, [cp("2026-03-26T00:00:00Z", state="made_up_state")], []
                         ).summary()["invalid_enum_values"] != [])


def test_file_loading(tmp: Path) -> None:
    print("run_eval file handling")
    check("missing file -> empty + reason", load_my_checkpoints(tmp / "nope.json")[0] == []
          and "no output file" in load_my_checkpoints(tmp / "nope.json")[1])
    (tmp / "empty.json").write_text("")
    check("empty file -> empty + reason", load_my_checkpoints(tmp / "empty.json")[1].endswith("all misses"))
    (tmp / "bad.json").write_text("{ nope")
    check("malformed JSON -> empty + reason", "not valid JSON" in load_my_checkpoints(tmp / "bad.json")[1])
    (tmp / "wrapped.json").write_text(json.dumps({"checkpoints": [cp("2026-03-26T00:00:00Z")]}))
    check("accepts {'checkpoints': [...]} wrapper", len(load_my_checkpoints(tmp / "wrapped.json")[0]) == 1)
    (tmp / "arr.json").write_text(json.dumps([cp("2026-03-26T00:00:00Z")]))
    check("accepts a bare array", len(load_my_checkpoints(tmp / "arr.json")[0]) == 1)
    (tmp / "obj.json").write_text(json.dumps({"foo": 1}))
    check("unexpected shape -> empty + reason", load_my_checkpoints(tmp / "obj.json")[0] == [])


def test_cli_end_to_end(tmp: Path) -> None:
    print("run_eval end to end")
    out = tmp / "output"
    out.mkdir(parents=True, exist_ok=True)
    results = tmp / "eval_results"
    rc = run_eval_main(["all", "--output-dir", str(out), "--results-dir", str(results)])
    check("'all' runs with zero output files", rc == 0)
    written = sorted(p.name for p in results.glob("*_report.json"))
    check("a report per scenario", written == ["scenario_01_report.json", "scenario_02_report.json",
                                              "scenario_03_report.json"], str(written))
    rep = json.loads((results / "scenario_01_report.json").read_text())
    check("report has detail + summary",
          {"checkpoint_results", "false_positive_results", "summary"} <= set(rep))
    check("report keeps the narrative for the writeup", bool(rep["true_narrative"]))
    rc2 = run_eval_main(["scenario_99", "--output-dir", str(out), "--results-dir", str(results)])
    check("unknown scenario exits non-zero, no traceback", rc2 == 1)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c360_eval_test_"))
    try:
        test_as_of_matching()
        test_timeliness()
        test_checkpoints()
        test_false_positives()
        test_summary()
        test_file_loading(tmp)
        test_cli_end_to_end(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
