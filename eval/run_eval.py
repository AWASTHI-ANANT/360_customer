#!/usr/bin/env python3
"""Score checkpoint output against ground truth and write a report.

    python -m eval.run_eval                 # every scenario found
    python -m eval.run_eval scenario_01
    python -m eval.run_eval all --output-dir output

Reads   ground_truth/{sid}.json   (falls back to the dataset's
        {data_root}/{sid}/ground_truth.json)
        output/{sid}_checkpoints.json
Writes  eval_results/{sid}_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from c360.loader import DEFAULT_DATA_ROOT, find_scenarios, load_scenario
from eval.harness import WEIGHTS, ScenarioScorer

ROOT = Path(__file__).resolve().parent.parent

TICK, CROSS, DASH = "PASS", "FAIL", "n/a "
# Timeliness verdicts that shouldn't read as failures in the report.
_TIMELINESS_MARK = {"on_time": "PASS", "early": "EARLY", "late": "LATE", "never": "NEVER"}


def _mark(ok: Optional[bool]) -> str:
    return DASH if ok is None else (TICK if ok else CROSS)


def _pct(value: Optional[float]) -> str:
    return " n/a " if value is None else f"{value * 100:5.1f}%"


def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def find_ground_truths(gt_dir: Path, data_root: Optional[Path]) -> dict[str, Path]:
    """scenario_id -> ground truth path, from ground_truth/ or the dataset folders.

    Keyed on the *folder*/file name, not the scenario_id inside the JSON -- in
    this dataset those disagree (scenario_01 contains
    "scenario_05_major_medical_event").
    """
    found: dict[str, Path] = {}
    if gt_dir.is_dir():
        for p in sorted(gt_dir.glob("*.json")):
            found[p.stem] = p
    for folder in find_scenarios(data_root):
        gt = folder / "ground_truth.json"
        if gt.exists():
            found.setdefault(folder.name, gt)
    return found


def resolve_ground_truth(sid: str, gt_dir: Path, data_root: Optional[Path]) -> Optional[Path]:
    direct = gt_dir / f"{sid}.json"
    if direct.exists():
        return direct
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    fallback = root / sid / "ground_truth.json"
    return fallback if fallback.exists() else None


def load_my_checkpoints(path: Path) -> tuple[list[dict], str]:
    """Tolerate a missing, empty or wrapped output file -- never crash."""
    if not path.exists():
        return [], f"no output file at {path} -- scoring as all misses"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [], f"cannot read {path}: {exc}"
    if not text:
        return [], f"{path.name} is empty -- scoring as all misses"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], f"{path.name} is not valid JSON ({exc.msg}) -- scoring as all misses"
    # Accept a bare array or {"checkpoints": [...]}.
    if isinstance(data, dict):
        data = data.get("checkpoints") or []
    if not isinstance(data, list):
        return [], f"{path.name} is not a checkpoint array -- scoring as all misses"
    return data, ""


def score_one(
    sid: str,
    gt_dir: Path,
    output_dir: Path,
    results_dir: Path,
    data_root: Optional[Path],
) -> Optional[dict]:
    gt_path = resolve_ground_truth(sid, gt_dir, data_root)
    if gt_path is None:
        print(f"\n=== {sid} === no ground truth found (looked in {gt_dir}/ and the dataset folder)")
        return None
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))

    my_path = output_dir / f"{sid}_checkpoints.json"
    my_checkpoints, warning = load_my_checkpoints(my_path)

    # Events come from the loader so event_id -> event_time works for FP windows.
    try:
        events = load_scenario(sid, data_root=data_root).all_events
    except Exception as exc:  # noqa: BLE001 - a bad scenario folder shouldn't kill the sweep
        print(f"    ! cannot load events for {sid} ({exc}) -- FP checks will be unresolved")
        events = []

    scorer = ScenarioScorer(ground_truth, my_checkpoints, events)
    report = scorer.report()
    summary = report["summary"]

    gt_sid = ground_truth.get("scenario_id", sid)
    header = f"=== {sid} ==="
    if gt_sid != sid:
        header += f" (ground truth calls itself {gt_sid!r})"
    print(f"\n{header}")
    print(f"    ground truth : {gt_path}")
    print(f"    my output    : {my_path if my_path.exists() else '(missing)'}")
    if warning:
        print(f"    ! {warning}")
    if summary["checkpoints_dropped"]:
        print(f"    ! {summary['checkpoints_dropped']} checkpoint(s) dropped: bad as_of_time")
    if summary["invalid_enum_values"]:
        print(f"    ! values outside the allowed enums (not scored):")
        for problem in summary["invalid_enum_values"]:
            print(f"        {problem}")

    print(f"\n    checkpoints ({summary['checkpoints_expected']} expected, "
          f"{summary['checkpoints_submitted']} submitted)")
    for row in report["checkpoint_results"]:
        exp, act = row["expected"], row["actual"]
        tl = row["timeliness"]
        print(f"      {row['as_of_time']}  {'(no checkpoint yet)' if not row['matched'] else 'vs mine @ ' + str(act['as_of_time'])}")
        print(f"        state  {_mark(row['state_match'])}  expected {exp['inferred_state']!r:36} got {act['inferred_state']!r}")
        print(f"        band   {_mark(row['band_match'])}  expected {exp['confidence_band']!r:36} got {act['confidence_band']!r}  (distance {row['band_distance']})")
        print(f"        action {_mark(row['action_match'])}  expected {exp['action']!r:36} got {act['action']!r}")
        print(f"        hitl   {_mark(row['hitl_match'])}  expected {exp['hitl_status']!r:36} got {act['hitl_status']!r}")
        if tl is not None:
            print(f"        timing {_TIMELINESS_MARK[tl]:4}  first state+band+action match vs "
                  f"ideal lead {exp['ideal_action_lead_time_days']}d")
        if row["subtype_match"] is not None:
            print(f"        subtype {_mark(row['subtype_match'])} expected {exp['action_subtype']!r} got {act['action_subtype']!r}  (free text, unscored)")

    print(f"\n    false-positive checks ({summary['false_positive_checks']})")
    for fp in report["false_positive_results"]:
        mark = DASH if not fp["resolved"] else _mark(fp["passed"])
        window = f"{fp['window'][0]} .. {fp['window'][1]}" if fp["window"] else "unresolved"
        print(f"      {mark}  {fp['event_id']}  {window}")
        print(f"            must not fire: {', '.join(fp['must_not_trigger_action'])}")
        print(f"            {fp['reason']}")
        for v in fp["violating_checkpoints"]:
            print(f"            !! {v['as_of_time']} fired {v['action']!r} (state {v['inferred_state']!r})")

    print(f"\n    summary")
    print(f"      state accuracy       {_pct(summary['state_accuracy'])}")
    print(f"      band accuracy        {_pct(summary['band_accuracy'])}   avg distance {_num(summary['avg_band_distance'])}")
    print(f"      action accuracy      {_pct(summary['action_accuracy'])}")
    print(f"      hitl accuracy        {_pct(summary['hitl_accuracy'])}")
    print(f"      timeliness (on_time) {_pct(summary['timeliness'])}   {summary['timeliness_breakdown']}")
    print(f"      false-positive pass  {_pct(summary['false_positive_pass_rate'])}")
    if summary.get("false_positive_vacuous_passes"):
        print(f"      ! {summary['false_positive_vacuous_passes']} of those passed with no "
              f"checkpoint in the window (silence, not restraint)")
    if summary["components_skipped"]:
        print(f"      ! not applicable, weights redistributed: {', '.join(summary['components_skipped'])}")
    print(f"      OVERALL SCORE        {summary['overall_score']:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{sid}_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"      report -> {out_path}")
    return report


def print_combined(reports: list[dict]) -> None:
    """Unweighted mean across scenarios, skipping components that never applied."""
    print(f"\n{'=' * 64}\n=== combined across {len(reports)} scenario(s) ===")
    fields = [
        "state_accuracy",
        "band_accuracy",
        "action_accuracy",
        "hitl_accuracy",
        "timeliness",
        "false_positive_pass_rate",
        "overall_score",
    ]
    means: dict[str, Optional[float]] = {}
    for f in fields:
        vals = [r["summary"][f] for r in reports if r["summary"].get(f) is not None]
        means[f] = sum(vals) / len(vals) if vals else None

    totals = {v: 0 for v in ("on_time", "early", "late", "never")}
    for r in reports:
        for k, v in r["summary"]["timeliness_breakdown"].items():
            totals[k] += v

    print(f"\n  {'scenario':34} {'state':>7} {'band':>7} {'action':>7} {'timely':>7} {'fp':>7} {'overall':>8}")
    for r in reports:
        s = r["summary"]
        print(
            f"  {s['scenario_id'][:34]:34} {_pct(s['state_accuracy'])} {_pct(s['band_accuracy'])} "
            f"{_pct(s['action_accuracy'])} {_pct(s['timeliness'])} "
            f"{_pct(s['false_positive_pass_rate'])} {s['overall_score']:8.4f}"
        )
    print(f"  {'-' * 80}")
    print(
        f"  {'MEAN':34} {_pct(means['state_accuracy'])} {_pct(means['band_accuracy'])} "
        f"{_pct(means['action_accuracy'])} {_pct(means['timeliness'])} "
        f"{_pct(means['false_positive_pass_rate'])} "
        f"{means['overall_score'] if means['overall_score'] is not None else 0.0:8.4f}"
    )
    print(f"\n  hitl accuracy (mean)   {_pct(means['hitl_accuracy'])}")
    print(f"  timeliness totals      {totals}")
    print(f"  weights                {WEIGHTS}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenario", nargs="?", default="all", help="scenario id, or 'all' (default)")
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "ground_truth")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "eval_results")
    ap.add_argument("--data-root", type=Path, default=None, help="folder containing scenario_* dirs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.scenario == "all":
        scenarios = sorted(find_ground_truths(args.gt_dir, args.data_root))
        if not scenarios:
            print(
                f"no ground truth found in {args.gt_dir}/ or under "
                f"{args.data_root or DEFAULT_DATA_ROOT}",
                file=sys.stderr,
            )
            return 1
    else:
        scenarios = [args.scenario]

    reports = [
        r
        for sid in scenarios
        if (r := score_one(sid, args.gt_dir, args.output_dir, args.results_dir, args.data_root))
        is not None
    ]
    if not reports:
        return 1
    if len(reports) > 1:
        print_combined(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
