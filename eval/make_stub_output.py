#!/usr/bin/env python3
"""Emit a trivial baseline checkpoint file per scenario, for exercising the harness.

This is scaffolding, NOT inference: it says "nothing is happening" on every
simulated day. Its only jobs are (a) give run_eval real-shaped input before the
agents exist, and (b) establish the floor score a do-nothing system gets.
Replace it with the real pipeline's output.

    python -m eval.make_stub_output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c360 import SimulatedClock, find_scenarios, iso, load_scenario

ROOT = Path(__file__).resolve().parent.parent


def stub_for(scenario_id: str, data_root: Path | None) -> list[dict]:
    scn = load_scenario(scenario_id, data_root=data_root)
    clock = SimulatedClock.from_config(scn.replay_config)
    checkpoints: list[dict] = []

    @clock.on_day_boundary
    def _checkpoint(sim_date):
        checkpoints.append(
            {
                "as_of_time": iso(clock.now()),
                "inferred_state": "no_significant_event",
                "confidence_band": "low",
                "action": "no_action",
                "action_subtype": None,
                "hitl_status": "auto_approved",
                "notes": "stub baseline: no inference yet",
            }
        )

    clock.fast_forward(scn.live_events)
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", default="all")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output")
    ap.add_argument("--data-root", type=Path, default=None)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = (
        [p.name for p in find_scenarios(args.data_root)]
        if args.scenario == "all"
        else [args.scenario]
    )
    for sid in ids:
        checkpoints = stub_for(sid, args.data_root)
        path = args.output_dir / f"{sid}_checkpoints.json"
        path.write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
        print(f"{path}  ({len(checkpoints)} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
