"""Print the span tree for one simulated day.

    python -m tracing.trace_view --scenario scenario_03 --date 2026-03-08
    python -m tracing.trace_view --scenario scenario_03 --date 2026-03-08 --full

A day's traces are every trace whose root span's sim_time falls on that date:
the day-boundary pass at 00:00 plus one trace per event ingested that day.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return spans


def render(spans: list[dict], day: str, full: bool = False) -> list[str]:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    width = None if full else 110
    out: list[str] = []
    for trace_id in sorted(by_trace):
        members = by_trace[trace_id]
        roots = [s for s in members if s.get("parent_span_id") is None]
        if not roots or not str(roots[0].get("sim_time") or "").startswith(day):
            continue
        children: dict[str, list[dict]] = defaultdict(list)
        for s in members:
            if s.get("parent_span_id"):
                children[s["parent_span_id"]].append(s)

        def walk(span: dict, prefix: str, last: bool, depth: int) -> None:
            branch = "" if depth == 0 else ("└─ " if last else "├─ ")
            head = (f"{prefix}{branch}{span['name']} [{span['kind']}] "
                    f"{span.get('latency_ms', 0):.1f}ms {span['status']}")
            if depth == 0:
                head = f"{trace_id} {head}  sim={span.get('sim_time')}"
            out.append(head)
            pad = prefix + ("" if depth == 0 else ("   " if last else "│  "))
            detail = []
            if span.get("inputs") not in (None, "[]"):
                detail.append(f"in : {span['inputs']}")
            if span.get("outputs") not in (None, "[]", "null"):
                detail.append(f"out: {span['outputs']}")
            if span.get("error"):
                detail.append(f"err: {span['error']}")
            for d in detail:
                out.append(f"{pad}    {d if width is None else d[:width]}")
            kids = sorted(children.get(span["span_id"], []), key=lambda s: s["span_id"])
            for i, kid in enumerate(kids):
                walk(kid, pad, i == len(kids) - 1, depth + 1)

        for root in roots:
            walk(root, "", True, 0)
        out.append("")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (simulated)")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "out" / "logs")
    ap.add_argument("--full", action="store_true", help="do not truncate inputs/outputs")
    args = ap.parse_args(argv)

    path = args.log_dir / f"{args.scenario}_trace.jsonl"
    if not path.exists():
        print(f"no trace at {path} -- run `python run_skeleton.py -s {args.scenario}` first",
              file=sys.stderr)
        return 1
    lines = render(load(path), args.date, args.full)
    if not lines:
        print(f"no spans for {args.scenario} on {args.date}", file=sys.stderr)
        return 1
    print(f"# {args.scenario} {args.date} ({path})\n")
    print("\n".join(lines).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
