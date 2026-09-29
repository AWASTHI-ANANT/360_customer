"""@traced(kind): one JSONL line per call to logs/{scenario}_trace.jsonl.

Parent/child links come from contextvars, so nesting follows the real call
stack with no plumbing: a call made while another traced call is running
becomes its child. A call with no traced parent starts a new trace -- in the
replay those roots are the per-event and per-day callbacks in run_skeleton.py,
so a trace is "one event" or "one simulated day".

Line fields: trace_id, span_id, parent_span_id, name, kind, sim_time,
latency_ms, inputs, outputs, status, error.

Inputs and outputs are summarised (event ids, finding keys, dates) rather than
dumped, then masked with c360.masking and truncated, so a trace never holds a
customer name or account id.

ASSUMPTIONS (spec was silent):
- Ids are sequential counters per configure() ("T00012", "S000345"), not
  random, so two runs of the same scenario produce the same ids.
- Unconfigured (tests calling agents directly), the decorator is a pass-through
  that writes nothing.
- sim_time is the simulated clock's now() at span start.
- latency_ms is wall-clock and is the only non-deterministic field.
"""

from __future__ import annotations

import contextvars
import functools
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

# c360 is imported inside functions: c360.llm_client imports this module, so a
# top-level import of c360 here would be circular.

MAX_FIELD_CHARS = 300
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:]+Z?)?")

_current_span: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("span", default=None)
_current_trace: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace", default=None)


@dataclass
class _Tracer:
    path: Path
    clock: Any
    profile: Any
    traces: int = 0
    spans: int = 0

    def sim_time(self) -> Optional[str]:
        from c360.loader import iso

        try:
            return iso(self.clock.now()) if self.clock is not None else None
        except Exception:  # noqa: BLE001 - a clock hiccup must not break a call
            return None

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


_tracer: Optional[_Tracer] = None


def configure(scenario_id: str, log_dir: str | Path, clock=None, profile=None,
              fresh: bool = True) -> Path:
    """Point tracing at logs/{scenario_id}_trace.jsonl for one run."""
    global _tracer
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{scenario_id}_trace.jsonl"
    if fresh:
        path.write_text("", encoding="utf-8")
    _tracer = _Tracer(path=path, clock=clock, profile=profile)
    return path


def disable() -> None:
    global _tracer
    _tracer = None


def _summarise(value: Any) -> Any:
    """Compact, JSON-able view of an argument or return value."""
    from c360.loader import iso

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    event_id = getattr(value, "event_id", None)
    if isinstance(event_id, str):
        return f"{event_id} {getattr(value, 'source_system', '')}/{getattr(value, 'event_type', '')}"
    if hasattr(value, "key") and hasattr(value, "confidence_band"):  # a Finding
        return f"finding {value.key} [{value.confidence_band}]"
    if hasattr(value, "chunk_id"):  # a PolicyChunk
        return f"chunk {value.chunk_id}"
    if isinstance(value, (list, tuple)):
        return [_summarise(v) for v in value[:8]] + (["..."] if len(value) > 8 else [])
    if isinstance(value, dict):
        return {str(k): _summarise(v) for k, v in list(value.items())[:12]}
    return type(value).__name__  # contexts, stores, agents: the name is enough


def _clean(value: Any, profile) -> str:
    from c360.masking import mask

    text = json.dumps(_summarise(value), default=str)
    text, mapping = mask(text, profile)
    # mask()'s phone pattern also swallows ISO dates; a date identifies no one
    # and a trace without dates is unreadable, so those are put back.
    for token, original in mapping.items():
        if _DATE_RE.fullmatch(original):
            text = text.replace(token, original)
    if len(text) > MAX_FIELD_CHARS:
        text = text[: MAX_FIELD_CHARS - 3] + "..."
    return text


def traced(kind: str, name: Optional[str] = None) -> Callable:
    """Decorator: record one span per call. Methods drop `self` and ctx args."""

    def deco(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tracer = _tracer
            if tracer is None:
                return fn(*args, **kwargs)
            tracer.spans += 1
            span_id = f"S{tracer.spans:06d}"
            parent = _current_span.get()
            trace_id = _current_trace.get()
            if trace_id is None:
                tracer.traces += 1
                trace_id = f"T{tracer.traces:05d}"
            t_tok = _current_trace.set(trace_id)
            s_tok = _current_span.set(span_id)
            sim_time = tracer.sim_time()
            shown_args = [a for a in args if type(a).__name__ not in _SKIP_ARG_TYPES]
            record: dict[str, Any] = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent,
                "name": span_name,
                "kind": kind,
                "sim_time": sim_time,
                "inputs": _clean({"args": shown_args, **kwargs} if kwargs else shown_args,
                                 tracer.profile),
            }
            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:MAX_FIELD_CHARS],
                              outputs=None)
                raise
            else:
                record.update(status="ok", error=None, outputs=_clean(result, tracer.profile))
                return result
            finally:
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                _current_span.reset(s_tok)
                _current_trace.reset(t_tok)
                tracer.write(record)

        return wrapper

    return deco


# Arguments that are plumbing, not data: never rendered into a span.
_SKIP_ARG_TYPES = {
    "AgentContext", "SignalAgent", "SupportAgent", "SynthesisAgent", "ActionAgent",
    "GuardrailAgent", "LLMClient", "CheckpointWriter", "MemoryStore", "ApprovalLog",
}
