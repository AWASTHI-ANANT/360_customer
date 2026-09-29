"""Span tracing for the replay: @traced decorator and the trace_view CLI."""

from .tracer import configure, disable, traced

__all__ = ["configure", "disable", "traced"]
