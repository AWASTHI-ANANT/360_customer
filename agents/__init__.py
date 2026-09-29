"""Agents. Signal and support write findings; synthesis picks a state; action picks an action."""

from .action_agent import ActionAgent
from .base import Agent, AgentContext, Finding
from .signal_agent import SignalAgent
from .support_agent import SupportAgent
from .synthesis_agent import SynthesisAgent

__all__ = [
    "ActionAgent",
    "Agent",
    "AgentContext",
    "Finding",
    "SignalAgent",
    "SupportAgent",
    "SynthesisAgent",
]
