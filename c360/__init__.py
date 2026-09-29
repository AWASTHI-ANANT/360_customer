"""c360 -- Customer 360 agentic system (Inter IIT Prepathon).

Flat package: import agents alongside these modules later.

    from c360 import load_scenario, SimulatedClock, DailyLogWriter, MemoryStore
"""

from .clock import SimulatedClock
from .daily_log_writer import DailyLogWriter, summarize_event
from .derive import derive, derive_all
from .event_store import EventStore
from .llm_client import LLMClient, LLMUnavailable, complete_json
from .masking import mask, mask_event_text, unmask
from .loader import (
    DEFAULT_DATA_ROOT,
    KNOWN_SOURCE_SYSTEMS,
    Account,
    CustomerProfile,
    Event,
    ReplayConfig,
    Scenario,
    ScenarioLoadError,
    find_scenarios,
    iso,
    load_ground_truth,
    load_scenario,
    parse_ts,
    resolve_scenario_dir,
)
from .memory_store import (
    CONFIDENCE_BANDS,
    Episode,
    EpisodicMemory,
    Finding,
    Hypothesis,
    MemoryStore,
    WorkingMemory,
)

__all__ = [
    "Account",
    "CONFIDENCE_BANDS",
    "CustomerProfile",
    "DEFAULT_DATA_ROOT",
    "DailyLogWriter",
    "Episode",
    "EpisodicMemory",
    "Event",
    "EventStore",
    "Finding",
    "Hypothesis",
    "KNOWN_SOURCE_SYSTEMS",
    "LLMClient",
    "LLMUnavailable",
    "MemoryStore",
    "ReplayConfig",
    "Scenario",
    "ScenarioLoadError",
    "SimulatedClock",
    "WorkingMemory",
    "complete_json",
    "derive",
    "derive_all",
    "find_scenarios",
    "iso",
    "load_ground_truth",
    "load_scenario",
    "mask",
    "mask_event_text",
    "parse_ts",
    "resolve_scenario_dir",
    "summarize_event",
    "unmask",
]
