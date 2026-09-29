"""LLM provider configuration.

Determinism note: sampling parameters (temperature/top_p/top_k) were REMOVED on
the Claude 4.7+ family -- sending temperature to claude-opus-5 returns HTTP 400.
Run-to-run reproducibility therefore comes from the on-disk response cache in
c360/llm_client.py, which keys on (model, system_prompt, user_prompt). A scored
re-run replays the cache and never calls the API. TEMPERATURE below is sent only
for models that still accept it.
"""

import os

PROVIDER = "anthropic"

# Per the Claude API reference: use the exact ID, never a date-suffixed variant.
MODEL = os.environ.get("C360_MODEL", "claude-opus-5")

# Models that still accept sampling params. Opus 5 / Sonnet 5 / Opus 4.7+ do not.
TEMPERATURE_CAPABLE_MODELS = {"claude-haiku-4-5"}
TEMPERATURE = 0

MAX_TOKENS = 2000

# Thinking is on by default on Opus 5; effort is the cost/quality dial.
# Classification of one short message does not need deep reasoning.
EFFORT = os.environ.get("C360_EFFORT", "low")

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

# Where cached responses and prompt traces go.
CACHE_DIR = "out/llm_cache"
TRACE_DIR = "out/logs"

# Hard offline switch: set C360_LLM_OFFLINE=1 to force every agent onto its
# deterministic fallback path (used by the test suite and by scored runs).
OFFLINE = os.environ.get("C360_LLM_OFFLINE", "") not in ("", "0", "false", "False")

# Retries for a malformed/refused response before the agent falls back.
MAX_RETRIES = 1


def temperature_for(model: str):
    """None when the model rejects sampling params."""
    return TEMPERATURE if model in TEMPERATURE_CAPABLE_MODELS else None
