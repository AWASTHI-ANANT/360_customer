"""Single-entry LLM client: complete_json(system, user, schema_name) -> dict.

Determinism: responses are cached on disk under a hash of
(model, system_prompt, user_prompt). A re-run replays the cache byte for byte
and never calls the API, which is what makes scored runs reproducible --
sampling parameters cannot do it, because temperature was removed on the
Claude 4.7+ family and returns HTTP 400 on claude-opus-5 (see config/llm.py).

Offline: with no SDK, no credentials, or C360_LLM_OFFLINE=1, complete_json
raises LLMUnavailable so the calling agent takes its deterministic fallback.

Every call is traced to logs/{scenario_id}_llm_trace.jsonl. Prompts are expected
to be masked before they get here; the client never sees a raw name.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import llm as llm_config
from tracing import traced

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No usable LLM (missing SDK, missing key, offline, or call failed)."""


class LLMInvalidJSON(ValueError):
    """The model returned something that isn't a JSON object."""


_client: Any = None
_client_error: Optional[str] = None


def _get_client():
    """Lazily build the SDK client. Returns None when unusable."""
    global _client, _client_error
    if _client is not None or _client_error is not None:
        return _client
    if llm_config.OFFLINE:
        _client_error = "C360_LLM_OFFLINE is set"
        return None
    try:
        import anthropic  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError:
        _client_error = "anthropic SDK not installed (pip install anthropic)"
        log.warning("%s -- agents will use deterministic fallbacks", _client_error)
        return None
    # An unset ANTHROPIC_API_KEY does not necessarily mean no credentials: the
    # SDK also resolves ANTHROPIC_AUTH_TOKEN and an `ant auth login` profile.
    # Construct and let the SDK decide; an auth failure surfaces on first call.
    try:
        _client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001
        _client_error = f"cannot construct client: {exc}"
        log.warning("%s -- agents will use deterministic fallbacks", _client_error)
        return None
    return _client


def available() -> bool:
    """True when a real call could be attempted."""
    return _get_client() is not None


def unavailable_reason() -> Optional[str]:
    _get_client()
    return _client_error


def cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
    h = hashlib.sha256()
    for part in (model, system_prompt, user_prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response.

    Tolerates ```json fences and leading prose, because a refusal to follow
    "return ONLY JSON" shouldn't cost us the whole event.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw[3:]
        if raw.lstrip().lower().startswith("json"):
            raw = raw.lstrip()[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise LLMInvalidJSON(f"no JSON object in response: {raw[:200]!r}") from None
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise LLMInvalidJSON(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


class LLMClient:
    """Holds cache/trace paths for one scenario."""

    def __init__(
        self,
        scenario_id: str = "scenario",
        cache_dir: str | Path | None = None,
        trace_dir: str | Path | None = None,
        model: Optional[str] = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.model = model or llm_config.MODEL
        self.cache_dir = Path(cache_dir or llm_config.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = Path(trace_dir or llm_config.TRACE_DIR) / f"{scenario_id}_llm_trace.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    # --- trace --------------------------------------------------------------

    def _trace(self, record: dict[str, Any]) -> None:
        record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()

    # --- cache --------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt cache entry %s -- ignoring", path.name)
            return None

    def _cache_write(self, key: str, payload: dict) -> None:
        tmp = self._cache_path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self._cache_path(key))

    # --- the one call -------------------------------------------------------

    @traced("llm")
    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str = "generic",
        event_id: Optional[str] = None,
    ) -> dict:
        """Return the model's JSON object. Raises LLMUnavailable / LLMInvalidJSON."""
        key = cache_key(self.model, system_prompt, user_prompt)
        cached = self._cache_read(key)
        if cached is not None:
            self.cache_hits += 1
            self._trace(
                {
                    "scenario_id": self.scenario_id,
                    "schema_name": schema_name,
                    "event_id": event_id,
                    "cache_key": key,
                    "cached": True,
                    "model": cached.get("model", self.model),
                    "response": cached.get("parsed"),
                }
            )
            return cached["parsed"]

        client = _get_client()
        if client is None:
            self._trace(
                {
                    "scenario_id": self.scenario_id,
                    "schema_name": schema_name,
                    "event_id": event_id,
                    "cache_key": key,
                    "cached": False,
                    "error": f"unavailable: {_client_error}",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            raise LLMUnavailable(str(_client_error))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": llm_config.MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            # Effort is the cost/quality dial on Opus 5; classification of one
            # short message does not need deep reasoning.
            "output_config": {"effort": llm_config.EFFORT},
        }
        temp = llm_config.temperature_for(self.model)
        if temp is not None:
            kwargs["temperature"] = temp

        started = time.monotonic()
        self.calls += 1
        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - mapped to one failure mode for the caller
            self.failures += 1
            self._trace(
                {
                    "scenario_id": self.scenario_id,
                    "schema_name": schema_name,
                    "event_id": event_id,
                    "cache_key": key,
                    "cached": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        elapsed = time.monotonic() - started

        # stop_details is populated only on a refusal; always check first.
        if getattr(response, "stop_reason", None) == "refusal":
            self.failures += 1
            detail = getattr(response, "stop_details", None)
            self._trace(
                {
                    "scenario_id": self.scenario_id,
                    "schema_name": schema_name,
                    "event_id": event_id,
                    "cache_key": key,
                    "cached": False,
                    "error": f"refusal: {getattr(detail, 'category', None)}",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            raise LLMUnavailable(f"model refused: {getattr(detail, 'category', 'unknown')}")

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        trace: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "schema_name": schema_name,
            "event_id": event_id,
            "cache_key": key,
            "cached": False,
            "model": self.model,
            "effort": llm_config.EFFORT,
            "request_id": getattr(response, "_request_id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "elapsed_sec": round(elapsed, 3),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": text,
        }
        try:
            parsed = _extract_json(text)
        except (LLMInvalidJSON, json.JSONDecodeError) as exc:
            self.failures += 1
            trace["error"] = f"invalid JSON: {exc}"
            self._trace(trace)
            raise LLMInvalidJSON(str(exc)) from exc

        trace["response"] = parsed
        self._trace(trace)
        self._cache_write(key, {"model": self.model, "parsed": parsed, "raw": text})
        return parsed


# Module-level convenience so agents can call one function, as specified.
_default: Optional[LLMClient] = None


def configure(scenario_id: str, **kwargs) -> LLMClient:
    """Point the module-level client at one scenario's cache/trace."""
    global _default
    _default = LLMClient(scenario_id, **kwargs)
    return _default


def get_client() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default


def complete_json(system_prompt: str, user_prompt: str, schema_name: str = "generic", **kw) -> dict:
    return get_client().complete_json(system_prompt, user_prompt, schema_name, **kw)
