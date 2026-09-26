"""LiteLLM wrappers that pace requests and wait out rate limits.

The Gemini free tier allows only a few requests per minute per model. LiteLLM's own
retries fire within seconds, and each retry spends quota, so they make a 429 worse.
Here calls are spaced at LLM_RPM, and a 429 waits for the delay the provider asks for
before trying again. Set LLM_RPM=0 to disable pacing on a paid key.

Every call's tokens and list-price cost are added to the current usage tracker, and
calls are traced to Langfuse when LANGFUSE_PUBLIC_KEY is set.
"""

import asyncio
import contextvars
import logging
import os
import re
import threading
import time
from contextlib import contextmanager

import litellm

from filings_agent.config import settings

log = logging.getLogger(__name__)

TRANSIENT = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
)
DEFAULT_WAIT = 60.0

_lock = threading.Lock()
_next_slot = 0.0

# Per-question usage and trace metadata; a ContextVar keeps concurrent questions apart.
_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar("usage", default=None)
_trace: contextvars.ContextVar[dict | None] = contextvars.ContextVar("trace", default=None)

if os.environ.get("LANGFUSE_PUBLIC_KEY") and "langfuse_otel" not in litellm.callbacks:
    litellm.callbacks.append("langfuse_otel")


@contextmanager
def track(trace_name: str, session_id: str | None = None):
    """Collect token and cost totals for the calls inside, grouped as one Langfuse trace."""
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "wait_s": 0.0}
    meta = {"trace_name": trace_name[:200]}
    if session_id:
        meta["session_id"] = session_id
    usage_token, trace_token = _usage.set(usage), _trace.set(meta)
    try:
        yield usage
    finally:
        _usage.reset(usage_token)
        _trace.reset(trace_token)


def _add_wait(seconds: float) -> None:
    """Count time spent pacing or backing off, so reported latency can exclude it."""
    usage = _usage.get()
    if usage is not None:
        usage["wait_s"] += seconds


def _record(resp) -> None:
    usage = _usage.get()
    if usage is None:
        return
    usage["calls"] += 1
    if resp.usage:
        usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
        usage["completion_tokens"] += resp.usage.completion_tokens or 0
    try:
        usage["cost_usd"] += litellm.completion_cost(completion_response=resp)
    except Exception as err:  # noqa: BLE001 - LiteLLM raises bare Exception for unpriced models
        log.debug("no price for %s: %s", resp.model if hasattr(resp, "model") else "?", err)


def _reserve_slot() -> float:
    """Return how long the caller must wait before its request may be sent."""
    global _next_slot
    if settings.llm_rpm <= 0:
        return 0.0
    with _lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / settings.llm_rpm
        return start - now


def retry_delay(err: Exception) -> float:
    """Seconds to wait after a transient error, using the provider's hint when it gives one."""
    match = re.search(r"retry in ([\d.]+)s", str(err)) or re.search(
        r'"retryDelay":\s*"(\d+)s"', str(err)
    )
    return float(match.group(1)) + 1 if match else DEFAULT_WAIT


def _defaults(kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    metadata = {**(_trace.get() or {}), **kwargs.pop("metadata", {})}
    return {"model": settings.llm_model, "num_retries": 0, "metadata": metadata, **kwargs}


def completion(**kwargs):
    for attempt in range(settings.llm_retries + 1):
        wait = _reserve_slot()
        _add_wait(wait)
        time.sleep(wait)
        try:
            resp = litellm.completion(**_defaults(kwargs))
            _record(resp)
            return resp
        except TRANSIENT as err:
            if attempt == settings.llm_retries:
                raise
            wait = retry_delay(err)
            _add_wait(wait)
            time.sleep(wait)


async def acompletion(**kwargs):
    for attempt in range(settings.llm_retries + 1):
        wait = _reserve_slot()
        _add_wait(wait)
        await asyncio.sleep(wait)
        try:
            resp = await litellm.acompletion(**_defaults(kwargs))
            _record(resp)
            return resp
        except TRANSIENT as err:
            if attempt == settings.llm_retries:
                raise
            wait = retry_delay(err)
            _add_wait(wait)
            await asyncio.sleep(wait)
