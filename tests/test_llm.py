import time

from filings_agent import llm
from filings_agent.config import settings


def test_reserve_slot_spaces_requests(monkeypatch):
    monkeypatch.setattr(settings, "llm_rpm", 60)
    monkeypatch.setattr(llm, "_next_slot", 0.0)
    waits = [llm._reserve_slot() for _ in range(3)]
    assert waits[0] == 0
    assert 0.9 < waits[1] <= 1.0
    assert 1.9 < waits[2] <= 2.0


def test_zero_rpm_disables_pacing(monkeypatch):
    monkeypatch.setattr(settings, "llm_rpm", 0)
    monkeypatch.setattr(llm, "_next_slot", time.monotonic() + 100)
    assert llm._reserve_slot() == 0


def test_retry_delay_uses_provider_hint():
    err = Exception('Quota exceeded. Please retry in 26.115287313s. "retryDelay": "26s"')
    assert 27 < llm.retry_delay(err) < 28


def test_retry_delay_defaults_without_hint():
    assert llm.retry_delay(Exception("connection reset")) == llm.DEFAULT_WAIT


def test_completion_waits_out_rate_limit(monkeypatch):
    import litellm

    calls, sleeps = [], []
    monkeypatch.setattr(settings, "llm_rpm", 0)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise litellm.RateLimitError("Please retry in 5s", "gemini", "gemini-3.8-flash")
        return "ok"

    monkeypatch.setattr(litellm, "completion", fake_completion)
    assert llm.completion(messages=[]) == "ok"
    assert len(calls) == 2 and calls[0]["num_retries"] == 0
    assert 6.0 in sleeps


def test_track_sums_usage_and_tags_trace(monkeypatch):
    import litellm

    seen = []
    monkeypatch.setattr(settings, "llm_rpm", 0)

    class Usage:
        prompt_tokens, completion_tokens = 100, 20

    class Resp:
        usage = Usage()

    def fake_completion(**kwargs):
        seen.append(kwargs["metadata"])
        return Resp()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "cost_per_token", lambda model, prompt_tokens, completion_tokens: (0.0008, 0.0002))
    with llm.track("What was TCS headcount?", session_id="run-1") as usage:
        llm.completion(messages=[])
        llm.completion(messages=[], metadata={"generation_name": "judge"})
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 200 and usage["completion_tokens"] == 40
    assert abs(usage["cost_usd"] - 0.002) < 1e-9
    assert seen[0] == {"trace_name": "What was TCS headcount?", "session_id": "run-1"}
    assert seen[1]["generation_name"] == "judge"
    llm.completion(messages=[])  # outside track(): no trace metadata, nothing recorded
    assert seen[-1] == {}


def test_cost_uses_requested_model_name(monkeypatch):
    import litellm

    class Usage:
        prompt_tokens, completion_tokens = 1000, 100

    class Resp:
        usage = Usage()
        model = "openai/gpt-oss-20b"  # what Groq echoes back, without the provider prefix

    monkeypatch.setattr(settings, "llm_rpm", 0)
    monkeypatch.setattr(litellm, "completion", lambda **kw: Resp())
    with llm.track("q") as usage:
        llm.completion(model="groq/openai/gpt-oss-20b", messages=[])
    assert usage["cost_usd"] > 0


def test_request_too_large_is_not_retried():
    assert not llm.retryable(Exception("Request too large for model on output tokens per minute"))
    assert llm.retryable(Exception("Rate limit reached, please try again in 2s"))
