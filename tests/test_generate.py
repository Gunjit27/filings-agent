import litellm

from evals import generate


class _Msg:
    def __init__(self, content):
        self.content = content


class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


def test_draft_parses_question(monkeypatch):
    monkeypatch.setattr(
        generate.llm, "completion", lambda **kw: _Resp('{"question": "Q?", "expected": "42"}')
    )
    assert generate.draft("TCS_FY26_annual_report", "text") == {"question": "Q?", "expected": "42"}


def test_draft_skips_when_model_says_so(monkeypatch):
    monkeypatch.setattr(generate.llm, "completion", lambda **kw: _Resp('{"skip": true}'))
    assert generate.draft("TCS_FY26_annual_report", "text") is None


def test_draft_survives_groq_json_rejection(monkeypatch):
    def reject(**kw):
        raise litellm.BadRequestError("json_validate_failed", "groq/openai/gpt-oss-20b", "groq")

    monkeypatch.setattr(generate.llm, "completion", reject)
    assert generate.draft("TCS_FY26_annual_report", "text") is None
