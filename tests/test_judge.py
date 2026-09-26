import asyncio

from evals import run as eval_run
from filings_agent.config import settings

Q = {"question": "Growth?", "expected": "33.7%"}


def votes(monkeypatch, verdicts):
    calls = []

    async def fake(prompt):
        calls.append(prompt)
        return verdicts[len(calls) - 1]

    monkeypatch.setattr(eval_run, "judge_once", fake)
    monkeypatch.setattr(settings, "judge_votes", 3)
    return calls


def test_majority_wins_and_stops_early(monkeypatch):
    calls = votes(monkeypatch, [True, True, False])
    assert asyncio.run(eval_run.judge(Q, "33.7%"))
    assert len(calls) == 2  # two YES already make a majority of three


def test_split_vote_takes_the_third(monkeypatch):
    calls = votes(monkeypatch, [True, False, False])
    assert not asyncio.run(eval_run.judge(Q, "33.7%"))
    assert len(calls) == 3


def test_refusals_skip_the_judge(monkeypatch):
    calls = votes(monkeypatch, [True])
    assert not asyncio.run(eval_run.judge(Q, "The filings I have do not cover this."))
    assert asyncio.run(eval_run.judge({"expected": "UNANSWERABLE"}, "They do not cover this."))
    assert calls == []
