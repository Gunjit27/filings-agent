"""Run the eval set and write a markdown score table.

Usage: python -m evals.run [--sample N] [--seed S]
Metrics per question:
  correct     LLM judge compares the answer with the expected one (or a refusal for UNANSWERABLE)
  cited_doc   at least one citation comes from expected_doc
  latency_s   agent time, excluding time spent waiting on the free-tier rate limit
  tokens, cost_usd   summed over the agent's LLM calls (cost at LiteLLM list prices)
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import time
from pathlib import Path

from filings_agent import llm
from filings_agent.agent.graph import agent_session, ask
from filings_agent.config import settings

HERE = Path(__file__).parent
REFUSAL = "do not cover"

JUDGE = """Question: {q}
Expected answer: {expected}
Model answer: {answer}
Does the model answer give what the question asks for, matching the expected answer?
Numbers may be rounded or in other units (₹13,417.66 billion = ₹13.42 trillion). Ignore extra
detail in either answer that the question did not ask for.
Reply with exactly YES or NO."""


async def judge(q: dict, answer: str) -> bool:
    if q["expected"] == "UNANSWERABLE":
        return REFUSAL in answer.lower()
    prompt = JUDGE.format(q=q["question"], expected=q["expected"], answer=answer)
    resp = await llm.acompletion(
        model=settings.judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        # Groq's free tier caps output tokens per minute, and a request is refused
        # outright if its expected output exceeds the cap, so keep the verdict short.
        max_tokens=512,
        reasoning_effort="low",
    )
    verdict = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.DOTALL)
    return verdict.strip().upper().startswith("YES")


def load_questions(sample: int | None, seed: int) -> list[dict]:
    lines = (HERE / "questions.jsonl").read_text().splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    if sample and sample < len(questions):
        # Fixed seed, so every PR is scored on the same questions and runs are comparable.
        questions = random.Random(seed).sample(questions, sample)
    return questions


async def main(sample: int | None, seed: int) -> None:
    questions = load_questions(sample, seed)
    run_id = time.strftime("eval-%Y%m%d-%H%M%S")
    rows = []
    async with agent_session(check_index=True) as graph:
        for i, q in enumerate(questions, 1):
            row = {"id": q["id"], "type": q["type"], "correct": False, "cited_doc": False}
            try:
                with llm.track(q["question"], session_id=run_id) as usage:
                    t0 = time.perf_counter()
                    try:
                        result = await ask(graph, q["question"])
                    finally:
                        row["latency_s"] = round(time.perf_counter() - t0 - usage["wait_s"], 2)
                        row["tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                        row["cost_usd"] = usage["cost_usd"]
                row["answer"] = result["answer"]
                row["steps"] = result["steps"]
                row["cited_doc"] = not q["expected_doc"] or any(
                    c.startswith(q["expected_doc"]) for c in result["citations"]
                )
                row["correct"] = await judge(q, result["answer"] or "")
            except Exception as err:  # noqa: BLE001 - one bad question shouldn't lose the run
                row["error"] = f"{type(err).__name__}: {err}"[:500]
            rows.append(row)
            detail = row.get("error") or " ".join((row.get("answer") or "").split())
            if not detail:
                detail = f"(empty answer after {row.get('steps')} steps)"
            print(
                f"[{i}/{len(questions)}] {q['id']} correct={row['correct']} "
                f"{row['latency_s']:.1f}s {row['tokens']} tok | {detail[:160]}"
            )
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summary = summarize(rows, settings.llm_model)
    (out / "summary.md").write_text(summary)
    print(summary)
    errors = sum("error" in r for r in rows)
    if errors * 2 > len(rows):
        raise SystemExit(f"{errors} of {len(rows)} questions errored; see the log above")


def summarize(rows: list[dict], model: str) -> str:
    def pct(key, subset):
        return f"{100 * sum(r[key] for r in subset) / len(subset):.0f}%" if subset else "-"

    lines = [
        f"### Eval results ({len(rows)} questions, `{model}`)",
        "",
        "| Type | N | Accuracy | Cited right doc |",
        "|---|---|---|---|",
    ]
    for t in sorted({r["type"] for r in rows}) + ["all"]:
        subset = rows if t == "all" else [r for r in rows if r["type"] == t]
        label = "**all**" if t == "all" else t
        lines.append(
            f"| {label} | {len(subset)} | {pct('correct', subset)} | {pct('cited_doc', subset)} |"
        )
    if rows:
        lat = sorted(r["latency_s"] for r in rows)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        tokens = statistics.mean(r["tokens"] for r in rows)
        cost = statistics.mean(r["cost_usd"] for r in rows)
        lines += [
            "",
            (
                f"Latency p50 {statistics.median(lat):.1f}s, p95 {p95:.1f}s · "
                f"{tokens:,.0f} tokens and ${cost:.4f} per question (list price)"
            ),
        ]
    errors = sum("error" in r for r in rows)
    if errors:
        lines += ["", f"⚠️ {errors} question(s) errored and are scored as wrong."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="score a fixed random sample of N questions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main(args.sample, args.seed))
