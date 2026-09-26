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
Does the model answer state the same facts as the expected answer (numbers may be rounded)?
Reply with exactly YES or NO."""


async def judge(q: dict, answer: str) -> bool:
    if q["expected"] == "UNANSWERABLE":
        return REFUSAL in answer.lower()
    prompt = JUDGE.format(q=q["question"], expected=q["expected"], answer=answer)
    resp = await llm.acompletion(
        model=settings.judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


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
    async with agent_session() as graph:
        for i, q in enumerate(questions, 1):
            with llm.track(q["question"], session_id=run_id) as usage:
                t0 = time.perf_counter()
                result = await ask(graph, q["question"])
                latency = time.perf_counter() - t0 - usage["wait_s"]
            correct = await judge(q, result["answer"] or "")
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "correct": correct,
                    "cited_doc": not q["expected_doc"]
                    or any(c.startswith(q["expected_doc"]) for c in result["citations"]),
                    "latency_s": round(latency, 2),
                    "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                    "cost_usd": usage["cost_usd"],
                    "answer": result["answer"],
                }
            )
            snippet = " ".join((result["answer"] or "").split())[:120]
            print(f"[{i}/{len(questions)}] {q['id']} correct={correct} {latency:.1f}s | {snippet}")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    summary = summarize(rows, settings.llm_model)
    (out / "summary.md").write_text(summary)
    print(summary)


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
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="score a fixed random sample of N questions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main(args.sample, args.seed))
