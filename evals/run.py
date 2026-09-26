"""Run the eval set and write a markdown score table.

Usage: python -m evals.run [--limit N]
Metrics per question:
  correct        LLM judge compares answer with expected (or refusal for UNANSWERABLE)
  cited_doc      at least one citation comes from expected_doc
  citation_valid every citation was actually retrieved (enforced by the agent's verifier)
  latency_s
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import litellm

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
    resp = await litellm.acompletion(
        model=settings.llm_model,
        messages=[{"role": "user", "content": JUDGE.format(q=q["question"], expected=q["expected"], answer=answer)}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip().upper().startswith("YES")


async def main(limit: int | None) -> None:
    lines = (HERE / "questions.jsonl").read_text().splitlines()
    questions = [json.loads(line) for line in lines if line.strip()][:limit]
    rows = []
    async with agent_session() as graph:
        for q in questions:
            t0 = time.perf_counter()
            result = await ask(graph, q["question"])
            latency = time.perf_counter() - t0
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "correct": await judge(q, result["answer"]),
                    "cited_doc": not q["expected_doc"]
                    or any(c.startswith(q["expected_doc"]) for c in result["citations"]),
                    "latency_s": round(latency, 2),
                }
            )
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.md").write_text(summarize(rows))
    print(summarize(rows))


def summarize(rows: list[dict]) -> str:
    def pct(key, subset):
        return f"{100 * sum(r[key] for r in subset) / len(subset):.0f}%" if subset else "-"

    lines = ["| Type | N | Accuracy | Cited right doc |", "|---|---|---|---|"]
    for t in sorted({r["type"] for r in rows}) + ["all"]:
        subset = rows if t == "all" else [r for r in rows if r["type"] == t]
        lines.append(f"| {t} | {len(subset)} | {pct('correct', subset)} | {pct('cited_doc', subset)} |")
    lat = [r["latency_s"] for r in rows]
    if lat:
        p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
        lines.append(f"\nLatency p50 {statistics.median(lat):.1f}s, p95 {p95:.1f}s")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    asyncio.run(main(ap.parse_args().limit))
