"""Score retrieval alone: does search put the right passage in front of the agent?

Usage: python -m evals.retrieval [--k 4]
No LLM calls, so it costs nothing and can run after every ingest. Per answerable question:
  doc_hit       a top-k result comes from expected_doc
  evidence_hit  a top-k result contains the question's evidence string (the key figure or
                name), ignoring spaces, commas and ₹
"""

import argparse
import json
import re
from pathlib import Path

from filings_agent import retrieval
from filings_agent.config import settings

HERE = Path(__file__).parent


def normalize(text: str) -> str:
    return re.sub(r"[\s,₹]", "", text).lower()


def score(questions: list[dict], k: int) -> list[dict]:
    rows = []
    for q in questions:
        if q["expected"] == "UNANSWERABLE" or not q.get("evidence"):
            continue
        hits = retrieval.search(q["question"], company=q.get("company"), k=k)
        texts = normalize(" ".join(h["text"] for h in hits))
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "doc_hit": any(h["doc_id"] == q["expected_doc"] for h in hits),
                "evidence_hit": normalize(q["evidence"]) in texts,
                "top": hits[0]["chunk_id"] if hits else "",
            }
        )
    return rows


def summary(rows: list[dict], k: int) -> str:
    rerank = settings.rerank_model or "off"
    mode = "hybrid" if retrieval.has_sparse() else "dense only"
    n = len(rows) or 1
    lines = [
        f"### Retrieval: `{settings.qdrant_collection}` ({mode}, rerank: {rerank}), top {k}",
        "",
        (
            f"Right filing: {sum(r['doc_hit'] for r in rows) / n:.0%} · "
            f"key figure retrieved: {sum(r['evidence_hit'] for r in rows) / n:.0%} "
            f"({len(rows)} questions)"
        ),
        "",
        "| id | type | filing | figure | top hit |",
        "|---|---|---|---|---|",
    ]
    mark = {True: "✅", False: "❌"}
    lines += [
        f"| {r['id']} | {r['type']} | {mark[r['doc_hit']]} | {mark[r['evidence_hit']]} "
        f"| {r['top']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    lines = (HERE / "questions.jsonl").read_text().splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    print(summary(score(questions, args.k), args.k))


if __name__ == "__main__":
    main()
