"""Draft candidate eval questions from indexed filings, for human review.

Samples chunks per document, asks the LLM for one factual question answerable from that chunk
alone, and writes evals/candidates.jsonl. Review, fix and move good ones into questions.jsonl;
compare-type and unanswerable questions are written by hand.

Usage: python -m evals.generate --per-doc 6
"""

import argparse
import json
import random
from pathlib import Path

import litellm
from qdrant_client import models

from filings_agent import llm
from filings_agent.config import settings
from filings_agent.retrieval import list_documents, qdrant

HERE = Path(__file__).parent
MIN_CHARS = 600  # skip near-empty pages (covers, dividers)

PROMPT = """Below is one page excerpt from {doc_id}.
Write ONE question an equity analyst might ask that this excerpt answers with a specific fact
(a number, name, date or short phrase). The question must name the company and fiscal year and
make sense without seeing the excerpt.

Reply as JSON: {{"question": "...", "expected": "short exact answer with units"}}
If the excerpt has no such fact, reply {{"skip": true}}.

Excerpt:
{text}"""


def sample_chunks(doc_id: str, n: int, rng: random.Random) -> list[dict]:
    points, _ = qdrant().scroll(
        settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ),
        limit=5000,
    )
    pool = [p.payload for p in points if len(p.payload["text"]) >= MIN_CHARS]
    return rng.sample(pool, min(n, len(pool)))


def draft(doc_id: str, text: str) -> dict | None:
    """One question/answer pair for a passage, or None when the model finds no usable fact."""
    try:
        resp = llm.completion(
            messages=[{"role": "user", "content": PROMPT.format(doc_id=doc_id, text=text)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        qa = json.loads(resp.choices[0].message.content)
    except (litellm.BadRequestError, json.JSONDecodeError) as err:
        # Groq rejects replies that fail its JSON check; one bad passage shouldn't end the run.
        print(f"  skipped a passage in {doc_id}: {str(err)[:120]}")
        return None
    if qa.get("skip") or not qa.get("question") or not qa.get("expected"):
        return None
    return qa


def main(per_doc: int, seed: int) -> None:
    rng = random.Random(seed)
    out = HERE / "candidates.jsonl"
    n = 0
    with out.open("w") as f:
        for doc_id in list_documents():
            for chunk in sample_chunks(doc_id, per_doc, rng):
                qa = draft(doc_id, chunk["text"])
                if not qa:
                    continue
                n += 1
                row = {
                    "id": f"gen{n:03d}",
                    "type": "lookup",
                    "company": chunk["company"],
                    "question": qa["question"],
                    "expected": qa["expected"],
                    "expected_doc": doc_id,
                    "source_chunk": chunk["chunk_id"],
                    "source_text": chunk["text"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    print(f"wrote {n} candidates to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.per_doc, args.seed)
