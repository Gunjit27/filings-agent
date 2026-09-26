from evals.run import summarize


def test_summary_table_and_footer():
    rows = [
        {"type": "lookup", "correct": True, "cited_doc": True, "latency_s": 2.0, "tokens": 1000, "cost_usd": 0.001},
        {"type": "lookup", "correct": False, "cited_doc": True, "latency_s": 4.0, "tokens": 3000, "cost_usd": 0.003},
        {"type": "unanswerable", "correct": True, "cited_doc": True, "latency_s": 1.0, "tokens": 500, "cost_usd": 0.0},
    ]
    out = summarize(rows, "groq/openai/gpt-oss-20b")
    assert "| lookup | 2 | 50% | 100% |" in out
    assert "| **all** | 3 | 67% | 100% |" in out
    assert "p50 2.0s" in out and "p95 4.0s" in out
