"""Hybrid search and reranking against a local Qdrant index, with stub models."""

import pytest

from filings_agent import retrieval
from filings_agent.config import settings
from tests.test_e2e import build_index, load_stubs

CHUNKS = [
    (f"ICICIBANK_FY25_annual_report_p{p}_0", "ICICIBANK", "FY25", p, text)
    for p, text in enumerate(
        [
            "Retail advances grew 8.9% during the year.",
            "Business banking advances grew 33.7% to Rs 2,633.67 billion.",
            "Total deposits rose to Rs 16,103.48 billion.",
            "The bank opened 460 new branches.",
            "Rural advances grew 5.2% during the year.",
            "Corporate advances grew 4.1% during the year.",
        ],
        start=10,
    )
]


@pytest.fixture
def index(tmp_path, monkeypatch, request):
    hybrid = getattr(request, "param", True)
    build_index(tmp_path / "qdrant_local", CHUNKS, hybrid=hybrid)
    stub = load_stubs()
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(retrieval, "embedder", stub.StubEmbedder())
    monkeypatch.setattr(retrieval, "sparse_embedder", stub.StubSparseEmbedder())
    monkeypatch.setattr(retrieval, "reranker", stub.StubReranker())
    for cached in (retrieval.qdrant, retrieval.has_sparse):
        cached.cache_clear()
    yield
    retrieval.qdrant().close()
    for cached in (retrieval.qdrant, retrieval.has_sparse):
        cached.cache_clear()


def test_hybrid_search_ranks_the_exact_passage_first(index):
    assert retrieval.has_sparse()
    hits = retrieval.search("business banking advances growth", company="ICICIBANK", k=2)
    assert hits[0]["chunk_id"] == "ICICIBANK_FY25_annual_report_p11_0"
    assert len(hits) == 2
    assert set(hits[0]) == {"chunk_id", "doc_id", "page", "text"}


def test_filters_apply_to_both_searches(index):
    assert retrieval.search("business banking advances", company="TCS") == []


@pytest.mark.parametrize("index", [False], indirect=True)
def test_dense_only_index_still_works(index):
    assert not retrieval.has_sparse()
    hits = retrieval.search("business banking advances growth", k=2)
    assert hits[0]["chunk_id"] == "ICICIBANK_FY25_annual_report_p11_0"


def test_retrieval_eval_scores_hits(index):
    from evals import retrieval as retrieval_eval

    questions = [
        {"id": "a", "type": "lookup", "company": "ICICIBANK", "expected": "33.7%",
         "question": "business banking advances growth", "evidence": "2,633.67",
         "expected_doc": "ICICIBANK_FY25_annual_report"},
        {"id": "b", "type": "lookup", "company": "ICICIBANK", "expected": "x",
         "question": "new branches opened", "evidence": "999",
         "expected_doc": "ICICIBANK_FY25_annual_report"},
        {"id": "c", "type": "unanswerable", "company": "WIPRO", "expected": "UNANSWERABLE",
         "question": "Wipro revenue", "expected_doc": ""},
    ]
    rows = retrieval_eval.score(questions, k=2)
    assert [(r["id"], r["doc_hit"], r["evidence_hit"]) for r in rows] == [
        ("a", True, True),
        ("b", True, False),
    ]
    assert "key figure retrieved: 50%" in retrieval_eval.summary(rows, 2)
