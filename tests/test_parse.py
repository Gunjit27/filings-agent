from filings_agent.agent.graph import CITATION_RE
from filings_agent.ingest.parse import clean, split_page


def test_short_page_is_one_chunk():
    assert split_page("hello") == ["hello"]


def test_empty_page_has_no_chunks():
    assert split_page("") == []


def test_long_page_splits_with_overlap():
    text = "\n".join(f"line {i} " + "x" * 90 for i in range(60))
    parts = split_page(text, max_chars=1000, overlap=100)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)
    assert parts[0][-50:] in text and parts[1][:20] in parts[0] + parts[1]


def test_clean_collapses_whitespace():
    assert clean("a   b\n\n\n\nc") == "a b\n\nc"


def test_citation_regex_matches_chunk_ids():
    text = "Headcount was 607,979 [TCS_FY25_annual_report_p45_0] and rose [INFY_FY24_annual_report_p3_1]."
    assert CITATION_RE.findall(text) == ["TCS_FY25_annual_report_p45_0", "INFY_FY24_annual_report_p3_1"]


def test_normalize_citations_handles_gpt_oss_style():
    from filings_agent.agent.graph import normalize_citations

    text = (
        "Expenses were ₹85,950 crore【INFY_FY25_annual_report_p324_0†L4-L9】 and"
        " rose [TCS_FY26_annual_report_p1_0, TCS_FY26_annual_report_p2_1]. See [note](x)."
    )
    assert normalize_citations(text) == (
        "Expenses were ₹85,950 crore[INFY_FY25_annual_report_p324_0] and"
        " rose [TCS_FY26_annual_report_p1_0][TCS_FY26_annual_report_p2_1]. See [note](x)."
    )


def test_fit_context_trims_oldest_tool_results_first():
    import json

    from filings_agent.agent.graph import fit_context

    def tool(cid):
        return {"role": "tool", "tool_call_id": cid, "content": json.dumps([{"chunk_id": cid, "text": "x" * 3000}])}

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, tool("A_FY25_annual_report_p1_0"), tool("A_FY25_annual_report_p2_0")]
    out = fit_context(msgs, max_chars=5000)
    assert out[2]["content"].startswith("[trimmed") and "A_FY25_annual_report_p1_0" in out[2]["content"]
    assert out[3] == msgs[3]  # the newest result is kept whole
    assert msgs[2]["content"].startswith("[{")  # the agent's state is not modified
    assert fit_context(msgs, max_chars=0) == msgs
