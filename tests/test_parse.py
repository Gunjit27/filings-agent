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
