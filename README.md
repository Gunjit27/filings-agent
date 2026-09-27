# filings-agent

A research agent that answers questions about Indian listed companies from their annual reports,
with a page-level citation on every claim. Built with LangGraph, an MCP tool server, Qdrant and
Groq-hosted open models; evaluated in CI on every pull request and traced in Langfuse.

**Indexed:** 7 annual reports from TCS, Infosys, Reliance, HDFC Bank
and ICICI Bank (FY25 and FY26), 5,834 passages

<!-- demo GIF goes here -->

## Eval results

<!-- eval-results:start -->
20 hand-verified questions, `groq/openai/gpt-oss-20b` as the agent. Numbers are the mean of two
runs on the same code, because single runs vary by a few questions.

| Metric | Result |
|---|---|
| Answer accuracy (LLM judge vs. verified answer) | 73% (runs: 70%, 75%) |
| Answers citing the correct filing | 83% |
| Correct refusals on unanswerable questions | 100% (2 of 2) |
| Latency (p50) | 4.4 s |
| Cost per question | $0.0004 |
<!-- eval-results:end -->

The questions are in `evals/questions.jsonl`: 15 single-fact lookups, 3 year-over-year
comparisons and 2 questions the filings can't answer, where the right response is a refusal.
Each expected answer was checked against the page it came from. gpt-oss-120b grades each answer
against the expected one; it samples up to three verdicts and takes the majority, since a single
verdict sometimes flipped between runs. "Citing the correct filing" checks that a citation points
at the filing that holds the answer.

### Retrieval

`evals/retrieval.py` checks search on its own, with no LLM calls: for each question, is the
passage with the answer in the top 4 results?

| Search | Right filing | Passage with the answer |
|---|---|---|
| Dense only (bge-small) | 100% | 50% |
| Dense + BM25 + MiniLM reranker | 100% | 72% |

Search almost always found the right report but often not the right page. Hybrid search and
reranking, together with a few agent fixes made at the same time (TCS digit parsing, a fallback
for years with no report, a retry on empty answers), took answer accuracy from 65% to 73%. The
reranker adds about 2 s per question on a CPU runner.

## How it works

1. **Ingest**: annual report PDFs are parsed page by page (PyMuPDF), embedded locally
   (fastembed: bge-small dense vectors plus BM25 sparse vectors) and stored in Qdrant with
   company, fiscal year and page metadata.
2. **Hybrid search**: dense and BM25 results are merged with reciprocal rank fusion, and a
   cross-encoder (MiniLM) reranks the top 20 down to 4. BM25 helps with exact labels like
   "business banking advances" that dense search misses.
3. **MCP server**: `search_filings`, `get_page` and `list_documents` are exposed as MCP tools,
   so any MCP client (this agent, Claude Desktop, Cursor) can use them.
4. **Agent**: a LangGraph loop calls the tools, drafts an answer, then a verifier checks that every
   cited chunk was actually retrieved. Unsupported citations trigger one retry, then get stripped.
5. **Evals**: the question set runs on every PR and posts accuracy, citation hit rate, p50/p95
   latency, tokens and cost per question as a PR comment. A retrieval-only check
   (`evals/retrieval.py`, no LLM calls) scores search after every ingest.
6. **Observability**: every LLM call is traced to Langfuse, grouped per question, with tokens,
   latency and cost.
7. **Demo**: a Streamlit app (`app/streamlit_app.py`) shows the answer with numbered citations,
   each linking to the page in the source PDF. Hosting is pending.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env            # add GROQ_API_KEY
python -m filings_agent.ingest.download
python -m filings_agent.ingest.index
python -m filings_agent.cli "How did Infosys's revenue change from FY25 to FY26?"
uv sync --extra app && streamlit run app/streamlit_app.py   # demo UI
```

Companies in scope are listed in `config/companies.yaml`; edit it and re-run ingest to change them.

## GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| CI | every push and PR | ruff, pytest; on PRs also scores a fixed 20-question sample and comments the table |
| Full eval | manual | scores every question and uploads the results |
| Ingest filings | manual, or a change to `config/companies.yaml` | downloads new reports, indexes them into Qdrant Cloud, then scores retrieval |
| Draft eval questions | manual | drafts candidate questions from indexed pages for human review |

Required repository secrets: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.
`GEMINI_API_KEY` is optional, for running with `LLM_MODEL=gemini/...`.
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are optional; when set, every LLM call is traced to Langfuse
(`LANGFUSE_HOST` defaults to the EU cloud).

## Known limitations

- Some PDFs (notably TCS's) embed fonts that PyMuPDF extracts as look-alike glyphs. Digits
  (`ϰ` for `4`) and the "ffi" ligature are mapped back during parsing; some words still come
  out with stray spaces.
- Tables are indexed as flattened text, so multi-column tables lose their row and column structure.
