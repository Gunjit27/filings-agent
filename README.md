# filings-agent

A research agent that answers questions about Indian listed companies from their annual reports,
with a page-level citation on every claim. Built with LangGraph, an MCP tool server, Qdrant and
Groq-hosted open models; evaluated in CI on every pull request and traced in Langfuse.

**Live demo:** _coming soon_ · **Indexed:** 7 annual reports from TCS, Infosys, Reliance, HDFC Bank
and ICICI Bank (FY25 and FY26), 5,834 passages

<!-- demo GIF goes here -->

## Eval results

<!-- eval-results:start -->
_Pending the first full run._
<!-- eval-results:end -->

20 hand-verified questions (`evals/questions.jsonl`): 15 single-fact lookups, 3 year-over-year
comparisons and 2 questions the filings can't answer, where the right response is a refusal.
Each expected answer was checked against the page it came from. An LLM judge from a different
model family (Qwen) grades answers; `Cited right doc` checks that a citation points at the filing
that holds the answer.

## How it works

1. **Ingest**: annual report PDFs are parsed page by page (PyMuPDF), embedded locally
   (fastembed, bge-small) and stored in Qdrant with company, fiscal year and page metadata.
2. **MCP server**: `search_filings`, `get_page` and `list_documents` are exposed as MCP tools,
   so any MCP client (this agent, Claude Desktop, Cursor) can use them.
3. **Agent**: a LangGraph loop calls the tools, drafts an answer, then a verifier checks that every
   cited chunk was actually retrieved. Unsupported citations trigger one retry, then get stripped.
4. **Evals**: the question set runs on every PR and posts accuracy, citation hit rate, p50/p95
   latency, tokens and cost per question as a PR comment.
5. **Observability**: every LLM call is traced to Langfuse, grouped per question, with tokens,
   latency and cost.
6. **Demo**: a Streamlit app (`app/streamlit_app.py`) shows the answer with numbered citations,
   each linking to the page in the source PDF. It deploys to a Hugging Face Space.

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
| Ingest filings | manual, or a change to `config/companies.yaml` | downloads new reports and indexes them into Qdrant Cloud |
| Draft eval questions | manual | drafts candidate questions from indexed pages for human review |

Required repository secrets: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`.
`GEMINI_API_KEY` is optional, for running with `LLM_MODEL=gemini/...`.
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are optional; when set, every LLM call is traced to Langfuse
(`LANGFUSE_HOST` defaults to the EU cloud).

## Known limitations

- Some PDFs (notably TCS's) embed fonts that PyMuPDF extracts as look-alike glyphs, for example
  `ϰ` for `4`, which hurts retrieval of numbers from those pages.
- Tables are indexed as flattened text, so multi-column tables lose their row and column structure.
- Five of the ten configured companies have no report URLs yet (`config/companies.yaml`).
