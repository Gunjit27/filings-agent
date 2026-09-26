# filings-agent

A research agent that answers questions about Indian listed companies from their annual reports
and BSE disclosures, with a page-level citation on every claim.

> Status: work in progress. Demo link, GIF and eval results land here as milestones complete.

## How it works

1. **Ingest**: annual report PDFs are parsed page by page (PyMuPDF), embedded locally
   (fastembed, bge-small) and stored in Qdrant with company, fiscal year and page metadata.
2. **MCP server**: `search_filings`, `get_page` and `list_documents` are exposed as MCP tools,
   so any MCP client (this agent, Claude Desktop, Cursor) can use them.
3. **Agent**: a LangGraph loop calls the tools, drafts an answer, then a verifier checks that every
   cited chunk was actually retrieved. Unsupported citations trigger one retry, then get stripped.
4. **Evals**: a question set (lookups, year-over-year, cross-company, unanswerable) runs on every PR
   and posts accuracy, citation and latency numbers as a PR comment.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env            # add GROQ_API_KEY
python -m filings_agent.ingest.download
python -m filings_agent.ingest.index
python -m filings_agent.cli "How did TCS's attrition change from FY25 to FY26?"
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
