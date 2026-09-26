# filings-agent: design draft (2026-09-26)

Draft built on the defaults proposed in the thread. Changes once the user answers.

## What it does
Ask a question about an Indian listed company ("How did TCS's attrition change FY25 to FY26?", "What related-party transactions did Reliance disclose?") and get an answer where every claim cites a filing, page and quote.

## Scope (default)
- 10 Nifty 50 companies: Reliance, TCS, Infosys, HDFC Bank, ICICI Bank, ITC, L&T, Bharti Airtel, HUL, Tata Motors.
- FY25 + FY26 annual reports (PDF, from company IR / BSE) plus BSE corporate announcements for the last 12 months.

## Architecture
```
ingest/        download PDFs + BSE announcements -> parse (pymupdf) -> chunk by page/section
               -> embed (fastembed bge-small, local, free) -> Qdrant (payload: company, fy, doc, page)
mcp_server/    FastMCP server exposing tools:
                 search_filings(query, company?, fy?, doc_type?) -> chunks with ids
                 get_page(doc_id, page) -> full page text
                 list_documents(company) -> available filings
                 compare_metric(company, metric, fys) -> extracted values + sources
agent/         LangGraph graph: plan -> tool loop (via MCP client) -> draft answer
               -> verify_citations (each claim must map to a retrieved chunk; else retry/drop)
               -> final answer {answer, citations[]}
api/           FastAPI /ask (streams steps)
ui/            Streamlit or Gradio demo showing answer + clickable citations
evals/         ~100 questions (JSONL): factual lookup, numeric, multi-year compare,
               multi-company, unanswerable. Metrics: answer correctness (LLM judge + exact
               numeric match), citation precision/recall, refusal on unanswerable, p50/p95
               latency, cost per question.
.github/workflows/eval.yml   runs evals on every PR, posts a markdown score table as PR comment,
                             fails if accuracy drops > N points vs main baseline.
observability  Langfuse tracing on every LLM + tool call (latency, tokens, cost).
```

## LLM
Provider-agnostic via LiteLLM. Default Gemini Flash (API; free tier covers CI evals). Ollama supported for local dev only, since it cannot run in Actions or on a free host.

## Hosting
Hugging Face Spaces (Docker) + Qdrant Cloud free tier. Secrets: LLM key, QDRANT_URL/KEY, LANGFUSE keys, as GitHub + HF secrets.

## Milestones
1. Ingest 2 companies, search tool, basic agent answering with citations (CLI).
2. MCP server + LangGraph citation verifier.
3. Eval set (100 Qs) + GitHub Actions PR comment.
4. Langfuse tracing, cost/latency in eval table.
5. All 10 companies, demo UI, deploy, README with GIF and results.

## Resume bullet (target)
"Built a citation-grounded research agent over Indian company filings (LangGraph + MCP); 100-question eval suite gated in CI, X% answer accuracy, Y% citation precision, p95 Z s, ₹W/query tracked in Langfuse."

## Open questions (asked in thread)
Repo name/creation, LLM provider, company scope, hosting.
