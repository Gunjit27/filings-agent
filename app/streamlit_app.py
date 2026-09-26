"""Demo UI: ask a question, get a cited answer. Run: streamlit run app/streamlit_app.py"""

import asyncio
import re
import time

import streamlit as st

from filings_agent import llm
from filings_agent.agent.graph import CITATION_RE, agent_session, ask
from filings_agent.config import load_companies, report_url, settings
from filings_agent.retrieval import get_chunks, list_documents

EXAMPLES = [
    "How did TCS's employee headcount change between FY25 and FY26?",
    "What was Infosys's attrition rate in FY26?",
    "What did HDFC Bank report as its gross NPA ratio for FY26?",
    "Compare the dividend per share declared by TCS and Infosys for FY26.",
    "What will Reliance's revenue be in FY30?",
]

st.set_page_config(page_title="Filings Agent", page_icon="📑", layout="wide")


@st.cache_data(ttl=3600)
def indexed_documents() -> list[str]:
    return list_documents()


async def _run(question: str) -> tuple[dict, dict]:
    with llm.track(question, session_id="demo") as usage:
        async with agent_session() as graph:
            result = await ask(graph, question)
    return result, usage


def number_citations(answer: str, citations: list[str]) -> str:
    """Replace [chunk_id] markers with [1], [2]... matching the sources list."""
    index = {c: i for i, c in enumerate(citations, 1)}
    return CITATION_RE.sub(lambda m: f"[{index[m.group(1)]}]" if m.group(1) in index else "", answer)


with st.sidebar:
    st.header("Filings Agent")
    st.caption(
        "A LangGraph agent that answers questions about Indian listed companies from their "
        "annual reports, calling search tools on an MCP server. Every claim cites a page."
    )
    docs = indexed_documents()
    names = {c["id"]: c["name"] for c in load_companies()}
    st.subheader(f"{len(docs)} filings indexed")
    for d in docs:
        company, fy, _ = d.split("_", 2)
        st.markdown(f"- {names.get(company, company)} · {fy}")
    st.caption(f"Model: `{settings.llm_model}`")

st.title("Ask the annual reports")
example = st.selectbox("Try an example", [""] + EXAMPLES, index=0)
question = st.text_input("Question", value=example, placeholder="e.g. What was TCS's attrition in FY26?")

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching filings…"):
        t0 = time.perf_counter()
        result, usage = asyncio.run(_run(question.strip()))
        elapsed = time.perf_counter() - t0 - usage["wait_s"]

    st.markdown(number_citations(result["answer"] or "", result["citations"]))
    cols = st.columns(4)
    cols[0].metric("Latency", f"{elapsed:.1f}s")
    cols[1].metric("LLM calls", usage["calls"])
    cols[2].metric("Tokens", f"{usage['prompt_tokens'] + usage['completion_tokens']:,}")
    cols[3].metric("Cost (list price)", f"${usage['cost_usd']:.4f}")

    if result["citations"]:
        st.subheader("Sources")
        for i, chunk in enumerate(get_chunks(result["citations"]), 1):
            label = f"[{i}] {names.get(chunk['company'], chunk['company'])} {chunk['fy']} annual report, p. {chunk['page']}"
            with st.expander(label):
                url = report_url(chunk["doc_id"], chunk["page"])
                if url:
                    st.markdown(f"[Open the page in the report]({url})")
                st.text(re.sub(r"\s+\n", "\n", chunk["text"])[:1500])
