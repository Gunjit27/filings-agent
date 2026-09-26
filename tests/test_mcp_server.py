import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from filings_agent.agent.graph import mcp_tools_to_openai


async def _list_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "filings_agent.mcp_server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return (await session.list_tools()).tools


def test_server_tools_convert_to_openai_format():
    tools = mcp_tools_to_openai(asyncio.run(_list_tools()))
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    assert set(by_name) == {"search_filings", "get_page", "list_documents"}
    search = by_name["search_filings"]
    assert search["parameters"]["required"] == ["query"]
    assert "chunk_id" in search["description"]


def test_agent_session_passes_env_to_server(monkeypatch):
    """The stdio server must inherit QDRANT_URL etc.; the SDK default env drops them."""
    import asyncio
    from contextlib import asynccontextmanager

    from filings_agent.agent import graph

    seen = {}

    @asynccontextmanager
    async def fake_stdio(params):
        seen["env"] = params.env
        raise RuntimeError("stop")
        yield

    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant")
    monkeypatch.setattr(graph, "stdio_client", fake_stdio)

    async def run():
        async with graph.agent_session():
            pass

    try:
        asyncio.run(run())
    except RuntimeError:
        pass
    assert seen["env"]["QDRANT_URL"] == "https://example.qdrant"


def test_search_falls_back_when_no_report_for_that_year(monkeypatch):
    """FY24 figures live in FY25 reports, so a search for an unindexed year still finds them."""
    from filings_agent import mcp_server, retrieval

    calls = []

    def fake_search(query, company=None, fy=None, doc_type=None):
        calls.append(fy)
        return [] if fy == "FY24" else [{"chunk_id": "ICICIBANK_FY25_annual_report_p9_0"}]

    monkeypatch.setattr(retrieval, "search", fake_search)
    hits = mcp_server.search_filings("total deposits", company="ICICIBANK", fy="FY24")
    assert hits == [{"chunk_id": "ICICIBANK_FY25_annual_report_p9_0"}]
    assert calls == ["FY24", None]
    calls.clear()
    mcp_server.search_filings("total deposits", company="ICICIBANK", fy="FY25")
    assert calls == ["FY25"]
