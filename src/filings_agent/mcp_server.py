"""MCP server exposing filing search tools. Run: python -m filings_agent.mcp_server"""

from mcp.server.mcpserver import MCPServer

from filings_agent import retrieval

mcp = MCPServer("filings")


@mcp.tool()
def search_filings(
    query: str, company: str | None = None, fy: str | None = None, doc_type: str | None = None
) -> list[dict]:
    """Search Indian company filings (keyword + semantic).

    company: company id such as TCS, INFY, RELIANCE. fy: fiscal year of the report, FY25 or
    FY26. A report also gives the previous year's figures, so FY24 numbers are in FY25 reports.
    Returns chunks with chunk_id, doc_id, page and text. Cite chunk_id in answers.
    """
    hits = retrieval.search(query, company=company, fy=fy, doc_type=doc_type)
    if not hits and fy:
        # No report for that year is indexed (say FY24): search every year instead, since the
        # next year's report carries it as the comparison column. chunk_ids name the year.
        hits = retrieval.search(query, company=company, doc_type=doc_type)
    return hits


@mcp.tool()
def get_page(doc_id: str, page: int) -> str:
    """Return the full text of one page of a filing, for reading around a search hit."""
    return retrieval.get_page(doc_id, page)


@mcp.tool()
def list_documents(company: str | None = None) -> list[str]:
    """List indexed filings (doc_ids), optionally for one company."""
    return retrieval.list_documents(company)


if __name__ == "__main__":
    mcp.run()
