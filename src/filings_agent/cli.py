"""Ask a question from the terminal: python -m filings_agent.cli "question" """

import asyncio
import sys

from filings_agent.agent.graph import agent_session, ask


async def main(question: str) -> None:
    async with agent_session() as graph:
        result = await ask(graph, question)
    print(result["answer"])
    print("\nSources:", ", ".join(result["citations"]) or "none")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))
