"""LangGraph agent: tool loop over the MCP server, then citation verification."""

import json
import re
import sys
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

import litellm
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from filings_agent.agent.prompts import CITATION_RETRY, SYSTEM
from filings_agent.config import settings

CITATION_RE = re.compile(r"\[([A-Z0-9]+_FY\d{2}_[a-z_]+_p\d+_\d+)\]")


def _extend(left: list, right: list) -> list:
    return left + right


class State(TypedDict):
    messages: Annotated[list[dict], _extend]
    retrieved: Annotated[list[str], _extend]
    steps: int
    retries: int
    answer: str
    citations: list[str]


def mcp_tools_to_openai(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema},
        }
        for t in tools
    ]


def build_graph(session: ClientSession, tools: list[dict]):
    async def agent(state: State) -> dict:
        resp = await litellm.acompletion(
            model=settings.llm_model, messages=state["messages"], tools=tools, temperature=0
        )
        msg = resp.choices[0].message.model_dump(exclude_none=True)
        return {"messages": [msg], "steps": state["steps"] + 1}

    async def call_tools(state: State) -> dict:
        out, retrieved = [], []
        for call in state["messages"][-1]["tool_calls"]:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = await session.call_tool(call["function"]["name"], args)
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            retrieved += re.findall(r'"chunk_id":\s*"([^"]+)"', text)
            out.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        return {"messages": out, "retrieved": retrieved}

    def verify(state: State) -> dict:
        answer = state["messages"][-1].get("content") or ""
        cited = list(dict.fromkeys(CITATION_RE.findall(answer)))
        bad = [c for c in cited if c not in set(state["retrieved"])]
        if bad and state["retries"] < 1:
            return {
                "messages": [{"role": "user", "content": CITATION_RETRY.format(bad=bad)}],
                "retries": state["retries"] + 1,
            }
        for c in bad:  # still unsupported after a retry: strip them rather than show fake sources
            answer = answer.replace(f"[{c}]", "")
        return {"answer": answer, "citations": [c for c in cited if c not in bad]}

    def after_agent(state: State) -> str:
        last = state["messages"][-1]
        if last.get("tool_calls") and state["steps"] < settings.max_agent_steps:
            return "tools"
        return "verify"

    def after_verify(state: State) -> str:
        return END if state.get("answer") is not None and state["messages"][-1]["role"] != "user" else "agent"

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", call_tools)
    g.add_node("verify", verify)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", after_verify, {END: END, "agent": "agent"})
    return g.compile()


@asynccontextmanager
async def agent_session():
    params = StdioServerParameters(command=sys.executable, args=["-m", "filings_agent.mcp_server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = mcp_tools_to_openai((await session.list_tools()).tools)
        yield build_graph(session, tools)


async def ask(graph, question: str) -> dict:
    state = await graph.ainvoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            "retrieved": [],
            "steps": 0,
            "retries": 0,
            "answer": None,
            "citations": [],
        }
    )
    return {"answer": state["answer"], "citations": state["citations"], "steps": state["steps"]}
