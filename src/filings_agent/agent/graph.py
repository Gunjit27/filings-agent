"""LangGraph agent: tool loop over the MCP server, then citation verification."""

import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from filings_agent import llm
from filings_agent.agent.prompts import ANSWER_NOW, CITATION_RETRY, SYSTEM
from filings_agent.config import settings

log = logging.getLogger(__name__)

CHUNK_ID = r"[A-Z0-9]+_FY\d{2}_[a-z_]+_p\d+_\d+"
CITATION_RE = re.compile(rf"\[({CHUNK_ID})\]")


def normalize_citations(text: str) -> str:
    """Rewrite citations to [chunk_id], one id per bracket.

    gpt-oss often cites in its own style, 【chunk_id】 or 【chunk_id†L1-L4】, or puts
    several ids in one bracket; the verifier and the UI expect [chunk_id].
    """

    def fix(m: re.Match) -> str:
        ids = re.findall(CHUNK_ID, m.group(1))
        return "".join(f"[{i}]" for i in ids) if ids else m.group(0)

    return re.sub(r"[\[【]([^\]】]*)[\]】]", fix, text)


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
            "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
        }
        for t in tools
    ]


def fit_context(messages: list[dict], max_chars: int) -> list[dict]:
    """Shorten the oldest tool results until the conversation fits in max_chars.

    A trimmed result keeps its chunk_ids, so the model can still cite what it read.
    """
    if max_chars <= 0:
        return messages
    messages = list(messages)

    def size() -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)

    for i, m in enumerate(messages):
        if size() <= max_chars:
            break
        if m.get("role") == "tool" and not m["content"].startswith("[trimmed"):
            ids = re.findall(r'"chunk_id":\s*"([^"]+)"', m["content"])
            stub = f"[trimmed to fit the context; chunk_ids read: {', '.join(ids) or 'none'}]"
            messages[i] = {**m, "content": stub}
    return messages


def build_graph(session: ClientSession, tools: list[dict]):
    async def agent(state: State) -> dict:
        # On the last allowed step, or once over the token budget, answer with what it has.
        # This is asked in words: gpt-oss sometimes calls a tool even under
        # tool_choice="none", and Groq rejects the whole request when it does.
        budget = settings.question_token_budget
        last_step = state["steps"] + 1 >= settings.max_agent_steps or (
            budget > 0 and llm.used_tokens() >= budget
        )
        messages = fit_context(state["messages"], settings.max_context_chars)
        if last_step:
            messages = [*messages, {"role": "user", "content": ANSWER_NOW}]
        resp = await llm.acompletion(
            messages=messages,
            tools=tools,
            temperature=0,
            max_tokens=settings.max_answer_tokens,
        )
        msg = resp.choices[0].message.model_dump(exclude_none=True)
        if last_step:
            msg.pop("tool_calls", None)  # ignore a tool call made anyway; answer with the text
            msg["content"] = msg.get("content") or ""
        return {"messages": [msg], "steps": state["steps"] + 1}

    async def call_tools(state: State) -> dict:
        out, retrieved = [], []
        for call in state["messages"][-1]["tool_calls"]:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = await session.call_tool(call["function"]["name"], args)
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            if result.is_error:
                log.warning("tool %s failed: %s", call["function"]["name"], text[:300])
            retrieved += re.findall(r'"chunk_id":\s*"([^"]+)"', text)
            out.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        return {"messages": out, "retrieved": retrieved}

    def verify(state: State) -> dict:
        answer = normalize_citations(state["messages"][-1].get("content") or "")
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
async def agent_session(check_index: bool = False):
    """Start the MCP server and yield the agent graph.

    check_index fails fast when the server can't list any filings (index unreachable or
    empty), instead of letting every question quietly come back unanswered.
    """
    # The MCP SDK hands stdio servers only a minimal env (HOME, PATH) by default,
    # so pass ours through or the server can't see QDRANT_URL and its API key.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "filings_agent.mcp_server"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        if check_index:
            result = await session.call_tool("list_documents", {})
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            if result.is_error or not text.strip("[] \n"):
                raise RuntimeError(f"MCP server can't list any filings: {text[:300]}")
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
