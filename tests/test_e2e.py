"""End to end: eval runner -> LangGraph agent -> MCP server subprocess -> Qdrant.

Only the LLM is faked. The server runs as a real subprocess against a local Qdrant
index, the way CI runs it against Qdrant Cloud, so this catches wiring bugs such as
the server not inheriting the environment.
"""

import asyncio
import importlib.util
import json
import os
import re
from pathlib import Path

import litellm
from qdrant_client import QdrantClient, models

from evals import run as eval_run
from filings_agent.config import settings

STUB = Path(__file__).parent / "stub_embed"
CHUNKS = [
    ("TCS_FY26_annual_report_p45_0", "TCS", "FY26", 45, "Total headcount was 607,979 employees."),
    ("TCS_FY26_annual_report_p90_0", "TCS", "FY26", 90, "Revenue grew to Rs 2,55,324 crore."),
    ("INFY_FY26_annual_report_p12_0", "INFY", "FY26", 12, "Infosys paid dividends of Rs 85 per share."),
]
QUESTIONS = [
    {
        "id": "t1",
        "type": "lookup",
        "question": "What was TCS headcount in FY26?",
        "expected": "607,979",
        "expected_doc": "TCS_FY26_annual_report",
    },
    {
        "id": "t2",
        "type": "unanswerable",
        "question": "What was Wipro revenue in FY26?",
        "expected": "UNANSWERABLE",
        "expected_doc": "",
    },
]


def load_stubs():
    spec = importlib.util.spec_from_file_location("stub_embedder", STUB / "stub_embedder.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    return stub


def build_index(path: Path, chunks=CHUNKS, hybrid: bool = True) -> None:
    stub = load_stubs()
    emb, sparse = stub.StubEmbedder(), stub.StubSparseEmbedder()
    client = QdrantClient(path=str(path))
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE),
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}
        if hybrid
        else None,
    )

    def vector(text):
        dense = next(emb.embed([text])).tolist()
        if not hybrid:
            return dense
        s = next(sparse.embed([text]))
        return {"": dense, "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())}

    client.upsert(
        settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=i,
                vector=vector(text),
                payload={
                    "chunk_id": cid,
                    "doc_id": cid.rsplit("_p", 1)[0],
                    "company": company,
                    "fy": fy,
                    "doc_type": "annual_report",
                    "page": page,
                    "text": text,
                },
            )
            for i, (cid, company, fy, page, text) in enumerate(chunks)
        ],
    )
    client.close()


def response(content=None, tool_calls=None):
    return litellm.ModelResponse(
        choices=[litellm.Choices(message=litellm.Message(content=content, tool_calls=tool_calls))],
        usage=litellm.Usage(prompt_tokens=100, completion_tokens=20),
    )


async def fake_acompletion(model, messages, **kwargs):
    if model == settings.judge_model:
        return response("YES" if "607,979" in messages[-1]["content"] else "NO")
    last = messages[-1]
    if last["role"] == "user":
        company = "TCS" if "TCS" in last["content"] else "WIPRO"
        call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_filings",
                "arguments": json.dumps({"query": last["content"], "company": company}),
            },
        }
        return response(tool_calls=[call])
    ids = re.findall(r'"chunk_id":\s*"([^"]+)"', last["content"])
    if not ids:
        return response("The filings I have do not cover this.")
    return response(f"TCS had 607,979 employees【{ids[0]}†L1-L2】.")  # gpt-oss style


def test_eval_runs_end_to_end(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_index(tmp_path / "qdrant_local")
    # The server subprocess reads these from its environment; this process reads settings.
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(STUB), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(settings, "llm_rpm", 0)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(eval_run, "HERE", tmp_path)
    monkeypatch.setattr(eval_run, "load_questions", lambda sample, seed: QUESTIONS)

    asyncio.run(eval_run.main(sample=None, seed=0))

    rows = {r["id"]: r for r in json.loads((tmp_path / "results" / "results.json").read_text())}
    assert rows["t1"]["correct"] and rows["t1"]["cited_doc"]
    assert "[TCS_FY26_annual_report_p45_0]" in rows["t1"]["answer"]
    assert rows["t2"]["correct"]
    assert "| **all** | 2 | 100% |" in (tmp_path / "results" / "summary.md").read_text()


def test_last_step_answers_even_if_model_calls_a_tool(tmp_path, monkeypatch):
    """gpt-oss may call a tool when told to answer; the agent must still finish with text."""
    from filings_agent.agent.graph import agent_session, ask
    from filings_agent.agent.prompts import ANSWER_NOW

    build_index(tmp_path / "qdrant_local")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(STUB), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(settings, "llm_rpm", 0)
    monkeypatch.setattr(settings, "max_agent_steps", 2)
    seen = []

    async def stubborn(model, messages, **kwargs):
        seen.append(kwargs.get("tool_choice"))
        call = {
            "id": f"call_{len(seen)}",
            "type": "function",
            "function": {"name": "search_filings", "arguments": '{"query": "TCS headcount"}'},
        }
        if messages[-1]["content"] == ANSWER_NOW:
            return response("TCS had 607,979 employees [TCS_FY26_annual_report_p45_0].", [call])
        return response(tool_calls=[call])

    monkeypatch.setattr(litellm, "acompletion", stubborn)

    async def run():
        async with agent_session() as graph:
            return await ask(graph, "What was TCS headcount in FY26?")

    result = asyncio.run(run())
    assert result["citations"] == ["TCS_FY26_annual_report_p45_0"]
    assert "none" not in seen  # never tool_choice="none", which Groq rejects for gpt-oss


def test_empty_answer_is_asked_again(tmp_path, monkeypatch):
    """An answer with no text (all reasoning, no content) gets one more try."""
    from filings_agent.agent.graph import agent_session, ask
    from filings_agent.agent.prompts import ANSWER_NOW

    build_index(tmp_path / "qdrant_local")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(STUB), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setattr(settings, "llm_rpm", 0)
    efforts = []

    async def silent_once(model, messages, **kwargs):
        efforts.append(kwargs.get("reasoning_effort"))
        if messages[-1]["content"] == ANSWER_NOW:
            return response("The filings I have do not cover this.")
        return response(content=None)

    monkeypatch.setattr(litellm, "acompletion", silent_once)

    async def run():
        async with agent_session() as graph:
            return await ask(graph, "What was TCS headcount in FY26?")

    result = asyncio.run(run())
    assert result["answer"] == "The filings I have do not cover this."
    assert efforts == [None, "low"]
