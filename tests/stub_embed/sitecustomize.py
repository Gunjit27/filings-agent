"""Loaded at interpreter start when this folder is on PYTHONPATH (see tests/test_e2e.py).

Swaps the fastembed model for a stub, so the MCP server subprocess can run without
downloading a model.
"""

from stub_embedder import StubEmbedder

try:
    import filings_agent.retrieval as _retrieval
except ImportError:
    _retrieval = None

if _retrieval is not None:
    _retrieval.embedder = StubEmbedder()
