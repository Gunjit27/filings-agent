"""Hybrid search over indexed filings: dense + BM25, fused, then reranked.

Dense vectors find passages that mean the same thing; BM25 finds exact terms such as
"business banking" or a figure's label, which dense search often misses. The two lists
are merged with reciprocal rank fusion, and a cross-encoder picks the best few.
"""

import uuid
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from filings_agent.config import settings


@lru_cache
def embedder() -> TextEmbedding:
    return TextEmbedding(settings.embed_model)


@lru_cache
def sparse_embedder() -> SparseTextEmbedding:
    return SparseTextEmbedding(settings.sparse_model)


@lru_cache
def reranker() -> TextCrossEncoder | None:
    return TextCrossEncoder(settings.rerank_model) if settings.rerank_model else None


SPARSE = "bm25"


def sparse_vector(embedding) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


@lru_cache
def has_sparse() -> bool:
    """True when the collection has BM25 vectors; older, dense-only indexes still work."""
    params = qdrant().get_collection(settings.qdrant_collection).config.params
    return SPARSE in (params.sparse_vectors or {})


@lru_cache
def qdrant() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return QdrantClient(path=str(settings.data_dir.parent / "qdrant_local"))


def _filter(**fields: str | None) -> models.Filter | None:
    must = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in fields.items()
        if v
    ]
    return models.Filter(must=must) if must else None


def search(
    query: str,
    company: str | None = None,
    fy: str | None = None,
    doc_type: str | None = None,
    k: int = 4,
) -> list[dict]:
    vector = next(embedder().query_embed(query)).tolist()
    query_filter = _filter(company=company, fy=fy, doc_type=doc_type)
    n = max(k, settings.search_candidates) if reranker() else k
    if has_sparse():
        sparse = sparse_vector(next(sparse_embedder().query_embed(query)))
        hits = qdrant().query_points(
            settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=vector, filter=query_filter, limit=n),
                models.Prefetch(query=sparse, using=SPARSE, filter=query_filter, limit=n),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=n,
        ).points
    else:
        hits = qdrant().query_points(
            settings.qdrant_collection, query=vector, query_filter=query_filter, limit=n
        ).points
    payloads = [h.payload for h in hits]
    if reranker() and len(payloads) > k:
        scores = list(reranker().rerank(query, [p["text"] for p in payloads]))
        order = sorted(range(len(payloads)), key=lambda i: scores[i], reverse=True)
        payloads = [payloads[i] for i in order]
    # Only what the agent needs: chunk_id already names the company, year and page.
    return [
        {"chunk_id": p["chunk_id"], "doc_id": p["doc_id"], "page": p["page"], "text": p["text"]}
        for p in payloads[:k]
    ]


def get_page(doc_id: str, page: int) -> str:
    points, _ = qdrant().scroll(
        settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="page", match=models.MatchValue(value=page)),
            ]
        ),
        limit=20,
    )
    points.sort(key=lambda p: p.payload["chunk_id"])
    return "\n".join(p.payload["text"] for p in points)


def list_documents(company: str | None = None) -> list[str]:
    seen, offset = set(), None
    while True:
        points, offset = qdrant().scroll(
            settings.qdrant_collection,
            scroll_filter=_filter(company=company),
            limit=1000,
            offset=offset,
            with_payload=["doc_id"],
        )
        seen.update(p.payload["doc_id"] for p in points)
        if offset is None:
            return sorted(seen)


def point_id(chunk_id: str) -> str:
    """Qdrant point id for a chunk; the indexer derives it the same way."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_chunks(chunk_ids: list[str]) -> list[dict]:
    """Payloads for the given chunk ids, in the order asked for; unknown ids are skipped."""
    points = qdrant().retrieve(settings.qdrant_collection, ids=[point_id(c) for c in chunk_ids])
    by_id = {p.payload["chunk_id"]: p.payload for p in points}
    return [by_id[c] for c in chunk_ids if c in by_id]
