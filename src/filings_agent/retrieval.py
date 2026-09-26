"""Vector search over indexed filings."""

from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from filings_agent.config import settings


@lru_cache
def embedder() -> TextEmbedding:
    return TextEmbedding(settings.embed_model)


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
    k: int = 6,
) -> list[dict]:
    vector = next(embedder().query_embed(query)).tolist()
    hits = qdrant().query_points(
        settings.qdrant_collection,
        query=vector,
        query_filter=_filter(company=company, fy=fy, doc_type=doc_type),
        limit=k,
    ).points
    return [{**h.payload, "score": round(h.score, 3)} for h in hits]


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
