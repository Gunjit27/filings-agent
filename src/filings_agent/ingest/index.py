"""Embed parsed chunks and upsert them into Qdrant."""

import uuid

from qdrant_client import models

from filings_agent.config import settings
from filings_agent.ingest.parse import parse_pdf
from filings_agent.retrieval import embedder, list_documents, qdrant

BATCH = 64


def ensure_collection() -> None:
    client = qdrant()
    if client.collection_exists(settings.qdrant_collection):
        return
    dim = len(next(embedder().embed(["probe"])))
    client.create_collection(
        settings.qdrant_collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field in ("company", "fy", "doc_type", "doc_id"):
        client.create_payload_index(settings.qdrant_collection, field, "keyword")
    client.create_payload_index(settings.qdrant_collection, "page", "integer")


def index_all(reindex: bool = False) -> None:
    ensure_collection()
    client = qdrant()
    done = set() if reindex else set(list_documents())
    for pdf in sorted((settings.data_dir / "raw").glob("*_annual_report.pdf")):
        if pdf.stem in done:
            print(f"{pdf.name}: already indexed, skipping")
            continue
        company, fy, _ = pdf.stem.split("_", 2)
        chunks = parse_pdf(pdf, company, fy)
        print(f"{pdf.name}: {len(chunks)} chunks")
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vectors = list(embedder().embed([c.text for c in batch]))
            client.upsert(
                settings.qdrant_collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                        vector=v.tolist(),
                        payload=c.payload(),
                    )
                    for c, v in zip(batch, vectors)
                ],
            )
            print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}")


if __name__ == "__main__":
    import sys

    index_all(reindex="--reindex" in sys.argv)
