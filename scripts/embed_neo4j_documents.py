from __future__ import annotations

import argparse
import json

from project01.knowledge.embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Create embeddings for Neo4j DocumentChunk nodes.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--create-index", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit phải >= 1")

    embedder = SentenceTransformerEmbedder(EmbeddingConfig(
        model_name=args.model, dimension=args.dimension, batch_size=args.batch_size,
    ))
    repository = Neo4jGraphRepository.connect()
    try:
        if args.create_index:
            repository.create_vector_index(args.dimension)
        pending = repository.get_chunks_missing_embeddings(limit=args.limit)
        processed = 0
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            vectors = embedder.embed_documents([item["text"] for item in batch])
            rows = [
                {"id": item["id"], "embedding": vector.tolist(),
                 "embedding_model": args.model, "embedding_dimension": args.dimension}
                for item, vector in zip(batch, vectors)
            ]
            processed += repository.store_chunk_embeddings(rows)
        print(json.dumps({
            "status": "success",
            "model": args.model,
            "dimension": args.dimension,
            "pending_chunks": len(pending),
            "embedded_chunks": processed,
            "index_created_or_verified": bool(args.create_index),
        }, ensure_ascii=False, indent=2))
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
