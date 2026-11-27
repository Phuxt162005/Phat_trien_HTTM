from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.knowledge.ingestion import Neo4jDocumentWriter, ingest_path, write_jsonl
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract local PDF/text/HTML files, chunk them and optionally store provenance in Neo4j."
    )
    parser.add_argument("input", type=Path, help="File or directory containing source documents.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--publisher")
    parser.add_argument("--license")
    parser.add_argument("--source-url", help="Canonical source URL, if known.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("data/processed/document_chunks.jsonl"))
    parser.add_argument("--neo4j", action="store_true", help="Also upsert document/chunk provenance to Neo4j.")
    args = parser.parse_args()

    chunks = ingest_path(
        args.input,
        source_name=args.source_name,
        publisher=args.publisher,
        license=args.license,
        source_url=args.source_url,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    output = write_jsonl(chunks, args.output)
    stored = None
    if args.neo4j:
        repository = Neo4jGraphRepository.connect()
        try:
            stored = Neo4jDocumentWriter(repository).upsert(chunks)
        finally:
            repository.close()

    print(json.dumps({
        "status": "success",
        "input": str(args.input),
        "chunk_count": len(chunks),
        "unique_documents": len({item.document_id for item in chunks}),
        "source_name": args.source_name,
        "output": str(output),
        "neo4j_upserted_chunks": stored,
        "clinical_facts_created": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
