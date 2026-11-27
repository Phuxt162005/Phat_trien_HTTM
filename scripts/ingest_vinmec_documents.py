from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.knowledge.ingestion import Neo4jDocumentWriter, write_jsonl
from project01.knowledge.neo4j_repository import Neo4jGraphRepository
from project01.knowledge.vinmec_ingestion import build_vinmec_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk locally extracted Vinmec articles with per-document provenance."
    )
    parser.add_argument("--input", type=Path, default=Path("data/documents/Vinmec"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/vinmec_chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Không tìm thấy thư mục tài liệu: {args.input}")

    chunks, skipped = build_vinmec_chunks(
        args.input, chunk_size=args.chunk_size, overlap=args.overlap,
    )
    output = write_jsonl(chunks, args.output)
    stored = None
    if args.neo4j and chunks:
        repository = Neo4jGraphRepository.connect()
        try:
            stored = Neo4jDocumentWriter(repository).upsert(chunks)
        finally:
            repository.close()

    print(json.dumps({
        "status": "success",
        "documents": len({item.document_id for item in chunks}),
        "chunks": len(chunks),
        "skipped_documents": len(skipped),
        "skip_reasons": skipped,
        "output": str(output),
        "neo4j_upserted_chunks": stored,
        "clinical_facts_created": 0,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
