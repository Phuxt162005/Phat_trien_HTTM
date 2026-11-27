from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.knowledge.ingestion import (
    Neo4jDocumentWriter,
    ingest_file,
    write_jsonl,
)
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


def build_chunks(root: Path, chunk_size: int, overlap: int):
    chunks = []
    seen_hashes: set[str] = set()
    skipped: list[dict[str, str]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        content_path = folder / "content.md"
        metadata_path = folder / "metadata.json"
        if not content_path.is_file() or not metadata_path.is_file():
            skipped.append({"folder": str(folder), "reason": "missing_content_or_metadata"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rights_note = str(metadata.get("rights_note") or "").strip()
        source_url = str(metadata.get("canonical_url") or metadata.get("source_url") or "").strip()
        if not rights_note or not source_url:
            skipped.append({"folder": str(folder), "reason": "missing_rights_note_or_source_url"})
            continue
        document_chunks = ingest_file(
            content_path,
            source_name="Vinmec",
            publisher=str(metadata.get("publisher") or "Vinmec"),
            license=rights_note,
            source_url=source_url,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not document_chunks:
            skipped.append({"folder": str(folder), "reason": "no_extractable_text"})
            continue
        digest = document_chunks[0].content_sha256
        if digest in seen_hashes:
            skipped.append({"folder": str(folder), "reason": "duplicate_content"})
            continue
        seen_hashes.add(digest)
        title = str(metadata.get("title") or folder.name)
        document_chunks = [
            item.__class__(
                **{
                    **item.__dict__,
                    "title": title,
                    "file_path": str(content_path),
                }
            )
            for item in document_chunks
        ]
        chunks.extend(document_chunks)
    return chunks, skipped


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

    chunks, skipped = build_chunks(args.input, args.chunk_size, args.overlap)
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
