from __future__ import annotations

import json
from pathlib import Path

from .ingestion import DocumentChunk, ingest_file


def build_vinmec_chunks(
    root: str | Path, *, chunk_size: int = 1000, overlap: int = 150
) -> tuple[list[DocumentChunk], list[dict[str, str]]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    chunks: list[DocumentChunk] = []
    seen_hashes: set[str] = set()
    skipped: list[dict[str, str]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        content_path = folder / "content.md"
        metadata_path = folder / "metadata.json"
        if not content_path.is_file() or not metadata_path.is_file():
            skipped.append({"folder": str(folder), "reason": "missing_content_or_metadata"})
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append({"folder": str(folder), "reason": f"invalid_metadata: {exc}"})
            continue
        rights_note = str(metadata.get("rights_note") or "").strip()
        source_url = str(metadata.get("canonical_url") or metadata.get("source_url") or "").strip()
        if not rights_note or not source_url:
            skipped.append({"folder": str(folder), "reason": "missing_rights_note_or_source_url"})
            continue
        try:
            document_chunks = ingest_file(
                content_path,
                source_name="Vinmec",
                publisher=str(metadata.get("publisher") or "Vinmec"),
                license=rights_note,
                source_url=source_url,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            skipped.append({"folder": str(folder), "reason": f"ingestion_error: {exc}"})
            continue
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
            DocumentChunk(
                id=item.id,
                document_id=item.document_id,
                source_id=item.source_id,
                title=title,
                text=item.text,
                page_start=item.page_start,
                page_end=item.page_end,
                chunk_index=item.chunk_index,
                source_name=item.source_name,
                publisher=item.publisher,
                license=item.license,
                source_url=item.source_url,
                file_path=str(content_path),
                content_sha256=item.content_sha256,
            )
            for item in document_chunks
        ]
        chunks.extend(document_chunks)
    return chunks, skipped
