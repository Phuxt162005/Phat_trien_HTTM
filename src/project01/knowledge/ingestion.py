from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    source_id: str
    title: str
    text: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    source_name: str
    publisher: str | None
    license: str | None
    source_url: str | None
    file_path: str
    content_sha256: str


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _normalize_text(value: str) -> str:
    value = value.replace("\x00", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if chunk_size < 100:
        raise ValueError("chunk_size phải >= 100 ký tự.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải >= 0 và nhỏ hơn chunk_size.")
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        part = normalized[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError('Cài PDF dependency bằng: pip install -e ".[rag]"') from exc
    reader = PdfReader(str(path))
    pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    if pages and not any(text.strip() for _, text in pages):
        raise ValueError(
            f"PDF '{path.name}' không trích xuất được lớp văn bản; "
            "có thể đây là PDF scan và cần OCR trước khi ingestion."
        )
    return pages


def _read_text_file(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    if suffix in {".html", ".htm"}:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise RuntimeError('Cài HTML dependency bằng: pip install -e ".[rag]"') from exc
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        return [(None, soup.get_text(separator=" ", strip=True))]
    raise ValueError(f"Định dạng chưa hỗ trợ: {path.suffix}")


def ingest_file(
    path: str | Path,
    *,
    source_name: str,
    publisher: str | None = None,
    license: str | None = None,
    source_url: str | None = None,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    if not source_name.strip():
        raise ValueError("source_name không được để trống.")
    pages = _read_text_file(file_path)
    page_texts = [(page, _normalize_text(text)) for page, text in pages if _normalize_text(text)]
    if not page_texts:
        return []
    document_text = "\n\n".join(text for _, text in page_texts)
    content_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
    source_identity = "|".join([source_name.strip(), publisher or "", source_url or "", license or ""])
    source_id = _stable_id("src", source_identity)
    document_id = _stable_id("doc", f"{file_path.name}|{content_hash}")
    title = file_path.stem.replace("_", " ").strip()
    output: list[DocumentChunk] = []
    for page_number, page_text in page_texts:
        # Chunk each PDF page independently so every chunk has exact page
        # provenance. Non-PDF formats use page_number=None.
        for part in chunk_text(page_text, chunk_size, overlap):
            page_range = [page_number] if page_number is not None else []
            output.append(_make_chunk(
                part, len(output), document_id, source_id, title, page_range,
                source_name, publisher, license, source_url, file_path, content_hash,
            ))
    return output


def _make_chunk(
    text: str, index: int, document_id: str, source_id: str, title: str,
    pages: list[int], source_name: str, publisher: str | None, license: str | None,
    source_url: str | None, path: Path, content_hash: str,
) -> DocumentChunk:
    identity = f"{document_id}|{index}|{text}"
    return DocumentChunk(
        id=_stable_id("chunk", identity),
        document_id=document_id,
        source_id=source_id,
        title=title,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        chunk_index=index,
        source_name=source_name,
        publisher=publisher,
        license=license,
        source_url=source_url,
        file_path=str(path),
        content_sha256=content_hash,
    )


def ingest_path(
    path: str | Path,
    *,
    source_name: str,
    publisher: str | None = None,
    license: str | None = None,
    source_url: str | None = None,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    root = Path(path)
    files: Iterable[Path] = (
        sorted(item for item in root.rglob("*") if item.suffix.lower() in {".pdf", ".txt", ".md", ".html", ".htm"})
        if root.is_dir() else [root]
    )
    chunks: list[DocumentChunk] = []
    seen_content_hashes: set[str] = set()
    for file in files:
        file_chunks = ingest_file(
            file, source_name=source_name, publisher=publisher, license=license,
            source_url=source_url, chunk_size=chunk_size, overlap=overlap,
        )
        if not file_chunks:
            continue
        content_hash = file_chunks[0].content_sha256
        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)
        chunks.extend(file_chunks)
    return chunks


def write_jsonl(chunks: list[DocumentChunk], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return target


class Neo4jDocumentWriter:
    """Idempotently stores source, document and chunk provenance; no clinical facts."""

    UPSERT_QUERY = """
    UNWIND $rows AS row
    MERGE (source:KnowledgeSource {id: row.source_id})
    ON CREATE SET source.name = row.source_name,
                  source.publisher = row.publisher,
                  source.license = row.license,
                  source.url = row.source_url
    MERGE (doc:MedicalDocument {id: row.document_id})
    ON CREATE SET doc.title = row.title,
                  doc.file_path = row.file_path,
                  doc.content_sha256 = row.content_sha256,
                  doc.url = row.source_url
    MERGE (doc)-[:FROM_SOURCE]->(source)
    MERGE (chunk:DocumentChunk {id: row.id})
    SET chunk.text = row.text,
        chunk.chunk_index = row.chunk_index,
        chunk.page_start = row.page_start,
        chunk.page_end = row.page_end,
        chunk.content_sha256 = row.content_sha256
    MERGE (doc)-[:HAS_CHUNK]->(chunk)
    MERGE (chunk)-[:PART_OF]->(doc)
    RETURN count(chunk) AS upserted
    """

    def __init__(self, repository):
        self.repository = repository

    def upsert(self, chunks: list[DocumentChunk], batch_size: int = 100) -> int:
        if batch_size < 1:
            raise ValueError("batch_size phải >= 1.")
        total = 0
        for offset in range(0, len(chunks), batch_size):
            rows = [asdict(item) for item in chunks[offset:offset + batch_size]]
            with self.repository.driver.session(database=self.repository.database) as session:
                record = session.run(self.UPSERT_QUERY, {"rows": rows}).single()
                total += int(record["upserted"]) if record else 0
        return total
