from __future__ import annotations

import json

import pytest

from project01.knowledge.ingestion import (
    DocumentChunk,
    Neo4jDocumentWriter,
    chunk_text,
    ingest_file,
    write_jsonl,
)


def test_chunk_text_validates_and_preserves_content():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=180, overlap=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert "word0" in chunks[0]
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=50)
    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=100, overlap=100)


def test_ingest_text_file_tracks_hash_source_and_chunk_ids(tmp_path):
    source = tmp_path / "ada_guidance.txt"
    source.write_text("Diabetes guidance. " * 100, encoding="utf-8")
    chunks = ingest_file(
        source, source_name="ADA", publisher="American Diabetes Association",
        license="review-required", source_url="https://diabetes.org",
        chunk_size=180, overlap=30,
    )
    assert len(chunks) > 1
    assert len({item.id for item in chunks}) == len(chunks)
    assert all(item.document_id == chunks[0].document_id for item in chunks)
    assert all(item.source_id == chunks[0].source_id for item in chunks)
    assert all(item.content_sha256 == chunks[0].content_sha256 for item in chunks)
    assert chunks[0].source_name == "ADA"
    assert chunks[0].publisher == "American Diabetes Association"


def test_empty_document_returns_no_chunks(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("   ", encoding="utf-8")
    assert ingest_file(source, source_name="Test") == []


def test_write_jsonl_writes_metadata(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Clinical reference " * 30, encoding="utf-8")
    chunks = ingest_file(source, source_name="Test", chunk_size=120, overlap=20)
    output = write_jsonl(chunks, tmp_path / "nested" / "chunks.jsonl")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(chunks)
    assert rows[0]["source_name"] == "Test"
    assert rows[0]["content_sha256"]


def test_neo4j_writer_uses_parameterized_idempotent_merge():
    class FakeRecord(dict):
        pass

    class FakeResult:
        def single(self):
            return FakeRecord(upserted=1)

    class FakeSession:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, cypher, parameters=None, **kwargs):
            self.calls.append((cypher, parameters or kwargs))
            return FakeResult()

    class FakeDriver:
        def __init__(self):
            self.session_instance = FakeSession()

        def session(self, database=None):
            return self.session_instance

    class FakeRepository:
        def __init__(self):
            self.driver = FakeDriver()
            self.database = "neo4j"

    repository = FakeRepository()
    writer = Neo4jDocumentWriter(repository)
    item = DocumentChunk(
        id="chunk_1", document_id="doc_1", source_id="src_1",
        title="Guidance", text="Evidence text", page_start=1, page_end=1,
        chunk_index=0, source_name="WHO", publisher="WHO", license="CC BY",
        source_url="https://who.int", file_path="guide.pdf", content_sha256="hash",
    )
    assert writer.upsert([item]) == 1
    cypher, params = repository.driver.session_instance.calls[0]
    assert "MERGE (source:KnowledgeSource" in cypher
    assert "MERGE (doc:MedicalDocument" in cypher
    assert "MERGE (chunk:DocumentChunk" in cypher
    assert params["rows"][0]["id"] == "chunk_1"
    with pytest.raises(ValueError):
        writer.upsert([item], batch_size=0)
