from __future__ import annotations

import json

from project01.knowledge.vinmec_ingestion import build_vinmec_chunks


def _write_article(root, folder, text, *, url="https://www.vinmec.com/vie/a", rights=True):
    path = root / folder
    path.mkdir(parents=True)
    (path / "content.md").write_text(text, encoding="utf-8")
    metadata = {
        "title": "Diabetes article",
        "publisher": "Vinmec",
        "canonical_url": url,
        "rights_note": "Permission verified" if rights else "",
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_build_vinmec_chunks_preserves_per_document_provenance(tmp_path):
    _write_article(tmp_path, "VIN-001", "Diabetes guidance. " * 100)
    chunks, skipped = build_vinmec_chunks(tmp_path, chunk_size=180, overlap=30)
    assert chunks
    assert skipped == []
    assert all(item.source_name == "Vinmec" for item in chunks)
    assert all(item.source_url == "https://www.vinmec.com/vie/a" for item in chunks)
    assert all(item.license == "Permission verified" for item in chunks)
    assert all(item.title == "Diabetes article" for item in chunks)


def test_build_vinmec_chunks_skips_missing_rights_and_duplicate_content(tmp_path):
    _write_article(tmp_path, "VIN-001", "Same text. " * 100)
    _write_article(
        tmp_path, "VIN-002", "Other text. " * 100,
        url="https://www.vinmec.com/vie/b", rights=False,
    )
    _write_article(
        tmp_path, "VIN-003", "Same text. " * 100,
        url="https://www.vinmec.com/vie/c",
    )
    chunks, skipped = build_vinmec_chunks(tmp_path, chunk_size=180, overlap=30)
    assert chunks
    assert len({item.document_id for item in chunks}) == 1
    assert len(skipped) == 2
    assert {item["reason"] for item in skipped} == {
        "missing_rights_note_or_source_url", "duplicate_content"
    }


def test_build_vinmec_chunks_reports_missing_metadata(tmp_path):
    folder = tmp_path / "VIN-404"
    folder.mkdir()
    (folder / "content.md").write_text("Some content", encoding="utf-8")
    chunks, skipped = build_vinmec_chunks(tmp_path)
    assert chunks == []
    assert skipped[0]["reason"] == "missing_content_or_metadata"
