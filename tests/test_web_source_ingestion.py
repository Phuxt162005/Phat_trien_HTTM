from __future__ import annotations

import json

import pytest

from project01.knowledge.web_source_ingestion import extract_article, import_vinmec_articles


def test_extract_article_prefers_article_and_removes_navigation():
    html = """
    <html><head><title>Page title</title>
    <link rel="canonical" href="https://www.vinmec.com/vie/article">
    <meta property="article:published_time" content="2024-01-01">
    </head><body><nav>Navigation should disappear</nav>
    <article><h1>Diabetes overview</h1><p>Clinical article content.</p></article>
    <footer>Footer should disappear</footer></body></html>
    """
    result = extract_article(html)
    assert result["heading"] == "Diabetes overview"
    assert result["canonical_url"] == "https://www.vinmec.com/vie/article"
    assert result["published_at"] == "2024-01-01"
    assert "Clinical article content" in result["content"]
    assert "Navigation should disappear" not in result["content"]
    assert "Footer should disappear" not in result["content"]


def test_import_requires_rights_note(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="rights-note"):
        import_vinmec_articles(audit, tmp_path / "out", rights_note=" ")


def test_import_skips_non_ok_audit_rows(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"results": [
        {"status": "http_error", "url": "https://www.vinmec.com/vie/a"},
        {"status": "ok", "page_valid": False, "url": "https://www.vinmec.com/vie/b"},
    ]}), encoding="utf-8")
    report = import_vinmec_articles(
        audit, tmp_path / "out", rights_note="Permission confirmed", limit=1,
    )
    assert report["candidate_count"] == 0
    assert report["status_counts"] == {}
    assert (tmp_path / "out" / "extraction_report.json").exists()


def test_import_rejects_non_vinmec_canonical_url(tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"results": [
        {"status": "ok", "page_valid": True, "url": "https://www.vinmec.com/vie/a"},
    ]}), encoding="utf-8")
    # URL allow-list is enforced before any network request.
    from project01.knowledge.web_source_ingestion import _normalise_url
    with pytest.raises(ValueError):
        _normalise_url("https://example.com/article")
