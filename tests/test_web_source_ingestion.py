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



def test_import_saves_article_and_metadata_with_permission(tmp_path, monkeypatch):
    import project01.knowledge.web_source_ingestion as importer

    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"results": [
        {
            "status": "ok", "page_valid": True,
            "catalog_id": "VIN-001",
            "catalog_title": "Catalog title",
            "url": "https://www.vinmec.com/vie/a",
        }
    ]}), encoding="utf-8")
    html = """
    <html><head><title>Vinmec title</title>
    <link rel="canonical" href="/vie/canonical-a">
    <meta property="article:modified_time" content="2025-01-02">
    </head><body><article><h1>Diabetes article</h1><p>
    """ + ("Long enough article content about diabetes. " * 30) + """
    </p></article></body></html>
    """
    monkeypatch.setattr(importer, "_robots_allowed", lambda *_args: True)
    monkeypatch.setattr(importer, "_fetch", lambda *_args: (
        html, "https://www.vinmec.com/vie/a", 200
    ))

    report = importer.import_vinmec_articles(
        audit, tmp_path / "documents",
        rights_note="Permission verified by project owner",
        delay=0,
    )

    assert report["status_counts"] == {"saved": 1}
    article_dir = next((tmp_path / "documents").glob("VIN-001-*"))
    metadata = json.loads((article_dir / "metadata.json").read_text(encoding="utf-8"))
    body = (article_dir / "content.md").read_text(encoding="utf-8")
    assert metadata["canonical_url"] == "https://www.vinmec.com/vie/canonical-a"
    assert metadata["modified_at"] == "2025-01-02"
    assert metadata["rights_note"] == "Permission verified by project owner"
    assert "Long enough article content" in body
    assert (tmp_path / "documents" / "extraction_report.json").exists()



def test_import_rejects_redirected_category_or_missing_heading(tmp_path, monkeypatch):
    import project01.knowledge.web_source_ingestion as importer

    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"results": [
        {
            "status": "ok", "page_valid": True,
            "catalog_id": "VIN-002",
            "url": "https://www.vinmec.com/vie/a",
        }
    ]}), encoding="utf-8")
    html = (
        "<html><head><title>Vinmec</title></head><body>"
        "<main><p>" + ("Generic listing text. " * 80) + "</p></main>"
        "</body></html>"
    )
    monkeypatch.setattr(importer, "_robots_allowed", lambda *_args: True)
    monkeypatch.setattr(importer, "_fetch", lambda *_args: (
        html, "https://www.vinmec.com/vie/bai-viet", 200
    ))
    report = importer.import_vinmec_articles(
        audit, tmp_path / "documents",
        rights_note="Permission verified", delay=0,
    )
    assert report["status_counts"] == {"invalid_or_insufficient_article": 1}
    assert not list((tmp_path / "documents").glob("VIN-002-*/content.md"))


def test_extract_article_removes_in_article_promotions_and_related_links():
    html = """
    <html><head><title>Diabetes</title></head><body>
    <article>
      <h1>Tiểu đường type 2</h1>
      <p>Đái tháo đường type 2 là bệnh rối loạn chuyển hóa glucose.</p>
      <p>Biến chứng có thể ảnh hưởng đến tim, thận và thần kinh.</p>
      <p>TẢI ĐÂY</p>
      <p>Tải và đặt lịch khám tự động trên ứng dụng My Vinmec để quản lý lịch khám.</p>
      <div class="related-articles">
        <h2>Bài viết liên quan</h2>
        <a href="/vie/other">Bài viết về chăm sóc sức khỏe</a>
      </div>
      <p>Để lại thông tin tư vấn, chúng tôi sẽ liên hệ với bạn.</p>
    </article>
    </body></html>
    """
    result = extract_article(html)
    content = result["content"]

    assert "Đái tháo đường type 2 là bệnh rối loạn chuyển hóa glucose." in content
    assert "Biến chứng có thể ảnh hưởng đến tim, thận và thần kinh." in content
    assert "TẢI ĐÂY" not in content
    assert "My Vinmec" not in content
    assert "Bài viết liên quan" not in content
    assert "chúng tôi sẽ liên hệ với bạn" not in content


def test_extract_article_normalizes_whitespace():
    html = """
    <html><body><article>
      <h1>Tiểu đường</h1>
      <p>Glucose     huyết     tăng.</p>
    </article></body></html>
    """
    result = extract_article(html)
    assert "Glucose huyết tăng." in result["content"]
