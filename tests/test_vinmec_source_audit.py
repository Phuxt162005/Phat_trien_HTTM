from __future__ import annotations

from project01.knowledge.source_audit import (
    MetadataParser,
    catalog_urls,
    validate_article_page,
)


def test_catalog_urls_extracts_unique_urls_and_metadata():
    catalog = """
| ID | Tài liệu | Chủ đề | Liên kết |
|---|---|---|---|
| VIN-001 | Bài A | Tổng quan | https://www.vinmec.com/vie/a |
| VIN-002 | Bài B | Tim mạch | https://www.vinmec.com/vie/b |
| VIN-003 | Bài A trùng | Tổng quan | https://www.vinmec.com/vie/a |
"""
    records = catalog_urls(catalog)
    assert len(records) == 2
    assert records[0]["catalog_id"] == "VIN-001"
    assert records[0]["catalog_title"] == "Bài A"
    assert records[1]["topic"] == "Tim mạch"
    with_s = catalog_urls("| VIN-004 | Bài có chữ s | Tổng quan | https://www.vinmec.com/vie/bai-viet/suc-khoe-nguoi-benh-vi |")
    assert with_s[0]["url"].endswith("/suc-khoe-nguoi-benh-vi")


def test_metadata_parser_reads_title_canonical_and_dates():
    parser = MetadataParser()
    parser.feed(
        '<html><head><title>Vinmec Article</title>'
        '<link rel="canonical" href="https://www.vinmec.com/canonical">'
        '<meta property="article:published_time" content="2024-01-02">'
        '<meta property="article:modified_time" content="2025-03-04">'
        '<meta name="description" content="Short description">'
        '</head><body><h1>Article heading</h1><p>Body text</p></body></html>'
    )
    assert parser.title == "Vinmec Article"
    assert parser.canonical == "https://www.vinmec.com/canonical"
    assert parser.published == "2024-01-02"
    assert parser.modified == "2025-03-04"
    assert parser.description == "Short description"
    assert parser.h1 == "Article heading"
    assert parser.body_text_chars > 0


def test_validate_article_page_rejects_category_page_even_with_http_200():
    parser = MetadataParser()
    parser.feed(
        "<html><head><title>Bài viết | Vinmec</title></head>"
        "<body><h1>Bài viết</h1><p>" + ("Navigation content " * 100) + "</p></body></html>"
    )
    valid, reason = validate_article_page(parser, "https://www.vinmec.com/vie/bai-viet/")
    assert not valid
    assert "danh mục" in reason


def test_validate_article_page_accepts_article_with_heading_and_body():
    parser = MetadataParser()
    parser.feed(
        "<html><head><title>Tiểu đường | Vinmec</title></head><body>"
        "<h1>Tổng quan về bệnh đái tháo đường</h1><p>"
        + ("Nội dung bài viết về bệnh đái tháo đường. " * 30)
        + "</p></body></html>"
    )
    valid, reason = validate_article_page(parser, "https://www.vinmec.com/vie/bai-viet/tieu-duong-vi")
    assert valid
    assert reason is None
