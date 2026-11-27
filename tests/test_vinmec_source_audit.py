from __future__ import annotations

from scripts.audit_vinmec_sources import MetadataParser, catalog_urls


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


def test_metadata_parser_reads_title_canonical_and_dates():
    parser = MetadataParser()
    parser.feed(
        '<html><head><title>Vinmec Article</title>'
        '<link rel="canonical" href="https://www.vinmec.com/canonical">'
        '<meta property="article:published_time" content="2024-01-02">'
        '<meta property="article:modified_time" content="2025-03-04">'
        '<meta name="description" content="Short description">'
        '</head><body>Body</body></html>'
    )
    assert parser.title == "Vinmec Article"
    assert parser.canonical == "https://www.vinmec.com/canonical"
    assert parser.published == "2024-01-02"
    assert parser.modified == "2025-03-04"
    assert parser.description == "Short description"
