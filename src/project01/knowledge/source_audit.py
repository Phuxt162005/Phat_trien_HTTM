from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

URL_RE = re.compile(r"https?://[^\\s|)]+", re.IGNORECASE)


def catalog_urls(text: str) -> list[dict[str, str]]:
    """Read Vinmec catalog rows and return unique URL records."""
    records: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0].lower() == "id" or set(cells[0]) <= {"-", ":"}:
            continue
        match = URL_RE.search(cells[3])
        if not match:
            continue
        url = match.group(0).rstrip(".,;")
        if url not in records:
            records[url] = {
                "catalog_id": cells[0],
                "catalog_title": cells[1],
                "topic": cells[2],
                "url": url,
            }
    return list(records.values())


class MetadataParser(HTMLParser):
    """Extract page metadata without persisting article body text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.canonical: str | None = None
        self.published: str | None = None
        self.modified: str | None = None
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = {key.lower(): value for key, value in attrs if key}
        if tag.lower() == "title":
            self.in_title = True
        if tag.lower() == "link" and values.get("rel", "").lower() == "canonical":
            self.canonical = values.get("href")
        if tag.lower() == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            content = values.get("content")
            if not content:
                return
            if key in {"article:published_time", "datepublished", "date"}:
                self.published = self.published or content
            elif key in {"article:modified_time", "datemodified", "last-modified"}:
                self.modified = self.modified or content
            elif key in {"description", "og:description"}:
                self.description = self.description or content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and data.strip():
            self.title_parts.append(data.strip())

    @property
    def title(self) -> str | None:
        value = " ".join(self.title_parts).strip()
        return value or None


def audit_url(record: dict[str, str], timeout: float = 20) -> dict[str, object]:
    url = record["url"]
    result: dict[str, object] = {
        **record,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "error",
    }
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.vinmec.com", "vinmec.com"}:
        result["error"] = "URL không thuộc HTTPS domain Vinmec được cho phép."
        return result
    request = Request(
        url,
        headers={
            "User-Agent": "Project01-Diabetes-Source-Auditor/1.0 (metadata-only; contact project maintainers)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            result["http_status"] = response.status
            result["final_url"] = response.geturl()
            result["content_type"] = content_type
            if "html" not in content_type.lower():
                result["status"] = "non_html"
                return result
            body = response.read(1_000_000).decode(
                response.headers.get_content_charset() or "utf-8", errors="replace"
            )
            parser = MetadataParser()
            parser.feed(body)
            result.update({
                "status": "ok",
                "page_title": parser.title,
                "canonical_url": parser.canonical,
                "published_at": parser.published,
                "modified_at": parser.modified,
                "description": parser.description,
                "content_saved": False,
            })
    except HTTPError as exc:
        result.update({"http_status": exc.code, "error": str(exc)})
        result["status"] = "blocked" if exc.code in {401, 403, 429} else "http_error"
    except (URLError, TimeoutError, OSError, UnicodeError) as exc:
        result["error"] = str(exc)
        result["status"] = "network_error"
    return result
