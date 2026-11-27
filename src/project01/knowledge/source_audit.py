from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

URL_RE = re.compile(r"https?://[^\\s|)]+", re.IGNORECASE)
GENERIC_TITLES = {"bai viet", "vinmec", "trang chu", "suc khoe tong hop"}


def _normalise(value: str | None) -> str:
    value = (value or "").lower()
    value = re.sub(r"\\s+", " ", value)
    return value.strip(" -|")


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
    """Extract metadata and article-text length without storing the article body."""

    IGNORED_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.in_title = False
        self.in_h1 = False
        self.ignored_depth = 0
        self.body_text_parts: list[str] = []
        self.in_body = False
        self.canonical: str | None = None
        self.published: str | None = None
        self.modified: str | None = None
        self.description: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        values = {key.lower(): value for key, value in attrs if key}
        if tag == "title":
            self.in_title = True
        elif tag == "h1":
            self.in_h1 = True
        elif tag == "body":
            self.in_body = True
        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
        if tag == "link" and values.get("rel", "").lower() == "canonical":
            self.canonical = values.get("href")
        if tag == "meta":
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
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        elif tag == "h1":
            self.in_h1 = False
        elif tag == "body":
            self.in_body = False
        if tag in self.IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.in_title and data.strip():
            self.title_parts.append(data.strip())
        if self.in_h1 and data.strip():
            self.h1_parts.append(data.strip())
        if self.in_body and data.strip():
            self.body_text_parts.append(data.strip())

    @property
    def title(self) -> str | None:
        value = " ".join(self.title_parts).strip()
        return value or None

    @property
    def h1(self) -> str | None:
        value = " ".join(self.h1_parts).strip()
        return value or None

    @property
    def body_text_chars(self) -> int:
        return len(" ".join(self.body_text_parts))


def validate_article_page(
    parser: MetadataParser, final_url: str, min_body_chars: int = 500
) -> tuple[bool, str | None]:
    """Reject category/redirect pages and pages without meaningful article text."""
    path = urlparse(final_url).path.rstrip("/")
    title = _normalise(parser.title)
    heading = _normalise(parser.h1)
    if path in {"/vie/bai-viet", "/vie"}:
        return False, "Trang đích là trang danh mục/tổng hợp, không phải bài viết."
    if not title or title in GENERIC_TITLES:
        return False, "Tiêu đề trang rỗng hoặc là tiêu đề chung."
    if not heading or heading in GENERIC_TITLES:
        return False, "Không tìm thấy tiêu đề H1 đặc trưng của bài viết."
    if parser.body_text_chars < min_body_chars:
        return False, f"Nội dung trang quá ngắn ({parser.body_text_chars} ký tự)."
    return True, None


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
            "User-Agent": "Project01-Diabetes-Source-Auditor/1.1 (metadata-only; contact project maintainers)",
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
            body = response.read(2_000_000).decode(
                response.headers.get_content_charset() or "utf-8", errors="replace"
            )
            page = MetadataParser()
            page.feed(body)
            is_article, validation_error = validate_article_page(page, response.geturl())
            result.update({
                "page_title": page.title,
                "page_h1": page.h1,
                "canonical_url": page.canonical,
                "published_at": page.published,
                "modified_at": page.modified,
                "description": page.description,
                "article_text_chars": page.body_text_chars,
                "content_saved": False,
                "page_valid": is_article,
            })
            if not is_article:
                result.update({"status": "invalid_page", "validation_error": validation_error})
            else:
                result["status"] = "ok"
    except HTTPError as exc:
        result.update({"http_status": exc.code, "error": str(exc)})
        result["status"] = "blocked" if exc.code in {401, 403, 429} else "http_error"
    except (URLError, TimeoutError, OSError, UnicodeError) as exc:
        result["error"] = str(exc)
        result["status"] = "network_error"
    return result
