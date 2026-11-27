from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


URL_RE = re.compile(r"https?://[^\s|)]+", re.IGNORECASE)


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


def audit_url(record: dict[str, str], timeout: float) -> dict[str, object]:
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
            # Metadata-only audit: parse the response in memory; never save article body.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Vinmec catalog URLs and page metadata; does not save article content."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/sources/Vinmec/SOURCE_CATALOG.md"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/vinmec_source_audit.json"),
    )
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="Optional maximum number of unique URLs to check.")
    args = parser.parse_args()
    if args.timeout <= 0 or args.delay < 0 or (args.limit is not None and args.limit < 1):
        parser.error("timeout phải > 0, delay >= 0 và limit nếu có phải >= 1.")

    records = catalog_urls(args.input.read_text(encoding="utf-8"))
    if args.limit:
        records = records[:args.limit]
    results = []
    for index, record in enumerate(records):
        if index and args.delay:
            time.sleep(args.delay)
        results.append(audit_url(record, args.timeout))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "catalog": str(args.input),
            "unique_url_count": len(records),
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for item in results:
        key = str(item["status"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({
        "status": "completed",
        "unique_url_count": len(records),
        "status_counts": counts,
        "report_path": str(args.output),
        "article_content_saved": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
