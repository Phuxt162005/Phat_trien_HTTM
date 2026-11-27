from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from urllib.request import Request, urlopen



USER_AGENT = "Project01-Diabetes-Source-Importer/1.0"
ALLOWED_HOSTS = {"www.vinmec.com", "vinmec.com"}


def _safe_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return value[:80] or "article"


def _normalise_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Chỉ chấp nhận URL HTTPS thuộc domain Vinmec.")
    return url


def _fetch(url: str, timeout: float, max_bytes: int = 4_000_000):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            raise ValueError(f"Content-Type không phải HTML: {content_type}")
        final_url = response.geturl()
        _normalise_url(final_url)
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f"Trang vượt giới hạn {max_bytes} bytes.")
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace"), final_url, response.status


def _robots_allowed(url: str, timeout: float) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        request = Request(robots_url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            parser.parse(response.read(512_000).decode("utf-8", errors="replace").splitlines())
        return parser.can_fetch(USER_AGENT, url)
    except HTTPError as exc:
        # A missing robots.txt is allowed; access-denied and server errors fail closed.
        if exc.code == 404:
            return True
        return False
    except (URLError, TimeoutError, OSError):
        return False


def extract_article(html: str) -> dict[str, str | None]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError('Cài HTML dependency bằng: pip install -e ".[rag]"') from exc
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    h1 = soup.find("h1")
    heading = h1.get_text(" ", strip=True) if h1 else None

    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical = canonical_tag.get("href") if canonical_tag else None

    def meta(*keys: str) -> str | None:
        for key in keys:
            tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
            if tag and tag.get("content"):
                return str(tag["content"]).strip()
        return None

    main = soup.find("article") or soup.find("main")
    if main is None:
        main = soup.find(attrs={"itemprop": "articleBody"})
    if main is None:
        main = soup.body or soup
    for element in main(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        element.decompose()

    text = unescape(main.get_text(separator="\n", strip=True))
    lines = [re.sub(r"\\s+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return {
        "title": title,
        "heading": heading,
        "canonical_url": canonical,
        "published_at": meta("article:published_time", "datePublished", "date"),
        "modified_at": meta("article:modified_time", "dateModified"),
        "author": meta("author", "article:author"),
        "description": meta("description", "og:description"),
        "content": text,
    }


def import_vinmec_articles(
    audit_path: str | Path,
    output_dir: str | Path,
    *,
    rights_note: str,
    timeout: float = 25,
    delay: float = 1.5,
    min_chars: int = 500,
    limit: int | None = None,
) -> dict[str, object]:
    if not rights_note.strip():
        raise ValueError("Cần ghi rõ căn cứ/quyền lưu trữ nội dung trong --rights-note.")
    if timeout <= 0 or delay < 0 or min_chars < 100:
        raise ValueError("Tham số timeout/delay/min_chars không hợp lệ.")

    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    rows = audit.get("results", [])
    candidates = [
        row for row in rows
        if row.get("status") == "ok" and row.get("page_valid") is True and row.get("url")
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit phải >= 1.")
        candidates = candidates[:limit]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seen_canonical: set[str] = set()
    seen_hashes: set[str] = set()
    results: list[dict[str, object]] = []

    for index, row in enumerate(candidates):
        if index and delay:
            time.sleep(delay)
        url = _normalise_url(str(row["url"]))
        item: dict[str, object] = {
            "catalog_id": row.get("catalog_id"),
            "url": url,
            "status": "error",
        }
        try:
            if not _robots_allowed(url, timeout):
                item.update({"status": "robots_disallowed_or_unavailable"})
                results.append(item)
                continue

            html, final_url, http_status = _fetch(url, timeout)
            article = extract_article(html)
            canonical = urljoin(final_url, str(article.get("canonical_url") or final_url))
            canonical = _normalise_url(canonical)
            if canonical in seen_canonical:
                item.update({"status": "duplicate_canonical", "canonical_url": canonical})
                results.append(item)
                continue
            content = str(article.get("content") or "").strip()
            heading = str(article.get("heading") or "").strip()
            final_path = urlparse(final_url).path.rstrip("/")
            if final_path in {"/vie", "/vie/bai-viet"} or not heading or len(content) < min_chars:
                item.update({
                    "status": "invalid_or_insufficient_article",
                    "canonical_url": canonical,
                    "content_chars": len(content),
                    "has_article_heading": bool(heading),
                })
                results.append(item)
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                item.update({
                    "status": "duplicate_content",
                    "canonical_url": canonical,
                    "content_sha256": content_hash,
                })
                results.append(item)
                continue

            title = str(article.get("heading") or article.get("title") or row.get("catalog_title") or "Vinmec article")
            folder = root / f"{row.get('catalog_id', 'VIN')}-{_safe_slug(title)}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "content.md").write_text(
                f"# {title}\n\n{content}\n", encoding="utf-8"
            )
            metadata = {
                "catalog_id": row.get("catalog_id"),
                "title": title,
                "publisher": "Vinmec",
                "source_url": url,
                "final_url": final_url,
                "canonical_url": canonical,
                "http_status": http_status,
                "published_at": article.get("published_at"),
                "modified_at": article.get("modified_at"),
                "author": article.get("author"),
                "description": article.get("description"),
                "content_sha256": content_hash,
                "content_chars": len(content),
                "rights_note": rights_note,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "content_file": "content.md",
            }
            (folder / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            seen_canonical.add(canonical)
            seen_hashes.add(content_hash)
            item.update({
                "status": "saved",
                "canonical_url": canonical,
                "content_chars": len(content),
                "content_sha256": content_hash,
                "output_dir": str(folder),
            })
        except HTTPError as exc:
            item.update({"status": "http_error", "http_status": exc.code, "error": str(exc)})
        except (URLError, TimeoutError, OSError, UnicodeError, ValueError) as exc:
            item.update({"status": "error", "error": str(exc)})
        results.append(item)

    counts: dict[str, int] = {}
    for item in results:
        status = str(item["status"])
        counts[status] = counts.get(status, 0) + 1
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_path": str(audit_path),
        "output_dir": str(root),
        "rights_note": rights_note,
        "candidate_count": len(candidates),
        "status_counts": counts,
        "results": results,
    }
    (root / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
