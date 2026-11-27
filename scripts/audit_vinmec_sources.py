from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from project01.knowledge.source_audit import audit_url, catalog_urls


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
