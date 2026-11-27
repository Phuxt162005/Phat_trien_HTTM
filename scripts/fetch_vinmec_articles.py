from __future__ import annotations

import argparse
import json

from project01.knowledge.web_source_ingestion import import_vinmec_articles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract audited Vinmec HTML articles to local files after rights verification."
    )
    parser.add_argument("--audit", default="data/processed/vinmec_source_audit.json")
    parser.add_argument("--output", default="data/documents/Vinmec")
    parser.add_argument("--rights-note", required=True,
                        help="Căn cứ/quyền lưu trữ nội dung đã được xác minh.")
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = import_vinmec_articles(
        args.audit, args.output, rights_note=args.rights_note,
        timeout=args.timeout, delay=args.delay,
        min_chars=args.min_chars, limit=args.limit,
    )
    print(json.dumps({
        "status": "completed",
        "candidate_count": report["candidate_count"],
        "status_counts": report["status_counts"],
        "report_path": f"{args.output}/extraction_report.json",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
