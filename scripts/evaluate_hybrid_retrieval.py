from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project01.knowledge.retrieval_evaluation import evaluate_retrieval_dataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON không hợp lệ: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: mỗi dòng phải là JSON object.")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Đánh giá offline chất lượng truy xuất Hybrid RAG bằng qrels do người đánh giá gán."
    )
    parser.add_argument("--qrels", type=Path, required=True, help="JSONL: query_id + relevant_ids/relevance")
    parser.add_argument("--runs", type=Path, required=True, help="JSONL: query_id + results đã xếp hạng")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10], help="Các cutoff, mặc định: 5 10")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn JSON báo cáo")
    args = parser.parse_args()

    qrels = {}
    for row in _read_jsonl(args.qrels):
        query_id = str(row.get("query_id", "")).strip()
        relevance = row.get("relevance", row.get("relevant_ids"))
        if not query_id or not isinstance(relevance, (dict, list)):
            raise ValueError("Qrels cần query_id và relevance (object) hoặc relevant_ids (array).")
        if query_id in qrels:
            raise ValueError(f"query_id bị trùng trong qrels: {query_id}")
        qrels[query_id] = relevance

    runs = {}
    for row in _read_jsonl(args.runs):
        query_id = str(row.get("query_id", "")).strip()
        results = row.get("results")
        if not query_id or not isinstance(results, list):
            raise ValueError("Runs cần query_id và results (array).")
        if query_id in runs:
            raise ValueError(f"query_id bị trùng trong runs: {query_id}")
        runs[query_id] = results

    report = evaluate_retrieval_dataset(qrels, runs, k_values=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "completed",
        "query_count": report["query_count"],
        "macro_metrics": report["macro"],
        "report_path": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
