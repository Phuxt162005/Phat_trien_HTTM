from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project01.knowledge.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from project01.knowledge.hybrid_retriever import HybridRetriever
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


def _read_queries(path: Path) -> list[dict[str, str]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON không hợp lệ: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: mỗi dòng phải là JSON object.")
            query_id = str(row.get("query_id", "")).strip()
            query = row.get("query")
            if not query_id or not isinstance(query, str) or not query.strip():
                raise ValueError(f"{path}:{line_number}: cần query_id và query không rỗng.")
            if query_id in seen:
                raise ValueError(f"query_id bị trùng: {query_id}")
            seen.add(query_id)
            rows.append({"query_id": query_id, "query": query.strip()})
    if not rows:
        raise ValueError("File câu hỏi không có query hợp lệ.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chạy HybridRetriever trên bộ query JSONL và lưu ranked runs JSONL."
    )
    parser.add_argument("--queries", type=Path, required=True, help="JSONL: query_id + query")
    parser.add_argument("--output", type=Path, required=True, help="JSONL kết quả theo thứ tự xếp hạng")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.limit <= 25:
        parser.error("--limit phải nằm trong khoảng 1..25")
    if args.candidate_multiplier < 1:
        parser.error("--candidate-multiplier phải >= 1")

    queries = _read_queries(args.queries)
    embedder = SentenceTransformerEmbedder(EmbeddingConfig(
        model_name=args.model, dimension=args.dimension, batch_size=args.batch_size,
    ))
    repository = Neo4jGraphRepository.connect()
    completed = 0
    try:
        retriever = HybridRetriever(repository, embedder)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for row in queries:
                response = retriever.retrieve(
                    row["query"], limit=args.limit,
                    candidate_multiplier=args.candidate_multiplier,
                )
                record = {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "results": response["results"],
                    "retrieval": response["retrieval"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed += 1
    finally:
        repository.close()

    print(json.dumps({
        "status": "completed",
        "query_count": completed,
        "model": args.model,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
