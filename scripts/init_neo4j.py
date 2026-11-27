from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.config.settings import ROOT_DIR
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Neo4j constraints and full-text index.")
    parser.add_argument("--schema", type=Path, default=ROOT_DIR / "knowledge" / "neo4j" / "schema.cypher")
    args = parser.parse_args()
    repository = Neo4jGraphRepository.connect()
    try:
        health = repository.health_check()
        statement_count = repository.apply_schema(args.schema)
        print(json.dumps({
            "status": "success",
            "health": health,
            "schema_statements_applied": statement_count,
            "schema_path": str(args.schema),
            "medical_facts_seeded": False,
        }, ensure_ascii=False, indent=2))
    finally:
        repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
