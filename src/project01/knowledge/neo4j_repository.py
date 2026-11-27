from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .neo4j_config import Neo4jSettings


FULLTEXT_INDEX = "diabetes_knowledge_fulltext"
SEARCH_QUERY = """
CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score
WITH node, score ORDER BY score DESC LIMIT $limit
OPTIONAL MATCH (node)-[rel]-(neighbor)
WHERE neighbor IS NOT NULL
  AND NOT neighbor:MedicalDocument
  AND NOT neighbor:KnowledgeSource
  AND NOT neighbor:DocumentChunk
OPTIONAL MATCH (node)-[:SUPPORTED_BY|PART_OF]->(linked_doc:MedicalDocument)
WITH node, score, rel, neighbor,
     coalesce(linked_doc, CASE WHEN node:MedicalDocument THEN node ELSE null END) AS doc
OPTIONAL MATCH (doc)-[:FROM_SOURCE]->(source:KnowledgeSource)
OPTIONAL MATCH (node)-[:FROM_SOURCE]->(node_source:KnowledgeSource)
RETURN labels(node) AS labels,
       properties(node) AS entity,
       score AS score,
       collect(DISTINCT CASE WHEN neighbor IS NULL THEN null ELSE {
           labels: labels(neighbor), properties: properties(neighbor),
           relationship: type(rel),
           direction: CASE WHEN startNode(rel) = node THEN 'outgoing' ELSE 'incoming' END
       } END)[0..10] AS related,
       collect(DISTINCT CASE WHEN doc IS NULL THEN null ELSE {
           document_id: doc.id, title: doc.title, url: doc.url,
           page: coalesce(node.page_start, doc.page),
           page_start: node.page_start, page_end: node.page_end,
           chunk_id: CASE WHEN node:DocumentChunk THEN node.id ELSE null END,
           source: CASE WHEN source IS NULL AND node_source IS NULL THEN null ELSE {
               name: coalesce(source.name, node_source.name),
               url: coalesce(source.url, node_source.url),
               publisher: coalesce(source.publisher, node_source.publisher),
               license: coalesce(source.license, node_source.license)
           } END
       } END) AS evidence
"""


def _lucene_terms(text: str) -> str:
    terms = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    if not terms:
        return ""
    return " AND ".join('"' + term.replace('"', '') + '"' for term in terms[:12])


class Neo4jGraphRepository:
    """Small read-oriented repository; driver creation is lazy and optional."""

    def __init__(self, driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database

    @classmethod
    def connect(cls, settings: Neo4jSettings | None = None) -> "Neo4jGraphRepository":
        settings = settings or Neo4jSettings.from_env()
        settings.validate()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError('Cài dependency Neo4j bằng: pip install -e ".[neo4j]"') from exc
        driver = GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password))
        return cls(driver, database=settings.database)

    def close(self) -> None:
        self.driver.close()

    def health_check(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            record = session.run("RETURN 1 AS ok").single()
        return {"connected": bool(record and record["ok"] == 1), "database": self.database}

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query không được để trống.")
        if not 1 <= int(limit) <= 25:
            raise ValueError("limit phải nằm trong khoảng 1..25.")
        lucene_query = _lucene_terms(query)
        if not lucene_query:
            return []
        parameters = {
            "index_name": FULLTEXT_INDEX,
            "query": lucene_query,
            "limit": int(limit),
        }
        with self.driver.session(database=self.database) as session:
            # Pass parameters as a mapping to avoid colliding with Session.run's
            # positional statement argument.
            result = session.run(SEARCH_QUERY, parameters)
            return [record.data() for record in result]

    def apply_schema(self, schema_path: str | Path) -> int:
        """Apply idempotent schema statements; does not create or seed medical facts."""
        statements = [part.strip() for part in Path(schema_path).read_text(encoding="utf-8").split(";") if part.strip()]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()
        return len(statements)
