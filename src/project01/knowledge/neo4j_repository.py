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

    VECTOR_INDEX = "diabetes_document_chunk_vector"
    VECTOR_SEARCH_QUERY = """
    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
    YIELD node, score
    OPTIONAL MATCH (node)-[:PART_OF]->(doc:MedicalDocument)
    OPTIONAL MATCH (doc)-[:FROM_SOURCE]->(source:KnowledgeSource)
    RETURN node.id AS id,
           node.text AS text,
           node.chunk_index AS chunk_index,
           node.page_start AS page_start,
           node.page_end AS page_end,
           doc.id AS document_id,
           doc.title AS title,
           doc.url AS url,
           score AS score,
           CASE WHEN source IS NULL THEN null ELSE {
               name: source.name, publisher: source.publisher,
               license: source.license, url: source.url
           } END AS source
    ORDER BY score DESC
    """

    def create_vector_index(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")
        statement = (
            "CREATE VECTOR INDEX " + self.VECTOR_INDEX +
            " IF NOT EXISTS FOR (chunk:DocumentChunk) ON (chunk.embedding) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: " + str(int(dimension)) + ", "
            "`vector.similarity_function`: 'cosine'}}"
        )
        with self.driver.session(database=self.database) as session:
            session.run(statement).consume()

    def get_chunks_missing_embeddings(self, limit: int = 1000) -> list[dict[str, Any]]:
        if not 1 <= int(limit) <= 100000:
            raise ValueError("limit phải nằm trong khoảng 1..100000.")
        query = """
        MATCH (chunk:DocumentChunk)
        WHERE chunk.embedding IS NULL AND chunk.text IS NOT NULL
        RETURN chunk.id AS id, chunk.text AS text
        ORDER BY chunk.id
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, {"limit": int(limit)})
            return [record.data() for record in result]

    def store_chunk_embeddings(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        for row in rows:
            if not row.get("id") or not row.get("embedding"):
                raise ValueError("Mỗi embedding row cần id và embedding.")
        query = """
        UNWIND $rows AS row
        MATCH (chunk:DocumentChunk {id: row.id})
        SET chunk.embedding = row.embedding,
            chunk.embedding_model = row.embedding_model,
            chunk.embedding_dimension = row.embedding_dimension
        RETURN count(chunk) AS stored
        """
        with self.driver.session(database=self.database) as session:
            record = session.run(query, {"rows": rows}).single()
            return int(record["stored"]) if record else 0

    def vector_search(self, embedding: list[float], limit: int = 8) -> list[dict[str, Any]]:
        if not embedding or any(not isinstance(value, (int, float)) for value in embedding):
            raise ValueError("embedding phải là danh sách số không rỗng.")
        if not 1 <= int(limit) <= 75:
            raise ValueError("limit phải nằm trong khoảng 1..75.")
        parameters = {
            "index_name": self.VECTOR_INDEX,
            "limit": int(limit),
            "embedding": [float(value) for value in embedding],
        }
        with self.driver.session(database=self.database) as session:
            result = session.run(self.VECTOR_SEARCH_QUERY, parameters)
            return [record.data() for record in result]

    def apply_schema(self, schema_path: str | Path) -> int:
        """Apply idempotent schema statements; does not create or seed medical facts."""
        statements = [part.strip() for part in Path(schema_path).read_text(encoding="utf-8").split(";") if part.strip()]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()
        return len(statements)
