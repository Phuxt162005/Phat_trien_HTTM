from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .neo4j_config import Neo4jSettings


FULLTEXT_INDEX = "diabetes_knowledge_fulltext"
SEARCH_QUERY = """
MATCH (node)
SEARCH node IN (
    FULLTEXT INDEX diabetes_knowledge_fulltext
    FOR $query
    LIMIT $limit
)
SCORE AS score
WITH node, score ORDER BY score DESC LIMIT $limit
OPTIONAL MATCH (node)-[rel]-(neighbor)
WHERE neighbor IS NOT NULL
  AND NOT neighbor:MedicalDocument
  AND NOT neighbor:KnowledgeSource
  AND NOT neighbor:DocumentChunk
OPTIONAL MATCH (node)-[:PART_OF]->(linked_doc:MedicalDocument)
WITH node, score, rel, neighbor,
     coalesce(linked_doc, CASE WHEN node:MedicalDocument THEN node ELSE null END) AS doc
OPTIONAL MATCH (doc)-[:CITES_SOURCE|FROM_SOURCE]->(source:KnowledgeSource)
OPTIONAL MATCH (node)-[:CITES_SOURCE|FROM_SOURCE]->(node_source:KnowledgeSource)
RETURN labels(node) AS labels,
       properties(node) AS entity,
       score AS score,
       collect(DISTINCT CASE WHEN neighbor IS NULL THEN null ELSE {
           labels: labels(neighbor), properties: properties(neighbor),
           relationship: type(rel),
           direction: CASE WHEN startNode(rel) = node THEN 'outgoing' ELSE 'incoming' END
       } END)[0..10] AS related,
       collect(DISTINCT CASE WHEN doc IS NULL THEN null ELSE {
           document_id: coalesce(doc.id, doc.doc_id),
           title: coalesce(doc.title, doc.label, doc.question),
           url: coalesce(doc.url, doc.source_url),
           page: properties(node)['page_start'],
           page_start: properties(node)['page_start'],
           page_end: properties(node)['page_end'],
           chunk_id: CASE WHEN node:DocumentChunk THEN node.id ELSE null END,
           source: CASE WHEN source IS NULL AND node_source IS NULL THEN null ELSE {
               name: coalesce(source.name, source.source_name, node_source.name, node_source.source_name, doc.source_name, doc.publisher),
               url: coalesce(source.url, source.source_url, node_source.url, node_source.source_url, doc.url, doc.source_url),
               publisher: coalesce(source.publisher, node_source.publisher, doc.publisher),
               license: coalesce(source.license, node_source.license, doc.license)
           } END
       } END) AS evidence
"""


def _lucene_terms(text: str) -> str:
    """Build a recall-oriented Lucene query from meaningful words.

    Requiring every token with AND made natural-language questions (especially
    Vietnamese questions with stop words) return no lexical candidates. OR lets
    Neo4j rank documents matching informative terms; the result limit and
    full-text score still control the candidate set.
    """
    terms = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
    stop_words = {
        "la", "gì", "gi", "của", "cua", "và", "va", "các", "cac", "những",
        "nhung", "cho", "một", "mot", "trong", "được", "duoc", "theo", "về",
        "ve", "nào", "nao", "như", "nhu", "hay", "có", "co", "không", "khong",
        "thế", "the", "what", "is", "are", "the", "of", "for", "and", "to",
        "a", "an", "with", "in", "on", "how", "does", "do", "can", "i", "you",
    }
    filtered = []
    seen = set()
    for term in terms:
        normalized = term.casefold()
        if normalized in stop_words or normalized in seen:
            continue
        seen.add(normalized)
        filtered.append(term)
    if not filtered:
        filtered = terms
    if not filtered:
        return ""
    return " OR ".join('"' + term.replace('"', '') + '"' for term in filtered[:12])


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
            "query": lucene_query,
            "limit": int(limit),
        }
        with self.driver.session(database=self.database) as session:
            # Pass parameters as a mapping to avoid colliding with Session.run's
            # positional statement argument.
            result = session.run(SEARCH_QUERY, parameters)
            return [record.data() for record in result]

    def fulltext_search_chunks(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search indexed document chunks by lexical terms and retain provenance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query không được để trống.")
        if not 1 <= int(limit) <= 25:
            raise ValueError("limit phải nằm trong khoảng 1..25.")
        lucene_query = _lucene_terms(query)
        if not lucene_query:
            return []
        statement = """
        MATCH (node:DocumentChunk)
        SEARCH node IN (
            FULLTEXT INDEX diabetes_knowledge_fulltext
            FOR $query
            LIMIT $limit
        )
        SCORE AS score
        OPTIONAL MATCH (node)-[:PART_OF]->(doc:MedicalDocument)
        OPTIONAL MATCH (doc)-[:CITES_SOURCE|FROM_SOURCE]->(source:KnowledgeSource)
        RETURN node.id AS id,
               node.text AS text,
               node.chunk_index AS chunk_index,
               properties(node).page_start AS page_start,
               properties(node).page_end AS page_end,
               doc.id AS document_id,
               doc.title AS title,
               doc.url AS url,
               score AS score,
               CASE WHEN source IS NULL THEN null ELSE {
                   name: coalesce(source.name, source.source_name),
                   publisher: source.publisher,
                   license: source.license,
                   url: coalesce(source.url, source.source_url)
               } END AS source
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(statement, {
                "query": lucene_query,
                "limit": int(limit),
            })
            return [record.data() for record in result]

    VECTOR_INDEX = "diabetes_document_chunk_vector"
    VECTOR_SEARCH_QUERY = """
    MATCH (node:DocumentChunk)
    SEARCH node IN (
        VECTOR INDEX diabetes_document_chunk_vector
        FOR $embedding
        LIMIT $limit
    ) SCORE AS score
    OPTIONAL MATCH (node)-[:PART_OF]->(doc:MedicalDocument)
    OPTIONAL MATCH (doc)-[:CITES_SOURCE|FROM_SOURCE]->(source:KnowledgeSource)
    RETURN node.id AS id,
           node.text AS text,
           node.chunk_index AS chunk_index,
           properties(node).page_start AS page_start,
           properties(node).page_end AS page_end,
           doc.id AS document_id,
           doc.title AS title,
           doc.url AS url,
           score AS score,
           CASE WHEN source IS NULL THEN null ELSE {
               name: coalesce(source.name, source.source_name),
               publisher: source.publisher,
               license: source.license,
               url: coalesce(source.url, source.source_url)
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
        if embedding is None:
            raise ValueError("embedding phải là danh sách số không rỗng.")
        try:
            # Accept Python sequences and NumPy arrays without evaluating an
            # ndarray's ambiguous truth value.
            values = [float(value) for value in embedding]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding phải là vector một chiều gồm các giá trị số.") from exc
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("embedding phải là vector số hữu hạn, không rỗng.")
        if not 1 <= int(limit) <= 75:
            raise ValueError("limit phải nằm trong khoảng 1..75.")
        parameters = {
            "index_name": self.VECTOR_INDEX,
            "limit": int(limit),
            "embedding": values,
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
