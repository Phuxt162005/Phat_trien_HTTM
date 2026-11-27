from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from project01.agents.knowledge_agent import KnowledgeGraphAgent
from project01.knowledge.neo4j_config import Neo4jSettings
from project01.knowledge.neo4j_repository import Neo4jGraphRepository, _lucene_terms


class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeResult(list):
    def single(self):
        return self[0] if self else None

    def consume(self):
        return None


class FakeSession:
    def __init__(self, records=None):
        self.records = records or []
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, cypher, parameters=None, **kwargs):
        params = dict(parameters or {})
        params.update(kwargs)
        self.calls.append((cypher, params))
        if "RETURN 1 AS ok" in cypher:
            return FakeResult([FakeRecord(ok=1)])
        if "RETURN count(chunk) AS stored" in cypher:
            return FakeResult([FakeRecord(stored=len(params.get("rows", [])))])
        return FakeResult(self.records)


class FakeDriver:
    def __init__(self, records=None):
        self.session_instance = FakeSession(records)
        self.closed = False

    def session(self, database=None):
        return self.session_instance

    def close(self):
        self.closed = True


def test_neo4j_settings_validation():
    with pytest.raises(ValueError, match="NEO4J_URI"):
        Neo4jSettings("", "", "").validate()
    Neo4jSettings("neo4j+s://example.databases.neo4j.io", "neo4j", "secret").validate()


def test_lucene_query_is_tokenized():
    assert _lucene_terms("diabetes type-2") == '"diabetes" AND "type" AND "2"'
    assert _lucene_terms("!!!") == ""


def test_repository_health_and_search_are_parameterized():
    driver = FakeDriver([FakeRecord(labels=["Disease"], entity={"id": "d1"}, score=0.9, related=[], evidence=[])])
    repository = Neo4jGraphRepository(driver, "neo4j")
    assert repository.health_check() == {"connected": True, "database": "neo4j"}
    results = repository.search("diabetes", limit=3)
    assert results[0]["entity"]["id"] == "d1"
    cypher, params = driver.session_instance.calls[-1]
    assert "db.index.fulltext.queryNodes" in cypher
    assert params == {
        "index_name": "diabetes_knowledge_fulltext",
        "query": '"diabetes"',
        "limit": 3,
    }
    with pytest.raises(ValueError):
        repository.search(" ")
    with pytest.raises(ValueError):
        repository.search("diabetes", limit=0)


def test_knowledge_agent_returns_provenance_and_warning_when_missing():
    class Repository:
        def __init__(self, records):
            self.records = records

        def search(self, query, limit=8):
            return self.records

    source = {"name": "WHO", "url": "https://example.org/source", "license": "CC BY"}
    agent = KnowledgeGraphAgent(Repository([{
        "labels": ["Disease"], "entity": {"id": "d1", "name": "Example"},
        "score": 0.8, "related": [], "evidence": [{"title": "Doc", "source": source}],
    }]))
    result = agent.retrieve("example")
    assert result.status == "success"
    assert result.data["evidence"][0]["sources"][0]["source"]["name"] == "WHO"
    empty = KnowledgeGraphAgent(Repository([])).retrieve("unknown")
    assert empty.status == "warning"
    assert empty.data["evidence"] == []


def test_schema_is_idempotent_and_does_not_seed_facts():
    schema = Path("knowledge/neo4j/schema.cypher").read_text(encoding="utf-8")
    assert "IF NOT EXISTS" in schema
    assert "CREATE FULLTEXT INDEX diabetes_knowledge_fulltext" in schema
    assert "CREATE (:Disease" not in schema


def test_repository_vector_methods_are_parameterized():
    driver = FakeDriver([
        FakeRecord(id="chunk-1", text="Evidence", score=0.91, source={"name": "WHO"})
    ])
    repository = Neo4jGraphRepository(driver, "neo4j")
    assert repository.get_chunks_missing_embeddings(limit=5)[0]["id"] == "chunk-1"
    query, params = driver.session_instance.calls[-1]
    assert "chunk.embedding IS NULL" in query
    assert params == {"limit": 5}

    assert repository.store_chunk_embeddings([{
        "id": "chunk-1", "embedding": [0.1, 0.2],
        "embedding_model": "test", "embedding_dimension": 2,
    }]) == 1
    query, params = driver.session_instance.calls[-1]
    assert "SET chunk.embedding = row.embedding" in query
    assert params["rows"][0]["id"] == "chunk-1"

    results = repository.vector_search(np.array([0.1, 0.2], dtype=np.float32), limit=3)
    assert results[0]["id"] == "chunk-1"
    query, params = driver.session_instance.calls[-1]
    assert "SEARCH node IN" in query
    assert "VECTOR INDEX diabetes_document_chunk_vector" in query
    assert "FOR $embedding" in query
    assert params["index_name"] == "diabetes_document_chunk_vector"
    assert params["embedding"] == [0.1, 0.2]


def test_repository_vector_methods_validate_inputs():
    repository = Neo4jGraphRepository(FakeDriver(), "neo4j")
    with pytest.raises(ValueError):
        repository.get_chunks_missing_embeddings(limit=0)
    with pytest.raises(ValueError):
        repository.store_chunk_embeddings([{"id": "x"}])
    with pytest.raises(ValueError):
        repository.vector_search([], limit=2)
    with pytest.raises(ValueError):
        repository.vector_search(np.array([[0.1, 0.2]], dtype=np.float32), limit=2)
    with pytest.raises(ValueError):
        repository.vector_search([0.1], limit=0)


def test_repository_creates_dimension_specific_vector_index():
    driver = FakeDriver()
    repository = Neo4jGraphRepository(driver, "neo4j")
    repository.create_vector_index(dimension=384)
    query, _ = driver.session_instance.calls[-1]
    assert "CREATE VECTOR INDEX diabetes_document_chunk_vector IF NOT EXISTS" in query
    assert "`vector.dimensions`: 384" in query
    with pytest.raises(ValueError):
        repository.create_vector_index(dimension=0)
