from __future__ import annotations

from pathlib import Path

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


class FakeSession:
    def __init__(self, records=None):
        self.records = records or []
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        if "RETURN 1 AS ok" in query:
            return FakeResult([FakeRecord(ok=1)])
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
    query, params = driver.session_instance.calls[-1]
    assert "db.index.fulltext.queryNodes" in query
    assert params["query"] == '"diabetes"'
    assert params["limit"] == 3
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
