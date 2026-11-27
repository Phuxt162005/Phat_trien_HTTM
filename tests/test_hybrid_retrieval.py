from __future__ import annotations

import numpy as np
import pytest

from project01.knowledge.embeddings import (
    EmbeddingConfig,
    SentenceTransformerEmbedder,
    reciprocal_rank_fusion,
)
from project01.knowledge.hybrid_retriever import HybridRetriever


class FakeEmbeddingModel:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        return np.ones((len(texts), self.dimension), dtype=float)


def test_embedder_adds_e5_prefix_and_normalizes():
    model = FakeEmbeddingModel()
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="fake", dimension=4), model=model,
    )
    embedder.embed_documents(["medical chunk"])
    embedder.embed_query("diabetes symptoms")
    assert model.inputs == ["passage: medical chunk", "query: diabetes symptoms"]


def test_embedder_rejects_dimension_mismatch():
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="fake", dimension=3),
        model=FakeEmbeddingModel(dimension=4),
    )
    with pytest.raises(ValueError, match="dimension"):
        embedder.embed_query("test")


def test_embedder_rejects_empty_text():
    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="fake", dimension=4), model=FakeEmbeddingModel(),
    )
    with pytest.raises(ValueError):
        embedder.embed_documents(["valid", ""])


def test_rrf_fuses_and_deduplicates_ranked_lists():
    result = reciprocal_rank_fusion([
        [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
        [{"id": "b", "title": "Bee"}, {"id": "c", "text": "C"}],
    ], k=60, limit=3)
    assert [item["id"] for item in result] == ["b", "a", "c"]
    assert result[0]["retrieval_ranks"] == [2, 1]
    assert result[0]["title"] == "Bee"


def test_rrf_validates_arguments_and_ignores_missing_ids():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)
    assert reciprocal_rank_fusion([[{"text": "no id"}]]) == []


def test_hybrid_retriever_calls_vector_and_graph_and_returns_fusion():
    class Repository:
        def vector_search(self, embedding, limit):
            assert len(embedding) == 4
            return [{"id": "chunk-1", "text": "vector evidence"}]

        def fulltext_search_chunks(self, query, limit):
            return [{
                "id": "chunk-1", "text": "keyword evidence",
                "title": "Guideline", "url": "https://example.org",
            }]

        def search(self, query, limit):
            return [{
                "entity": {"id": "disease-1", "name": "Diabetes"},
                "evidence": [{"chunk_id": "chunk-1", "title": "Guideline"}],
            }]

    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="fake", dimension=4), model=FakeEmbeddingModel(),
    )
    result = HybridRetriever(Repository(), embedder).retrieve("diabetes")
    assert result["retrieval"]["vector_candidates"] == 1
    assert result["retrieval"]["keyword_candidates"] == 1
    assert result["retrieval"]["graph_candidates"] == 1
    assert result["retrieval"]["retrieval_channels"] == ["vector", "keyword", "graph"]
    assert result["results"][0]["id"] == "chunk-1"
    assert result["results"][0]["retrieval_methods"] == ["vector", "keyword", "graph"]
    assert result["retrieval"]["fusion"] == "reciprocal_rank_fusion"


def test_hybrid_retriever_validates_query_and_limit():
    class NeverCalled:
        def vector_search(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def search(self, *args, **kwargs):
            raise AssertionError("should not be called")

    embedder = SentenceTransformerEmbedder(
        EmbeddingConfig(model_name="fake", dimension=4), model=FakeEmbeddingModel(),
    )
    retriever = HybridRetriever(NeverCalled(), embedder)
    with pytest.raises(ValueError):
        retriever.retrieve(" ")
    with pytest.raises(ValueError):
        retriever.retrieve("diabetes", limit=0)


def test_hybrid_fusion_preserves_graph_citations_for_vector_match():
    result = reciprocal_rank_fusion([
        [{"id": "chunk-1", "text": "Vector text", "retrieval_method": "vector"}],
        [{
            "id": "chunk-1", "title": "Guideline", "url": "https://example.org",
            "sources": [{"title": "Guideline", "page_start": 3}],
            "retrieval_method": "graph",
        }],
    ])
    assert result[0]["title"] == "Guideline"
    assert result[0]["url"] == "https://example.org"
    assert result[0]["sources"][0]["page_start"] == 3
    assert result[0]["retrieval_methods"] == ["vector", "graph"]
