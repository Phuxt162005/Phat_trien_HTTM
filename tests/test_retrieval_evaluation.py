from __future__ import annotations

import pytest

from project01.knowledge.retrieval_evaluation import (
    evaluate_ranked_query,
    evaluate_retrieval_dataset,
)


def test_ranked_query_metrics_and_citation_quality():
    metrics = evaluate_ranked_query(
        {"relevant-a": 2, "relevant-b": 1},
        [
            {"id": "irrelevant", "title": "Noise", "url": "https://example.org/noise"},
            {"id": "relevant-a", "title": "Guideline", "url": "https://example.org/a"},
            {"id": "relevant-b", "title": "No source URL"},
        ],
        k_values=[1, 2, 3],
    )
    assert metrics["precision@2"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert metrics["ndcg@2"] < 1.0
    assert metrics["citation_coverage@3"] == pytest.approx(2 / 3)
    assert metrics["relevant_citation_accuracy@2"] == pytest.approx(1.0)


def test_binary_qrels_accept_iterables_and_deduplicate_results():
    metrics = evaluate_ranked_query(
        ["chunk-1"],
        [{"id": "chunk-1"}, {"id": "chunk-1"}, {"id": "chunk-2"}],
        k_values=[1, 2],
    )
    assert metrics["retrieved_count"] == 2
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert metrics["reciprocal_rank"] == 1.0


def test_dataset_macro_averages_and_requires_every_query_run():
    report = evaluate_retrieval_dataset(
        {"q1": ["a"], "q2": ["b"]},
        {"q1": [{"id": "a"}], "q2": [{"id": "x"}]},
        k_values=[1],
    )
    assert report["query_count"] == 2
    assert report["macro"]["recall@1"] == pytest.approx(0.5)
    assert report["methodology"]["averaging"] == "unweighted_macro_over_queries"
    with pytest.raises(ValueError, match="Thiếu kết quả"):
        evaluate_retrieval_dataset({"q1": ["a"], "q2": ["b"]}, {"q1": []})


def test_evaluation_rejects_invalid_cutoffs_and_empty_qrels():
    with pytest.raises(ValueError):
        evaluate_ranked_query(["a"], [{"id": "a"}], k_values=[0])
    with pytest.raises(ValueError, match="không được để trống"):
        evaluate_retrieval_dataset({}, {})
