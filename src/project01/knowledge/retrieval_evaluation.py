from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2
from typing import Any
from urllib.parse import urlparse


def _validate_k_values(k_values: Iterable[int]) -> list[int]:
    values = sorted(set(int(value) for value in k_values))
    if not values or any(value < 1 for value in values):
        raise ValueError("k_values phải chứa ít nhất một số nguyên dương.")
    return values


def _has_citation(item: Mapping[str, Any]) -> bool:
    """Return whether a result carries a minimally usable source citation."""
    url = item.get("url") or item.get("source_url")
    title = item.get("title") or item.get("document_title")
    source = item.get("source")
    source_name = source.get("name") if isinstance(source, Mapping) else source
    valid_url = False
    if isinstance(url, str):
        parsed = urlparse(url.strip())
        valid_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    return bool(valid_url and (title or source_name))


def evaluate_ranked_query(
    relevant: Mapping[str, float] | Iterable[str],
    results: Sequence[Mapping[str, Any]],
    *,
    k_values: Iterable[int] = (5, 10),
) -> dict[str, Any]:
    """Evaluate one ranked result list against human-authored relevance judgments.

    relevant may be an iterable of relevant IDs (binary relevance) or a mapping
    from ID to a positive graded relevance score. Duplicate result IDs are ignored
    after their first occurrence.
    """
    ks = _validate_k_values(k_values)
    if isinstance(relevant, Mapping):
        grades = {str(key): float(value) for key, value in relevant.items() if float(value) > 0}
    else:
        grades = {str(value): 1.0 for value in relevant}
    if any(not key for key in grades):
        raise ValueError("Relevant IDs không được để trống.")
    if any(not isinstance(item, Mapping) for item in results):
        raise ValueError("Mỗi kết quả truy xuất phải là một object.")
    ranked: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        identity = item.get("id") or item.get("chunk_id")
        if identity is None or not str(identity).strip() or str(identity) in seen:
            continue
        seen.add(str(identity))
        ranked.append(item)

    ranked_ids = [str(item.get("id") or item.get("chunk_id")) for item in ranked]
    output: dict[str, Any] = {"relevant_count": len(grades), "retrieved_count": len(ranked)}
    first_relevant_rank = next(
        (rank for rank, identity in enumerate(ranked_ids, start=1) if identity in grades),
        None,
    )
    output["reciprocal_rank"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    for k in ks:
        top = ranked[:k]
        top_ids = ranked_ids[:k]
        hits = sum(identity in grades for identity in top_ids)
        output[f"precision@{k}"] = hits / k
        output[f"recall@{k}"] = hits / len(grades) if grades else 0.0

        dcg = sum(
            (2 ** grades.get(identity, 0.0) - 1.0) / log2(rank + 1)
            for rank, identity in enumerate(top_ids, start=1)
        )
        ideal_grades = sorted(grades.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** grade - 1.0) / log2(rank + 1)
            for rank, grade in enumerate(ideal_grades, start=1)
        )
        output[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

        cited = sum(_has_citation(item) for item in top)
        output[f"citation_coverage@{k}"] = cited / len(top) if top else 0.0
        relevant_top = [item for item in top if str(item.get("id") or item.get("chunk_id")) in grades]
        cited_relevant = sum(_has_citation(item) for item in relevant_top)
        output[f"relevant_citation_accuracy@{k}"] = (
            cited_relevant / len(relevant_top) if relevant_top else 0.0
        )
    return output


def evaluate_retrieval_dataset(
    qrels: Mapping[str, Mapping[str, float] | Iterable[str]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k_values: Iterable[int] = (5, 10),
) -> dict[str, Any]:
    """Compute per-query metrics and unweighted macro averages."""
    ks = _validate_k_values(k_values)
    if not qrels:
        raise ValueError("Qrels không được để trống.")
    missing = sorted(set(qrels) - set(runs))
    if missing:
        raise ValueError(f"Thiếu kết quả truy xuất cho query_id: {', '.join(missing)}")

    per_query = {
        query_id: evaluate_ranked_query(qrels[query_id], runs.get(query_id, []), k_values=ks)
        for query_id in sorted(qrels)
    }
    metric_names = [
        "reciprocal_rank",
        *[f"{metric}@{k}" for k in ks for metric in (
            "precision", "recall", "ndcg", "citation_coverage", "relevant_citation_accuracy"
        )],
    ]
    macro = {
        name: sum(float(metrics[name]) for metrics in per_query.values()) / len(per_query)
        for name in metric_names
    }
    return {
        "query_count": len(per_query),
        "k_values": ks,
        "macro": macro,
        "per_query": per_query,
        "methodology": {
            "averaging": "unweighted_macro_over_queries",
            "precision_denominator": "requested_k",
            "relevance_judgments": "human_authored_qrels",
            "citation_rule": "valid_http_or_https_url_and_title_or_source_name",
        },
    }
