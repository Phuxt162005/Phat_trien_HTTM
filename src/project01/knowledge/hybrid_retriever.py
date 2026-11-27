from __future__ import annotations

from typing import Any

from .embeddings import reciprocal_rank_fusion


class HybridRetriever:
    """Combines vector document search and graph full-text retrieval using RRF."""

    def __init__(self, repository, embedder, *, rrf_k: int = 60):
        if rrf_k < 1:
            raise ValueError("rrf_k phải >= 1.")
        self.repository = repository
        self.embedder = embedder
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, limit: int = 8, candidate_multiplier: int = 3) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query không được để trống.")
        if not 1 <= limit <= 25:
            raise ValueError("limit phải nằm trong khoảng 1..25.")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier phải >= 1.")

        candidate_limit = min(75, limit * candidate_multiplier)
        vector = self.embedder.embed_query(query)
        vector_results = self.repository.vector_search(vector, limit=candidate_limit)
        graph_results = self.repository.search(query, limit=candidate_limit)

        vector_ranked = []
        for item in vector_results:
            vector_ranked.append({
                **item,
                "id": item.get("id") or item.get("chunk_id"),
                "retrieval_method": "vector",
            })

        graph_ranked = []
        for item in graph_results:
            entity = item.get("entity") or {}
            evidence_items = item.get("evidence") or []
            chunk_id = next((ev.get("chunk_id") for ev in evidence_items if ev.get("chunk_id")), None)
            identity = chunk_id or entity.get("id")
            graph_ranked.append({
                **item,
                "id": identity,
                "entity_id": entity.get("id"),
                "retrieval_method": "graph",
                "entity": entity,
                "sources": evidence_items,
            })

        fused = reciprocal_rank_fusion(
            [vector_ranked, graph_ranked], k=self.rrf_k, limit=limit,
        )
        return {
            "query": query,
            "results": fused,
            "retrieval": {
                "vector_candidates": len(vector_ranked),
                "graph_candidates": len(graph_ranked),
                "fused_results": len(fused),
                "fusion": "reciprocal_rank_fusion",
            },
            "limitations": (
                "Kết quả truy xuất là tài liệu tham khảo; cần kiểm tra nguồn, "
                "ngày cập nhật và nội dung trước khi sử dụng trong hỗ trợ quyết định y khoa."
            ),
        }
