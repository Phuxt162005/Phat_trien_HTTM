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

        candidate_limit = min(25, limit * candidate_multiplier)
        vector = self.embedder.embed_query(query)
        vector_results = self.repository.vector_search(vector, limit=min(75, candidate_limit))
        keyword_search = getattr(self.repository, "fulltext_search_chunks", None)
        keyword_results = keyword_search(query, limit=candidate_limit) if callable(keyword_search) else []
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
            document_id = next((ev.get("document_id") for ev in evidence_items if ev.get("document_id")), None)
            identity = chunk_id or document_id or entity.get("id") or entity.get("doc_id")
            source = next((ev.get("source") for ev in evidence_items if ev.get("source")), None)
            if source is None and any(entity.get(key) for key in ("source_name", "source_url", "publisher", "license")):
                source = {
                    "name": entity.get("source_name"),
                    "url": entity.get("source_url"),
                    "publisher": entity.get("publisher"),
                    "license": entity.get("license"),
                }
            graph_ranked.append({
                **item,
                "id": identity,
                "entity_id": entity.get("id") or entity.get("doc_id"),
                "document_id": document_id or entity.get("doc_id") or entity.get("id"),
                "retrieval_method": "graph",
                "entity": entity,
                "sources": evidence_items,
                "source": source,
                "title": next((ev.get("title") for ev in evidence_items if ev.get("title")), None)
                         or entity.get("title") or entity.get("label") or entity.get("question"),
                "url": next((ev.get("url") for ev in evidence_items if ev.get("url")), None)
                       or entity.get("url") or entity.get("source_url"),
                "page_start": next((ev.get("page_start") for ev in evidence_items if ev.get("page_start") is not None), None),
                "page_end": next((ev.get("page_end") for ev in evidence_items if ev.get("page_end") is not None), None),
            })

        keyword_ranked = [
            {**item, "id": item.get("id") or item.get("chunk_id"), "retrieval_method": "keyword"}
            for item in keyword_results
        ]

        fused = reciprocal_rank_fusion(
            [vector_ranked, keyword_ranked, graph_ranked],
            k=self.rrf_k, limit=limit,
        )
        return {
            "query": query,
            "results": fused,
            "retrieval": {
                "vector_candidates": len(vector_ranked),
                "keyword_candidates": len(keyword_ranked),
                "graph_candidates": len(graph_ranked),
                "fused_results": len(fused),
                "retrieval_channels": ["vector", "keyword", "graph"],
                "fusion": "reciprocal_rank_fusion",
            },
            "limitations": (
                "Kết quả truy xuất là tài liệu tham khảo; cần kiểm tra nguồn, "
                "ngày cập nhật và nội dung trước khi sử dụng trong hỗ trợ quyết định y khoa."
            ),
        }
