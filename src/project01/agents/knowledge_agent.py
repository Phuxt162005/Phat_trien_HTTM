from __future__ import annotations

from typing import Any

from .contracts import AgentResult, BaseAgent


class KnowledgeGraphAgent(BaseAgent):
    """Retrieves graph entities and their source provenance; does not generate diagnoses."""

    def __init__(self, repository):
        super().__init__("KnowledgeGraphAgent")
        self.repository = repository

    def retrieve(self, query: str, limit: int = 8) -> AgentResult:
        try:
            records = self.repository.search(query, limit=limit)
            if not records:
                return AgentResult(
                    self.name,
                    "warning",
                    {
                        "query": query,
                        "evidence": [],
                        "message": "Không tìm thấy tri thức phù hợp trong Knowledge Graph.",
                        "clinical_decision": False,
                    },
                )
            evidence: list[dict[str, Any]] = []
            for item in records:
                evidence.append({
                    "labels": item.get("labels", []),
                    "entity": item.get("entity", {}),
                    "score": item.get("score"),
                    "related": item.get("related", []),
                    "sources": [source for source in item.get("evidence", []) if source],
                })
            has_source = any(item["sources"] for item in evidence)
            return AgentResult(
                self.name,
                "success" if has_source else "warning",
                {
                    "query": query,
                    "evidence": evidence,
                    "source_attribution_available": has_source,
                    "clinical_decision": False,
                    "limitations": "Tri thức truy xuất là bằng chứng tham khảo, không phải chẩn đoán hoặc chỉ định điều trị.",
                },
            )
        except Exception as exc:
            return AgentResult(
                self.name,
                "failed",
                {"query": query, "clinical_decision": False},
                errors=[str(exc)],
            )
