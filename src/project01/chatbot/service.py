from __future__ import annotations

import re
from typing import Any, Protocol


class Retriever(Protocol):
    def retrieve(self, query: str, *, limit: int = 8, candidate_multiplier: int = 3) -> dict[str, Any]: ...


_MEDICAL_TERMS = (
    "đái tháo đường", "tiểu đường", "diabetes", "đường huyết", "glucose",
    "hba1c", "insulin", "hạ đường huyết", "tăng đường huyết", "biến chứng",
    "triệu chứng", "chẩn đoán", "xét nghiệm", "bệnh thận", "bệnh võng mạc",
    "bệnh thần kinh", "dinh dưỡng", "điều trị", "phòng ngừa", "thuốc",
)
_GREETING_PATTERNS = (
    r"^(xin chào|chào|hello|hi|hey)( bạn)?[!. ]*$",
    r"^(cảm ơn|cam on|thanks|thank you)[!. ]*$",
)
_URGENT_TERMS = (
    "bất tỉnh", "mất ý thức", "co giật", "khó thở", "đau ngực",
    "lơ mơ", "ngất", "unconscious", "seizure", "chest pain",
    "difficulty breathing",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _is_greeting(text: str) -> bool:
    normalized = _normalize(text)
    return any(re.match(pattern, normalized) for pattern in _GREETING_PATTERNS)


def _is_medical_query(text: str) -> bool:
    normalized = _normalize(text)
    return any(term in normalized for term in _MEDICAL_TERMS)


def _is_urgent(text: str) -> bool:
    normalized = _normalize(text)
    return any(term in normalized for term in _URGENT_TERMS)


class ChatbotService:
    """Evidence-first chatbot facade over HybridRetriever.

    This service deliberately does not invent medical answers. It returns short
    source excerpts and provenance until a separately evaluated answer
    generation model is configured.
    """

    def __init__(self, retriever: Retriever, *, result_limit: int = 5):
        if not 1 <= result_limit <= 10:
            raise ValueError("result_limit phải nằm trong khoảng 1..10.")
        self.retriever = retriever
        self.result_limit = result_limit

    def answer(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "needs_clarification",
                "intent": "empty",
                "answer": "Bạn hãy nhập câu hỏi cụ thể để tôi có thể hỗ trợ.",
                "citations": [],
                "retrieval": None,
                "clinical_decision": False,
            }

        clean_query = query.strip()
        if _is_greeting(clean_query):
            return {
                "status": "success",
                "intent": "greeting",
                "answer": "Xin chào! Tôi có thể hỗ trợ tìm hiểu thông tin về bệnh đái tháo đường dựa trên tài liệu có nguồn dẫn.",
                "citations": [],
                "retrieval": None,
                "clinical_decision": False,
            }

        if _is_urgent(clean_query):
            return {
                "status": "urgent",
                "intent": "medical_urgent",
                "answer": "Các dấu hiệu bạn mô tả có thể cần được xử trí khẩn cấp. Hãy liên hệ cấp cứu hoặc đến cơ sở y tế gần nhất ngay; không chờ câu trả lời từ chatbot.",
                "citations": [],
                "retrieval": None,
                "clinical_decision": False,
            }

        if not _is_medical_query(clean_query):
            return {
                "status": "out_of_scope",
                "intent": "general",
                "answer": "Tôi hiện tập trung hỗ trợ thông tin về bệnh đái tháo đường. Bạn có thể hỏi về triệu chứng, yếu tố nguy cơ, xét nghiệm, biến chứng, dinh dưỡng hoặc theo dõi bệnh.",
                "citations": [],
                "retrieval": None,
                "clinical_decision": False,
            }

        try:
            result = self.retriever.retrieve(clean_query, limit=self.result_limit)
        except Exception:
            return {
                "status": "retrieval_unavailable",
                "intent": "medical_information",
                "answer": "Hiện không truy xuất được kho tri thức. Vui lòng thử lại sau; tôi không đưa ra câu trả lời y khoa khi chưa lấy được bằng chứng.",
                "citations": [],
                "retrieval": None,
                "clinical_decision": False,
            }

        ranked = result.get("results", [])
        citations: list[dict[str, Any]] = []
        for item in ranked:
            text = str(item.get("text") or "").strip()
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            source = item.get("source") or {}
            source_name = source.get("name") if isinstance(source, dict) else str(source or "")
            if not text and not (title and url):
                continue
            citations.append({
                "index": len(citations) + 1,
                "id": item.get("id") or item.get("chunk_id") or item.get("document_id"),
                "title": title or "Tài liệu tham khảo",
                "source": source_name or None,
                "url": url or None,
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "excerpt": text[:700] if text else None,
                "retrieval_methods": item.get("retrieval_methods", []),
            })
            if len(citations) >= self.result_limit:
                break

        if not citations:
            return {
                "status": "no_evidence",
                "intent": "medical_information",
                "answer": "Tôi chưa tìm thấy tài liệu phù hợp trong kho tri thức cho câu hỏi này. Tôi không thể kết luận khi chưa có bằng chứng đáng tin cậy.",
                "citations": [],
                "retrieval": result.get("retrieval"),
                "clinical_decision": False,
            }

        lines = [
            "Tôi tìm thấy các tài liệu có liên quan dưới đây. Đây là thông tin tham khảo được trích từ nguồn, chưa phải kết luận cá nhân hóa:"
        ]
        for citation in citations[:3]:
            excerpt = citation.get("excerpt")
            if excerpt:
                lines.append(f"[{citation['index']}] {excerpt}")
            else:
                lines.append(f"[{citation['index']}] {citation['title']}")
        lines.append(
            "Thông tin này không thay thế chẩn đoán hoặc tư vấn của nhân viên y tế. Không tự ý thay đổi thuốc hay phác đồ điều trị."
        )
        return {
            "status": "success",
            "intent": "medical_information",
            "answer": "\n\n".join(lines),
            "citations": citations,
            "retrieval": result.get("retrieval"),
            "clinical_decision": False,
        }
