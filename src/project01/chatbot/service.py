from __future__ import annotations

import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, *, limit: int = 8, candidate_multiplier: int = 3) -> dict[str, Any]: ...


class ChatGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


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
_SYSTEM_PROMPT = """Bạn là trợ lý cung cấp thông tin giáo dục sức khỏe về bệnh đái tháo đường.
Chỉ sử dụng thông tin được cung cấp trong phần BẰNG CHỨNG. Không bịa kiến thức,
số liệu, nguồn hoặc trích dẫn. Nếu bằng chứng không đủ hoặc mâu thuẫn, nói rõ
điều đó và đề nghị người dùng hỏi nhân viên y tế. Dẫn nguồn bằng đúng ký hiệu
[1], [2] tương ứng với danh sách bằng chứng. Không chẩn đoán, kê đơn, thay đổi
liều thuốc hoặc đưa ra phác đồ cá nhân. Trả lời bằng tiếng Việt, rõ ràng, trực tiếp.
Lịch sử hội thoại chỉ dùng để hiểu câu hỏi nối tiếp; không được xem là bằng chứng y khoa."""


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


def _clean_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    cleaned = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:2000]})
    return cleaned


class ChatbotService:
    """Evidence-first chatbot with optional LLM synthesis and deterministic fallback."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        generator: ChatGenerator | None = None,
        result_limit: int = 5,
    ):
        if not 1 <= result_limit <= 10:
            raise ValueError("result_limit phải nằm trong khoảng 1..10.")
        self.retriever = retriever
        self.generator = generator
        self.result_limit = result_limit

    def answer(self, query: str, *, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {
                "status": "needs_clarification", "intent": "empty",
                "answer": "Bạn hãy nhập câu hỏi cụ thể để tôi có thể hỗ trợ.",
                "citations": [], "retrieval": None, "generation": "not_used",
                "clinical_decision": False,
            }

        clean_query = query.strip()
        if _is_greeting(clean_query):
            return {
                "status": "success", "intent": "greeting",
                "answer": "Xin chào! Tôi có thể hỗ trợ tìm hiểu thông tin về bệnh đái tháo đường dựa trên tài liệu có nguồn dẫn.",
                "citations": [], "retrieval": None, "generation": "not_used",
                "clinical_decision": False,
            }

        if _is_urgent(clean_query):
            return {
                "status": "urgent", "intent": "medical_urgent",
                "answer": "Các dấu hiệu bạn mô tả có thể cần được xử trí khẩn cấp. Hãy liên hệ cấp cứu hoặc đến cơ sở y tế gần nhất ngay; không chờ câu trả lời từ chatbot.",
                "citations": [], "retrieval": None, "generation": "not_used",
                "clinical_decision": False,
            }

        if not _is_medical_query(clean_query) and not history:
            return {
                "status": "out_of_scope", "intent": "general",
                "answer": "Tôi hiện tập trung hỗ trợ thông tin về bệnh đái tháo đường. Bạn có thể hỏi về triệu chứng, yếu tố nguy cơ, xét nghiệm, biến chứng, dinh dưỡng hoặc theo dõi bệnh.",
                "citations": [], "retrieval": None, "generation": "not_used",
                "clinical_decision": False,
            }

        try:
            result = self.retriever.retrieve(clean_query, limit=self.result_limit)
        except Exception:
            return {
                "status": "retrieval_unavailable", "intent": "medical_information",
                "answer": "Hiện không truy xuất được kho tri thức. Vui lòng thử lại sau; tôi không đưa ra câu trả lời y khoa khi chưa lấy được bằng chứng.",
                "citations": [], "retrieval": None, "generation": "not_used",
                "clinical_decision": False,
            }

        citations: list[dict[str, Any]] = []
        for item in result.get("results", []):
            excerpt = str(item.get("text") or "").strip()
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            source = item.get("source") or {}
            source_name = source.get("name") if isinstance(source, dict) else str(source or "")
            if not excerpt and not (title and url):
                continue
            citations.append({
                "index": len(citations) + 1,
                "id": item.get("id") or item.get("chunk_id") or item.get("document_id"),
                "title": title or "Tài liệu tham khảo",
                "source": source_name or None,
                "url": url or None,
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "excerpt": excerpt[:1200] if excerpt else None,
                "retrieval_methods": item.get("retrieval_methods", []),
            })
            if len(citations) >= self.result_limit:
                break

        if not citations:
            return {
                "status": "no_evidence", "intent": "medical_information",
                "answer": "Tôi chưa tìm thấy tài liệu phù hợp trong kho tri thức cho câu hỏi này. Tôi không thể kết luận khi chưa có bằng chứng đáng tin cậy.",
                "citations": [], "retrieval": result.get("retrieval"),
                "generation": "not_used", "clinical_decision": False,
            }

        fallback = [
            "Tôi tìm thấy các tài liệu có liên quan dưới đây. Đây là thông tin tham khảo được trích từ nguồn, chưa phải kết luận cá nhân hóa:"
        ]
        for citation in citations[:3]:
            if citation.get("excerpt"):
                fallback.append(f"[{citation['index']}] {citation['excerpt']}")
            else:
                fallback.append(f"[{citation['index']}] {citation['title']}")
        fallback.append(
            "Thông tin này không thay thế chẩn đoán hoặc tư vấn của nhân viên y tế. Không tự ý thay đổi thuốc hay phác đồ điều trị."
        )
        answer = "\n\n".join(fallback)
        generation = "disabled"
        status = "success"

        if self.generator is not None:
            evidence = "\n\n".join(
                f"[{c['index']}] Tiêu đề: {c['title']}\nNguồn: {c.get('source')}\n"
                f"URL: {c.get('url')}\nTrang: {c.get('page_start')}–{c.get('page_end')}\n"
                f"Trích đoạn: {c.get('excerpt') or ''}"
                for c in citations
            )
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            messages.extend(_clean_history(history))
            messages.append({
                "role": "user",
                "content": (
                    f"CÂU HỎI HIỆN TẠI:\n{clean_query}\n\n"
                    f"BẰNG CHỨNG ĐƯỢC PHÉP SỬ DỤNG:\n{evidence}\n\n"
                    "Hãy trả lời chỉ dựa trên bằng chứng trên. Giữ ký hiệu trích dẫn [n]."
                ),
            })
            try:
                generated = self.generator.generate(messages)
            except Exception:
                logger.exception("LLM generation failed; using evidence-based fallback.")
                generation = "llm_error"
                status = "degraded"
            else:
                referenced = {int(value) for value in re.findall(r"\[(\d+)\]", generated)}
                valid_references = {item["index"] for item in citations}
                invalid_references = referenced - valid_references
                if not referenced or invalid_references:
                    logger.warning(
                        "LLM response rejected because citations are missing or invalid "
                        "(referenced=%s, valid=%s). Using evidence-based fallback.",
                        sorted(referenced),
                        sorted(valid_references),
                    )
                    generation = "llm_invalid_citations"
                    status = "degraded"
                else:
                    answer = generated
                    generation = "llm"

        return {
            "status": status, "intent": "medical_information",
            "answer": answer, "citations": citations,
            "retrieval": result.get("retrieval"),
            "generation": generation, "clinical_decision": False,
        }
