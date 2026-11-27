from __future__ import annotations

from project01.chatbot import ChatbotService


class FakeRetriever:
    def __init__(self, result=None, error=None):
        self.result = result or {"results": [], "retrieval": {"fusion": "rrf"}}
        self.error = error
        self.calls = []

    def retrieve(self, query, *, limit=8, candidate_multiplier=3):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.result


def test_greeting_does_not_call_retriever():
    retriever = FakeRetriever()
    response = ChatbotService(retriever).answer("Xin chào")
    assert response["intent"] == "greeting"
    assert retriever.calls == []


def test_out_of_scope_does_not_call_retriever():
    retriever = FakeRetriever()
    response = ChatbotService(retriever).answer("Viết giúp tôi bài thơ")
    assert response["status"] == "out_of_scope"
    assert retriever.calls == []


def test_medical_query_returns_source_excerpt_and_provenance():
    retriever = FakeRetriever({
        "results": [{
            "id": "chunk-1",
            "title": "Tổng quan bệnh đái tháo đường",
            "text": "Đái tháo đường là bệnh rối loạn chuyển hóa glucose.",
            "url": "https://example.org/diabetes",
            "source": {"name": "Example Medical"},
            "page_start": 2,
            "page_end": 2,
            "retrieval_methods": ["vector", "keyword"],
        }],
        "retrieval": {"fusion": "reciprocal_rank_fusion"},
    })
    response = ChatbotService(retriever).answer("Bệnh đái tháo đường là gì?")
    assert response["status"] == "success"
    assert response["clinical_decision"] is False
    assert response["citations"][0]["url"] == "https://example.org/diabetes"
    assert "[1]" in response["answer"]


def test_no_evidence_abstains():
    response = ChatbotService(FakeRetriever()).answer("Triệu chứng bệnh tiểu đường")
    assert response["status"] == "no_evidence"
    assert response["citations"] == []


def test_retrieval_failure_does_not_fabricate_answer():
    response = ChatbotService(FakeRetriever(error=RuntimeError("offline"))).answer(
        "Đường huyết là gì?"
    )
    assert response["status"] == "retrieval_unavailable"
    assert response["citations"] == []


def test_urgent_symptom_routes_to_emergency_without_retrieval():
    retriever = FakeRetriever()
    response = ChatbotService(retriever).answer("Tôi bị tiểu đường và bất tỉnh")
    assert response["status"] == "urgent"
    assert retriever.calls == []


def test_urgent_symptom_is_not_blocked_by_topic_filter():
    retriever = FakeRetriever()
    response = ChatbotService(retriever).answer("Tôi đang khó thở")
    assert response["status"] == "urgent"
    assert retriever.calls == []


def test_empty_query_requests_clarification():
    response = ChatbotService(FakeRetriever()).answer("  ")
    assert response["status"] == "needs_clarification"
