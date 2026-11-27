from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from project01.chatbot import ChatbotService
from project01.knowledge.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from project01.knowledge.hybrid_retriever import HybridRetriever
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = None
    app.state.chatbot = None
    app.state.startup_error = None
    try:
        repository = Neo4jGraphRepository.connect()
        embedder = SentenceTransformerEmbedder(EmbeddingConfig(
            model_name="intfloat/multilingual-e5-small",
            dimension=384,
        ))
        app.state.chatbot = ChatbotService(HybridRetriever(repository, embedder))
    except Exception as exc:
        app.state.startup_error = str(exc)
    yield
    if repository is not None:
        repository.close()


app = FastAPI(
    title="Diabetes Decision Support API",
    version="0.1.0",
    description="Evidence-grounded information retrieval API; not a diagnostic service.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, Any]:
    if app.state.chatbot is None:
        return {"status": "degraded", "chatbot_ready": False}
    return {"status": "ok", "chatbot_ready": True}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    service = app.state.chatbot
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot chưa sẵn sàng. Kiểm tra kết nối Neo4j và cấu hình môi trường.",
        )
    return service.answer(payload.query)
