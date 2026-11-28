from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from project01.chatbot import ChatbotService
from project01.chatbot.llm import LLMConfig, OpenAICompatibleChatClient
from project01.knowledge.embeddings import EmbeddingConfig, SentenceTransformerEmbedder
from project01.knowledge.hybrid_retriever import HybridRetriever
from project01.knowledge.neo4j_repository import Neo4jGraphRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = None
    app.state.chatbot = None
    app.state.startup_error = None
    app.state.llm_status = "disabled"
    try:
        repository = Neo4jGraphRepository.connect()
        embedder = SentenceTransformerEmbedder(EmbeddingConfig(
            model_name="intfloat/multilingual-e5-small",
            dimension=384,
        ))
        generator = None
        llm_config = LLMConfig.from_env()
        if llm_config is not None:
            generator = OpenAICompatibleChatClient(llm_config)
            app.state.llm_status = f"configured:{llm_config.provider}/{llm_config.model}"
        app.state.chatbot = ChatbotService(
            HybridRetriever(repository, embedder),
            generator=generator,
        )
    except Exception as exc:
        app.state.startup_error = str(exc)
    yield
    if repository is not None:
        repository.close()


app = FastAPI(
    title="Diabetes Decision Support API",
    version="0.2.0",
    description="Evidence-grounded information retrieval and optional LLM synthesis; not a diagnostic service.",
    lifespan=lifespan,
)


class ChatTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)


@app.get("/health")
def health() -> dict[str, Any]:
    if app.state.chatbot is None:
        return {"status": "degraded", "chatbot_ready": False, "llm": app.state.llm_status}
    return {"status": "ok", "chatbot_ready": True, "llm": app.state.llm_status}


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    service = app.state.chatbot
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot chưa sẵn sàng. Kiểm tra kết nối Neo4j và cấu hình môi trường.",
        )
    history = [turn.model_dump() for turn in payload.history]
    return service.answer(payload.query, history=history)
