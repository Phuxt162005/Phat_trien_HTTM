from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_DIMENSION = 384


@dataclass
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = 32
    device: str | None = None

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name không được để trống.")
        if self.dimension < 1:
            raise ValueError("dimension phải >= 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size phải >= 1.")


class SentenceTransformerEmbedder:
    """Lazy multilingual E5 embedder; query/passages use their required prefixes."""

    def __init__(self, config: EmbeddingConfig | None = None, model: Any = None):
        self.config = config or EmbeddingConfig()
        self.config.validate()
        self._model = model

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('Cài embedding dependency bằng: pip install -e ".[rag]"') from exc
            kwargs = {}
            if self.config.device:
                kwargs["device"] = self.config.device
            self._model = SentenceTransformer(self.config.model_name, **kwargs)
        return self._model

    def _encode(self, texts: Sequence[str]):
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.config.dimension:
            raise ValueError(
                f"Embedding dimension không khớp: nhận {vectors.shape}, "
                f"mong đợi (*, {self.config.dimension})."
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]):
        cleaned = [str(text).strip() for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Document text không được rỗng.")
        return self._encode([f"passage: {text}" for text in cleaned])

    def embed_query(self, query: str):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query không được để trống.")
        return self._encode([f"query: {query.strip()}"])[0]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists by stable chunk/entity ID using RRF."""
    if k < 1 or limit < 1:
        raise ValueError("k và limit phải >= 1.")
    fused: dict[str, dict[str, Any]] = {}
    for results in ranked_lists:
        seen: set[str] = set()
        for rank, item in enumerate(results, start=1):
            key = str(item.get("id") or item.get("chunk_id") or item.get("entity_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            entry = fused.setdefault(key, {
                **item, "id": key, "rrf_score": 0.0,
                "retrieval_ranks": [], "retrieval_methods": [],
            })
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["retrieval_ranks"].append(rank)
            method = item.get("retrieval_method")
            if method and method not in entry["retrieval_methods"]:
                entry["retrieval_methods"].append(method)
            for field in (
                "text", "title", "url", "page", "page_start", "page_end",
                "document_id", "chunk_id", "source", "sources", "entity", "related",
            ):
                if entry.get(field) in (None, "", [], {}) and item.get(field) not in (None, "", [], {}):
                    entry[field] = item[field]
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]


def _record_to_dict(record: Any) -> dict[str, Any]:
    return record.data() if hasattr(record, "data") else dict(record)
