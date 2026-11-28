from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 45.0
    temperature: float = 0.2
    max_tokens: int = 900

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        provider = os.getenv("LLM_PROVIDER", "").strip().casefold()
        if not provider or provider in {"none", "disabled", "off"}:
            return None
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "").strip()
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
            api_key = os.getenv("LLM_API_KEY") or "ollama"
        elif provider in {"openai", "openai_compatible"}:
            model = os.getenv("LLM_MODEL", "").strip()
            base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
            api_key = os.getenv("LLM_API_KEY", "").strip() or None
        else:
            raise ValueError("LLM_PROVIDER chỉ hỗ trợ ollama, openai hoặc openai_compatible.")
        if not model:
            raise ValueError("Chưa cấu hình model: kiểm tra LLM_MODEL hoặc OLLAMA_MODEL.")
        if provider != "ollama" and not api_key:
            raise ValueError("Thiếu LLM_API_KEY cho provider đã chọn.")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL/OLLAMA_BASE_URL phải là HTTP(S) URL.")
        return cls(
            provider=provider,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "900")),
        )


class OpenAICompatibleChatClient:
    """Minimal OpenAI Chat Completions client using only the Python standard library."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, messages: list[dict[str, str]]) -> str:
        endpoint = self.config.base_url
        if self.config.provider == "ollama" and not endpoint.endswith("/v1") and not endpoint.endswith("/v1/chat/completions"):
            endpoint += "/v1"
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM provider không khả dụng: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM provider trả về response không đúng định dạng Chat Completions.") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM provider trả về câu trả lời rỗng.")
        return content.strip()
