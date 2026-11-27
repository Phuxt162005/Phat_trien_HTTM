from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    message_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentResult:
    agent: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    messages: list[AgentMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def send(self, receiver: str, message_type: str, payload: dict[str, Any] | None = None) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            payload=payload or {},
        )
