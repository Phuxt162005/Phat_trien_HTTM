from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        return cls(
            uri=os.getenv("NEO4J_URI", "").strip(),
            username=os.getenv("NEO4J_USERNAME", "neo4j").strip(),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
        )

    def validate(self) -> None:
        missing = []
        if not self.uri:
            missing.append("NEO4J_URI")
        if not self.username:
            missing.append("NEO4J_USERNAME")
        if not self.password:
            missing.append("NEO4J_PASSWORD")
        if missing:
            raise ValueError("Thiếu cấu hình Neo4j: " + ", ".join(missing))
        if not self.uri.startswith(("neo4j://", "neo4j+s://", "neo4j+ssc://", "bolt://", "bolt+s://", "bolt+ssc://")):
            raise ValueError("NEO4J_URI phải là URI Neo4j/Bolt hợp lệ.")
