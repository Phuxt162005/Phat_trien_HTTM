from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import AgentResult, BaseAgent
from .patient_data_agent import PatientDataAgent
from .rule_based_agent import RuleBasedAgent
from ..data.loader import DatasetConfig, load_dataset


class CoordinatorAgent(BaseAgent):
    """Initial deterministic workflow. Model/RAG agents are integrated in later phases."""

    def __init__(self, data_agent: PatientDataAgent | None = None, rule_agent: RuleBasedAgent | None = None):
        super().__init__("CoordinatorAgent")
        self.data_agent = data_agent or PatientDataAgent()
        self.rule_agent = rule_agent or RuleBasedAgent()
        self.trace: list[dict[str, Any]] = []

    def audit_dataset(self, config: DatasetConfig, data_dir: Path) -> AgentResult:
        self.trace = []
        try:
            df = load_dataset(config, data_dir)
            self.trace.append({"step": "load_dataset", "status": "success", "dataset": config.name})
            quality = self.data_agent.validate_frame(df, target=config.target)
            self.trace.append({"step": self.data_agent.name, "status": quality.status})
            if quality.status != "success":
                return AgentResult(self.name, "error", {"dataset": config.name, "quality": quality.data, "trace": self.trace}, errors=quality.errors)
            rules = self.rule_agent.review_data_quality(quality.data)
            self.trace.append({"step": self.rule_agent.name, "status": rules.status})
            return AgentResult(self.name, "success", {
                "dataset": config.name,
                "task": config.task,
                "target": config.target,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "class_distribution": {str(k): int(v) for k, v in df[config.target].value_counts(dropna=False).items()},
                "quality": quality.data,
                "rule_review": rules.data,
                "trace": self.trace,
            })
        except Exception as exc:
            self.trace.append({"step": "load_or_audit", "status": "error", "error": str(exc)})
            return AgentResult(self.name, "error", {"dataset": config.name, "trace": self.trace}, errors=[str(exc)])
