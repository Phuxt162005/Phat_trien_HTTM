from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .contracts import AgentResult, BaseAgent
from ..data.loader import DatasetConfig
from ..ml.evaluation import evaluate_predictions
from ..ml.models import _fit_and_evaluate, soft_voting
from ..ml.preprocessing import PreparedData, prepare_dataset


class ModelAAgent(BaseAgent):
    """Logistic Regression model agent."""

    def __init__(self) -> None:
        super().__init__("ModelAAgent")

    def run(self, prepared: PreparedData) -> AgentResult:
        estimator = LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        )
        output = _fit_and_evaluate("Logistic Regression", estimator, prepared)
        return AgentResult(
            self.name,
            "success",
            {"output": output, "validation_score": output.validation_score,
             "validation_metric": output.validation_metric},
            [self.send("ConsensusAgent", "model_ready", {
                "model": self.name, "validation_score": output.validation_score
            })],
        )


class ModelBAgent(BaseAgent):
    """Random Forest model agent."""

    def __init__(self, n_estimators: int = 120, n_jobs: int = 1) -> None:
        super().__init__("ModelBAgent")
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def run(self, prepared: PreparedData) -> AgentResult:
        estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            bootstrap=True,
            max_samples=0.8,
            random_state=42,
            n_jobs=self.n_jobs,
        )
        output = _fit_and_evaluate("Random Forest", estimator, prepared)
        return AgentResult(
            self.name,
            "success",
            {"output": output, "validation_score": output.validation_score,
             "validation_metric": output.validation_metric},
            [self.send("ConsensusAgent", "model_ready", {
                "model": self.name, "validation_score": output.validation_score
            })],
        )


class ConsensusAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ConsensusAgent")

    def run(self, model_results: list[AgentResult]) -> AgentResult:
        if len(model_results) < 2:
            return AgentResult(self.name, "failed", errors=["Consensus cần ít nhất hai mô hình."])
        outputs = [result.data["output"] for result in model_results]
        voting = soft_voting({str(i): output for i, output in enumerate(outputs)})
        return AgentResult(
            self.name, "success", voting,
            [self.send("ExplanationAgent", "consensus_ready", {"rows": len(voting["predictions"])}),
             self.send("EvaluationAgent", "consensus_ready", {"rows": len(voting["predictions"])})],
        )


class ExplanationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("ExplanationAgent")

    def run(self, random_forest_result: AgentResult, consensus_result: AgentResult,
            profile: dict[str, Any]) -> AgentResult:
        model = random_forest_result.data["output"].model
        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        try:
            feature_names = preprocessor.get_feature_names_out()
            importances = estimator.feature_importances_
            top = (pd.DataFrame({"feature": feature_names, "importance": importances})
                   .sort_values("importance", ascending=False).head(15)
                   .reset_index(drop=True))
        except (AttributeError, ValueError):
            top = pd.DataFrame(columns=["feature", "importance"])
        probabilities = consensus_result.data["probabilities"]
        summary = {
            "dataset": profile["dataset"],
            "task": profile["task"],
            "mean_max_probability": float(np.mean(probabilities.max(axis=1))),
            "low_confidence_count_below_0_60": int(np.sum(probabilities.max(axis=1) < 0.60)),
            "top_features": top.to_dict("records"),
            "explanation_scope": "Random Forest feature importance; not causal explanation",
        }
        return AgentResult(
            self.name, "success", {"summary": summary, "top_features": top},
            [self.send("SafetyReviewAgent", "explanation_ready", {"created": True})],
        )


class EvaluationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("EvaluationAgent")

    def run(self, consensus_result: AgentResult, prepared: PreparedData) -> AgentResult:
        metrics = evaluate_predictions(
            prepared.y_test,
            consensus_result.data["predictions"],
            consensus_result.data["probabilities"],
            consensus_result.data["classes"],
            prepared.profile["task_kind"],
        )
        return AgentResult(
            self.name, "success", metrics,
            [self.send("SafetyReviewAgent", "evaluation_ready", {"complete": True})],
        )


class SafetyReviewAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("SafetyReviewAgent")

    def run(self, consensus_result: AgentResult, explanation_result: AgentResult,
            evaluation_result: AgentResult) -> AgentResult:
        probabilities = np.asarray(consensus_result.data["probabilities"], dtype=float)
        predictions = np.asarray(consensus_result.data["predictions"])
        valid = (
            probabilities.ndim == 2
            and np.isfinite(probabilities).all()
            and (probabilities >= 0).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
            and len(predictions) == len(probabilities)
        )
        if not valid:
            return AgentResult(
                self.name, "failed",
                {"valid_output": False},
                errors=["Safety Review thất bại: đầu ra dự đoán không hợp lệ."],
            )
        warnings = [
            "Đây là kết quả dự đoán của mô hình, không phải chẩn đoán y khoa.",
            "Cần kiểm chứng trên dữ liệu độc lập và được chuyên gia y tế đánh giá trước khi ứng dụng.",
            "Feature importance không chứng minh quan hệ nhân quả.",
        ]
        if explanation_result.data["summary"]["low_confidence_count_below_0_60"]:
            warnings.append("Có mẫu có xác suất lớp cao nhất dưới 0.60; cần thận trọng khi diễn giải.")
        return AgentResult(
            self.name, "success",
            {"valid_output": True, "warnings": warnings,
             "evaluation_summary": {
                 key: value for key, value in evaluation_result.data.items()
                 if isinstance(value, (int, float))
             }},
            [self.send("MLCoordinatorAgent", "review_passed", {"valid_output": True})],
        )


class DeepLearningAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("DeepLearningAgent")

    def run(self, prepared: PreparedData) -> AgentResult:
        from ..ml.deep_learning import train_deep_learning_model

        output = train_deep_learning_model(prepared)
        metrics = evaluate_predictions(
            prepared.y_test, output.test_predictions,
            output.test_probabilities, output.classes,
            prepared.profile["task_kind"],
        )
        return AgentResult(
            self.name, "success",
            {"output": output, "metrics": metrics,
             "validation_score": output.validation_score,
             "validation_metric": output.validation_metric,
             "best_epoch": output.best_epoch,
             "epochs_ran": output.epochs_ran,
             "stopped_early": output.stopped_early},
            [self.send("SafetyReviewAgent", "deep_learning_ready", {"complete": True})],
        )


class MLCoordinatorAgent(BaseAgent):
    """Coordinates preprocessing, model agents, consensus, evaluation and safety review."""

    def __init__(self, rf_estimators: int = 120, n_jobs: int = 1) -> None:
        super().__init__("MLCoordinatorAgent")
        self.model_a = ModelAAgent()
        self.model_b = ModelBAgent(rf_estimators, n_jobs)
        self.consensus = ConsensusAgent()
        self.explanation = ExplanationAgent()
        self.evaluation = EvaluationAgent()
        self.safety = SafetyReviewAgent()
        self.deep_learning_agent = DeepLearningAgent()
        self.message_log: list[dict[str, Any]] = []
        self.agent_log: list[dict[str, str]] = []

    def run(self, frame: pd.DataFrame, config: DatasetConfig,
            artifact_dir: Path | None = None,
            include_deep_learning: bool = False) -> AgentResult:
        self.message_log, self.agent_log = [], []
        if not config.has_target or not config.target:
            return AgentResult(
                self.name, "failed",
                errors=[f"{config.name}: dataset không có target, không thể chạy supervised learning."],
            )
        try:
            prepared = prepare_dataset(frame, config)
            self._record("PreprocessingAgent", "success", [])
            model_a = self.model_a.run(prepared)
            model_b = self.model_b.run(prepared)
            self._record_result(model_a)
            self._record_result(model_b)
            consensus = self.consensus.run([model_a, model_b])
            self._record_result(consensus)
            explanation = self.explanation.run(model_b, consensus, prepared.profile)
            evaluation = self.evaluation.run(consensus, prepared)
            self._record_result(explanation)
            self._record_result(evaluation)
            safety = self.safety.run(consensus, explanation, evaluation)
            self._record_result(safety)
            if safety.status != "success":
                return AgentResult(self.name, "failed", {"trace": self.agent_log},
                                   errors=safety.errors)

            outputs = {
                "logistic_regression": model_a.data["output"],
                "random_forest": model_b.data["output"],
            }
            deep_learning = None
            if include_deep_learning:
                deep_learning_result = self.deep_learning_agent.run(prepared)
                self._record_result(deep_learning_result)
                deep_learning = deep_learning_result.data["output"]
                metrics_dl = deep_learning_result.data["metrics"]
            else:
                metrics_dl = None

            metrics = {
                "individual": {
                    name: evaluate_predictions(
                        prepared.y_test, output.test_predictions,
                        output.test_probabilities, output.classes,
                        prepared.profile["task_kind"],
                    )
                    for name, output in outputs.items()
                },
                "soft_voting": evaluation.data,
            }
            if metrics_dl is not None:
                metrics["deep_learning"] = metrics_dl
            if artifact_dir is not None:
                target_dir = Path(artifact_dir) / config.name
                target_dir.mkdir(parents=True, exist_ok=True)
                for name, output in outputs.items():
                    joblib.dump(output.model, target_dir / f"{name}.joblib")
                if deep_learning is not None:
                    import torch
                    torch.save(
                        deep_learning.model.state_dict(),
                        target_dir / "deep_learning_mlp.pt",
                    )
                    joblib.dump(
                        deep_learning.preprocessor,
                        target_dir / "deep_learning_preprocessor.joblib",
                    )
                manifest = {
                    "dataset": config.name,
                    "task": config.task,
                    "target": config.target,
                    "profile": prepared.profile,
                    "metrics": metrics,
                    "deep_learning": None if deep_learning is None else {
                        "validation_score": deep_learning.validation_score,
                        "validation_metric": deep_learning.validation_metric,
                        "best_epoch": deep_learning.best_epoch,
                        "epochs_ran": deep_learning.epochs_ran,
                        "stopped_early": deep_learning.stopped_early,
                        "classes": [str(value) for value in deep_learning.classes],
                    },
                    "preprocessing": {
                        "numeric_columns": prepared.numeric_columns,
                        "categorical_columns": prepared.categorical_columns,
                        "excluded_features": prepared.excluded_features,
                        "constant_features": prepared.constant_features,
                    },
                    "models": {
                        name: {
                            "validation_score": output.validation_score,
                            "validation_metric": output.validation_metric,
                            "classes": [str(value) for value in output.classes],
                        }
                        for name, output in outputs.items()
                    },
                }
                (target_dir / "metrics.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            return AgentResult(
                self.name, "success",
                {
                    "profile": prepared.profile,
                    "models": outputs,
                    "consensus": consensus.data,
                    "explanation": explanation.data,
                    "metrics": metrics,
                    "deep_learning": deep_learning,
                    "safety_review": safety.data,
                    "agent_log": self.agent_log,
                    "message_log": self.message_log,
                },
            )
        except Exception as exc:
            return AgentResult(
                self.name, "failed",
                {"trace": self.agent_log},
                errors=[f"{type(exc).__name__}: {exc}"],
            )

    def _record(self, agent: str, status: str, messages: list[Any]) -> None:
        self.agent_log.append({"agent": agent, "status": status})
        for message in messages:
            self.message_log.append(message.__dict__)

    def _record_result(self, result: AgentResult) -> None:
        self._record(result.agent, result.status, result.messages)
        if result.status != "success":
            raise RuntimeError(f"{result.agent} thất bại: {result.errors}")
