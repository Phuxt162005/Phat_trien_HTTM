import numpy as np
import pandas as pd

from project01.agents.ml_agents import MLCoordinatorAgent
from project01.data.loader import DatasetConfig


def make_frame(n=180):
    rng = np.random.default_rng(23)
    age = rng.normal(50, 12, n)
    bmi = rng.normal(28, 5, n)
    gender = rng.choice(["Female", "Male"], n)
    target = ((age + bmi * 1.4 + rng.normal(0, 7, n)) > 90).astype(int)
    return pd.DataFrame({
        "age": age, "bmi": bmi, "gender": gender, "diabetes": target
    })


def test_ml_coordinator_runs_agent_flow_and_safety_review(tmp_path):
    config = DatasetConfig(
        "toy", "toy.csv", "diabetes", "binary_classification"
    )
    coordinator = MLCoordinatorAgent(rf_estimators=8, n_jobs=1)
    result = coordinator.run(make_frame(), config, artifact_dir=tmp_path)

    assert result.status == "success", result.errors
    assert result.data["safety_review"]["valid_output"] is True
    assert result.data["consensus"]["probabilities"].shape[1] == 2
    assert [item["agent"] for item in result.data["agent_log"]] == [
        "PreprocessingAgent", "ModelAAgent", "ModelBAgent",
        "ConsensusAgent", "ExplanationAgent", "EvaluationAgent",
        "SafetyReviewAgent",
    ]
    assert (tmp_path / "toy" / "logistic_regression.joblib").is_file()
    assert (tmp_path / "toy" / "random_forest.joblib").is_file()
    assert (tmp_path / "toy" / "metrics.json").is_file()
    assert len(result.data["message_log"]) >= 5


def test_ml_coordinator_rejects_unlabeled_dataset():
    config = DatasetConfig(
        "unlabeled", "unlabeled.csv", None,
        "unlabeled_health_tabular", has_target=False,
    )
    result = MLCoordinatorAgent(rf_estimators=4).run(
        pd.DataFrame({"age": [20, 30, 40]}), config
    )
    assert result.status == "failed"
    assert "không có target" in result.errors[0]



def test_ml_coordinator_supports_multiclass_dataset():
    rng = np.random.default_rng(37)
    n = 240
    age = rng.normal(50, 12, n)
    bmi = rng.normal(28, 5, n)
    target = np.digitize(age + bmi + rng.normal(0, 5, n), [70, 85])
    frame = pd.DataFrame({"age": age, "bmi": bmi, "Diabetes_012": target})
    config = DatasetConfig(
        "toy_multiclass", "toy.csv", "Diabetes_012", "multiclass_classification"
    )
    result = MLCoordinatorAgent(rf_estimators=6, n_jobs=1).run(frame, config)
    assert result.status == "success", result.errors
    assert result.data["metrics"]["soft_voting"]["confusion_matrix"] is not None
    assert "roc_auc_ovr_macro" in result.data["metrics"]["soft_voting"]
    assert len(result.data["consensus"]["classes"]) == 3
