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



def test_multiclass_random_forest_oversamples_training_partition_only():
    rng = np.random.default_rng(91)
    counts = {0: 180, 1: 8, 2: 35}
    labels = np.concatenate([
        np.full(count, label) for label, count in counts.items()
    ])
    frame = pd.DataFrame({
        "feature_a": rng.normal(size=len(labels)),
        "feature_b": rng.normal(size=len(labels)),
        "Diabetes_012": labels,
    })
    config = DatasetConfig(
        "imbalanced_toy", "toy.csv", "Diabetes_012", "multiclass_classification"
    )
    result = MLCoordinatorAgent(rf_estimators=8, n_jobs=1).run(frame, config)
    assert result.status == "success", result.errors

    rf_output = result.data["models"]["random_forest"]
    before = rf_output.train_class_counts_before
    after = rf_output.train_class_counts_after
    assert before["1"] < after["1"]
    assert after["1"] >= int(np.ceil(max(before.values()) * 0.20))
    assert sum(after.values()) > sum(before.values())

    # The test partition remains the original held-out data; no resampling is applied to it.
    test_rows = result.data["profile"]["test_rows"]
    assert test_rows > 0
    assert len(rf_output.test_predictions) == test_rows


def test_imbalance_comparison_selects_on_validation_and_evaluates_test_once():
    from project01.ml.imbalance_comparison import run_comparison

    rng = np.random.default_rng(103)
    counts = {0: 150, 1: 12, 2: 45}
    labels = np.concatenate([
        np.full(count, label) for label, count in counts.items()
    ])
    frame = pd.DataFrame({
        "feature_a": rng.normal(size=len(labels)),
        "feature_b": rng.normal(size=len(labels)),
        "Diabetes_012": labels,
    })
    config = DatasetConfig(
        "imbalance_comparison_toy", "toy.csv", "Diabetes_012",
        "multiclass_classification",
    )
    report, model = run_comparison(frame, config, n_estimators=5, n_jobs=1)

    assert report["selected_strategy"] in {
        "baseline", "class_weight_balanced", "random_over_20pct",
        "random_over_50pct", "random_over_100pct",
    }
    assert len(report["candidates"]) == 5
    assert report["test_evaluation_count"] == 1
    assert "f1_macro" in report["selected_validation_metrics"]
    assert "f1_macro" in report["final_test_metrics"]
    assert hasattr(model, "predict_proba")

def test_minority_threshold_tuning_uses_validation_and_evaluates_test_once():
    from project01.ml.threshold_tuning import run_minority_threshold_tuning

    rng = np.random.default_rng(207)
    counts = {0: 150, 1: 18, 2: 55}
    labels = np.concatenate([
        np.full(count, label) for label, count in counts.items()
    ])
    frame = pd.DataFrame({
        "feature_a": rng.normal(size=len(labels)),
        "feature_b": rng.normal(size=len(labels)),
        "Diabetes_012": labels,
    })
    config = DatasetConfig(
        "threshold_tuning_toy", "toy.csv", "Diabetes_012",
        "multiclass_classification",
    )
    report, model = run_minority_threshold_tuning(
        frame, config, n_estimators=5, n_jobs=1,
        thresholds=[0.10, 0.20, 0.30],
    )

    assert report["selected_threshold"] in [0.10, 0.20, 0.30]
    assert len(report["validation_threshold_candidates"]) == 3
    assert report["test_evaluation_count"] == 1
    assert "f1_macro" in report["selected_validation_metrics"]
    assert "f1_macro" in report["final_test_metrics"]
    assert hasattr(model, "predict_proba")

