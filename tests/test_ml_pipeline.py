import numpy as np
import pandas as pd
import pytest

from project01.data.loader import DatasetConfig
from project01.ml.evaluation import evaluate_predictions
from project01.ml.preprocessing import prepare_dataset
from project01.ml.models import soft_voting, train_baseline_models


def make_labeled_frame(n=240):
    rng = np.random.default_rng(17)
    age = rng.normal(50, 12, n)
    bmi = rng.normal(28, 5, n)
    gender = rng.choice(["Female", "Male"], n)
    target = ((age + bmi * 1.5 + rng.normal(0, 8, n)) > 95).astype(int)
    return pd.DataFrame({"age": age, "bmi": bmi, "gender": gender, "diabetes": target})


def test_preprocessing_splits_and_fits_without_test_leakage():
    frame = make_labeled_frame()
    config = DatasetConfig("toy", "toy.csv", "diabetes", "binary_classification")
    prepared = prepare_dataset(frame, config)
    assert len(prepared.X_train) + len(prepared.X_validation) + len(prepared.X_test) == len(frame)
    assert len(prepared.X_test) > 0
    assert prepared.profile["target"] == "diabetes"


def test_unlabeled_dataset_is_rejected_by_supervised_preprocessing():
    frame = pd.DataFrame({"age": [30, 40, 50, 60]})
    config = DatasetConfig(
        "unlabeled", "unlabeled.csv", None, "unlabeled_health_tabular",
        has_target=False,
    )
    with pytest.raises(ValueError, match="không có target"):
        prepare_dataset(frame, config)


def test_baseline_models_soft_voting_and_metrics():
    frame = make_labeled_frame(300)
    config = DatasetConfig("toy", "toy.csv", "diabetes", "binary")
    prepared = prepare_dataset(frame, config)
    outputs = train_baseline_models(prepared, rf_estimators=12, n_jobs=1)
    voting = soft_voting(outputs)
    metrics = evaluate_predictions(
        prepared.y_test, voting["predictions"], voting["probabilities"],
        voting["classes"], "binary",
    )
    assert set(outputs) == {"logistic_regression", "random_forest"}
    assert voting["probabilities"].shape == (len(prepared.y_test), 2)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    assert np.allclose(voting["probabilities"].sum(axis=1), 1.0)
