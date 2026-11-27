import numpy as np
import pandas as pd
import pytest

from project01.data.loader import DatasetConfig
from project01.ml.model_benchmark import apply_class_score_multiplier
from project01.ml.classwise_tuning import run_classwise_probability_tuning


def test_class_score_multiplier_keeps_all_classes_competing():
    probabilities = np.array([
        [0.40, 0.10, 0.50],
        [0.45, 0.20, 0.35],
    ])
    predictions = apply_class_score_multiplier(
        probabilities, [0, 1, 2], target_class=1, factor=3.0
    )
    assert predictions.tolist() == [2, 1]


def test_factor_one_matches_argmax():
    probabilities = np.array([
        [0.70, 0.10, 0.20],
        [0.10, 0.60, 0.30],
        [0.20, 0.10, 0.70],
    ])
    predictions = apply_class_score_multiplier(
        probabilities, [0, 1, 2], target_class=1, factor=1.0
    )
    assert predictions.tolist() == [0, 1, 2]


def test_multiplier_rejects_invalid_factor_or_class():
    probabilities = np.array([[0.7, 0.2, 0.1]])
    with pytest.raises(ValueError):
        apply_class_score_multiplier(probabilities, [0, 1, 2], 1, 0.5)
    with pytest.raises(ValueError):
        apply_class_score_multiplier(probabilities, [0, 1, 2], 4, 2.0)


def test_classwise_tuning_is_validation_only_and_preserves_three_classes():
    rng = np.random.default_rng(90)
    counts = {0: 100, 1: 25, 2: 55}
    labels = np.concatenate([
        np.full(count, label) for label, count in counts.items()
    ])
    frame = pd.DataFrame({
        "signal": labels + rng.normal(0, 0.8, len(labels)),
        "noise": rng.normal(size=len(labels)),
        "target": labels,
    })
    config = DatasetConfig(
        "classwise_toy", "toy.csv", "target", "multiclass_classification"
    )
    report, model = run_classwise_probability_tuning(
        frame,
        config,
        factors=[1.0, 1.5, 2.0],
        max_iter=12,
    )
    assert report["selection_partition"] == "validation_only"
    assert report["test_partition_accessed"] is False
    assert report["constraints"]["candidate_count"] == 3
    assert set(model.classes_) == {0, 1, 2}
