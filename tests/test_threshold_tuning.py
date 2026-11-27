import numpy as np
import pytest

from project01.ml.threshold_tuning import _apply_minority_factor


def test_minority_factor_keeps_all_classes_in_argmax_competition():
    probabilities = np.array([
        [0.40, 0.10, 0.50],  # class 2 remains the strongest after adjustment
        [0.45, 0.20, 0.35],  # class 1 wins after its score is multiplied
    ])
    predictions = _apply_minority_factor(
        probabilities, classes=[0, 1, 2], minority_label=1, factor=3.0
    )
    assert predictions.tolist() == [2, 1]


def test_factor_one_matches_original_argmax():
    probabilities = np.array([
        [0.70, 0.10, 0.20],
        [0.10, 0.60, 0.30],
        [0.20, 0.10, 0.70],
    ])
    predictions = _apply_minority_factor(
        probabilities, classes=[0, 1, 2], minority_label=1, factor=1.0
    )
    assert predictions.tolist() == [0, 1, 2]


def test_minority_factor_rejects_invalid_values():
    probabilities = np.array([[0.7, 0.2, 0.1]])
    with pytest.raises(ValueError, match=">= 1"):
        _apply_minority_factor(probabilities, [0, 1, 2], 1, 0.8)
    with pytest.raises(ValueError, match="Không tìm thấy"):
        _apply_minority_factor(probabilities, [0, 1, 2], 3, 2.0)
