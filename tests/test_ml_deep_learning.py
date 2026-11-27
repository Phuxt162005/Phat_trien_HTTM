import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from project01.data.loader import DatasetConfig
from project01.ml.deep_learning import train_deep_learning_model
from project01.ml.preprocessing import prepare_dataset


def test_mlp_trains_and_returns_valid_probabilities():
    rng = np.random.default_rng(31)
    n = 180
    age = rng.normal(48, 10, n)
    bmi = rng.normal(27, 4, n)
    target = ((age + bmi + rng.normal(0, 5, n)) > 75).astype(int)
    frame = pd.DataFrame({"age": age, "bmi": bmi, "diabetes": target})
    config = DatasetConfig("toy", "toy.csv", "diabetes", "binary_classification")
    prepared = prepare_dataset(frame, config)

    result = train_deep_learning_model(
        prepared, max_epochs=3, patience=2, batch_size=64,
        hidden_layers=(16,), random_state=7,
    )
    assert result.best_epoch >= 1
    assert result.epochs_ran <= 3
    assert result.test_probabilities.shape == (len(prepared.y_test), 2)
    assert np.allclose(result.test_probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert len(result.test_predictions) == len(prepared.y_test)
