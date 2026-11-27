import numpy as np
import pandas as pd

from project01.data.loader import DatasetConfig
from project01.ml.model_benchmark import run_model_benchmark


def test_model_benchmark_compares_models_without_test_access():
    rng = np.random.default_rng(321)
    counts = {0: 100, 1: 25, 2: 55}
    labels = np.concatenate([
        np.full(count, label) for label, count in counts.items()
    ])
    frame = pd.DataFrame({
        "signal": labels + rng.normal(0, 0.8, len(labels)),
        "noise": rng.normal(size=len(labels)),
        "binary": rng.integers(0, 2, len(labels)),
        "target": labels,
    })
    config = DatasetConfig(
        "benchmark_toy", "toy.csv", "target", "multiclass_classification"
    )
    report, model = run_model_benchmark(
        frame, config, n_estimators=6, n_jobs=1
    )

    assert report["candidate_count"] == 11
    assert report["test_partition_accessed"] is False
    assert report["selection_partition"] == "validation_only"
    assert report["selected_model"] in {
        "random_forest", "extra_trees", "hist_gradient_boosting"
    }
    assert "f1_macro" in report["selected_validation_metrics"]
    assert hasattr(model, "predict_proba")
