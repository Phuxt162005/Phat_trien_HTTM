import numpy as np
import pandas as pd

from project01.data.loader import DatasetConfig
from project01.ml.feature_diagnostics import run_feature_diagnostics


def test_feature_diagnostics_uses_validation_and_does_not_touch_test():
    rng = np.random.default_rng(23)
    n_per_class = 90
    labels = np.repeat([0, 1, 2], n_per_class)
    frame = pd.DataFrame({
        "signal": np.concatenate([
            rng.normal(0.0, 1.0, n_per_class),
            rng.normal(1.0, 1.0, n_per_class),
            rng.normal(2.0, 1.0, n_per_class),
        ]),
        "noise": rng.normal(size=len(labels)),
        "category": rng.choice(["A", "B", "C"], size=len(labels)),
        "target": labels,
    })
    config = DatasetConfig(
        "toy_multiclass", "toy.csv", "target", "multiclass_classification"
    )

    report = run_feature_diagnostics(
        frame,
        config,
        n_estimators=8,
        n_jobs=1,
        validation_sample_size=40,
        permutation_repeats=2,
    )

    assert report["analysis_partition"] == "validation_sample_only"
    assert report["test_partition_accessed"] is False
    assert report["validation_sample_size"] == 40
    assert report["validation_permutation_importance_macro_f1"]
    assert report["numeric_pairwise_effect_sizes_train"]
    assert "category" in report["categorical_class_profiles_train"]
