from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_predictions
from .models import _class_counts, _random_oversample
from .preprocessing import prepare_dataset


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _candidate_estimators(n_estimators: int, n_jobs: int):
    common_tree = {
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": n_jobs,
    }
    return {
        "random_forest": RandomForestClassifier(
            **common_tree, bootstrap=True, max_samples=0.8
        ),
        "extra_trees": ExtraTreesClassifier(
            **common_tree, bootstrap=False
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=max(80, n_estimators),
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def _strategies():
    return [
        ("baseline", None, None),
        ("class_weight_balanced", "balanced", None),
        ("random_over_20pct", None, 0.20),
        ("random_over_50pct", None, 0.50),
    ]


def run_model_benchmark(
    frame,
    config,
    n_estimators: int = 120,
    n_jobs: int = 1,
):
    """Compare tree ensembles using train/validation only; never access test."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Model benchmark hiện chỉ hỗ trợ bài toán đa lớp.")

    preprocessor = clone(prepared.preprocessor)
    X_train = _dense(preprocessor.fit_transform(prepared.X_train))
    X_validation = _dense(preprocessor.transform(prepared.X_validation))
    y_train = prepared.y_train.to_numpy()
    y_validation = prepared.y_validation.to_numpy()
    class_counts = _class_counts(y_train)
    minority_label = min(class_counts, key=class_counts.get)

    estimators = _candidate_estimators(n_estimators, n_jobs)
    strategies = _strategies()
    candidates: list[dict[str, Any]] = []

    for model_name, estimator_template in estimators.items():
        for strategy, class_weight, oversample_ratio in strategies:
            if class_weight is not None and model_name == "hist_gradient_boosting":
                # HistGradientBoosting does not expose class_weight consistently
                # across the project's supported sklearn range.
                continue
            estimator = clone(estimator_template)
            if class_weight is not None:
                estimator.set_params(class_weight=class_weight)

            if oversample_ratio is None:
                X_fit, y_fit = X_train, y_train
            else:
                X_fit, y_fit = _random_oversample(
                    X_train, y_train, minimum_ratio=oversample_ratio
                )

            estimator.fit(X_fit, y_fit)
            probabilities = estimator.predict_proba(X_validation)
            predictions = estimator.predict(X_validation)
            metrics = evaluate_predictions(
                y_validation, predictions, probabilities,
                estimator.classes_, prepared.profile["task_kind"],
            )
            minority_metrics = metrics["classification_report"][str(minority_label)]
            candidates.append({
                "model": model_name,
                "strategy": strategy,
                "validation_metrics": metrics,
                "minority_class": str(minority_label),
                "minority_precision": float(minority_metrics["precision"]),
                "minority_recall": float(minority_metrics["recall"]),
                "minority_f1": float(minority_metrics["f1-score"]),
                "train_class_counts_before": class_counts,
                "train_class_counts_after": _class_counts(y_fit),
            })

    # Macro F1 is the primary selection criterion; minority F1 and recall
    # are tie-breakers, not independent objectives that can hide poor precision.
    winner = max(
        candidates,
        key=lambda item: (
            item["validation_metrics"]["f1_macro"],
            item["minority_f1"],
            item["minority_recall"],
        ),
    )
    winner_estimator = clone(estimators[winner["model"]])
    selected_strategy = next(
        item for item in strategies if item[0] == winner["strategy"]
    )
    _, class_weight, oversample_ratio = selected_strategy
    if class_weight is not None:
        winner_estimator.set_params(class_weight=class_weight)

    # Refit selected pipeline on train+validation only. Do not inspect test.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = _dense(
        final_preprocessor.fit_transform(prepared.X_train_validation)
    )
    y_train_validation = prepared.y_train_validation.to_numpy()
    if oversample_ratio is None:
        X_final, y_final = X_train_validation, y_train_validation
    else:
        X_final, y_final = _random_oversample(
            X_train_validation, y_train_validation,
            minimum_ratio=oversample_ratio,
        )
    winner_estimator.fit(X_final, y_final)
    final_model = Pipeline([
        ("preprocess", final_preprocessor),
        ("model", winner_estimator),
    ])

    report = {
        "dataset": config.name,
        "task": config.task,
        "selection_partition": "validation_only",
        "selection_metric": "macro_f1_then_minority_f1_then_minority_recall",
        "minority_class": str(minority_label),
        "selected_model": winner["model"],
        "selected_strategy": winner["strategy"],
        "selected_validation_metrics": winner["validation_metrics"],
        "selected_validation_minority_metrics": {
            "precision": winner["minority_precision"],
            "recall": winner["minority_recall"],
            "f1": winner["minority_f1"],
        },
        "candidates": candidates,
        "candidate_count": len(candidates),
        "test_partition_accessed": False,
        "test_evaluation_status": (
            "not_evaluated; the test split was previously inspected during "
            "exploratory experiments and is not treated as a pristine holdout"
        ),
        "train_class_counts_before": class_counts,
        "train_validation_class_counts_after": _class_counts(y_final),
        "profile": prepared.profile,
        "caution": (
            "All model/strategy selection uses validation only. The selected model "
            "is refit on train+validation. No test metrics are produced. This is a "
            "model-development comparison, not clinical validation."
        ),
    }
    return report, final_model


def save_benchmark_artifacts(report: dict[str, Any], model, output_dir):
    output_dir = output_dir / report["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_benchmark.json"
    model_path = output_dir / "selected_benchmark_model.joblib"
    import json
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(model, model_path)
    return report_path, model_path
