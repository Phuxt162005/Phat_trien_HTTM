from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .evaluation import evaluate_predictions
from .model_benchmark import apply_class_score_multiplier, _dense
from .models import _class_counts, _random_oversample
from .preprocessing import prepare_dataset


def _recall(metrics: dict[str, Any], label) -> float:
    return float(metrics["classification_report"][str(label)]["recall"])


def run_classwise_probability_tuning(
    frame,
    config,
    factors: list[float] | None = None,
    max_macro_f1_drop: float = 0.02,
    min_other_class_recall_ratio: float = 0.90,
    max_iter: int = 120,
):
    """Tune minority score multiplier on validation for selected HGB/50% strategy."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Class-wise probability tuning chỉ hỗ trợ bài toán đa lớp.")

    counts = _class_counts(prepared.y_train.to_numpy())
    minority_label = min(counts, key=counts.get)
    if factors is None:
        factors = [round(float(x), 2) for x in np.arange(1.0, 12.01, 0.25)]
    if not factors or any(not np.isfinite(x) or x < 1.0 for x in factors):
        raise ValueError("Các hệ số phải hữu hạn và >= 1.")
    if max_macro_f1_drop < 0 or not 0 < min_other_class_recall_ratio <= 1:
        raise ValueError("Các ràng buộc bảo toàn không hợp lệ.")

    estimator_params = {
        "max_iter": max(80, int(max_iter)),
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": 42,
    }

    # Fit only on train. Validation is used only to choose the multiplier.
    validation_preprocessor = clone(prepared.preprocessor)
    X_train = _dense(validation_preprocessor.fit_transform(prepared.X_train))
    X_validation = _dense(validation_preprocessor.transform(prepared.X_validation))
    y_train = prepared.y_train.to_numpy()
    y_validation = prepared.y_validation.to_numpy()
    X_fit, y_fit = _random_oversample(X_train, y_train, minimum_ratio=0.50)
    validation_estimator = HistGradientBoostingClassifier(**estimator_params)
    validation_estimator.fit(X_fit, y_fit)
    probabilities = validation_estimator.predict_proba(X_validation)
    classes = list(validation_estimator.classes_)

    baseline_predictions = validation_estimator.predict(X_validation)
    baseline_metrics = evaluate_predictions(
        y_validation, baseline_predictions, probabilities, classes, "multiclass"
    )
    other_classes = [label for label in classes if label != minority_label]
    baseline_other_recall = {
        str(label): _recall(baseline_metrics, label) for label in other_classes
    }

    candidates = []
    for factor in factors:
        predictions = apply_class_score_multiplier(
            probabilities, classes, minority_label, float(factor)
        )
        metrics = evaluate_predictions(
            y_validation, predictions, probabilities, classes, "multiclass"
        )
        minority = metrics["classification_report"][str(minority_label)]
        other_recall = {
            str(label): _recall(metrics, label) for label in other_classes
        }
        recall_ok = all(
            other_recall[str(label)] >= baseline_other_recall[str(label)]
            * min_other_class_recall_ratio
            for label in other_classes
        )
        macro_ok = (
            metrics["f1_macro"] >= baseline_metrics["f1_macro"] - max_macro_f1_drop
        )
        candidates.append({
            "factor": float(factor),
            "metrics": metrics,
            "minority_precision": float(minority["precision"]),
            "minority_recall": float(minority["recall"]),
            "minority_f1": float(minority["f1-score"]),
            "other_class_recall": other_recall,
            "preserves_constraints": bool(recall_ok and macro_ok),
        })

    feasible = [item for item in candidates if item["preserves_constraints"]]
    if feasible:
        pool = feasible
        objective = "minority_f1_then_macro_f1_then_precision"
        winner = max(pool, key=lambda item: (
            item["minority_f1"],
            item["metrics"]["f1_macro"],
            item["minority_precision"],
        ))
    else:
        pool = candidates
        objective = "fallback_macro_f1_then_minority_f1_then_precision"
        winner = max(pool, key=lambda item: (
            item["metrics"]["f1_macro"],
            item["minority_f1"],
            item["minority_precision"],
        ))

    # Refit the chosen estimator on train+validation only; never access test.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = _dense(
        final_preprocessor.fit_transform(prepared.X_train_validation)
    )
    y_train_validation = prepared.y_train_validation.to_numpy()
    X_final, y_final = _random_oversample(
        X_train_validation, y_train_validation, minimum_ratio=0.50
    )
    final_estimator = HistGradientBoostingClassifier(**estimator_params)
    final_estimator.fit(X_final, y_final)
    final_model = Pipeline([
        ("preprocess", final_preprocessor),
        ("model", final_estimator),
    ])

    report = {
        "dataset": config.name,
        "model": "hist_gradient_boosting",
        "training_strategy": "random_over_50pct",
        "minority_class": str(minority_label),
        "selected_factor": winner["factor"],
        "selection_objective": objective,
        "constraints": {
            "max_macro_f1_drop": max_macro_f1_drop,
            "minimum_other_class_recall_ratio": min_other_class_recall_ratio,
            "baseline_other_class_recall": baseline_other_recall,
            "feasible_candidate_count": len(feasible),
            "candidate_count": len(candidates),
        },
        "baseline_validation_metrics": baseline_metrics,
        "selected_validation_metrics": winner["metrics"],
        "selected_validation_minority_metrics": {
            "precision": winner["minority_precision"],
            "recall": winner["minority_recall"],
            "f1": winner["minority_f1"],
        },
        "selected_validation_other_class_recall": winner["other_class_recall"],
        "candidates": candidates,
        "selection_partition": "validation_only",
        "test_partition_accessed": False,
        "test_evaluation_status": (
            "not_evaluated; the test split was previously inspected in exploratory "
            "experiments and is not treated as a pristine holdout"
        ),
        "train_class_counts_before": counts,
        "train_validation_class_counts_after": _class_counts(y_final),
        "profile": prepared.profile,
        "caution": (
            "The class score multiplier changes the decision rule, not probability "
            "calibration. Review precision and false positives for every class. "
            "This is a research prototype, not a clinical diagnostic model."
        ),
    }
    return report, final_model


def save_classwise_tuning_artifacts(report: dict[str, Any], model, output_dir):
    output_dir = Path(output_dir) / report["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classwise_probability_tuning.json"
    model_path = output_dir / "hist_gradient_boosting_classwise.joblib"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(model, model_path)
    return report_path, model_path
