from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

from .evaluation import evaluate_predictions
from .preprocessing import prepare_dataset


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _apply_minority_threshold(probabilities, classes, minority_label, threshold):
    classes = list(classes)
    if minority_label not in classes:
        raise ValueError(f"Không tìm thấy lớp thiểu số {minority_label!r} trong model.classes_.")
    minority_index = classes.index(minority_label)
    other_indices = [index for index in range(len(classes)) if index != minority_index]
    if not other_indices:
        raise ValueError("Threshold tuning cần ít nhất hai lớp.")
    predictions = np.asarray(classes)[np.argmax(probabilities[:, other_indices], axis=1)]
    selected = probabilities[:, minority_index] >= threshold
    predictions[selected] = minority_label
    return predictions


def run_minority_threshold_tuning(
    frame,
    config,
    n_estimators: int = 120,
    n_jobs: int = 1,
    thresholds: list[float] | None = None,
):
    """Tune a minority-class probability threshold on validation, then test once."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Threshold tuning hiện chỉ hỗ trợ bài toán đa lớp.")

    classes = sorted(prepared.y_train.unique().tolist())
    minority_label = min(
        prepared.y_train.value_counts().to_dict(),
        key=prepared.y_train.value_counts().to_dict().get,
    )
    if thresholds is None:
        thresholds = [round(value, 2) for value in np.arange(0.01, 0.51, 0.01)]
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("Các threshold phải nằm trong khoảng (0, 1).")

    common = {
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "max_samples": 0.8,
        "random_state": 42,
        "n_jobs": n_jobs,
    }

    validation_preprocessor = clone(prepared.preprocessor)
    X_train = _dense(validation_preprocessor.fit_transform(prepared.X_train))
    X_validation = _dense(validation_preprocessor.transform(prepared.X_validation))
    validation_estimator = RandomForestClassifier(**common)
    validation_estimator.fit(X_train, prepared.y_train.to_numpy())
    validation_probabilities = validation_estimator.predict_proba(X_validation)
    validation_classes = list(validation_estimator.classes_)

    baseline_predictions = validation_estimator.predict(X_validation)
    baseline_metrics = evaluate_predictions(
        prepared.y_validation, baseline_predictions, validation_probabilities,
        validation_classes, prepared.profile["task_kind"],
    )

    candidates = []
    for threshold in thresholds:
        predictions = _apply_minority_threshold(
            validation_probabilities, validation_classes, minority_label, threshold
        )
        metrics = evaluate_predictions(
            prepared.y_validation, predictions, validation_probabilities,
            validation_classes, prepared.profile["task_kind"],
        )
        minority_metrics = metrics["classification_report"][str(minority_label)]
        candidates.append({
            "threshold": float(threshold),
            "metrics": metrics,
            "minority_precision": float(minority_metrics["precision"]),
            "minority_recall": float(minority_metrics["recall"]),
            "minority_f1": float(minority_metrics["f1-score"]),
        })

    # Optimize minority F1 first; use macro F1 and precision to break ties.
    winner = max(
        candidates,
        key=lambda item: (
            item["minority_f1"],
            item["metrics"]["f1_macro"],
            item["minority_precision"],
        ),
    )
    selected_threshold = winner["threshold"]

    # Refit on train+validation and evaluate the held-out test partition once.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = _dense(
        final_preprocessor.fit_transform(prepared.X_train_validation)
    )
    final_estimator = RandomForestClassifier(**common)
    final_estimator.fit(X_train_validation, prepared.y_train_validation.to_numpy())
    final_model = Pipeline([
        ("preprocess", final_preprocessor),
        ("model", final_estimator),
    ])
    test_probabilities = final_model.predict_proba(prepared.X_test)
    test_predictions = _apply_minority_threshold(
        test_probabilities, list(final_estimator.classes_),
        minority_label, selected_threshold,
    )
    test_metrics = evaluate_predictions(
        prepared.y_test, test_predictions, test_probabilities,
        final_estimator.classes_, prepared.profile["task_kind"],
    )

    report = {
        "dataset": config.name,
        "task": config.task,
        "minority_class": str(minority_label),
        "selection_objective": "minority_f1_then_macro_f1_then_minority_precision",
        "baseline_validation_metrics_argmax": baseline_metrics,
        "selected_threshold": selected_threshold,
        "selected_validation_metrics": winner["metrics"],
        "validation_threshold_candidates": candidates,
        "final_test_metrics": test_metrics,
        "test_evaluation_count": 1,
        "profile": prepared.profile,
        "caution": (
            "Threshold tuning changes the decision rule, not the underlying model "
            "probability quality. Review minority precision and false positives."
        ),
    }
    return report, final_model
