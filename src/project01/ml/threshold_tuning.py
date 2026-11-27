from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

from .evaluation import evaluate_predictions
from .preprocessing import prepare_dataset


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _apply_minority_factor(probabilities, classes, minority_label, factor):
    """Increase one class score, then keep argmax competition across every class."""
    probabilities = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Kích thước xác suất không khớp với danh sách classes.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Ma trận xác suất chứa giá trị không hợp lệ.")
    if factor < 1.0:
        raise ValueError("Hệ số điều chỉnh phải >= 1.")
    if minority_label not in classes:
        raise ValueError(f"Không tìm thấy lớp thiểu số {minority_label!r} trong model.classes_.")
    adjusted = probabilities.copy()
    adjusted[:, classes.index(minority_label)] *= float(factor)
    # Do not convert this into one-vs-rest thresholding: all classes still compete.
    return np.asarray(classes)[np.argmax(adjusted, axis=1)]


def _class_recall(metrics, label):
    return float(metrics["classification_report"][str(label)]["recall"])


def run_minority_threshold_tuning(
    frame,
    config,
    n_estimators: int = 120,
    n_jobs: int = 1,
    factors: list[float] | None = None,
):
    """Tune a minority score multiplier on validation; do not consume the test set."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Điều chỉnh hệ số lớp thiểu số hiện chỉ hỗ trợ bài toán đa lớp.")

    counts = prepared.y_train.value_counts()
    minority_label = counts.idxmin()
    classes = sorted(prepared.y_train.unique().tolist())
    if factors is None:
        factors = [round(float(value), 2) for value in np.arange(1.0, 6.01, 0.25)]
    if not factors or any(not np.isfinite(value) or value < 1.0 for value in factors):
        raise ValueError("Các hệ số điều chỉnh phải là số hữu hạn và >= 1.")

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
    other_classes = [label for label in validation_classes if label != minority_label]
    baseline_other_recall = {
        str(label): _class_recall(baseline_metrics, label) for label in other_classes
    }

    candidates = []
    for factor in factors:
        predictions = _apply_minority_factor(
            validation_probabilities, validation_classes, minority_label, float(factor)
        )
        metrics = evaluate_predictions(
            prepared.y_validation, predictions, validation_probabilities,
            validation_classes, prepared.profile["task_kind"],
        )
        report = metrics["classification_report"]
        minority_metrics = report[str(minority_label)]
        other_recall = {
            str(label): float(report[str(label)]["recall"]) for label in other_classes
        }
        preserves_other_classes = all(
            other_recall[str(label)] >= baseline_other_recall[str(label)] * 0.75
            for label in other_classes
        )
        preserves_macro_f1 = metrics["f1_macro"] >= baseline_metrics["f1_macro"] - 0.03
        candidates.append({
            "factor": float(factor),
            "metrics": metrics,
            "minority_precision": float(minority_metrics["precision"]),
            "minority_recall": float(minority_metrics["recall"]),
            "minority_f1": float(minority_metrics["f1-score"]),
            "other_class_recall": other_recall,
            "meets_preservation_constraints": bool(
                preserves_other_classes and preserves_macro_f1
            ),
        })

    feasible = [item for item in candidates if item["meets_preservation_constraints"]]
    selection_pool = feasible if feasible else candidates
    if feasible:
        selection_objective = (
            "minority_f1_then_macro_f1_then_other_class_recall_then_precision"
        )
        winner = max(
            selection_pool,
            key=lambda item: (
                item["minority_f1"],
                item["metrics"]["f1_macro"],
                min(item["other_class_recall"].values(), default=0.0),
                item["minority_precision"],
            ),
        )
    else:
        selection_objective = (
            "fallback_macro_f1_then_minority_f1_then_other_class_recall"
        )
        winner = max(
            selection_pool,
            key=lambda item: (
                item["metrics"]["f1_macro"],
                item["minority_f1"],
                min(item["other_class_recall"].values(), default=0.0),
            ),
        )

    # Fit the deployable estimator on train+validation. The held-out test split is
    # intentionally not read here; it has already been inspected in prior runs.
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

    report = {
        "dataset": config.name,
        "task": config.task,
        "minority_class": str(minority_label),
        "selected_factor": winner["factor"],
        "selection_objective": selection_objective,
        "preservation_constraints": {
            "macro_f1_drop_max": 0.03,
            "other_class_recall_minimum_ratio": 0.75,
            "baseline_other_class_recall": baseline_other_recall,
            "feasible_candidate_count": len(feasible),
            "candidate_count": len(candidates),
        },
        "baseline_validation_metrics_argmax": baseline_metrics,
        "selected_validation_metrics": winner["metrics"],
        "selected_validation_other_class_recall": winner["other_class_recall"],
        "validation_factor_candidates": candidates,
        "test_evaluation_status": (
            "not_evaluated_in_this_run; keep test data untouched for a final "
            "evaluation after the decision rule and model are frozen"
        ),
        "profile": prepared.profile,
        "caution": (
            "The multiplier changes class-score competition; it does not improve "
            "probability calibration or establish clinical validity. Review minority "
            "precision, false positives, macro F1 and recall for every class."
        ),
    }
    return report, final_model
