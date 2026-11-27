from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import PreparedData


RANDOM_STATE = 42


@dataclass
class ModelOutput:
    name: str
    model: Pipeline
    validation_score: float
    validation_metric: str
    classes: list[Any]
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    train_class_counts_before: dict[str, int]
    train_class_counts_after: dict[str, int]


def _validation_score(y_true, predictions, probabilities, task: str, classes: list[Any]) -> tuple[float, str]:
    from sklearn.metrics import f1_score, roc_auc_score

    if task == "binary":
        positive_label = classes[-1]
        positive_index = classes.index(positive_label)
        score = roc_auc_score(
            (np.asarray(y_true) == positive_label).astype(int),
            probabilities[:, positive_index],
        )
        return float(score), "roc_auc"
    return float(f1_score(y_true, predictions, average="macro", zero_division=0)), "f1_macro"


def _class_counts(values) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(label): int(count) for label, count in zip(unique, counts)}


def _random_oversample(X: np.ndarray, y: np.ndarray, minimum_ratio: float = 0.20):
    """Oversample minority classes on the training partition only."""
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    target_count = max(1, int(np.ceil(int(counts.max()) * minimum_ratio)))
    rng = np.random.default_rng(RANDOM_STATE)
    X_parts, y_parts = [X], [y]
    for label, count in zip(unique, counts):
        if count < target_count:
            indices = np.flatnonzero(y == label)
            sampled = rng.choice(indices, size=target_count - int(count), replace=True)
            X_parts.append(X[sampled])
            y_parts.append(y[sampled])
    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    order = rng.permutation(len(y_out))
    return X_out[order], y_out[order]


def _fit_and_evaluate(
    name: str, estimator, prepared: PreparedData,
    oversample_minority: bool = False,
    minimum_class_ratio: float = 0.20,
) -> ModelOutput:
    # Fit preprocessing on training data only.
    validation_preprocessor = clone(prepared.preprocessor)
    X_train = validation_preprocessor.fit_transform(prepared.X_train)
    X_validation = validation_preprocessor.transform(prepared.X_validation)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_validation, "toarray"):
        X_validation = X_validation.toarray()
    X_train, X_validation = np.asarray(X_train), np.asarray(X_validation)
    y_train = prepared.y_train.to_numpy()
    before_counts = _class_counts(y_train)

    if oversample_minority:
        X_fit, y_fit = _random_oversample(X_train, y_train, minimum_class_ratio)
    else:
        X_fit, y_fit = X_train, y_train
    after_counts = _class_counts(y_fit)

    validation_estimator = clone(estimator)
    validation_estimator.fit(X_fit, y_fit)
    val_probabilities = validation_estimator.predict_proba(X_validation)
    val_predictions = validation_estimator.predict(X_validation)
    classes = list(validation_estimator.classes_)
    score, metric = _validation_score(
        prepared.y_validation, val_predictions, val_probabilities,
        prepared.profile["task_kind"], classes,
    )

    # Refit on train+validation; never resample or fit on test data.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = final_preprocessor.fit_transform(prepared.X_train_validation)
    if hasattr(X_train_validation, "toarray"):
        X_train_validation = X_train_validation.toarray()
    X_train_validation = np.asarray(X_train_validation)
    y_train_validation = prepared.y_train_validation.to_numpy()
    if oversample_minority:
        X_final, y_final = _random_oversample(
            X_train_validation, y_train_validation, minimum_class_ratio
        )
    else:
        X_final, y_final = X_train_validation, y_train_validation

    final_estimator = clone(estimator)
    final_estimator.fit(X_final, y_final)
    final_model = Pipeline([
        ("preprocess", final_preprocessor),
        ("model", final_estimator),
    ])
    test_probabilities = final_model.predict_proba(prepared.X_test)
    test_predictions = final_model.predict(prepared.X_test)
    return ModelOutput(
        name=name,
        model=final_model,
        validation_score=score,
        validation_metric=metric,
        classes=list(final_estimator.classes_),
        test_predictions=np.asarray(test_predictions),
        test_probabilities=np.asarray(test_probabilities),
        train_class_counts_before=before_counts,
        train_class_counts_after=after_counts,
    )


def train_baseline_models(
    prepared: PreparedData,
    rf_estimators: int = 120,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
) -> dict[str, ModelOutput]:
    """Train Logistic Regression and Random Forest; validation selects, test evaluates."""
    logistic = LogisticRegression(
        max_iter=1500, class_weight="balanced", random_state=random_state
    )
    forest = RandomForestClassifier(
        n_estimators=rf_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.8,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return {
        "logistic_regression": _fit_and_evaluate("Logistic Regression", logistic, prepared),
        "random_forest": _fit_and_evaluate("Random Forest", forest, prepared),
    }


def soft_voting(outputs: dict[str, ModelOutput]) -> dict[str, Any]:
    if len(outputs) < 2:
        raise ValueError("Soft voting cần ít nhất hai mô hình.")
    class_orders = [tuple(output.classes) for output in outputs.values()]
    if len(set(class_orders)) != 1:
        raise ValueError("Các mô hình không có cùng thứ tự lớp; không thể soft voting.")
    probabilities = np.mean(
        [output.test_probabilities for output in outputs.values()], axis=0
    )
    classes = np.asarray(next(iter(outputs.values())).classes)
    return {
        "classes": classes.tolist(),
        "probabilities": probabilities,
        "predictions": classes[np.argmax(probabilities, axis=1)],
        "participating_models": list(outputs),
    }
