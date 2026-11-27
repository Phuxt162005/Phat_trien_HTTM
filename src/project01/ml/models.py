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


def _fit_and_evaluate(name: str, estimator, prepared: PreparedData) -> ModelOutput:
    train_model = Pipeline([
        ("preprocess", clone(prepared.preprocessor)),
        ("model", estimator),
    ])
    train_model.fit(prepared.X_train, prepared.y_train)
    val_probabilities = train_model.predict_proba(prepared.X_validation)
    val_predictions = train_model.predict(prepared.X_validation)
    classes = list(train_model.named_steps["model"].classes_)
    score, metric = _validation_score(
        prepared.y_validation, val_predictions, val_probabilities,
        prepared.profile["task_kind"], classes,
    )

    final_model = Pipeline([
        ("preprocess", clone(prepared.preprocessor)),
        ("model", clone(estimator)),
    ])
    final_model.fit(prepared.X_train_validation, prepared.y_train_validation)
    test_probabilities = final_model.predict_proba(prepared.X_test)
    test_predictions = final_model.predict(prepared.X_test)
    return ModelOutput(
        name=name,
        model=final_model,
        validation_score=score,
        validation_metric=metric,
        classes=list(final_model.named_steps["model"].classes_),
        test_predictions=np.asarray(test_predictions),
        test_probabilities=np.asarray(test_probabilities),
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
