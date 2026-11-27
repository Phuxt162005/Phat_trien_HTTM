from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, probabilities, classes, task: str) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Kích thước ma trận xác suất không khớp số mẫu/lớp.")
    if len(y_pred) != len(y_true):
        raise ValueError("Số dự đoán không khớp số nhãn thực tế.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Xác suất dự đoán chứa giá trị không hợp lệ.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Tổng xác suất mỗi mẫu phải bằng 1.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
    }
    if task == "binary":
        positive_label = classes[-1]
        positive_index = classes.index(positive_label)
        metrics["roc_auc"] = float(roc_auc_score(
            (y_true == positive_label).astype(int), probabilities[:, positive_index]
        ))
    else:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(
            y_true, probabilities, labels=classes, multi_class="ovr", average="macro"
        ))
    return metrics
