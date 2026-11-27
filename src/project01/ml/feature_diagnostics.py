from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .preprocessing import prepare_dataset


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _safe_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _numeric_class_profiles(frame: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    for column in numeric_columns:
        per_class = {}
        for label in sorted(pd.Series(target).dropna().unique().tolist()):
            values = pd.to_numeric(frame.loc[target.index[target == label], column], errors="coerce").dropna()
            per_class[str(label)] = {
                "count": int(values.size),
                "mean": _safe_number(values.mean()) if values.size else None,
                "std": _safe_number(values.std(ddof=1)) if values.size > 1 else None,
                "median": _safe_number(values.median()) if values.size else None,
                "q25": _safe_number(values.quantile(0.25)) if values.size else None,
                "q75": _safe_number(values.quantile(0.75)) if values.size else None,
            }
        profiles[column] = per_class
    return profiles


def _pairwise_numeric_effects(frame: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    labels = sorted(pd.Series(target).dropna().unique().tolist())
    for left, right in combinations(labels, 2):
        for column in numeric_columns:
            left_values = pd.to_numeric(frame.loc[target.index[target == left], column], errors="coerce").dropna()
            right_values = pd.to_numeric(frame.loc[target.index[target == right], column], errors="coerce").dropna()
            if len(left_values) < 2 or len(right_values) < 2:
                continue
            pooled_variance = (
                (len(left_values) - 1) * left_values.var(ddof=1)
                + (len(right_values) - 1) * right_values.var(ddof=1)
            ) / (len(left_values) + len(right_values) - 2)
            pooled_std = float(np.sqrt(max(pooled_variance, 0.0)))
            effect = (
                float(left_values.mean() - right_values.mean()) / pooled_std
                if pooled_std > 0 else 0.0
            )
            rows.append({
                "class_pair": [str(left), str(right)],
                "feature": column,
                "standardized_mean_difference": _safe_number(effect),
                "absolute_effect_size": _safe_number(abs(effect)),
            })
    rows.sort(key=lambda item: item["absolute_effect_size"] or 0.0, reverse=True)
    return rows


def _categorical_class_profiles(frame: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    categorical_columns = [
        column for column in frame.columns
        if column not in frame.select_dtypes(include=["number", "bool"]).columns
    ]
    labels = sorted(pd.Series(target).dropna().unique().tolist())
    for column in categorical_columns:
        per_class = {}
        for label in labels:
            values = frame.loc[target.index[target == label], column].fillna("<MISSING>").astype(str)
            counts = values.value_counts(normalize=True).head(5)
            per_class[str(label)] = {
                str(category): float(proportion) for category, proportion in counts.items()
            }
        profiles[column] = per_class
    return profiles



DIABETES04_ORDINAL_FEATURES = {"Age", "Education", "Income", "GenHlth"}
DIABETES04_CONTINUOUS_FEATURES = {"BMI", "MentHlth", "PhysHlth"}


def _classify_feature_types(frame: pd.DataFrame, dataset_name: str) -> dict[str, str]:
    """Classify encoded numeric fields using known dataset semantics where available."""
    result: dict[str, str] = {}
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column]):
            result[column] = "categorical"
        elif dataset_name == "diabetes04" and column in DIABETES04_ORDINAL_FEATURES:
            result[column] = "ordinal"
        elif dataset_name == "diabetes04" and column in DIABETES04_CONTINUOUS_FEATURES:
            result[column] = "continuous"
        else:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            unique = set(values.unique().tolist())
            if len(unique) <= 2 and unique.issubset({0, 1, 0.0, 1.0}):
                result[column] = "binary"
            else:
                result[column] = "continuous"
    return result


def _typed_feature_profiles(
    frame: pd.DataFrame, target: pd.Series, feature_types: dict[str, str]
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    labels = sorted(pd.Series(target).dropna().unique().tolist())
    for column, kind in feature_types.items():
        per_class: dict[str, Any] = {}
        for label in labels:
            values = frame.loc[target.index[target == label], column]
            if kind == "binary":
                numeric = pd.to_numeric(values, errors="coerce").dropna()
                per_class[str(label)] = {
                    "count": int(numeric.size),
                    "positive_rate": _safe_number(numeric.mean()) if numeric.size else None,
                }
            elif kind == "ordinal":
                distribution = values.fillna("<MISSING>").astype(str).value_counts(normalize=True)
                per_class[str(label)] = {
                    "count": int(len(values)),
                    "category_proportions": {
                        str(key): float(value) for key, value in distribution.items()
                    },
                }
            elif kind == "categorical":
                distribution = values.fillna("<MISSING>").astype(str).value_counts(normalize=True).head(10)
                per_class[str(label)] = {
                    "count": int(len(values)),
                    "category_proportions": {
                        str(key): float(value) for key, value in distribution.items()
                    },
                }
            else:
                numeric = pd.to_numeric(values, errors="coerce").dropna()
                per_class[str(label)] = {
                    "count": int(numeric.size),
                    "mean": _safe_number(numeric.mean()) if numeric.size else None,
                    "std": _safe_number(numeric.std(ddof=1)) if numeric.size > 1 else None,
                    "median": _safe_number(numeric.median()) if numeric.size else None,
                    "q25": _safe_number(numeric.quantile(0.25)) if numeric.size else None,
                    "q75": _safe_number(numeric.quantile(0.75)) if numeric.size else None,
                }
        profiles[column] = {"type": kind, "by_class": per_class}
    return profiles

def run_feature_diagnostics(
    frame: pd.DataFrame,
    config,
    n_estimators: int = 120,
    n_jobs: int = 1,
    validation_sample_size: int = 10000,
    permutation_repeats: int = 3,
) -> dict[str, Any]:
    """Diagnose class overlap and validation permutation importance without test access."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Feature diagnostics hiện chỉ hỗ trợ bài toán đa lớp.")

    preprocessor = clone(prepared.preprocessor)
    X_train = _dense(preprocessor.fit_transform(prepared.X_train))
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
        n_jobs=n_jobs,
    )
    estimator.fit(X_train, prepared.y_train.to_numpy())
    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    sample_size = min(int(validation_sample_size), len(prepared.X_validation))
    if sample_size < 1:
        raise ValueError("Validation không có mẫu để phân tích.")
    sample = prepared.X_validation.sample(
        n=sample_size, random_state=42
    ) if sample_size < len(prepared.X_validation) else prepared.X_validation
    sample_y = prepared.y_validation.loc[sample.index]

    permutation = permutation_importance(
        model,
        sample,
        sample_y,
        scoring="f1_macro",
        n_repeats=permutation_repeats,
        random_state=42,
        n_jobs=n_jobs,
    )
    importance = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(
            prepared.X_validation.columns,
            permutation.importances_mean,
            permutation.importances_std,
        )
    ]
    importance.sort(key=lambda item: item["importance_mean"], reverse=True)

    validation_predictions = model.predict(sample)
    confusion_pairs = []
    for actual in sorted(pd.Series(sample_y).unique().tolist()):
        for predicted in sorted(pd.Series(validation_predictions).unique().tolist()):
            if actual == predicted:
                continue
            count = int(((sample_y.to_numpy() == actual) & (validation_predictions == predicted)).sum())
            if count:
                confusion_pairs.append({
                    "actual_class": str(actual),
                    "predicted_class": str(predicted),
                    "count": count,
                })
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    feature_types = _classify_feature_types(prepared.X_train, config.name)
    return {
        "dataset": config.name,
        "feature_types": feature_types,
        "typed_feature_profiles_train": _typed_feature_profiles(
            prepared.X_train, prepared.y_train, feature_types
        ),
        "target": config.target,
        "task": config.task,
        "analysis_partition": "validation_sample_only",
        "validation_sample_size": int(sample_size),
        "validation_total_rows": int(len(prepared.X_validation)),
        "test_partition_accessed": False,
        "class_distribution_train": {
            str(label): int(count)
            for label, count in prepared.y_train.value_counts().items()
        },
        "validation_confusion_pairs_sample": confusion_pairs,
        "validation_permutation_importance_macro_f1": importance,
        "numeric_class_profiles_train": _numeric_class_profiles(
            prepared.X_train, prepared.y_train
        ),
        "numeric_pairwise_effect_sizes_train": _pairwise_numeric_effects(
            prepared.X_train, prepared.y_train
        ),
        "categorical_class_profiles_train": _categorical_class_profiles(
            prepared.X_train, prepared.y_train
        ),
        "caution": (
            "Permutation importance measures predictive contribution on this validation "
            "sample, not causality. Standardized mean differences describe marginal "
            "numeric separation and do not establish clinical relevance. Test data is "
            "not accessed."
        ),
    }
