from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.loader import DatasetConfig


RANDOM_STATE = 42


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_train_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_train_validation: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_features: list[str]
    constant_features: list[str]
    profile: dict[str, Any]


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("Không còn feature hợp lệ sau khi loại target/constant columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_dataset(
    frame: pd.DataFrame,
    config: DatasetConfig,
    test_size: float = 0.15,
    validation_size: float = 0.1765,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """Split first; fit all imputers/scalers/encoders on training data only."""
    if not config.has_target or not config.target:
        raise ValueError(f"{config.name}: dataset không có target, không thể chạy supervised learning.")
    if config.target not in frame.columns:
        raise ValueError(f"{config.name}: thiếu target '{config.target}'.")
    if frame[config.target].isna().any():
        raise ValueError(f"{config.name}: target chứa giá trị thiếu.")
    y = frame[config.target].copy()
    if y.nunique() < 2:
        raise ValueError(f"{config.name}: target phải có ít nhất hai lớp.")

    excluded = [column for column in config.excluded_features if column in frame.columns]
    X = frame.drop(columns=[config.target, *excluded]).copy()
    constant = [column for column in X.columns if X[column].nunique(dropna=False) <= 1]
    if constant:
        X = X.drop(columns=constant)
    if X.shape[1] == 0:
        raise ValueError(f"{config.name}: không còn feature sau khi loại cột không hợp lệ.")

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_validation,
    )

    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    profile = {
        "dataset": config.name,
        "task": config.task,
        "target": config.target,
        "classes": sorted(y.dropna().unique().tolist()),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "rows": int(len(frame)),
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
    }
    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_train_validation=X_train_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_train_validation=y_train_validation,
        y_test=y_test,
        preprocessor=preprocessor,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_features=excluded,
        constant_features=constant,
        profile=profile,
    )
