from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from project01.config.settings import CONFIG_PATH, DATA_DIR, ARTIFACT_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml.evaluation import evaluate_predictions
from project01.ml.models import _class_counts, _random_oversample
from project01.ml.preprocessing import prepare_dataset


def _build_candidates(n_estimators: int, n_jobs: int):
    common = {
        "n_estimators": n_estimators,
        "min_samples_leaf": 2,
        "bootstrap": True,
        "max_samples": 0.8,
        "random_state": 42,
        "n_jobs": n_jobs,
    }
    candidates = [
        ("baseline", None, None),
        ("class_weight_balanced", "balanced_subsample", None),
        ("random_over_20pct", None, 0.20),
        ("random_over_50pct", None, 0.50),
        ("random_over_100pct", None, 1.00),
    ]
    return candidates, common


def run_comparison(frame, config, n_estimators: int = 120, n_jobs: int = 1):
    """Select an imbalance strategy on validation; evaluate the winner on test once."""
    prepared = prepare_dataset(frame, config)
    if prepared.profile["task_kind"] != "multiclass":
        raise ValueError("Thí nghiệm này hiện được thiết kế cho dataset đa lớp.")

    candidates, common = _build_candidates(n_estimators, n_jobs)
    validation_preprocessor = clone(prepared.preprocessor)
    X_train = np.asarray(validation_preprocessor.fit_transform(prepared.X_train))
    X_validation = np.asarray(validation_preprocessor.transform(prepared.X_validation))
    y_train = prepared.y_train.to_numpy()
    y_validation = prepared.y_validation.to_numpy()
    before_counts = _class_counts(y_train)
    minority_label = min(before_counts, key=before_counts.get)
    results = []

    for name, class_weight, oversample_ratio in candidates:
        if oversample_ratio is None:
            X_fit, y_fit = X_train, y_train
        else:
            X_fit, y_fit = _random_oversample(
                X_train, y_train, minimum_ratio=oversample_ratio
            )

        estimator = RandomForestClassifier(**common, class_weight=class_weight)
        estimator.fit(X_fit, y_fit)
        probabilities = estimator.predict_proba(X_validation)
        predictions = estimator.predict(X_validation)
        metrics = evaluate_predictions(
            y_validation, predictions, probabilities,
            estimator.classes_, prepared.profile["task_kind"],
        )
        minority_report = metrics["classification_report"].get(str(minority_label), {})
        results.append({
            "strategy": name,
            "validation_metrics": metrics,
            "validation_minority_class": str(minority_label),
            "validation_minority_recall": float(minority_report.get("recall", 0.0)),
            "train_class_counts_before": before_counts,
            "train_class_counts_after": _class_counts(y_fit),
        })

    # Macro F1 is primary; minority-class recall breaks ties.
    winner = max(
        results,
        key=lambda item: (
            item["validation_metrics"]["f1_macro"],
            item["validation_minority_recall"],
        ),
    )
    _, class_weight, oversample_ratio = next(
        item for item in candidates if item[0] == winner["strategy"]
    )

    # Refit on train+validation. Test is untouched until this one final evaluation.
    final_preprocessor = clone(prepared.preprocessor)
    X_train_validation = np.asarray(
        final_preprocessor.fit_transform(prepared.X_train_validation)
    )
    y_train_validation = prepared.y_train_validation.to_numpy()
    if oversample_ratio is not None:
        X_final, y_final = _random_oversample(
            X_train_validation, y_train_validation,
            minimum_ratio=oversample_ratio,
        )
    else:
        X_final, y_final = X_train_validation, y_train_validation

    final_estimator = RandomForestClassifier(**common, class_weight=class_weight)
    final_estimator.fit(X_final, y_final)
    final_model = Pipeline([
        ("preprocess", final_preprocessor),
        ("model", final_estimator),
    ])
    test_probabilities = final_model.predict_proba(prepared.X_test)
    test_predictions = final_model.predict(prepared.X_test)
    test_metrics = evaluate_predictions(
        prepared.y_test, test_predictions, test_probabilities,
        final_estimator.classes_, prepared.profile["task_kind"],
    )

    report = {
        "dataset": config.name,
        "task": config.task,
        "selection_metric": "validation_macro_f1_then_minority_recall",
        "minority_class": str(minority_label),
        "selected_strategy": winner["strategy"],
        "selected_validation_metrics": winner["validation_metrics"],
        "candidates": results,
        "final_test_metrics": test_metrics,
        "test_evaluation_count": 1,
        "train_validation_class_counts_before": _class_counts(y_train_validation),
        "train_validation_class_counts_after": _class_counts(y_final),
        "profile": prepared.profile,
    }
    return report, final_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiclass imbalance strategies using validation only."
    )
    parser.add_argument("--dataset", default="diabetes04")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    if args.dataset not in configs:
        raise SystemExit(f"Dataset '{args.dataset}' không có trong cấu hình.")
    config = configs[args.dataset]
    if not config.has_target or not config.target:
        raise SystemExit(f"{args.dataset} không có target hợp lệ.")

    frame = load_dataset(config, args.data_dir)
    report, model = run_comparison(
        frame, config, n_estimators=args.rf_estimators, n_jobs=args.n_jobs
    )
    output_dir = args.artifact_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "imbalance_comparison.json"
    model_path = output_dir / "random_forest_selected_imbalance.joblib"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(model, model_path)

    print(json.dumps({
        "dataset": report["dataset"],
        "selected_strategy": report["selected_strategy"],
        "validation_macro_f1": report["selected_validation_metrics"]["f1_macro"],
        "test_macro_f1": report["final_test_metrics"]["f1_macro"],
        "test_minority_recall": report["final_test_metrics"]["classification_report"]
            .get(report["minority_class"], {}).get("recall", 0.0),
        "report_path": str(report_path),
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
