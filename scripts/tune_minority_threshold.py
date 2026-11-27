from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from project01.config.settings import CONFIG_PATH, DATA_DIR, ARTIFACT_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml.threshold_tuning import run_minority_threshold_tuning


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune minority-class decision threshold on validation data."
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
    report, model = run_minority_threshold_tuning(
        frame, config, n_estimators=args.rf_estimators, n_jobs=args.n_jobs
    )
    output_dir = args.artifact_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "minority_threshold_tuning.json"
    model_path = output_dir / "random_forest_minority_threshold.joblib"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(model, model_path)

    minority_metrics = report["final_test_metrics"]["classification_report"][
        report["minority_class"]
    ]
    print(json.dumps({
        "dataset": report["dataset"],
        "minority_class": report["minority_class"],
        "selected_threshold": report["selected_threshold"],
        "validation_minority_f1": report["selected_validation_metrics"]
            ["classification_report"][report["minority_class"]]["f1-score"],
        "test_macro_f1": report["final_test_metrics"]["f1_macro"],
        "test_minority_precision": minority_metrics["precision"],
        "test_minority_recall": minority_metrics["recall"],
        "test_minority_f1": minority_metrics["f1-score"],
        "report_path": str(report_path),
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
