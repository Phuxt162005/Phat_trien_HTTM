from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.config.settings import CONFIG_PATH, DATA_DIR, ARTIFACT_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml.feature_diagnostics import run_feature_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze feature separation and validation errors for multiclass data."
    )
    parser.add_argument("--dataset", default="diabetes04")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--validation-sample-size", type=int, default=10000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    if args.dataset not in configs:
        raise SystemExit(f"Dataset '{args.dataset}' không có trong cấu hình.")
    config = configs[args.dataset]
    if not config.has_target or not config.target:
        raise SystemExit(f"{args.dataset} không có target hợp lệ.")

    frame = load_dataset(config, args.data_dir)
    report = run_feature_diagnostics(
        frame,
        config,
        n_estimators=args.rf_estimators,
        n_jobs=args.n_jobs,
        validation_sample_size=args.validation_sample_size,
        permutation_repeats=args.permutation_repeats,
    )
    output_dir = args.artifact_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "feature_diagnostics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "dataset": report["dataset"],
        "validation_sample_size": report["validation_sample_size"],
        "test_partition_accessed": report["test_partition_accessed"],
        "top_predictive_features": report[
            "validation_permutation_importance_macro_f1"
        ][:10],
        "largest_validation_confusion_pairs": report[
            "validation_confusion_pairs_sample"
        ][:10],
        "report_path": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
