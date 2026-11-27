from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.config.settings import CONFIG_PATH, DATA_DIR, ARTIFACT_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml.classwise_tuning import (
    run_classwise_probability_tuning,
    save_classwise_tuning_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune a multiclass class-score multiplier on validation only."
    )
    parser.add_argument("--dataset", default="diabetes04")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-other-recall-ratio", type=float, default=0.90)
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    if args.dataset not in configs:
        raise SystemExit(f"Dataset '{args.dataset}' không có trong cấu hình.")
    config = configs[args.dataset]
    if not config.has_target or not config.target:
        raise SystemExit(f"{args.dataset} không có target hợp lệ.")

    frame = load_dataset(config, args.data_dir)
    report, model = run_classwise_probability_tuning(
        frame,
        config,
        max_iter=args.max_iter,
        max_macro_f1_drop=args.max_macro_f1_drop,
        min_other_class_recall_ratio=args.min_other_recall_ratio,
    )
    report_path, model_path = save_classwise_tuning_artifacts(
        report, model, args.artifact_dir
    )
    print(json.dumps({
        "dataset": report["dataset"],
        "model": report["model"],
        "training_strategy": report["training_strategy"],
        "minority_class": report["minority_class"],
        "selected_factor": report["selected_factor"],
        "selection_objective": report["selection_objective"],
        "validation_macro_f1_baseline": report["baseline_validation_metrics"]["f1_macro"],
        "validation_macro_f1_adjusted": report["selected_validation_metrics"]["f1_macro"],
        "minority_precision": report["selected_validation_minority_metrics"]["precision"],
        "minority_recall": report["selected_validation_minority_metrics"]["recall"],
        "minority_f1": report["selected_validation_minority_metrics"]["f1"],
        "feasible_candidate_count": report["constraints"]["feasible_candidate_count"],
        "test_partition_accessed": report["test_partition_accessed"],
        "report_path": str(report_path),
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
