from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.config.settings import CONFIG_PATH, DATA_DIR, ARTIFACT_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml.model_benchmark import run_model_benchmark, save_benchmark_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare tree ensemble models using train/validation only."
    )
    parser.add_argument("--dataset", default="diabetes04")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--estimators", type=int, default=120)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    if args.dataset not in configs:
        raise SystemExit(f"Dataset '{args.dataset}' không có trong cấu hình.")
    config = configs[args.dataset]
    if not config.has_target or not config.target:
        raise SystemExit(f"{args.dataset} không có target hợp lệ.")

    frame = load_dataset(config, args.data_dir)
    report, model = run_model_benchmark(
        frame, config, n_estimators=args.estimators, n_jobs=args.n_jobs
    )
    report_path, model_path = save_benchmark_artifacts(
        report, model, args.artifact_dir
    )
    print(json.dumps({
        "dataset": report["dataset"],
        "candidate_count": report["candidate_count"],
        "selected_model": report["selected_model"],
        "selected_strategy": report["selected_strategy"],
        "validation_macro_f1": report["selected_validation_metrics"]["f1_macro"],
        "minority_class": report["minority_class"],
        "minority_precision": report["selected_validation_minority_metrics"]["precision"],
        "minority_recall": report["selected_validation_minority_metrics"]["recall"],
        "minority_f1": report["selected_validation_minority_metrics"]["f1"],
        "test_partition_accessed": report["test_partition_accessed"],
        "report_path": str(report_path),
        "model_path": str(model_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
