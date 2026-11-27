from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.config.settings import ARTIFACT_DIR, CONFIG_PATH, DATA_DIR
from project01.data import load_dataset, load_dataset_configs
from project01.ml import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train baseline ML models for labeled diabetes datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=["diabetes01", "diabetes04"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--rf-estimators", type=int, default=120)
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    summary = {}
    for name in args.datasets:
        if name not in configs:
            raise SystemExit(f"Dataset '{name}' không có trong cấu hình.")
        config = configs[name]
        if not config.has_target:
            raise SystemExit(
                f"{name} không có target; không được dùng cho supervised classification."
            )
        print(f"\n=== Training {name} ===")
        frame = load_dataset(config, args.data_dir)
        result = run_experiment(
            frame, config, artifact_dir=args.artifact_dir,
            rf_estimators=args.rf_estimators,
        )
        summary[name] = {
            "profile": result["profile"],
            "metrics": result["metrics"],
        }
        print(json.dumps(summary[name], ensure_ascii=False, indent=2))

    summary_path = args.artifact_dir / "ml_experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
