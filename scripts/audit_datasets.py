from __future__ import annotations

import argparse
import json
from pathlib import Path

from project01.agents import CoordinatorAgent
from project01.config.settings import CONFIG_PATH, DATA_DIR, ROOT_DIR
from project01.data import load_dataset_configs


OFFICIAL_DATASETS = ("diabetes01", "diabetes02", "diabetes04")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the official diabetes datasets without modifying source CSV files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "data" / "manifests" / "original_dataset_audit.json")
    args = parser.parse_args()

    configs = load_dataset_configs(args.config)
    required = set(OFFICIAL_DATASETS)
    if not required.issubset(configs):
        raise SystemExit(f"Cấu hình thiếu dataset chính thức: {sorted(required - set(configs))}")

    coordinator = CoordinatorAgent()
    results = {}
    failed = False
    for name in OFFICIAL_DATASETS:
        result = coordinator.audit_dataset(configs[name], args.data_dir)
        results[name] = result.to_dict()
        print(f"[{result.status.upper()}] {name}")
        if result.status == "failed":
            failed = True
            for error in result.errors:
                print("  -", error)
        elif result.data:
            print("  shape:", result.data["shape"], "target:", result.data["target"], "has_target:", result.data["has_target"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Audit report:", args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
