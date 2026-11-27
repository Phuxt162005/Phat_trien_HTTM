from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    file: str
    target: str | None
    task: str
    excluded_features: tuple[str, ...] = ()
    role: str = "external"
    has_target: bool = True


def load_dataset_configs(path: Path) -> dict[str, DatasetConfig]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình dataset: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result: dict[str, DatasetConfig] = {}
    for name, item in raw.get("datasets", {}).items():
        has_target = bool(item.get("has_target", item.get("target") is not None))
        target_value = item.get("target")
        target = str(target_value) if target_value is not None else None
        if has_target and not target:
            raise ValueError(f"{name}: has_target=true nhưng chưa khai báo target.")
        if not has_target and target:
            raise ValueError(f"{name}: has_target=false nhưng target vẫn được khai báo.")
        result[name] = DatasetConfig(
            name=name,
            file=str(item["file"]),
            target=target,
            task=str(item["task"]),
            excluded_features=tuple(item.get("excluded_features", [])),
            role=str(item.get("role", "external")),
            has_target=has_target,
        )
    return result


def load_dataset(config: DatasetConfig, data_dir: Path) -> pd.DataFrame:
    path = data_dir / config.file
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu dataset {config.name}: {path}. "
            "Hãy đặt file CSV gốc vào data/raw/; không thay đổi file gốc."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{config.name}: tên cột bị trùng: {duplicated}")
    if config.has_target:
        if not config.target or config.target not in df.columns:
            raise ValueError(
                f"{config.name}: không tìm thấy target '{config.target}'. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )
        if df[config.target].isna().any():
            raise ValueError(f"{config.name}: target '{config.target}' có giá trị thiếu.")
    return df
