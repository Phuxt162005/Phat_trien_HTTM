from pathlib import Path

import pandas as pd
import pytest
import yaml

from project01.data.loader import DatasetConfig, load_dataset, load_dataset_configs


def test_load_dataset_without_target(tmp_path: Path):
    pd.DataFrame({"Age": [40, 55], "BMI": [23.1, 29.4]}).to_csv(tmp_path / "diabetes02.csv", index=False)
    config = DatasetConfig(
        name="diabetes02",
        file="diabetes02.csv",
        target=None,
        task="unlabeled_health_tabular",
        has_target=False,
    )
    loaded = load_dataset(config, tmp_path)
    assert loaded.shape == (2, 2)
    assert "Age" in loaded.columns


def test_load_dataset_requires_target_when_configured(tmp_path: Path):
    pd.DataFrame({"feature": [1, 2]}).to_csv(tmp_path / "data.csv", index=False)
    config = DatasetConfig(name="labeled", file="data.csv", target="diabetes", task="binary")
    with pytest.raises(ValueError, match="target"):
        load_dataset(config, tmp_path)


def test_config_marks_diabetes02_as_unlabeled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "datasets": {
                "diabetes02": {
                    "file": "diabetes02.csv",
                    "target": None,
                    "task": "unlabeled_health_tabular",
                    "has_target": False,
                }
            }
        }),
        encoding="utf-8",
    )
    config = load_dataset_configs(config_path)["diabetes02"]
    assert config.target is None
    assert config.has_target is False
