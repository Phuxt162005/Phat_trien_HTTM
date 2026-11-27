from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from ..data.loader import DatasetConfig
from .evaluation import evaluate_predictions
from .models import soft_voting, train_baseline_models
from .preprocessing import prepare_dataset


def run_experiment(
    frame,
    config: DatasetConfig,
    artifact_dir: Path | None = None,
    rf_estimators: int = 120,
) -> dict[str, Any]:
    """Run supervised ML for one labeled dataset and optionally persist artifacts."""
    prepared = prepare_dataset(frame, config)
    models = train_baseline_models(prepared, rf_estimators=rf_estimators)
    voting = soft_voting(models)

    individual_metrics = {
        name: evaluate_predictions(
            prepared.y_test, output.test_predictions, output.test_probabilities,
            output.classes, prepared.profile["task_kind"],
        )
        for name, output in models.items()
    }
    voting_metrics = evaluate_predictions(
        prepared.y_test, voting["predictions"], voting["probabilities"],
        voting["classes"], prepared.profile["task_kind"],
    )

    result = {
        "profile": prepared.profile,
        "preprocessing": {
            "numeric_columns": prepared.numeric_columns,
            "categorical_columns": prepared.categorical_columns,
            "excluded_features": prepared.excluded_features,
            "constant_features": prepared.constant_features,
        },
        "models": models,
        "soft_voting": voting,
        "metrics": {"individual": individual_metrics, "soft_voting": voting_metrics},
    }

    if artifact_dir is not None:
        target_dir = Path(artifact_dir) / config.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, output in models.items():
            joblib.dump(output.model, target_dir / f"{name}.joblib")
        manifest = {
            "dataset": config.name,
            "task": config.task,
            "target": config.target,
            "profile": prepared.profile,
            "metrics": result["metrics"],
            "preprocessing": result["preprocessing"],
            "models": {
                name: {
                    "validation_score": output.validation_score,
                    "validation_metric": output.validation_metric,
                    "classes": [str(value) for value in output.classes],
                }
                for name, output in models.items()
            },
        }
        (target_dir / "metrics.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
