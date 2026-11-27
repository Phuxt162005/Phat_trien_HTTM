from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import AgentResult, BaseAgent


class PatientDataAgent(BaseAgent):
    """Audits tabular data quality; it does not make a medical diagnosis."""

    def __init__(self) -> None:
        super().__init__("PatientDataAgent")

    def validate_frame(self, df: pd.DataFrame, target: str | None = None) -> AgentResult:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return AgentResult(self.name, "failed", errors=["Dữ liệu đầu vào phải là DataFrame không rỗng."])

        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        missing_by_column = df.isna().sum()
        constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
        numeric = df.select_dtypes(include=[np.number])
        infinite_by_column = {
            str(c): int(np.isinf(numeric[c].to_numpy()).sum())
            for c in numeric.columns
            if np.isinf(numeric[c].to_numpy()).any()
        }
        missing_target = bool(target and target not in df.columns)
        target_distribution: dict[str, int] | None = None
        if target and not missing_target:
            target_distribution = {
                str(k): int(v) for k, v in df[target].value_counts(dropna=False).items()
            }

        report: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "missing_cells": int(df.isna().sum().sum()),
            "missing_by_column": {
                str(k): int(v) for k, v in missing_by_column[missing_by_column > 0].items()
            },
            "missing_percent_by_column": {
                str(k): round(float(v) / len(df) * 100, 4)
                for k, v in missing_by_column[missing_by_column > 0].items()
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant_columns,
            "duplicate_columns": [str(c) for c in duplicate_columns],
            "infinite_values_by_column": infinite_by_column,
            "target": target,
            "target_exists": (not missing_target) if target else None,
            "target_distribution": target_distribution,
        }
        errors: list[str] = []
        if missing_target:
            errors.append(f"Không tìm thấy cột mục tiêu '{target}'.")
        if duplicate_columns:
            errors.append(f"Tên cột bị trùng: {duplicate_columns}")
        if infinite_by_column:
            errors.append(f"Có giá trị vô cực trong các cột: {list(infinite_by_column)}")
        status = "failed" if errors else ("warning" if report["missing_cells"] or report["duplicate_rows"] else "passed")
        return AgentResult(self.name, status, report, errors=errors)
