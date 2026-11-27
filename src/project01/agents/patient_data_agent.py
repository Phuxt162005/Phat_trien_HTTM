from __future__ import annotations

from typing import Any
import pandas as pd

from .contracts import AgentResult, BaseAgent


class PatientDataAgent(BaseAgent):
    """Validates tabular inputs; it does not make a medical diagnosis."""

    def __init__(self) -> None:
        super().__init__("PatientDataAgent")

    def validate_frame(self, df: pd.DataFrame, target: str | None = None) -> AgentResult:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return AgentResult(self.name, "error", errors=["Dữ liệu đầu vào phải là DataFrame không rỗng."])
        missing_target = bool(target and target not in df.columns)
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        missing_by_column = df.isna().sum()
        constant_columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        report: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": int(df.isna().sum().sum()),
            "missing_by_column": {str(k): int(v) for k, v in missing_by_column[missing_by_column > 0].items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": [str(c) for c in constant_columns],
            "duplicate_columns": [str(c) for c in duplicate_columns],
            "target": target,
            "target_exists": not missing_target if target else None,
        }
        errors: list[str] = []
        if missing_target:
            errors.append(f"Không tìm thấy cột mục tiêu '{target}'.")
        if duplicate_columns:
            errors.append(f"Tên cột bị trùng: {duplicate_columns}")
        return AgentResult(self.name, "error" if errors else "success", report, errors=errors)
