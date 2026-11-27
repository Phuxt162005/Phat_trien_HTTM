from __future__ import annotations

from .contracts import AgentResult, BaseAgent


class RuleBasedAgent(BaseAgent):
    """Runs technical/business validation rules, not clinical treatment rules."""

    def __init__(self) -> None:
        super().__init__("RuleBasedAgent")

    def review_data_quality(self, quality_report: dict) -> AgentResult:
        findings = []
        if quality_report.get("missing_cells", 0):
            findings.append({"level": "warning", "code": "MISSING_VALUES", "message": "Dữ liệu có giá trị thiếu; cần xử lý trong pipeline và chỉ fit phép biến đổi trên tập train."})
        if quality_report.get("duplicate_rows", 0):
            findings.append({"level": "info", "code": "DUPLICATE_ROWS", "message": "Phát hiện bản ghi trùng; cần xem xét trước khi loại bỏ."})
        if quality_report.get("constant_columns"):
            findings.append({"level": "info", "code": "CONSTANT_COLUMNS", "message": "Có cột hằng không mang thông tin phân biệt."})
        return AgentResult(self.name, "success", {"findings": findings, "passed": True})
