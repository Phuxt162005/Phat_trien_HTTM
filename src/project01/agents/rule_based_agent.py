from __future__ import annotations

from .contracts import AgentResult, BaseAgent


class RuleBasedAgent(BaseAgent):
    """Runs technical data-quality rules, not clinical treatment rules."""

    def __init__(self) -> None:
        super().__init__("RuleBasedAgent")

    def review_data_quality(self, quality_report: dict) -> AgentResult:
        findings = []
        if quality_report.get("missing_cells", 0):
            findings.append({"level": "warning", "code": "MISSING_VALUES", "message": "Dữ liệu có giá trị thiếu; cần xử lý trong pipeline và chỉ fit phép biến đổi trên tập train."})
        if quality_report.get("duplicate_rows", 0):
            findings.append({"level": "warning", "code": "DUPLICATE_ROWS", "message": "Phát hiện bản ghi trùng; cần đánh giá trước khi loại bỏ, không tự động xóa."})
        if quality_report.get("constant_columns"):
            findings.append({"level": "info", "code": "CONSTANT_COLUMNS", "message": "Có cột hằng không mang thông tin phân biệt."})
        if quality_report.get("infinite_values_by_column"):
            findings.append({"level": "error", "code": "INFINITE_VALUES", "message": "Có giá trị vô cực cần xử lý trước khi huấn luyện."})
        status = "failed" if any(x["level"] == "error" for x in findings) else ("warning" if any(x["level"] == "warning" for x in findings) else "passed")
        return AgentResult(self.name, status, {"findings": findings, "passed": status != "failed"})
