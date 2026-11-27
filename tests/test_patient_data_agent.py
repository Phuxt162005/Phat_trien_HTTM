import pandas as pd
from project01.agents.patient_data_agent import PatientDataAgent


def test_quality_report_detects_missing_and_duplicates():
    df = pd.DataFrame({"feature": [1, 1, None], "target": [0, 0, 1]})
    result = PatientDataAgent().validate_frame(df, target="target")
    assert result.status == "success"
    assert result.data["missing_cells"] == 1
    assert result.data["duplicate_rows"] == 1


def test_missing_target_fails():
    df = pd.DataFrame({"feature": [1, 2]})
    result = PatientDataAgent().validate_frame(df, target="target")
    assert result.status == "error"
