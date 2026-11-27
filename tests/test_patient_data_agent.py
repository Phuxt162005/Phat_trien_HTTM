import pandas as pd

from project01.agents.patient_data_agent import PatientDataAgent


def test_quality_report_detects_missing_and_duplicates():
    df = pd.DataFrame({"feature": [1, 1, None], "target": [0, 0, 1]})
    result = PatientDataAgent().validate_frame(df, target="target")
    assert result.status == "warning"
    assert result.data["missing_cells"] == 1
    assert result.data["duplicate_rows"] == 1
    assert result.data["target_distribution"] == {"0": 2, "1": 1}


def test_missing_target_fails():
    df = pd.DataFrame({"feature": [1, 2]})
    result = PatientDataAgent().validate_frame(df, target="target")
    assert result.status == "failed"


def test_unlabeled_dataset_is_valid_and_has_no_target_distribution():
    df = pd.DataFrame({"Age": [40, 55], "BMI": [23.1, 29.4]})
    result = PatientDataAgent().validate_frame(df, target=None)
    assert result.status == "passed"
    assert result.data["target"] is None
    assert result.data["target_exists"] is None
    assert result.data["target_distribution"] is None


def test_infinite_values_are_reported_as_failure():
    df = pd.DataFrame({"feature": [1.0, float("inf")]})
    result = PatientDataAgent().validate_frame(df)
    assert result.status == "failed"
    assert result.data["infinite_values_by_column"]["feature"] == 1
