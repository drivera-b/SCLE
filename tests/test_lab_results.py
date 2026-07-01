from src.dataset import data_dir
from src.lab_results import lab_csv_template, parse_lab_csv


def test_template_parses_with_expected_standard_units():
    result = parse_lab_csv(lab_csv_template())
    assert result.ok
    assert result.values["systolic_bp"] == 128.0
    assert result.values["hba1c"] == 5.5
    assert not result.warnings
    assert (data_dir() / "lab_results_template.csv").read_bytes() == lab_csv_template()


def test_lab_import_rejects_wrong_units_and_multiple_rows():
    wrong_units = parse_lab_csv(b"fasting_glucose,fasting_glucose_unit\n5.4,mmol/L\n")
    assert not wrong_units.ok
    assert any("mg/dL" in error for error in wrong_units.errors)

    multiple = parse_lab_csv(b"hba1c\n5.4\n5.8\n")
    assert not multiple.ok
    assert any("exactly one row" in error for error in multiple.errors)


def test_lab_import_accepts_aliases_and_warns_when_units_missing():
    result = parse_lab_csv(b"SBP,A1C,body mass index\n121,5.3,23.5\n")
    assert result.ok
    assert result.values == {"systolic_bp": 121.0, "hba1c": 5.3, "bmi": 23.5}
    assert result.warnings
