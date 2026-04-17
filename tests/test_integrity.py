import numpy as np

from cryptoflow.integrity import aa_test_check, build_srm_alert, srm_check
from cryptoflow.reporting import build_integrity_report


def test_srm_not_significant_on_balanced_split():
    res = srm_check(
        observed_counts={"control": 25000, "treatment": 24980},
        expected_weights={"control": 0.5, "treatment": 0.5},
    )
    assert res.significant is False


def test_srm_significant_on_mismatch():
    res = srm_check(
        observed_counts={"control": 25000, "treatment": 23000},
        expected_weights={"control": 0.5, "treatment": 0.5},
    )
    assert res.significant is True
    alert = build_srm_alert(res, experiment_id="exp_fee")
    assert alert is not None
    assert alert.check_name == "srm_guard"


def test_aa_requires_minimum_sample():
    try:
        aa_test_check(np.array([1.0]), np.array([1.0]))
        assert False, "must raise for too small AA sample"
    except ValueError:
        assert True


def test_integrity_report_contains_srm_payload():
    report = build_integrity_report(
        experiment_id="exp_fee",
        control_count=1000,
        treatment_count=1000,
    )
    assert "srm" in report
    assert "alerts" in report
    assert report["srm"]["significant"] is False
