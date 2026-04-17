"""
Lightweight reporting helpers to attach data-integrity diagnostics to summaries.
"""
from __future__ import annotations

import numpy as np

from cryptoflow.integrity import aa_test_check, build_aa_alert, build_srm_alert, srm_check


def build_integrity_report(
    *,
    experiment_id: str,
    control_count: int,
    treatment_count: int,
    expected_split: tuple[float, float] = (0.5, 0.5),
    aa_control: np.ndarray | None = None,
    aa_treatment: np.ndarray | None = None,
) -> dict:
    """
    Return structured integrity checks for API/reporting layers.
    """
    srm = srm_check(
        observed_counts={"control": control_count, "treatment": treatment_count},
        expected_weights={"control": expected_split[0], "treatment": expected_split[1]},
    )
    alerts = []
    srm_alert = build_srm_alert(srm, experiment_id=experiment_id)
    if srm_alert:
        alerts.append(srm_alert.to_dict())

    aa_payload = None
    if aa_control is not None and aa_treatment is not None:
        aa = aa_test_check(aa_control, aa_treatment)
        aa_payload = {
            "p_value": aa.p_value,
            "significant": aa.significant,
            "effect": aa.effect,
        }
        aa_alert = build_aa_alert(aa, experiment_id=experiment_id)
        if aa_alert:
            alerts.append(aa_alert.to_dict())

    return {
        "srm": {
            "p_value": srm.p_value,
            "significant": srm.significant,
            "observed_total": srm.observed_total,
        },
        "aa_test": aa_payload,
        "alerts": alerts,
    }
