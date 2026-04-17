"""
Data integrity checks for experimentation pipelines.

Includes:
- AA testing sanity checks (same-distribution groups must not diverge)
- Sample Ratio Mismatch (SRM) guardrails
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy import stats as spstats


Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class IntegrityAlert:
    check_name: str
    severity: Severity
    experiment_id: str
    reason: str
    suggested_action: str
    p_value: float | None = None
    details: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AAResult:
    p_value: float
    significant: bool
    effect: float
    check_name: str = "aa_test"


@dataclass(frozen=True)
class SRMResult:
    p_value: float
    significant: bool
    observed_total: int
    expected_total: float
    check_name: str = "srm_guard"


def aa_test_check(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> AAResult:
    """
    Run Welch t-test on two equivalent groups in an A/A setup.

    significant=True indicates integrity risk in assignment/logging/instrumentation.
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("AA test requires at least 2 observations per group")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    _, p_value = spstats.ttest_ind(treatment, control, equal_var=False)
    effect = float(np.mean(treatment) - np.mean(control))
    p = float(p_value)
    return AAResult(p_value=p, significant=bool(p <= alpha), effect=effect)


def srm_check(
    observed_counts: dict[str, int],
    expected_weights: dict[str, float],
    alpha: float = 0.001,
) -> SRMResult:
    """
    Pearson chi-square SRM check.

    Example:
    observed_counts = {"control": 25000, "treatment": 24500}
    expected_weights = {"control": 0.5, "treatment": 0.5}
    """
    if set(observed_counts) != set(expected_weights):
        raise ValueError("observed_counts and expected_weights must contain identical group keys")
    if any(v < 0 for v in observed_counts.values()):
        raise ValueError("observed counts must be non-negative")
    if any(w <= 0 for w in expected_weights.values()):
        raise ValueError("expected weights must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    keys = sorted(observed_counts.keys())
    observed = np.array([observed_counts[k] for k in keys], dtype=float)
    total = float(np.sum(observed))
    if total == 0:
        raise ValueError("observed counts sum to zero")

    weight_sum = float(sum(expected_weights.values()))
    expected = np.array([expected_weights[k] / weight_sum * total for k in keys], dtype=float)
    chi2, p_value = spstats.chisquare(observed, f_exp=expected)
    p = float(p_value)
    return SRMResult(
        p_value=p,
        significant=bool(p <= alpha),
        observed_total=int(total),
        expected_total=float(np.sum(expected)),
    )


def build_srm_alert(result: SRMResult, experiment_id: str) -> IntegrityAlert | None:
    if not result.significant:
        return None
    return IntegrityAlert(
        check_name=result.check_name,
        severity="critical",
        experiment_id=experiment_id,
        reason="Sample Ratio Mismatch detected against configured split.",
        suggested_action="Pause decisioning, verify assignment and ingestion lag before interpreting results.",
        p_value=result.p_value,
        details={
            "observed_total": float(result.observed_total),
            "expected_total": result.expected_total,
        },
    )


def build_aa_alert(result: AAResult, experiment_id: str) -> IntegrityAlert | None:
    if not result.significant:
        return None
    return IntegrityAlert(
        check_name=result.check_name,
        severity="critical",
        experiment_id=experiment_id,
        reason="AA test shows statistically significant difference in equivalent groups.",
        suggested_action="Audit randomization, exposure logging, and metric instrumentation before shipping changes.",
        p_value=result.p_value,
        details={"effect": result.effect},
    )
