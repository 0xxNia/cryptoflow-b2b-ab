"""
cryptoflow.reconciliation — Ex-ante vs ex-post reconciliation.

Closes the loop between the pre-launch forecast (preflight.PreflightReport)
and the post-launch observed outcome (pnl.FinancialPnL):

  planned_upside        vs.   actual_posterior_mean
  planned_fp_risk       vs.   actual_tail_probability × loss
  planned_duration      vs.   realized_duration

A systematic bias here is how we learn to forecast better over time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from preflight import PreflightReport
from pnl import FinancialPnL


class ReconciliationStatus(str, Enum):
    ON_TRACK          = "on_track"           # |delta_pct| ≤ on_track_tolerance
    OVERSHOOT         = "overshoot"          # actual > planned by more than tolerance
    UNDERSHOOT        = "undershoot"         # actual < planned by more than tolerance
    DIRECTIONAL_MISS  = "directional_miss"   # sign flip — planned gain, got loss (or vice versa)


@dataclass(frozen=True)
class ReconciliationReport:
    # Ex-ante (from preflight)
    planned_upside_usd:  float
    planned_fp_risk_usd: float
    planned_days:        float

    # Ex-post (from pnl + realised)
    actual_mean_usd:     float
    actual_tail_usd:     float   # $ risk at the same tail probability the plan used
    realized_days:       float

    # Deltas
    upside_delta_usd:    float   # actual − planned
    upside_bias_pct:     float   # (actual − planned) / |planned| × 100
    risk_delta_usd:      float   # (actual tail loss) − planned_fp_risk
    days_delta:          float

    status:              ReconciliationStatus
    message:             str


def reconcile(
    preflight:             PreflightReport,
    pnl:                   FinancialPnL,
    realized_days:         float,
    on_track_tolerance_pct: float = 30.0,
) -> ReconciliationReport:
    """
    Compare the pre-launch forecast to the post-launch realised outcome.

    Parameters
    ----------
    preflight:
        Report emitted at test setup time by `preflight.preflight_check`.
    pnl:
        FinancialPnL built from the final Bayesian posterior via
        `pnl.pnl_from_bayesian`.
    realized_days:
        Actual days the experiment ran (may differ from plan due to
        early stopping via mSPRT or guardrail kill).
    on_track_tolerance_pct:
        Absolute bias threshold (%) for the ON_TRACK verdict. Default 30 %.
    """
    planned_upside = preflight.expected_upside_usd
    actual_mean    = pnl.posterior_mean_usd

    upside_delta = actual_mean - planned_upside
    if planned_upside == 0:
        bias_pct = 0.0 if actual_mean == 0 else float("inf")
    else:
        bias_pct = upside_delta / abs(planned_upside) * 100.0

    # "Actual tail loss" at the same α used for the plan's false-positive risk
    #   plan_fp_risk = α × daily_value × horizon
    # so we compare to pnl's P(net loss) scaled to a comparable $ number.
    actual_tail_usd = pnl.prob_net_loss * max(0.0, -pnl.posterior_mean_usd + pnl.posterior_std_usd)
    risk_delta      = actual_tail_usd - preflight.false_positive_risk_usd

    # Classify
    if (planned_upside > 0) != (actual_mean > 0) and planned_upside != 0 and actual_mean != 0:
        status = ReconciliationStatus.DIRECTIONAL_MISS
    elif abs(bias_pct) <= on_track_tolerance_pct:
        status = ReconciliationStatus.ON_TRACK
    elif bias_pct > 0:
        status = ReconciliationStatus.OVERSHOOT
    else:
        status = ReconciliationStatus.UNDERSHOOT

    verb = {
        ReconciliationStatus.ON_TRACK:         "in line with",
        ReconciliationStatus.OVERSHOOT:        "materially above",
        ReconciliationStatus.UNDERSHOOT:       "materially below",
        ReconciliationStatus.DIRECTIONAL_MISS: "in the opposite direction from",
    }[status]

    message = (
        f"Actual ${actual_mean:,.0f} is {verb} planned ${planned_upside:,.0f} "
        f"({bias_pct:+.1f}%). Duration {realized_days:.0f}d vs plan {preflight.days_to_complete:.0f}d."
    )

    return ReconciliationReport(
        planned_upside_usd=planned_upside,
        planned_fp_risk_usd=preflight.false_positive_risk_usd,
        planned_days=preflight.days_to_complete,
        actual_mean_usd=actual_mean,
        actual_tail_usd=actual_tail_usd,
        realized_days=realized_days,
        upside_delta_usd=upside_delta,
        upside_bias_pct=bias_pct,
        risk_delta_usd=risk_delta,
        days_delta=realized_days - preflight.days_to_complete,
        status=status,
        message=message,
    )
