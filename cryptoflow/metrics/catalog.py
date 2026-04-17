"""
Metric catalog for growth / retention (not exchange core latency).

Each metric can declare a default CUPED covariate (pre-period proxy).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GrowthDomain(str, Enum):
    LOYALTY_VIP = "loyalty_vip"
    FEE_ELASTICITY = "fee_elasticity"
    INVOLUNTARY_CHURN_RECOVERY = "involuntary_churn_recovery"
    GENERIC_GROWTH = "generic_growth"


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    domain: GrowthDomain
    description: str
    cuped_covariate_key: str | None  # e.g. "pre_fee_usd_28d"


def metric_catalog() -> tuple[MetricDefinition, ...]:
    return (
        MetricDefinition(
            key="fee_revenue_usd_28d",
            domain=GrowthDomain.FEE_ELASTICITY,
            description="Net trading fees after rebates (28d window)",
            cuped_covariate_key="pre_fee_usd_28d",
        ),
        MetricDefinition(
            key="traded_volume_usd_28d",
            domain=GrowthDomain.FEE_ELASTICITY,
            description="Notional volume (28d)",
            cuped_covariate_key="pre_volume_usd_28d",
        ),
        MetricDefinition(
            key="vip_tier_migration",
            domain=GrowthDomain.LOYALTY_VIP,
            description="Probability of tier upgrade/downgrade in window",
            cuped_covariate_key="pre_rolling_volume_usd_90d",
        ),
        MetricDefinition(
            key="reactivation_7d_after_liquidation",
            domain=GrowthDomain.INVOLUNTARY_CHURN_RECOVERY,
            description="Binary: active within 7d after involuntary churn event",
            cuped_covariate_key="pre_activity_score_30d",
        ),
        MetricDefinition(
            key="churn_28d_scalper_segment",
            domain=GrowthDomain.FEE_ELASTICITY,
            description="Churn in scalper segment (for cannibalization readouts)",
            cuped_covariate_key="pre_trades_28d",
        ),
    )
