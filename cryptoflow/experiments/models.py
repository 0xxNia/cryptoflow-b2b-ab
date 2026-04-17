from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cryptoflow.regimes import MarketRegime


@dataclass(frozen=True)
class VariantConfig:
    variant_id: str
    payload: dict[str, Any] = field(default_factory=dict)  # fee bps, VIP threshold, etc.


@dataclass(frozen=True)
class ExperimentSpec:
    """Static definition stored in config service or DB."""

    experiment_id: str
    salt: str
    variants: tuple[VariantConfig, ...]
    unit_type: str = "user"  # user | wallet | session
    owner_team: str = "growth"
    hypothesis: str = ""


@dataclass(frozen=True)
class ExposureRecord:
    """One row for warehouse fact_experiment_exposure."""

    ts: datetime
    user_id: str
    experiment_id: str
    variant_id: str
    assignment_version: str
    market_regime: MarketRegime | None
    platform: str | None = None
    app_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
