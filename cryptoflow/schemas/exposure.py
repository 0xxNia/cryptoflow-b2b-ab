"""
Pydantic v2 models for cryptoflow.exposure_log.

ExposureEventPayload  — validated at the API boundary (external input).
ExposureEventRow      — enriched row ready for ClickHouse insertion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class TradingStyle(str, Enum):
    HODLER  = "hodler"
    SWING   = "swing"
    SCALPER = "scalper"


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE     = "moderate"
    DEGEN        = "degen"


class WalletSize(str, Enum):
    MINNOW  = "minnow"
    DOLPHIN = "dolphin"
    WHALE   = "whale"


class ChurnSensitivity(str, Enum):
    STICKY     = "sticky"
    NEUTRAL    = "neutral"
    MERCENARY  = "mercenary"


# ── Payload model (API boundary) ──────────────────────────────────────────────

class ExposureEventPayload(BaseModel):
    """
    Payload received from the assignment service when a user is enrolled
    in an A/B experiment.  Strictly validated before any downstream write.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    experiment_id: Annotated[
        str,
        Field(
            min_length=3,
            max_length=128,
            pattern=r"^[a-z0-9_\-]+$",
            description="Experiment slug, lowercase alphanumeric + underscores/hyphens",
        ),
    ]
    variant_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            pattern=r"^[a-z0-9_\-]+$",
            description='Variant label, e.g. "control" or "treatment"',
        ),
    ]
    user_id: Annotated[
        str,
        Field(min_length=1, max_length=64, description="Stable platform user ID"),
    ]
    timestamp: Annotated[
        datetime,
        Field(description="Assignment timestamp; timezone-aware required"),
    ]

    trading_style:     TradingStyle
    risk_profile:      RiskProfile
    wallet_size:       WalletSize
    churn_sensitivity: ChurnSensitivity

    expected_dtv_usd: Annotated[
        float,
        Field(gt=0.0, description="Expected Daily Trading Volume in USD"),
    ]
    pre_exp_dtv_30d: Annotated[
        float,
        Field(ge=0.0, description="Observed 30-day pre-experiment DTV (CUPED covariate)"),
    ]
    fee_elasticity: Annotated[
        float,
        Field(ge=0.0, le=50.0, description="Price-elasticity coefficient"),
    ]

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (use UTC)")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_variant_not_experiment(self) -> "ExposureEventPayload":
        if self.variant_id == self.experiment_id:
            raise ValueError("variant_id must differ from experiment_id")
        return self


# ── Row model (ClickHouse insertion) ─────────────────────────────────────────

class ExposureEventRow(ExposureEventPayload):
    """
    Enriched model with server-side fields appended before ClickHouse insert.
    Inherits all payload validators; ingest_time is assigned at write time.
    """

    ingest_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server ingestion timestamp",
    )

    def to_clickhouse_dict(self) -> dict:
        """Serialize to a dict whose keys match the ClickHouse column names."""
        return {
            "experiment_id":    self.experiment_id,
            "variant_id":       self.variant_id,
            "user_id":          self.user_id,
            "timestamp":        self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "trading_style":    self.trading_style.value,
            "risk_profile":     self.risk_profile.value,
            "wallet_size":      self.wallet_size.value,
            "churn_sensitivity": self.churn_sensitivity.value,
            "expected_dtv_usd": self.expected_dtv_usd,
            "pre_exp_dtv_30d":  self.pre_exp_dtv_30d,
            "fee_elasticity":   self.fee_elasticity,
            "ingest_time":      self.ingest_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
