"""
Pydantic v2 models for cryptoflow.transaction_events.

TransactionEventPayload — validated at the API boundary.
TransactionEventRow     — enriched row ready for ClickHouse insertion.

DESIGN NOTE:
  market_regime is an EVENT-level property, set by the platform
  market-state service at ingestion time based on current macro
  conditions (e.g. derived from realised volatility, BTC trend).
  It is NOT derived from user attributes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class MarketRegime(str, Enum):
    """
    Macro market state at the time of a transaction.
    Determined by the platform's market-context service, not by user profile.

    bull     — trending up, low realised vol (< 30-day historical avg)
    bear     — trending down, moderate vol
    high_vol — regime-agnostic spike; crypto VIX equivalent > threshold
    """
    BULL     = "bull"
    BEAR     = "bear"
    HIGH_VOL = "high_vol"


class EventType(str, Enum):
    TRADE        = "trade"
    DEPOSIT      = "deposit"
    WITHDRAWAL   = "withdrawal"
    LIQUIDATION  = "liquidation"


class TradeSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


# ── Payload model (API boundary) ──────────────────────────────────────────────

class TransactionEventPayload(BaseModel):
    """
    Payload submitted by the transaction processor for every financial event
    that occurs for a user enrolled in a running experiment.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    # Identifiers
    experiment_id: Annotated[
        str,
        Field(min_length=3, max_length=128, pattern=r"^[a-z0-9_\-]+$"),
    ]
    variant_id: Annotated[
        str,
        Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-]+$"),
    ]
    user_id: Annotated[str, Field(min_length=1, max_length=64)]

    # Timing
    timestamp: Annotated[
        datetime,
        Field(description="Event timestamp; must be timezone-aware"),
    ]

    # Event type
    event_type:    EventType
    market_regime: MarketRegime   # set by market-context service, not client

    # Asset details (required only for trade/liquidation, optional for deposit/withdrawal)
    coin: Annotated[
        str | None,
        Field(default=None, max_length=16, pattern=r"^[A-Z0-9]+$"),
    ] = None
    side: TradeSide | None = None

    # Financial values
    volume_usd: Annotated[
        float,
        Field(gt=0.0, description="Raw transaction volume in USD"),
    ]
    fee_usd: Annotated[
        float,
        Field(ge=0.0, description="Fee charged in USD"),
    ]

    # Denormalised user attributes (snapshot from enrollment record)
    trading_style: Annotated[str, Field(min_length=1, max_length=32)]
    wallet_size:   Annotated[str, Field(min_length=1, max_length=32)]

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _trade_requires_coin_and_side(self) -> "TransactionEventPayload":
        if self.event_type in (EventType.TRADE, EventType.LIQUIDATION):
            if self.coin is None:
                raise ValueError(f"coin is required for event_type={self.event_type.value}")
            if self.side is None:
                raise ValueError(f"side is required for event_type={self.event_type.value}")
        return self

    @model_validator(mode="after")
    def _fee_cannot_exceed_volume(self) -> "TransactionEventPayload":
        if self.fee_usd > self.volume_usd:
            raise ValueError(
                f"fee_usd ({self.fee_usd}) exceeds volume_usd ({self.volume_usd})"
            )
        return self


# ── Row model (ClickHouse insertion) ─────────────────────────────────────────

class TransactionEventRow(TransactionEventPayload):
    """
    Enriched model with server-side fields.
    volume_usd_w is the Winsorized volume computed at ingestion time
    using the 99th-percentile cap from the rolling 30-day window.
    """

    event_id:     uuid.UUID = Field(default_factory=uuid.uuid4)
    volume_usd_w: Annotated[
        float,
        Field(gt=0.0, description="Winsorized volume (99th-pct cap applied at ingestion)"),
    ]
    ingest_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_clickhouse_dict(self) -> dict:
        """Serialize to a dict whose keys match the ClickHouse column names."""
        return {
            "event_id":       str(self.event_id),
            "experiment_id":  self.experiment_id,
            "variant_id":     self.variant_id,
            "user_id":        self.user_id,
            "timestamp":      self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "event_type":     self.event_type.value,
            "market_regime":  self.market_regime.value,
            "coin":           self.coin or "",
            "side":           self.side.value if self.side else "",
            "volume_usd":     self.volume_usd,
            "fee_usd":        self.fee_usd,
            "volume_usd_w":   self.volume_usd_w,
            "trading_style":  self.trading_style,
            "wallet_size":    self.wallet_size,
            "ingest_time":    self.ingest_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
