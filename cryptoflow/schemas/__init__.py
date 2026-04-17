"""
cryptoflow.schemas — Pydantic v2 models for payload validation.

Validates incoming event payloads BEFORE ingestion into ClickHouse,
ensuring type safety, enum constraints, and business-rule invariants.
"""

from .exposure import (
    TradingStyle,
    RiskProfile,
    WalletSize,
    ChurnSensitivity,
    ExposureEventPayload,
    ExposureEventRow,
)
from .transaction import (
    MarketRegime,
    EventType,
    TradeSide,
    TransactionEventPayload,
    TransactionEventRow,
)

__all__ = [
    "TradingStyle",
    "RiskProfile",
    "WalletSize",
    "ChurnSensitivity",
    "ExposureEventPayload",
    "ExposureEventRow",
    "MarketRegime",
    "EventType",
    "TradeSide",
    "TransactionEventPayload",
    "TransactionEventRow",
]
