"""
Market regime tagging for stratified experiment readouts.

Regimes are coarse labels derived from macro/vol inputs (ingested upstream).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeInputs:
    """Example inputs; wire to your vol/trend series."""

    realized_vol_annualized: float
    trend_return_30d: float  # e.g. log return of benchmark


def classify_regime(inp: RegimeInputs, *, vol_high: float = 0.8, vol_low: float = 0.35) -> MarketRegime:
    """
    Simple rule-based classifier (replace with production model).

    - HIGH_VOL if annualized vol above threshold
    - BULL/BEAR from 30d trend sign when not chop
    - CHOP when trend magnitude small
    """
    if inp.realized_vol_annualized >= vol_high:
        return MarketRegime.HIGH_VOL

    t = inp.trend_return_30d
    if abs(t) < 0.02:
        return MarketRegime.CHOP
    if t > 0:
        return MarketRegime.BULL
    if t < 0:
        return MarketRegime.BEAR
    return MarketRegime.UNKNOWN


def vol_bucket(inp: RegimeInputs, *, vol_high: float = 0.8, vol_low: float = 0.35) -> str:
    if inp.realized_vol_annualized >= vol_high:
        return "high"
    if inp.realized_vol_annualized <= vol_low:
        return "low"
    return "mid"
