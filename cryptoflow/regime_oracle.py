"""
cryptoflow.regime_oracle — real-time market regime classifier.

Volatility does not change at midnight. When BTC breaks a support level at
14:30 UTC, every trade after that moment should be tagged `high_vol`, not
at the next daily cron. This module is the ingestion-time service responsible
for that tagging.

Architecture (production deployment)
────────────────────────────────────
    ┌──────────────────────┐        ┌──────────────────────┐
    │ Exchange WS feed     │──────▶ │  RegimeOracle        │
    │ (Binance / Coinbase) │  tick  │  • realized vol      │
    │                      │        │  • trend pct         │
    └──────────────────────┘        │  • hysteresis        │
                                    └──────────┬───────────┘
                                               │ regime flag
                                               ▼
                                    ┌──────────────────────┐
                                    │ Event ingestion path │
                                    │ tags each event with │
                                    │ current regime.      │
                                    └──────────────────────┘

For this codebase the WebSocket feed is stubbed — see `connect_binance_ws`.
Swap the stub with real `websockets.connect(...)` when wiring to prod.

The same oracle can also replay a historical OHLCV DataFrame
(`replay_bars`) so backfills and backtests produce regime flags that are
bit-identical to what the live service would have emitted.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

import numpy as np
import pandas as pd


Regime = Literal["bull", "bear", "high_vol"]
_REGIMES: Final[tuple[Regime, ...]] = ("bull", "bear", "high_vol")


# ── Thresholds ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OracleThresholds:
    """
    Tunable classification thresholds. Defaults derived from BTC/USD
    historical distributions (2017–2025):
      • ~60% annualized realized vol separates "normal" from "high_vol"
      • ±0.5% 24h return separates bull / bear / sideways
      • 3 consecutive confirming ticks required to flip regime (hysteresis)
    """
    realized_vol_annualized_high: float = 0.60
    trend_threshold_pct:          float = 0.005
    lookback_hours:               int   = 24
    hysteresis_ticks:             int   = 3
    min_ticks_to_classify:        int   = 10


# ── Events ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeChange:
    """Emitted whenever the oracle flips its current regime."""
    timestamp:       datetime
    from_regime:     Regime
    to_regime:       Regime
    realized_vol:    float   # annualized, at flip time
    trend_pct:       float   # over lookback window, at flip time
    reference_price: float


# ── Oracle ───────────────────────────────────────────────────────────────────

class RegimeOracle:
    """
    Streaming classifier. `update(timestamp, price)` → current regime.

    Not thread-safe; run one oracle per consumer loop and publish the flag
    to a Redis/Kafka topic for cross-process readers.
    """

    def __init__(
        self,
        thresholds:     OracleThresholds | None = None,
        initial_regime: Regime = "bull",
    ) -> None:
        self.thresholds     = thresholds or OracleThresholds()
        self._ticks:        deque[tuple[datetime, float]] = deque()
        self._current:      Regime = initial_regime
        self._history:      list[RegimeChange] = []
        self._candidate:    Regime | None = None
        self._candidate_n:  int = 0

    # ── Public read-only API ─────────────────────────────────────────────────

    @property
    def current_regime(self) -> Regime:
        return self._current

    @property
    def history(self) -> list[RegimeChange]:
        return list(self._history)

    def snapshot(self) -> dict:
        """Machine-readable current state for a status endpoint."""
        vol   = self._realized_vol()
        trend = self._trend_pct()
        last  = self._ticks[-1] if self._ticks else (None, None)
        return {
            "regime":           self._current,
            "realized_vol":     vol,
            "trend_pct":        trend,
            "reference_price":  last[1],
            "reference_time":   last[0],
            "ticks_in_window":  len(self._ticks),
            "pending_flip_to":  self._candidate,
            "pending_flip_n":   self._candidate_n,
        }

    # ── Tick ingestion ───────────────────────────────────────────────────────

    def update(self, timestamp: datetime, price: float) -> Regime:
        """Feed a price tick; return the (possibly-unchanged) current regime."""
        if price <= 0 or not np.isfinite(price):
            raise ValueError("price must be a positive finite number")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self._ticks.append((timestamp, float(price)))
        cutoff = timestamp - timedelta(hours=self.thresholds.lookback_hours)
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()

        proposed = self._classify()
        if proposed == self._current:
            self._candidate, self._candidate_n = None, 0
        elif proposed == self._candidate:
            self._candidate_n += 1
            if self._candidate_n >= self.thresholds.hysteresis_ticks:
                self._history.append(RegimeChange(
                    timestamp       = timestamp,
                    from_regime     = self._current,
                    to_regime       = proposed,
                    realized_vol    = self._realized_vol(),
                    trend_pct       = self._trend_pct(),
                    reference_price = price,
                ))
                self._current = proposed
                self._candidate, self._candidate_n = None, 0
        else:
            self._candidate, self._candidate_n = proposed, 1
        return self._current

    # ── Classification internals ─────────────────────────────────────────────

    def _classify(self) -> Regime:
        if len(self._ticks) < self.thresholds.min_ticks_to_classify:
            return self._current
        vol = self._realized_vol()
        if vol > self.thresholds.realized_vol_annualized_high:
            return "high_vol"
        trend = self._trend_pct()
        if trend >  self.thresholds.trend_threshold_pct: return "bull"
        if trend < -self.thresholds.trend_threshold_pct: return "bear"
        return self._current  # sideways — keep prior regime

    def _realized_vol(self) -> float:
        """Annualized realized volatility over the lookback window."""
        if len(self._ticks) < 2:
            return 0.0
        prices  = np.fromiter((p for _, p in self._ticks), dtype=np.float64)
        returns = np.diff(np.log(prices))
        if returns.size == 0:
            return 0.0
        span_hours = (self._ticks[-1][0] - self._ticks[0][0]).total_seconds() / 3600.0
        if span_hours <= 0:
            return 0.0
        hourly_var = float(np.var(returns, ddof=0)) * returns.size / span_hours
        return float(np.sqrt(hourly_var * 24 * 365))  # √(8760) annualization

    def _trend_pct(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        first, last = self._ticks[0][1], self._ticks[-1][1]
        return (last - first) / first

    # ── Bulk replay (backfills / backtests) ──────────────────────────────────

    def replay_bars(
        self,
        bars: pd.DataFrame,
        timestamp_col: str = "timestamp",
        price_col:     str = "close",
    ) -> pd.Series:
        """
        Replay a historical OHLCV DataFrame through the oracle and return a
        pd.Series of per-bar regime flags aligned with `bars[timestamp_col]`.

        The flags are bit-identical to what the live oracle would have
        published had it been running over the same period.
        """
        if timestamp_col not in bars.columns:
            raise KeyError(f"bars DataFrame missing {timestamp_col!r}")
        if price_col not in bars.columns:
            raise KeyError(f"bars DataFrame missing {price_col!r}")
        flags: list[Regime] = []
        for ts, px in zip(bars[timestamp_col], bars[price_col]):
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            flags.append(self.update(ts_py, float(px)))
        return pd.Series(flags, index=bars[timestamp_col].to_numpy(), name="regime")


# ── Live feed adapter (stub) ─────────────────────────────────────────────────

async def connect_binance_ws(
    oracle:  RegimeOracle,
    symbol:  str = "btcusdt",
    on_flip: "callable[[RegimeChange], None] | None" = None,
) -> None:
    """
    Stub for a production WebSocket consumer. In prod replace the body with:

        import websockets, json
        url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        async with websockets.connect(url) as ws:
            async for msg in ws:
                tick = json.loads(msg)
                ts    = datetime.fromtimestamp(tick["T"]/1000, tz=timezone.utc)
                price = float(tick["p"])
                prev  = oracle.current_regime
                new   = oracle.update(ts, price)
                if new != prev and on_flip is not None:
                    on_flip(oracle.history[-1])

    This stub exists so downstream code can `import connect_binance_ws` and
    wire it up without this module depending on `websockets` at import time.
    """
    raise NotImplementedError(
        "connect_binance_ws is a production stub — see docstring for the "
        "~10-line websockets.connect body to paste in when deploying."
    )
