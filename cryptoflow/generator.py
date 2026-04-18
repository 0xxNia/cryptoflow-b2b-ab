"""
cryptoflow.generator — Production-grade synthetic data generator.

Outputs DataFrames whose columns match the ClickHouse DDL defined in
sql/clickhouse/exposure_log.sql and sql/clickhouse/transaction_events.sql.

Statistical properties
──────────────────────
  Trade volumes  : Pareto(α = 1.5, scale = x_min)   power-law tail
  Trade counts   : NegativeBinomial(r = 2, p)        over-dispersed
  Market regimes : First-order Markov chain over {bull, bear, high_vol}
  Winsorization  : 99th-percentile cap applied to volume_usd_w at row construction
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd


# ── Persona parameter tables ──────────────────────────────────────────────────
# Mirrors generate_data.py, but uses spec numbers (swing 15/wk, scalper 250/wk,
# whale 50× volume) and targets the ClickHouse schema from Module 1.

STYLE_BASE: dict[str, dict] = {
    "hodler":  {"trades_week":   1, "avg_volume": 5_000, "churn_month": 0.03, "fee_elast": 0.10},
    "swing":   {"trades_week":  15, "avg_volume": 2_000, "churn_month": 0.05, "fee_elast": 0.80},
    "scalper": {"trades_week": 250, "avg_volume":   800, "churn_month": 0.12, "fee_elast": 4.50},
}
STYLE_WEIGHTS: dict[str, float] = {"hodler": 0.65, "swing": 0.30, "scalper": 0.05}

RISK_MULT: dict[str, dict] = {
    "conservative": {"t": 1.0, "v": 1.0, "c": 1.0},
    "moderate":     {"t": 1.5, "v": 1.2, "c": 1.5},
    "degen":        {"t": 2.5, "v": 3.0, "c": 5.0},   # 5× churn (spec)
}
RISK_WEIGHTS: dict[str, float] = {"conservative": 0.50, "moderate": 0.35, "degen": 0.15}

WALLET_MULT: dict[str, dict] = {
    "minnow":  {"t": 1.0, "v":  0.2, "c": 1.3},
    "dolphin": {"t": 1.0, "v":  1.0, "c": 1.0},
    "whale":   {"t": 0.5, "v": 50.0, "c": 0.4},   # 50× volume, 0.5× freq (spec)
}
WALLET_WEIGHTS: dict[str, float] = {"minnow": 0.80, "dolphin": 0.18, "whale": 0.02}

CHURN_MULT: dict[str, dict] = {
    "sticky":    {"t": 1.0, "c": 0.3, "fe": 0.2},
    "neutral":   {"t": 1.0, "c": 1.0, "fe": 1.0},
    "mercenary": {"t": 1.2, "c": 3.5, "fe": 3.0},
}
CHURN_WEIGHTS: dict[str, float] = {"sticky": 0.20, "neutral": 0.45, "mercenary": 0.35}

# ── Volume distribution ───────────────────────────────────────────────────────
# Pareto(α=1.5): E[X] = α·x_min/(α−1) = 3·x_min  ⟹  x_min = mean/3
PARETO_ALPHA: float = 1.5
VOLUME_CAP:   float = 5_000_000.0

# ── Trade count distribution ──────────────────────────────────────────────────
# NegBinom(r=2, p): variance = mean + mean²/r > mean (over-dispersed vs Poisson)
NEGBINOM_R: int = 2

# ── Event type mix ────────────────────────────────────────────────────────────
_EVENT_TYPES = ["trade", "deposit", "withdrawal", "liquidation"]
_EVENT_PROBS = np.array([0.82, 0.09, 0.06, 0.03])

# ── Asset universe ────────────────────────────────────────────────────────────
_COINS  = ["BTC", "ETH", "SOL", "BNB", "DOGE", "XRP", "MATIC", "AVAX"]
_COIN_W = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])

# ── Market regime Markov chain ────────────────────────────────────────────────
_REGIMES = ["bull", "bear", "high_vol"]

# Row i = transition probabilities FROM regime i (rows sum to 1.0).
# Stationary distribution ≈ [0.50 bull, 0.35 bear, 0.15 high_vol].
_REGIME_TRANSITION = np.array([
    [0.9400, 0.0400, 0.0200],   # from bull
    [0.0571, 0.9129, 0.0300],   # from bear
    [0.0667, 0.0700, 0.8633],   # from high_vol
])
_REGIME_INITIAL = np.array([0.50, 0.35, 0.15])

# ── Default fee rates ─────────────────────────────────────────────────────────
BASE_FEE_RATE:      float = 0.0025
TREATMENT_FEE_RATE: float = 0.0015


# ── Output container ──────────────────────────────────────────────────────────

class ExperimentData(NamedTuple):
    """Pair of DataFrames ready for ClickHouse insertion."""
    exposure:     pd.DataFrame  # matches cryptoflow.exposure_log schema
    transactions: pd.DataFrame  # matches cryptoflow.transaction_events schema


# ── Internal helpers ──────────────────────────────────────────────────────────

def _markov_regimes(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a daily market regime sequence via a first-order Markov chain.

    Returns int8 array of length n_days; values are indices into _REGIMES.
    The same regime sequence is shared across all users for the same calendar day,
    reflecting that market state is an EVENT-level property (not user-level).
    """
    state = int(rng.choice(len(_REGIMES), p=_REGIME_INITIAL))
    out   = np.empty(n_days, dtype=np.int8)
    for d in range(n_days):
        out[d] = state
        state  = int(rng.choice(len(_REGIMES), p=_REGIME_TRANSITION[state]))
    return out


def _pareto_volumes(n: int, mean_volume: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n volumes from Pareto(α=PARETO_ALPHA) with the specified mean.

    Inverse-CDF: x = x_min * (1 - u)^{-1/α}, u ~ Uniform(0,1).
    x_min = mean * (α − 1) / α.
    """
    x_min = max(mean_volume * (PARETO_ALPHA - 1.0) / PARETO_ALPHA, 1.0)
    u = rng.random(n)
    return np.minimum(x_min * (1.0 - u) ** (-1.0 / PARETO_ALPHA), VOLUME_CAP)


def _negbinom_counts(mean_n: float, n_weeks: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n_weeks trade-count observations from NegBinom(r=NEGBINOM_R, p=r/(r+mean_n)).

    Returns int32 array; variance = mean_n + mean_n²/r  (over-dispersed vs Poisson).
    """
    if mean_n <= 0 or n_weeks == 0:
        return np.zeros(n_weeks, dtype=np.int32)
    p = NEGBINOM_R / (NEGBINOM_R + mean_n)
    return rng.negative_binomial(NEGBINOM_R, p, size=n_weeks).astype(np.int32)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_experiment(
    n_users:            int      = 10_000,
    experiment_id:      str      = "fee_reduction_2024q1",
    start_date:         datetime | None = None,
    end_date:           datetime | None = None,
    control_variant:    str      = "control",
    treatment_variant:  str      = "treatment",
    control_fee_rate:   float    = BASE_FEE_RATE,
    treatment_fee_rate: float    = TREATMENT_FEE_RATE,
    winsorize_pct:      float    = 99.0,
    seed:               int      = 42,
) -> ExperimentData:
    """
    Generate a synthetic A/B experiment dataset matching the ClickHouse schema.

    Returns ExperimentData(exposure, transactions) where:
      exposure     matches cryptoflow.exposure_log DDL
      transactions matches cryptoflow.transaction_events DDL
    """
    rng = np.random.default_rng(seed)

    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    elif start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)

    if end_date is None:
        end_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    n_days  = (end_date - start_date).days
    n_weeks = n_days // 7
    if n_days <= 0:
        raise ValueError("end_date must be strictly after start_date")

    # Global daily market regime sequence (shared across all users)
    daily_regime = _markov_regimes(n_days, rng)

    # ── Sample persona attributes ─────────────────────────────────────────────
    styles   = rng.choice(list(STYLE_WEIGHTS),  n_users, p=list(STYLE_WEIGHTS.values()))
    risks    = rng.choice(list(RISK_WEIGHTS),   n_users, p=list(RISK_WEIGHTS.values()))
    wallets  = rng.choice(list(WALLET_WEIGHTS), n_users, p=list(WALLET_WEIGHTS.values()))
    churns   = rng.choice(list(CHURN_WEIGHTS),  n_users, p=list(CHURN_WEIGHTS.values()))
    variants = rng.choice([control_variant, treatment_variant], n_users, p=[0.5, 0.5])
    user_ids = np.array([f"u_{i:06d}" for i in range(n_users)])

    # ── Per-user behavioural parameters (vectorised) ─────────────────────────
    tw_base = np.array([
        min(
            STYLE_BASE[s]["trades_week"] * RISK_MULT[r]["t"]
            * WALLET_MULT[w]["t"] * CHURN_MULT[c]["t"],
            500,
        )
        for s, r, w, c in zip(styles, risks, wallets, churns)
    ])
    av_arr = np.array([
        STYLE_BASE[s]["avg_volume"] * RISK_MULT[r]["v"] * WALLET_MULT[w]["v"]
        for s, r, w in zip(styles, risks, wallets)
    ])
    cm_arr = np.array([
        min(
            STYLE_BASE[s]["churn_month"] * RISK_MULT[r]["c"]
            * WALLET_MULT[w]["c"] * CHURN_MULT[c]["c"],
            0.95,
        )
        for s, r, w, c in zip(styles, risks, wallets, churns)
    ])
    fe_arr = np.array([
        STYLE_BASE[s]["fee_elast"] * CHURN_MULT[c]["fe"]
        for s, c in zip(styles, churns)
    ])

    # Monthly → weekly churn probability
    cw_arr = 1.0 - (1.0 - cm_arr) ** (1.0 / 4.33)

    # Treatment: fee reduction → more trades via fee elasticity
    # fee_delta < 0 for a fee reduction, so multiplier > 1 for elastic users
    fee_delta = (treatment_fee_rate - control_fee_rate) / control_fee_rate
    tw_arr = tw_base.copy()
    t_mask = variants == treatment_variant
    tw_arr[t_mask] = np.minimum(
        tw_base[t_mask] * (1.0 - fe_arr[t_mask] * fee_delta),
        500,
    )

    # Expected DTV (USD/day) for exposure table
    expected_dtv = np.round(av_arr * (tw_base / 7.0), 4)
    pre_exp_dtv  = np.round(
        np.maximum(expected_dtv * rng.lognormal(0.0, 0.30, n_users), 0.0), 4
    )

    # Exposure timestamps: random within first 7 days
    exp_offsets = rng.integers(0, 7 * 24 * 3600, n_users)
    exposure_ts = [start_date + timedelta(seconds=int(o)) for o in exp_offsets]

    # ── Exposure DataFrame ────────────────────────────────────────────────────
    exposure_df = pd.DataFrame({
        "experiment_id":     experiment_id,
        "variant_id":        variants,
        "user_id":           user_ids,
        "timestamp":         exposure_ts,
        "trading_style":     styles,
        "risk_profile":      risks,
        "wallet_size":       wallets,
        "churn_sensitivity": churns,
        "expected_dtv_usd":  expected_dtv,
        "pre_exp_dtv_30d":   pre_exp_dtv,
        "fee_elasticity":    np.round(fe_arr, 4),
    })

    # ── Transaction events ────────────────────────────────────────────────────
    tx_rows: list[dict] = []

    for i in range(n_users):
        cw       = float(cw_arr[i])
        tw       = float(tw_arr[i])
        av       = float(av_arr[i])
        var      = str(variants[i])
        sty      = str(styles[i])
        wal      = str(wallets[i])
        uid      = str(user_ids[i])
        fee_rate = treatment_fee_rate if var == treatment_variant else control_fee_rate

        # Active weeks: geometric(weekly_churn) ∩ [0, n_weeks]
        if cw <= 0.0:
            active_weeks = n_weeks
        elif cw >= 1.0:
            active_weeks = 0
        else:
            active_weeks = min(int(rng.geometric(cw)), n_weeks)

        if active_weeks == 0:
            continue

        weekly_counts = _negbinom_counts(tw, active_weeks, rng)
        total_trades  = int(weekly_counts.sum())
        if total_trades == 0:
            continue

        volumes    = _pareto_volumes(total_trades, av, rng)
        active_sec = max(active_weeks * 7 * 24 * 3600, 1)
        ts_secs    = rng.integers(0, active_sec, total_trades)
        trade_days = np.minimum(ts_secs // (24 * 3600), n_days - 1)

        regime_idx = daily_regime[trade_days]
        ev_types   = rng.choice(_EVENT_TYPES, total_trades, p=_EVENT_PROBS)
        coins      = rng.choice(_COINS,       total_trades, p=_COIN_W)
        sides      = rng.choice(["buy", "sell"], total_trades)

        for j in range(total_trades):
            abs_ts = start_date + timedelta(seconds=int(ts_secs[j]))
            if abs_ts >= end_date:
                continue
            ev  = ev_types[j]
            vol = float(volumes[j])
            tx_rows.append({
                "event_id":      str(uuid.uuid4()),
                "experiment_id": experiment_id,
                "variant_id":    var,
                "user_id":       uid,
                "timestamp":     abs_ts,
                "event_type":    ev,
                "market_regime": _REGIMES[int(regime_idx[j])],
                "coin":          coins[j] if ev in ("trade", "liquidation") else "",
                "side":          sides[j] if ev in ("trade", "liquidation") else "",
                "volume_usd":    round(vol, 2),
                "fee_usd":       round(vol * fee_rate, 4),
                "trading_style": sty,
                "wallet_size":   wal,
            })

    if tx_rows:
        transactions_df = pd.DataFrame(tx_rows)
        cap = np.percentile(transactions_df["volume_usd"].values, winsorize_pct)
        transactions_df["volume_usd_w"] = (
            transactions_df["volume_usd"].clip(upper=cap).round(2)
        )
    else:
        transactions_df = pd.DataFrame(columns=[
            "event_id", "experiment_id", "variant_id", "user_id", "timestamp",
            "event_type", "market_regime", "coin", "side",
            "volume_usd", "fee_usd", "volume_usd_w", "trading_style", "wallet_size",
        ])

    return ExperimentData(exposure=exposure_df, transactions=transactions_df)


# ══════════════════════════════════════════════════════════════════════════════
# MCMC engine: vectorized Markov-chain user-trajectory simulator
# ══════════════════════════════════════════════════════════════════════════════
#
# Simulates daily state transitions (Active / Dormant / Churned) for N users
# simultaneously using broadcasted numpy ops — no per-user Python iteration.
# Produces a ClickHouse-ready event log where each row is a trade executed
# on a day the user was Active.
#
# Distinct from generate_experiment above: that function is persona-driven and
# emits A/B-framed exposure + transaction tables; CryptoMCSimulator is a pure
# behavioural engine parameterised by Markov transition matrices.

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# ── MCMC state constants ─────────────────────────────────────────────────────

STATE_ACTIVE:  Final[int] = 0
STATE_DORMANT: Final[int] = 1
STATE_CHURNED: Final[int] = 2
NUM_STATES:    Final[int] = 3

MIN_VOLUME_USD: Final[float] = 1.0
# Upper cap reuses VOLUME_CAP (= 5_000_000.0) defined earlier in the module.

# Default user-batch size. At 250k users × 60 days the per-batch peak memory
# stays in the ~250–350 MB range (state matrix + event DataFrame), so a single
# worker can safely stream 5–10M-user populations without OOM.
DEFAULT_BATCH_SIZE: Final[int] = 250_000


# ── MCMC profile configuration ───────────────────────────────────────────────

@dataclass(frozen=True)
class TradingProfile:
    """
    Immutable behavioural profile for the Markov-chain engine.

    transitions[i, j] = P(next_state = j | current_state = i).
    Row 2 (Churned) is the absorbing row [0, 0, 1].
    """
    name:        str
    weight:      float
    transitions: np.ndarray
    mu:          float           # Log-Normal location parameter
    sigma:       float           # Log-Normal scale parameter


MCMC_PROFILES: Final[tuple[TradingProfile, ...]] = (
    TradingProfile(
        name="HODLer",
        weight=0.65,
        transitions=np.array(
            [
                [0.05, 0.94,  0.01 ],   # Active  → ...
                [0.02, 0.975, 0.005],   # Dormant → ...
                [0.00, 0.00,  1.00 ],   # Churned (absorbing)
            ],
            dtype=np.float64,
        ),
        mu=8.5,
        sigma=1.2,
    ),
    TradingProfile(
        name="SwingTrader",
        weight=0.30,
        transitions=np.array(
            [
                [0.30, 0.65, 0.05],
                [0.15, 0.83, 0.02],
                [0.00, 0.00, 1.00],
            ],
            dtype=np.float64,
        ),
        mu=7.6,
        sigma=0.8,
    ),
    TradingProfile(
        name="Scalper",
        weight=0.05,
        transitions=np.array(
            [
                [0.80, 0.10, 0.10],
                [0.50, 0.30, 0.20],
                [0.00, 0.00, 1.00],
            ],
            dtype=np.float64,
        ),
        mu=6.6,
        sigma=0.5,
    ),
)


# ── Simulator ────────────────────────────────────────────────────────────────

class CryptoMCSimulator:
    """
    Vectorized Markov-Chain Monte Carlo simulator for synthetic user
    trajectories.

    Usage:
        sim    = CryptoMCSimulator(num_users=100_000, horizon_days=60,
                                   start_date="2026-01-01", seed=42)
        events = sim.generate_event_log()   # pd.DataFrame, ClickHouse-ready

    Memory profile at 500k users × 60 days:
        state matrix    : ~30 MB   (int8)
        per-day randoms : ~4  MB   (released each iteration)
        event dataframe : O(active user-days) rows
    """

    OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
        "timestamp", "user_id", "profile", "event_type", "volume_usd",
    )

    def __init__(
        self,
        num_users:    int,
        horizon_days: int,
        start_date:   str,
        seed:         int | None = None,
    ) -> None:
        if num_users    <= 0: raise ValueError("num_users must be positive")
        if horizon_days <= 0: raise ValueError("horizon_days must be positive")

        self.num_users    = int(num_users)
        self.horizon_days = int(horizon_days)
        self.start_date   = pd.Timestamp(start_date).normalize()
        self.rng          = np.random.default_rng(seed)

        # _cdf[p, s] is the CDF of the transition row for profile p in state s.
        # Shape: (num_profiles, num_states, num_states).
        self._cdf = np.stack([p.transitions.cumsum(axis=1) for p in MCMC_PROFILES])

        self._profile_weights = np.array(
            [p.weight for p in MCMC_PROFILES], dtype=np.float64,
        )
        self._profile_weights /= self._profile_weights.sum()
        self._profile_mu    = np.array([p.mu    for p in MCMC_PROFILES], dtype=np.float64)
        self._profile_sigma = np.array([p.sigma for p in MCMC_PROFILES], dtype=np.float64)
        self._profile_names = np.array([p.name  for p in MCMC_PROFILES], dtype=object)

        self._profiles: np.ndarray | None = None
        self._states:   np.ndarray | None = None

    # ── Profile assignment ───────────────────────────────────────────────────

    def _generate_profiles(self) -> np.ndarray:
        """Assign every user a profile index via weighted categorical sampling."""
        self._profiles = self.rng.choice(
            len(MCMC_PROFILES),
            size=self.num_users,
            p=self._profile_weights,
        ).astype(np.int8)
        return self._profiles

    # ── Markov chain ─────────────────────────────────────────────────────────

    def _simulate_markov_chain(self) -> np.ndarray:
        """
        Evolve all users simultaneously across horizon_days.

        For day t+1 the next state is drawn by comparing a uniform against the
        CDF of the transition row keyed by (profile, current_state). The single
        loop runs over days only (typically ~60); the user dimension is fully
        vectorized.
        """
        if self._profiles is None:
            self._generate_profiles()
        profiles = self._profiles  # type: ignore[assignment]

        states = np.empty((self.num_users, self.horizon_days), dtype=np.int8)
        states[:, 0] = STATE_ACTIVE   # day-0 cohort seeded as Active

        for t in range(1, self.horizon_days):
            current = states[:, t - 1]                      # (N,)
            cdfs    = self._cdf[profiles, current]          # (N, 3)
            r       = self.rng.random(size=self.num_users)  # (N,)
            # Count CDF thresholds that r has met or exceeded.
            # r ∈ [0, 1) and the last CDF entry is exactly 1.0, so the count
            # is always in {0, 1, 2} = {Active, Dormant, Churned}.
            states[:, t] = (r[:, np.newaxis] >= cdfs).sum(axis=1)

        self._states = states
        return states

    # ── Single-batch simulation (streaming primitive) ────────────────────────

    def _simulate_batch(self, batch_size: int) -> pd.DataFrame:
        """
        Simulate Markov chain + event extraction for one batch of users.

        Advances self.rng. Memory peak is bounded by batch_size:
          state matrix    : batch_size × horizon_days × 1 byte
          event DataFrame : O(active user-days in batch)
        Returns an empty, correctly-typed DataFrame when the batch yields no events.
        """
        # 1. Profiles for this batch
        profiles = self.rng.choice(
            len(MCMC_PROFILES), size=batch_size, p=self._profile_weights,
        ).astype(np.int8)

        # 2. Markov chain — only this batch's users materialized simultaneously
        states = np.empty((batch_size, self.horizon_days), dtype=np.int8)
        states[:, 0] = STATE_ACTIVE
        for t in range(1, self.horizon_days):
            cdfs = self._cdf[profiles, states[:, t - 1]]
            r    = self.rng.random(size=batch_size)
            states[:, t] = (r[:, np.newaxis] >= cdfs).sum(axis=1)

        # 3. Event extraction — (user, day) pairs where state is Active
        user_idx, day_idx = np.nonzero(states == STATE_ACTIVE)
        if user_idx.size == 0:
            return pd.DataFrame(
                {c: pd.Series(dtype=self._column_dtype(c)) for c in self.OUTPUT_COLUMNS}
            )

        event_profile_idx = profiles[user_idx]
        mu    = self._profile_mu[event_profile_idx]
        sigma = self._profile_sigma[event_profile_idx]

        volumes = self.rng.lognormal(mean=mu, sigma=sigma)
        np.clip(volumes, MIN_VOLUME_USD, VOLUME_CAP, out=volumes)

        user_uuids = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object,
        )
        timestamps = self.start_date + pd.to_timedelta(day_idx, unit="D")

        return pd.DataFrame(
            {
                "timestamp":  timestamps,
                "user_id":    user_uuids[user_idx],
                "profile":    self._profile_names[event_profile_idx],
                "event_type": "trade",
                "volume_usd": np.round(volumes, 2),
            },
            columns=list(self.OUTPUT_COLUMNS),
        )

    # ── Streaming API ────────────────────────────────────────────────────────

    def stream_event_log(
        self, batch_size: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Yield event DataFrames batch-by-batch without materializing the full log.

        Use this for populations larger than a few hundred thousand users.
        Each yielded DataFrame is sorted by (timestamp, user_id) within its batch;
        consumers that need globally sorted output must merge-sort across batches.
        """
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        remaining = self.num_users
        while remaining > 0:
            n = min(batch_size, remaining)
            df = self._simulate_batch(n)
            if not df.empty:
                df.sort_values(
                    by=["timestamp", "user_id"],
                    kind="stable", ignore_index=True, inplace=True,
                )
            yield df
            remaining -= n

    def write_parquet(
        self,
        output_dir:   str | Path,
        batch_size:   int | None = None,
        compression:  str  = "snappy",
    ) -> dict:
        """
        Stream the event log to a partitioned Parquet dataset.

        One file per batch: {output_dir}/part-00000.parquet, part-00001, ...
        This is the recommended path for populations ≥ 1M users; the resulting
        directory can be ingested directly by ClickHouse
        (`INSERT INTO ... FROM INFILE '*.parquet' FORMAT Parquet`), DuckDB
        (`read_parquet('{output_dir}/*.parquet')`), or Spark/Dask.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        total_events = 0
        total_bytes  = 0
        n_batches    = 0
        for batch_df in self.stream_event_log(batch_size):
            if batch_df.empty:
                continue
            path = out / f"part-{n_batches:05d}.parquet"
            batch_df.to_parquet(path, index=False, compression=compression)
            total_events += len(batch_df)
            total_bytes  += path.stat().st_size
            n_batches    += 1

        return {
            "output_dir":  str(out),
            "batches":     n_batches,
            "events":      total_events,
            "bytes":       total_bytes,
            "compression": compression,
        }

    # ── Event log orchestration (in-memory convenience) ──────────────────────

    def generate_event_log(
        self, batch_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Produce the full transactional event log as a single in-memory DataFrame.

        Streams internally in user batches to cap peak memory, then concatenates.
        For populations that do not comfortably fit in RAM, prefer
        `stream_event_log` or `write_parquet`.

        Columns: timestamp, user_id, profile, event_type, volume_usd.
        One row per (user, day) where the user's state was Active.
        """
        parts = [df for df in self.stream_event_log(batch_size) if not df.empty]
        if not parts:
            return pd.DataFrame(
                {c: pd.Series(dtype=self._column_dtype(c)) for c in self.OUTPUT_COLUMNS}
            )
        events_df = pd.concat(parts, ignore_index=True, copy=False)
        events_df.sort_values(
            by=["timestamp", "user_id"], kind="stable", ignore_index=True, inplace=True,
        )
        return events_df

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _column_dtype(col: str) -> str:
        return {
            "timestamp":  "datetime64[ns]",
            "user_id":    "object",
            "profile":    "object",
            "event_type": "object",
            "volume_usd": "float64",
        }[col]
