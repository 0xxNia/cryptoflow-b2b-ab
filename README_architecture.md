# CryptoFlow — Data Architecture

## Data Flow

```
generator.py
    │
    ├─► exposure_df          ──► ClickHouse: cryptoflow.exposure_log
    │   (one row per user)
    │
    └─► transactions_df      ──► ClickHouse: cryptoflow.transaction_events
        (one row per event)        │
                                   └─► MV: mv_daily_volume
                                       (SummingMergeTree by regime/style)
```

## Module Roles

| File | Purpose |
|---|---|
| `generator.py` | Synthetic data — Pareto volumes, NegBinom counts, Markov regimes |
| `schemas/exposure.py` | Pydantic validation for exposure events at API boundary |
| `schemas/transaction.py` | Pydantic validation for transaction events at API boundary |
| `stats.py` | CUPED · mSPRT · Bayesian (continuous + binary) · SRM |
| `market.py` | Winsorization · regime-conditional A/B segmentation |
| `sql/clickhouse/` | ClickHouse DDL — tables and materialized views |

## Statistical Distributions

**Trade volumes** — `Pareto(α = 1.5)`

Financial volume is power-law distributed. The Pareto with `α = 1.5` has
infinite variance, which captures whale behaviour realistically.
`x_min = mean/3` ensures the distribution mean matches persona parameters.

```
volume = x_min × (1 − u)^{−1/α},   u ~ Uniform(0, 1)
```

**Trade counts per week** — `NegativeBinomial(r = 2, p)`

Poisson (variance = mean) under-represents burstiness in real trading.
NegBinomial with `r = 2` gives `variance = mean + mean²/2`, matching
the over-dispersion observed in crypto transaction data.

```
p = r / (r + μ)   where μ = target_trades_per_week
```

**Market regime** — First-order Markov chain over `{bull, bear, high_vol}`

The regime sequence is shared across all users for the same calendar day —
regime is an event-level property, not a user property. The transition matrix
produces a stationary distribution of ≈ 50 % bull / 35 % bear / 15 % high_vol.

```
                 bull    bear  high_vol    stationary
    bull     [ 0.940,  0.040,  0.020 ]  →  50 %
    bear     [ 0.057,  0.913,  0.030 ]  →  35 %
    high_vol [ 0.067,  0.070,  0.863 ]  →  15 %
```

## Validation Layer (schemas/)

Every event passes through a Pydantic v2 model before reaching ClickHouse:

- `ExposureEventPayload` — validates enrollment events; enforces tz-aware
  timestamp, experiment slug pattern, and `variant_id ≠ experiment_id`.
- `TransactionEventPayload` — enforces coin + side for trade/liquidation
  events, and `fee_usd ≤ volume_usd`.
- `*Row` subclasses add server-side fields (`event_id`, `volume_usd_w`,
  `ingest_time`) and expose `to_clickhouse_dict()`.

## Statistical Methods

### CUPED  (`stats.cuped`)
```
Y_adj = Y − θ·(X − E[X]),   θ = Cov(Y, X) / Var(X)
```
Variance reduction ≈ 1 − (1 − ρ²) where ρ = corr(pre, post).
At ρ = 0.87 this yields ~75 % variance reduction, roughly doubling
effective sample size without extra traffic.

### mSPRT  (`stats.msprt`)
```
log Λ_t = −½ log(1 + τ²/V) + τ²·δ²/(2·V·(V + τ²))
```
Always-valid p-value: `p_t = min(1, 1/Λ_t)`.  Reject when `Λ_t ≥ 1/α`.
Prevents p-value inflation from repeated interim looks (peeking problem).

### Bayesian Expected Loss — continuous  (`stats.bayesian_ab`)
```
E[Loss | launch] = σ_δ·φ(z) − δ·(1 − Φ(z)),   z = δ/σ_δ
```
Converts to USD via `revenue_per_unit`.  Decision: launch when
`P(treatment wins) ≥ threshold_pct`.

### Bayesian Expected Loss — binary  (`stats.bayesian_ab_binary`)
Beta-Binomial conjugate model with Monte Carlo Expected Loss (50 k samples).
Used for conversion rate, retention, and activation metrics.

### SRM Guard  (`stats.srm_test`)
Chi-square goodness-of-fit on observed vs expected assignment counts.
Uses `α = 0.01` (stricter than usual) because SRM is an infrastructure
failure, not an experimental signal.

### Regime Analysis  (`market.regime_analysis`)
Aggregates transactions to per-user totals within each `market_regime`,
then runs Welch t-tests and reports effect, CI, and MDE per regime.
Answers: "was the treatment effective only in high-volatility markets?"

## ClickHouse Schema Notes

- `exposure_log` — `ReplacingMergeTree(timestamp)` deduplicates
  re-assignments; ORDER BY `(experiment_id, user_id)`.
- `transaction_events` — `MergeTree`; ORDER BY `(experiment_id, variant_id, user_id, timestamp)`.
- `mv_daily_volume` — `SummingMergeTree` pre-aggregates `volume_usd_w`
  and `fee_usd` by `(experiment_id, variant_id, event_date, market_regime, trading_style)`.
- Both tables use `SAMPLE BY intHash32(user_id)` for deterministic sampling
  in exploratory queries.
- `volume_usd_w` (Winsorized at 99th pct) is stored pre-computed to avoid
  repeated window-function overhead in aggregation queries.
