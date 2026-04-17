-- Dimension: market regime at timestamp (join on ts bucket / as-of).
CREATE TABLE IF NOT EXISTS dim_market_regime
(
    regime_ts           DateTime64(3, 'UTC'),
    regime_id           LowCardinality(String) COMMENT 'bull|bear|chop|high_vol|...',
    vol_annualized      Float64,
    trend_return_30d    Float64,
    benchmark_symbol    LowCardinality(String) DEFAULT 'BTC',
    ingest_version      UInt32 DEFAULT 1
)
ENGINE = MergeTree
ORDER BY (regime_ts, regime_id);
