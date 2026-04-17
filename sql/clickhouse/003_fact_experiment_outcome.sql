-- Fact: per-user per-window outcomes joined to experiments (grain: user × experiment × window).
CREATE TABLE IF NOT EXISTS fact_experiment_outcome
(
    window_start        DateTime64(3, 'UTC'),
    window_end          DateTime64(3, 'UTC'),
    user_id             String,
    experiment_id       LowCardinality(String),
    variant_id          LowCardinality(String),
    metric_key           LowCardinality(String),
    metric_value         Float64,
    cuped_covariate      Nullable(Float64),
    market_regime        LowCardinality(Nullable(String)),
    segment_key          LowCardinality(Nullable(String)) COMMENT 'scalper|vip|post_liquidation|...'
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(window_end)
ORDER BY (experiment_id, metric_key, window_end, user_id);
