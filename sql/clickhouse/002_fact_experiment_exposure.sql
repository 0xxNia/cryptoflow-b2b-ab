-- Fact: first-class exposure log for A/B (SQL-ready mart).
CREATE TABLE IF NOT EXISTS fact_experiment_exposure
(
    ts                  DateTime64(3, 'UTC'),
    user_id             String,
    experiment_id       LowCardinality(String),
    variant_id          LowCardinality(String),
    assignment_version  LowCardinality(String),
    market_regime       LowCardinality(Nullable(String)),
    unit_type           LowCardinality(String) DEFAULT 'user',
    platform            LowCardinality(Nullable(String)),
    app_version         Nullable(String),
    ingest_batch_id     String DEFAULT toString(generateUUIDv4())
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (experiment_id, ts, user_id);
