-- Snowflake: mirror of ClickHouse exposure mart (adjust clustering / streams).
CREATE TABLE IF NOT EXISTS fact_experiment_exposure (
    ts                  TIMESTAMP_NTZ,
    user_id             VARCHAR,
    experiment_id       VARCHAR,
    variant_id          VARCHAR,
    assignment_version  VARCHAR,
    market_regime       VARCHAR,
    unit_type           VARCHAR DEFAULT 'user',
    platform            VARCHAR,
    app_version         VARCHAR,
    ingest_batch_id     VARCHAR DEFAULT UUID_STRING()
);
