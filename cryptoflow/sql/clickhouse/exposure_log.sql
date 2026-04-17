-- ============================================================================
-- Table: cryptoflow.exposure_log
-- Purpose: Immutable record of every A/B assignment event.
--          One row per (experiment_id, user_id); duplicates resolved by
--          ReplacingMergeTree keeping the row with the latest timestamp.
-- Engine: ReplacingMergeTree — deduplicates re-assignments on merge.
-- Partition: (toDate(timestamp), experiment_id) — satisfies both daily
--            roll-up queries and per-experiment full scans.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS cryptoflow;

CREATE TABLE IF NOT EXISTS cryptoflow.exposure_log
(
    -- ── Identity ─────────────────────────────────────────────────────────────
    experiment_id       LowCardinality(String)
        COMMENT 'Experiment slug, e.g. "fee_reduction_2024q1"',
    variant_id          LowCardinality(String)
        COMMENT 'Variant label: "control" | "treatment" | "treatment_b"',
    user_id             String
        COMMENT 'Stable platform user identifier',

    -- ── Timing ───────────────────────────────────────────────────────────────
    timestamp           DateTime64(3, 'UTC')
        COMMENT 'Assignment event timestamp (millisecond precision)',
    assigned_date       Date
        MATERIALIZED toDate(timestamp)
        COMMENT 'Materialized partition key component',

    -- ── Behavioural profile at time of assignment (snapshot, NOT live) ───────
    trading_style       LowCardinality(String)
        COMMENT 'hodler | swing | scalper',
    risk_profile        LowCardinality(String)
        COMMENT 'conservative | moderate | degen',
    wallet_size         LowCardinality(String)
        COMMENT 'minnow | dolphin | whale',
    churn_sensitivity   LowCardinality(String)
        COMMENT 'sticky | neutral | mercenary',

    -- ── Derived behavioural metrics (pre-computed at assignment time) ─────────
    expected_dtv_usd    Float64
        COMMENT 'Expected Daily Trading Volume in USD (power-law mean)',
    pre_exp_dtv_30d     Float64
        COMMENT '30-day pre-experiment DTV — primary CUPED covariate',
    fee_elasticity      Float32
        COMMENT 'Price-elasticity coefficient for fee sensitivity analysis',

    -- ── Ingestion metadata ───────────────────────────────────────────────────
    ingest_time         DateTime
        DEFAULT now()
        COMMENT 'Row insertion timestamp (server-side)'
)
ENGINE = ReplacingMergeTree(timestamp)
PARTITION BY (toDate(timestamp), experiment_id)
ORDER BY   (experiment_id, user_id)
SAMPLE BY   intHash32(user_id)
SETTINGS
    index_granularity        = 8192,
    merge_with_ttl_timeout   = 86400;


-- Bloom-filter index for point lookups by user_id
ALTER TABLE cryptoflow.exposure_log
    ADD INDEX idx_user_id user_id TYPE bloom_filter(0.01) GRANULARITY 4;

-- ============================================================================
-- Materialized view: per-experiment assignment counts (SRM pre-check)
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS cryptoflow.mv_exposure_counts
ENGINE = AggregatingMergeTree()
PARTITION BY experiment_id
ORDER BY (experiment_id, variant_id)
AS
SELECT
    experiment_id,
    variant_id,
    countState()   AS user_count_state,
    minState(timestamp) AS first_assignment_state,
    maxState(timestamp) AS last_assignment_state
FROM cryptoflow.exposure_log
GROUP BY experiment_id, variant_id;
