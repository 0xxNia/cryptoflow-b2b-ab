-- ============================================================================
-- Table: cryptoflow.transaction_events
-- Purpose: Immutable event stream of all trade/deposit/withdrawal/liquidation
--          events for users enrolled in A/B experiments.
--
-- CRITICAL DESIGN NOTE:
--   market_regime is an EVENT property, not a user property.
--   It records the macro-market state AT THE MOMENT THE TRADE OCCURRED,
--   enabling regime-conditional A/B analysis (e.g., "was the fee-reduction
--   experiment effective during high_vol vs bear regimes?").
--
-- Engine: MergeTree — append-only event log; no deduplication needed as
--         each event_id is globally unique.
-- Partition: (toDate(timestamp), experiment_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS cryptoflow.transaction_events
(
    -- ── Identity ─────────────────────────────────────────────────────────────
    event_id            UUID
        DEFAULT generateUUIDv4()
        COMMENT 'Globally unique event identifier (UUID v4)',
    experiment_id       LowCardinality(String)
        COMMENT 'Parent experiment identifier',
    variant_id          LowCardinality(String)
        COMMENT 'A/B variant: "control" | "treatment"',
    user_id             String
        COMMENT 'Platform user identifier',

    -- ── Timing ───────────────────────────────────────────────────────────────
    timestamp           DateTime64(3, 'UTC')
        COMMENT 'Event occurrence timestamp (ms precision)',
    event_date          Date
        MATERIALIZED toDate(timestamp)
        COMMENT 'Materialized partition key component',

    -- ── Event classification ──────────────────────────────────────────────────
    event_type          Enum8(
                            'trade'        = 1,
                            'deposit'      = 2,
                            'withdrawal'   = 3,
                            'liquidation'  = 4
                        )
        COMMENT 'Type of financial event',

    -- ── Market context at event time (EVENT property, NOT user property) ─────
    -- This field is determined by the platform market-state service at the
    -- time the event is processed. Two users trading at the same moment will
    -- have the SAME market_regime value, regardless of their user profiles.
    market_regime       Enum8(
                            'bull'      = 1,
                            'bear'      = 2,
                            'high_vol'  = 3
                        )
        COMMENT 'Macro market regime at event time — event-level property',

    -- ── Trade details ─────────────────────────────────────────────────────────
    coin                LowCardinality(String)
        COMMENT 'Asset traded, e.g. "BTC", "ETH"',
    side                Enum8('buy' = 1, 'sell' = 2)
        COMMENT 'Trade direction',
    volume_usd          Float64
        COMMENT 'Raw trade volume in USD (Pareto-distributed, may contain whales)',
    fee_usd             Float64
        COMMENT 'Fee charged in USD',

    -- ── Winsorized volume (pre-computed at ingestion for performance) ─────────
    -- Capped at the 99th percentile of volume_usd computed over a rolling
    -- 30-day window per experiment. Prevents whale distortion in aggregations.
    volume_usd_w        Float64
        COMMENT 'Winsorized volume: min(volume_usd, p99_rolling_30d)',

    -- ── Denormalised user attributes (snapshot at event time) ────────────────
    -- Stored here to avoid joins during statistical aggregations on this table.
    trading_style       LowCardinality(String)
        COMMENT 'User trading style at event time',
    wallet_size         LowCardinality(String)
        COMMENT 'User wallet size tier at event time',

    -- ── Ingestion metadata ───────────────────────────────────────────────────
    ingest_time         DateTime
        DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY (toDate(timestamp), experiment_id)
ORDER BY   (experiment_id, variant_id, user_id, timestamp)
SAMPLE BY   intHash32(user_id)
SETTINGS
    index_granularity = 8192;


-- Bloom-filter indexes for selective point lookups
ALTER TABLE cryptoflow.transaction_events
    ADD INDEX idx_user_id    user_id     TYPE bloom_filter(0.01) GRANULARITY 4;
ALTER TABLE cryptoflow.transaction_events
    ADD INDEX idx_event_type event_type  TYPE set(4)             GRANULARITY 1;
ALTER TABLE cryptoflow.transaction_events
    ADD INDEX idx_regime     market_regime TYPE set(3)           GRANULARITY 1;


-- ============================================================================
-- Materialized view: daily per-variant volume aggregates
-- Used by the regime-conditional analysis pipeline (market.py)
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS cryptoflow.mv_daily_volume
ENGINE = SummingMergeTree()
PARTITION BY (experiment_id, market_regime)
ORDER BY    (experiment_id, variant_id, event_date, market_regime, trading_style)
AS
SELECT
    experiment_id,
    variant_id,
    event_date,
    market_regime,
    trading_style,
    count()                         AS event_count,
    sum(volume_usd_w)               AS total_volume_w,
    sum(fee_usd)                    AS total_fee,
    countIf(event_type = 'trade')   AS trade_count
FROM cryptoflow.transaction_events
GROUP BY
    experiment_id, variant_id, event_date,
    market_regime, trading_style;
