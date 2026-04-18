"""
cryptoflow.settlement — late-arriving-data guard for A/B analysis.

Crypto events (especially from mobile clients on flaky networks) routinely
arrive 12–48h after the original event time. Running mSPRT / SRM / CUPED on
raw same-day data produces "ragged" cohorts: users who will still generate
backfilled events are compared against users who already have complete
histories. The result is spurious Sample Ratio Mismatch alerts and biased
treatment effects.

This module defines a single policy object that is applied at the edge of
every statistical pipeline: raw events in → settled + deduped frame out.

Mirrors the SQL-side guard in
    cryptoflow/sql/clickhouse/transaction_events.sql :: transaction_events_settled
Keep DEFAULT_SETTLEMENT_HOURS in sync with the INTERVAL clause there.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import numpy as np
import pandas as pd


# Industry norm for crypto mobile telemetry: 99%+ of retries land within 24h,
# 48h gives comfortable tail coverage without starving near-real-time analysis.
DEFAULT_SETTLEMENT_HOURS: Final[float] = 48.0


# ── Policy ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SettlementPolicy:
    """
    Applies a settlement window + event_id dedup to an event frame.

    The guarantees this policy provides to downstream statistical code:
      1. No event with timestamp > (now − window) is ever returned — so
         cohorts are frozen at the cutoff and cannot grow retroactively.
      2. At most one row per event_id — the row with the latest ingest_time
         wins (matching the ClickHouse ReplacingMergeTree(ingest_time) DDL).
      3. A deterministic `unsettled_count` is computed before filtering, so
         ops can alert if the tail share exceeds a threshold.
    """
    window_hours:     float = DEFAULT_SETTLEMENT_HOURS
    timestamp_col:    str   = "timestamp"
    event_id_col:     str   = "event_id"
    ingest_time_col:  str   = "ingest_time"

    def __post_init__(self) -> None:
        if self.window_hours < 0:
            raise ValueError("window_hours must be non-negative")

    # ── Core primitives ──────────────────────────────────────────────────────

    def cutoff(self, now: datetime | None = None) -> datetime:
        """Return the upper-bound event timestamp that is considered settled."""
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now - timedelta(hours=self.window_hours)

    def apply(
        self,
        events: pd.DataFrame,
        now: datetime | None = None,
        dedup: bool = True,
    ) -> pd.DataFrame:
        """
        Return a copy of `events` with late-arriving rows removed and optional
        event_id deduplication (latest ingest_time wins). The input frame is
        not mutated.

        If `event_id_col` or `ingest_time_col` are not present, dedup is skipped
        silently — this lets the policy be applied to minimal simulator output
        (no event_id) as well as to fully-schema'd ClickHouse rows.
        """
        if events.empty:
            return events.copy()

        ts = pd.to_datetime(events[self.timestamp_col], utc=True)
        cutoff_ts = pd.Timestamp(self.cutoff(now))
        settled = events.loc[ts <= cutoff_ts].copy()

        if dedup and self.event_id_col in settled.columns:
            sort_cols = [self.event_id_col]
            if self.ingest_time_col in settled.columns:
                settled = settled.sort_values(
                    [self.event_id_col, self.ingest_time_col],
                    kind="stable",
                )
            settled = settled.drop_duplicates(subset=sort_cols, keep="last")
            settled.reset_index(drop=True, inplace=True)

        return settled

    def unsettled_share(
        self,
        events: pd.DataFrame,
        now: datetime | None = None,
    ) -> float:
        """Fraction of rows whose timestamp lies inside the unsettled window."""
        if events.empty:
            return 0.0
        ts = pd.to_datetime(events[self.timestamp_col], utc=True)
        cutoff_ts = pd.Timestamp(self.cutoff(now))
        return float((ts > cutoff_ts).mean())


# ── Cohort helpers for SRM and downstream stats ──────────────────────────────

def settled_cohort_counts(
    exposure:      pd.DataFrame,
    transactions:  pd.DataFrame,
    policy:        SettlementPolicy | None = None,
    now:           datetime | None = None,
    variant_col:   str = "variant_id",
    user_col:      str = "user_id",
    ts_col:        str = "timestamp",
) -> dict[str, int]:
    """
    Count users per variant whose exposure AND at least one transaction have
    settled under the policy. This is the correct denominator for
    `cryptoflow.stats.srm_test` — counting raw exposure rows triggers false
    SRM alerts whenever network lag slightly unbalances the two variants.

    Returns {variant_id: settled_user_count}.
    """
    if policy is None:
        policy = SettlementPolicy()

    cutoff_ts = pd.Timestamp(policy.cutoff(now))

    exp_ts = pd.to_datetime(exposure[ts_col], utc=True)
    settled_exposure = exposure.loc[exp_ts <= cutoff_ts]

    settled_tx = policy.apply(transactions, now=now)
    tx_users = set(settled_tx[user_col].unique()) if not settled_tx.empty else set()

    settled_exposure = settled_exposure[settled_exposure[user_col].isin(tx_users)]
    counts = settled_exposure.groupby(variant_col)[user_col].nunique().to_dict()
    return {str(v): int(c) for v, c in counts.items()}


def annotate_settlement(
    events: pd.DataFrame,
    policy: SettlementPolicy | None = None,
    now:    datetime | None = None,
) -> pd.DataFrame:
    """
    Return a copy of `events` with an extra boolean column `is_settled`.

    Useful for dashboards that want to render settled vs pending counts side
    by side without filtering rows out.
    """
    if policy is None:
        policy = SettlementPolicy()
    if events.empty:
        out = events.copy()
        out["is_settled"] = np.array([], dtype=bool)
        return out
    ts = pd.to_datetime(events[policy.timestamp_col], utc=True)
    cutoff_ts = pd.Timestamp(policy.cutoff(now))
    out = events.copy()
    out["is_settled"] = (ts <= cutoff_ts).to_numpy()
    return out
