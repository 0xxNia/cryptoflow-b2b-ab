"""
cryptoflow.market — Winsorization and regime-conditional A/B analysis.

Two public functions:
  winsorize(arr, pct)     — cap array values at the given percentile
  regime_analysis(...)    — A/B results segmented by market_regime

market_regime is an EVENT-level property (set by the platform's market-state
service at ingestion time), not a user-level attribute.  Segmenting by regime
answers "was the treatment effective in bull vs bear vs high-volatility markets?"
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as spstats

from stats import SRMResult, srm_test


# ─────────────────────────────────────────────────────────────────────────────
# Winsorization
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """
    Cap values at the given percentile.  Returns a new array.

    Used to tame whale-driven outliers in trade volumes before computing
    per-user aggregates for A/B analysis.  A 99th-pct cap preserves 99%
    of the distribution while removing the most extreme tail observations.
    """
    if not (0 < pct <= 100):
        raise ValueError("pct must be in (0, 100]")
    cap = float(np.percentile(arr, pct))
    return np.minimum(arr, cap)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeStats:
    """A/B statistics for a single market regime (or "overall" across all regimes)."""
    regime:              str
    n_control:           int
    n_treatment:         int
    control_mean:        float
    treatment_mean:      float
    effect:              float             # absolute: treatment_mean − control_mean
    relative_uplift_pct: float             # effect / control_mean × 100
    se:                  float             # standard error of the effect estimate
    t_stat:              float
    p_value:             float
    ci_95:               tuple[float, float]
    mde_80_pct:          float             # MDE at 80% power, α=0.05
    significant:         bool
    alpha:               float


@dataclass
class RegimeAnalysisResult:
    """Full regime-conditional A/B analysis result."""
    experiment_id:     str
    metric:            str
    control_variant:   str
    treatment_variant: str
    overall:           RegimeStats
    by_regime:         dict[str, RegimeStats]  # keys: "bull", "bear", "high_vol"
    srm:               SRMResult


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats_for_pair(
    control:   np.ndarray,
    treatment: np.ndarray,
    regime:    str,
    alpha:     float = 0.05,
) -> RegimeStats:
    """Compute Welch t-test statistics for one (control, treatment) pair."""
    n_c, n_t = len(control), len(treatment)

    if n_c < 2 or n_t < 2:
        mu_c = float(np.mean(control))   if n_c > 0 else float("nan")
        mu_t = float(np.mean(treatment)) if n_t > 0 else float("nan")
        return RegimeStats(
            regime=regime, n_control=n_c, n_treatment=n_t,
            control_mean=mu_c, treatment_mean=mu_t,
            effect=float("nan"), relative_uplift_pct=float("nan"),
            se=float("nan"), t_stat=float("nan"), p_value=float("nan"),
            ci_95=(float("nan"), float("nan")), mde_80_pct=float("nan"),
            significant=False, alpha=alpha,
        )

    mu_c, mu_t = float(np.mean(control)), float(np.mean(treatment))
    effect     = mu_t - mu_c

    var_c = float(np.var(control,   ddof=1)) / n_c
    var_t = float(np.var(treatment, ddof=1)) / n_t
    se    = float(np.sqrt(var_c + var_t))

    if se == 0:
        t_stat  = float("inf") if effect > 0 else float("-inf") if effect < 0 else 0.0
        p_value = 0.0 if effect != 0 else 1.0
    else:
        t_stat          = float(effect / se)
        _, p_value      = spstats.ttest_ind(treatment, control, equal_var=False)
        p_value         = float(p_value)

    ci = (effect - 1.96 * se, effect + 1.96 * se)

    # MDE at 80% power, α=alpha (two-sided Welch)
    # MDE = (z_{α/2} + z_{0.80}) × pooled_σ × √(2 / n̄)
    # where n̄ = harmonic mean of n_c and n_t
    pooled_var = (
        (n_c - 1) * np.var(control, ddof=1) + (n_t - 1) * np.var(treatment, ddof=1)
    ) / (n_c + n_t - 2)
    n_harm = 2.0 / (1.0 / n_c + 1.0 / n_t)
    z_sum  = float(spstats.norm.ppf(1.0 - alpha / 2.0) + spstats.norm.ppf(0.80))
    mde    = z_sum * float(np.sqrt(pooled_var)) * float(np.sqrt(2.0 / n_harm))

    rel_uplift = (effect / mu_c * 100.0) if mu_c != 0 else float("nan")

    return RegimeStats(
        regime=regime,
        n_control=n_c,
        n_treatment=n_t,
        control_mean=mu_c,
        treatment_mean=mu_t,
        effect=effect,
        relative_uplift_pct=rel_uplift,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_95=(float(ci[0]), float(ci[1])),
        mde_80_pct=float(mde),
        significant=p_value < alpha,
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def regime_analysis(
    transactions:      pd.DataFrame,
    control_variant:   str   = "control",
    treatment_variant: str   = "treatment",
    metric:            str   = "volume_usd_w",
    experiment_id:     str   = "",
    alpha:             float = 0.05,
) -> RegimeAnalysisResult:
    """
    Compute regime-conditional A/B test results.

    The function aggregates transactions to per-user totals within each
    market_regime, then runs Welch t-tests comparing control vs treatment.
    This surfaces whether the treatment effect varies across market conditions.

    Parameters
    ----------
    transactions:      DataFrame with columns: variant_id, market_regime,
                       user_id, and the target metric column.
    metric:            Column to aggregate per (user, regime) — default "volume_usd_w".
    alpha:             Significance level for all hypothesis tests.

    Returns
    -------
    RegimeAnalysisResult with overall stats, per-regime breakdown, and SRM check.
    """
    required = {"variant_id", "market_regime", "user_id", metric}
    missing  = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions DataFrame is missing columns: {missing}")

    # ── SRM guard ──────────────────────────────────────────────────────────────
    srm_counts = (
        transactions.groupby("variant_id")["user_id"].nunique().to_dict()
    )
    srm = srm_test(srm_counts)

    # ── Overall: per-user metric across all regimes ───────────────────────────
    user_overall = (
        transactions
        .groupby(["user_id", "variant_id"])[metric]
        .sum()
        .reset_index()
    )
    ctrl_overall = (
        user_overall.loc[user_overall["variant_id"] == control_variant, metric]
        .values.astype(float)
    )
    trt_overall = (
        user_overall.loc[user_overall["variant_id"] == treatment_variant, metric]
        .values.astype(float)
    )
    overall = _stats_for_pair(ctrl_overall, trt_overall, "overall", alpha)

    # ── Per-regime: per-user metric within each regime ────────────────────────
    user_by_regime = (
        transactions
        .groupby(["user_id", "variant_id", "market_regime"])[metric]
        .sum()
        .reset_index()
    )

    by_regime: dict[str, RegimeStats] = {}
    for regime in ("bull", "bear", "high_vol"):
        sub   = user_by_regime[user_by_regime["market_regime"] == regime]
        ctrl  = sub.loc[sub["variant_id"] == control_variant,   metric].values.astype(float)
        trt   = sub.loc[sub["variant_id"] == treatment_variant, metric].values.astype(float)
        by_regime[regime] = _stats_for_pair(ctrl, trt, regime, alpha)

    return RegimeAnalysisResult(
        experiment_id=experiment_id,
        metric=metric,
        control_variant=control_variant,
        treatment_variant=treatment_variant,
        overall=overall,
        by_regime=by_regime,
        srm=srm,
    )
