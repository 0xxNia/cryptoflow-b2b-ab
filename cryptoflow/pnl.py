"""
cryptoflow.pnl — Bayesian posterior → stakeholder-ready financial narrative.

A finance sponsor does not want to read `P(μ_t > μ_c) = 0.92`. They want:
    "If we roll out variant B, we expect +$1.2M over the next 12 months.
     There is an 8% chance we actually lose more than $50,000."

This module projects the per-user Bayesian posterior from `stats.bayesian_ab`
onto a rollout population + time horizon, producing:
  • expected $ impact ± 95% credible interval
  • P(loss > threshold) for a configurable list of $-thresholds
  • a ranked list of plain-English bullets ready for a Streamlit / PDF block
  • a tail-risk-aware decision verdict that overrides the per-user "launch"
    verdict when the total $ tail risk exceeds what finance tolerates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from scipy import stats as spstats

try:
    from .stats import BayesianResult
except ImportError:  # pragma: no cover — supports sys.path-based loading from dashboard.py
    from stats import BayesianResult  # type: ignore[no-redef]


DEFAULT_LOSS_THRESHOLDS_USD: Final[tuple[float, ...]] = (
    10_000.0, 50_000.0, 250_000.0, 1_000_000.0,
)
# If the tail probability of losing this much exceeds DEFAULT_MAX_TAIL_PROB,
# we override a "launch" verdict with "gather_more_data" — finance sign-off
# rule of thumb, tune per org.
DEFAULT_CATASTROPHIC_LOSS_USD: Final[float] = 250_000.0
DEFAULT_MAX_TAIL_PROB:          Final[float] = 0.05


# ── Result container ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FinancialPnL:
    """Dollarised projection of a Bayesian experiment outcome."""
    posterior_mean_usd: float
    posterior_std_usd:  float
    ci95_usd:           tuple[float, float]

    prob_net_loss:      float                 # P(total_impact < 0)
    tail_probabilities: dict[float, float]    # threshold_usd → P(loss > threshold)

    scale_to_usd:       float                 # multiplier used, exposed for audit
    horizon_label:      str                   # "over the next 12 months", etc.

    # Verdicts
    finance_decision:   str                   # "launch" | "hold" | "reject"
    recommendation:     str                   # one-sentence summary

    # Rendered bullets for a dashboard
    narrative:          list[str] = field(default_factory=list)


# ── Core projection ──────────────────────────────────────────────────────────

def financial_pnl(
    effect_mean_per_unit: float,
    effect_std_per_unit:  float,
    scale_to_usd:         float,
    loss_thresholds_usd:  tuple[float, ...] = DEFAULT_LOSS_THRESHOLDS_USD,
    horizon_label:        str   = "over the next 12 months",
    catastrophic_loss_usd: float = DEFAULT_CATASTROPHIC_LOSS_USD,
    max_tail_prob:        float = DEFAULT_MAX_TAIL_PROB,
    per_unit_decision:    str   = "gather_more_data",
) -> FinancialPnL:
    """
    Project a per-unit posterior N(effect_mean_per_unit, effect_std_per_unit)
    onto a rollout scale (`scale_to_usd` = users × horizon × revenue_per_unit)
    and compute the resulting $-denominated posterior + tail probabilities.

    Loss is defined as the negative of the rollout impact. If the per-user
    posterior says `effect ~ N(+0.10, 0.30)` and scale_to_usd = 10_000_000,
    the rollout impact is N(+$1M, $3M) — there is still meaningful probability
    of a multi-million-dollar loss despite a positive mean. That is the story
    finance wants to see.

    `per_unit_decision` is the upstream BayesianResult.decision value — used
    so this function can override it when the $-scale tail risk exceeds
    `max_tail_prob` at `catastrophic_loss_usd`.
    """
    if effect_std_per_unit < 0:
        raise ValueError("effect_std_per_unit must be non-negative")
    if not np.isfinite(effect_std_per_unit) or not np.isfinite(effect_mean_per_unit):
        raise ValueError("posterior parameters must be finite")
    if not (0 < max_tail_prob < 1):
        raise ValueError("max_tail_prob must be in (0, 1)")

    mean_usd = float(effect_mean_per_unit) * float(scale_to_usd)
    std_usd  = abs(float(effect_std_per_unit) * float(scale_to_usd))

    # P(loss > X) = P(total_impact < -X)
    def _tail(threshold: float) -> float:
        if std_usd == 0:
            return 1.0 if mean_usd < -threshold else 0.0
        return float(spstats.norm.cdf(-threshold, loc=mean_usd, scale=std_usd))

    tail_probs = {float(t): _tail(t) for t in sorted(loss_thresholds_usd)}
    prob_net_loss = _tail(0.0)

    ci95 = (mean_usd - 1.96 * std_usd, mean_usd + 1.96 * std_usd)

    # ── Finance-level decision override ──
    catastrophic_prob = _tail(catastrophic_loss_usd)
    if per_unit_decision == "launch" and catastrophic_prob > max_tail_prob:
        decision = "hold"
        rec = (
            f"Finance override: per-user test says LAUNCH, but "
            f"P(loss > {_fmt_usd(catastrophic_loss_usd)}) = {catastrophic_prob:.1%} "
            f"exceeds the {max_tail_prob:.0%} tail-risk ceiling. "
            "Gather more data before rollout."
        )
    elif per_unit_decision == "launch":
        decision = "launch"
        rec = (
            f"Launch: expected impact {_fmt_signed_usd(mean_usd)} {horizon_label}, "
            f"with a {catastrophic_prob:.1%} chance of losing more than "
            f"{_fmt_usd(catastrophic_loss_usd)}."
        )
    elif per_unit_decision == "reject":
        decision = "reject"
        rec = (
            f"Reject: treatment is worse in expectation "
            f"({_fmt_signed_usd(mean_usd)} {horizon_label}). Do not roll out."
        )
    else:
        decision = "hold"
        rec = (
            f"Hold: evidence is inconclusive. Expected impact "
            f"{_fmt_signed_usd(mean_usd)} {horizon_label} with 95% CI "
            f"[{_fmt_signed_usd(ci95[0])}, {_fmt_signed_usd(ci95[1])}]. "
            "Keep running."
        )

    narrative = _build_narrative(
        mean_usd=mean_usd,
        std_usd=std_usd,
        ci95=ci95,
        prob_net_loss=prob_net_loss,
        tail_probs=tail_probs,
        horizon_label=horizon_label,
        decision=decision,
        recommendation=rec,
    )

    return FinancialPnL(
        posterior_mean_usd=mean_usd,
        posterior_std_usd=std_usd,
        ci95_usd=(float(ci95[0]), float(ci95[1])),
        prob_net_loss=float(prob_net_loss),
        tail_probabilities=tail_probs,
        scale_to_usd=float(scale_to_usd),
        horizon_label=horizon_label,
        finance_decision=decision,
        recommendation=rec,
        narrative=narrative,
    )


# ── Convenience wrapper around stats.BayesianResult ──────────────────────────

def pnl_from_bayesian(
    result:                BayesianResult,
    rollout_users:         int,
    weekly_units_per_user: float,
    horizon_weeks:         int = 52,
    revenue_per_unit:      float = 1.0,
    loss_thresholds_usd:   tuple[float, ...] = DEFAULT_LOSS_THRESHOLDS_USD,
    catastrophic_loss_usd: float = DEFAULT_CATASTROPHIC_LOSS_USD,
    max_tail_prob:         float = DEFAULT_MAX_TAIL_PROB,
) -> FinancialPnL:
    """
    Build a FinancialPnL from a BayesianResult plus rollout assumptions.

    rollout_users         : projected rollout population (e.g. 5_000_000 MAU)
    weekly_units_per_user : how many metric units each user produces per week
                            (e.g. avg weekly trade count if `result` was run
                            on per-user weekly trade count)
    horizon_weeks         : business reporting horizon (52 = annualised)
    revenue_per_unit      : USD value of one metric unit. MUST match the
                            definition used when constructing BayesianResult.
    """
    if rollout_users <= 0 or horizon_weeks <= 0:
        raise ValueError("rollout_users and horizon_weeks must be positive")
    if weekly_units_per_user < 0 or revenue_per_unit < 0:
        raise ValueError("weekly_units_per_user and revenue_per_unit must be >= 0")

    scale = rollout_users * weekly_units_per_user * horizon_weeks * revenue_per_unit
    horizon_label = _horizon_label(horizon_weeks)

    return financial_pnl(
        effect_mean_per_unit=result.effect_mean,
        effect_std_per_unit=result.effect_std,
        scale_to_usd=scale,
        loss_thresholds_usd=loss_thresholds_usd,
        horizon_label=horizon_label,
        catastrophic_loss_usd=catastrophic_loss_usd,
        max_tail_prob=max_tail_prob,
        per_unit_decision=result.decision,
    )


# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_usd(x: float) -> str:
    a = abs(x)
    if a >= 1_000_000:
        return f"${a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"${a / 1_000:.0f}k"
    return f"${a:,.0f}"


def _fmt_signed_usd(x: float) -> str:
    sign = "+" if x >= 0 else "−"
    return f"{sign}{_fmt_usd(x)}"


def _horizon_label(weeks: int) -> str:
    if weeks >= 52 and weeks % 52 == 0:
        years = weeks // 52
        return f"over the next {years} year{'s' if years > 1 else ''}"
    if weeks >= 4 and weeks % 4 == 0:
        months = weeks // 4
        return f"over the next {months} months"
    return f"over the next {weeks} weeks"


def _build_narrative(
    mean_usd:       float,
    std_usd:        float,
    ci95:           tuple[float, float],
    prob_net_loss:  float,
    tail_probs:     dict[float, float],
    horizon_label:  str,
    decision:       str,
    recommendation: str,
) -> list[str]:
    bullets: list[str] = []
    bullets.append(
        f"Expected P&L impact {horizon_label}: "
        f"{_fmt_signed_usd(mean_usd)} (95% CI: "
        f"{_fmt_signed_usd(ci95[0])} .. {_fmt_signed_usd(ci95[1])})."
    )
    bullets.append(
        f"Probability of a net loss {horizon_label}: {prob_net_loss:.1%}."
    )
    for thr, p in tail_probs.items():
        bullets.append(
            f"Probability of losing more than {_fmt_usd(thr)}: {p:.1%}."
        )
    bullets.append(f"Recommendation: {recommendation}")
    return bullets
