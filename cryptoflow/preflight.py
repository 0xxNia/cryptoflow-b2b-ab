"""
cryptoflow.hypothesis — Pre-launch hypothesis constructor.

Forces the PM to think in units of business impact BEFORE running the test:

  • Primary metric + expected MDE  → required sample size & days
  • Guardrail metrics              → ceilings that must not be breached
  • Unit value in $                → expected upside and false-positive risk in $

Public API:
  ExperimentHypothesis   Pydantic model, validated at UI submit time
  BaselineStats          observed metric statistics from the data warehouse
  preflight_check()      returns a PreflightReport with feasibility + risk in $
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated

from calculator import required_sample_size


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class MetricDirection(str, Enum):
    """
    Semantic direction of the metric.

    HIGHER_IS_BETTER — DTV, revenue, retention, activation rate.
    LOWER_IS_BETTER  — churn, latency, error rate, margin call rate.

    For primary metrics, direction describes which way we WANT the treatment
    to move the metric.  For guardrails, the OPPOSITE direction triggers
    a breach (e.g., a LOWER_IS_BETTER guardrail like churn breaches when
    observed > baseline × (1 + max_degradation_pct/100)).
    """
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER  = "lower_is_better"


class MetricType(str, Enum):
    CONTINUOUS = "continuous"    # DTV, revenue, latency
    PROPORTION = "proportion"    # conversion, retention, activation


# ─────────────────────────────────────────────────────────────────────────────
# Input models (Pydantic — validated at API boundary)
# ─────────────────────────────────────────────────────────────────────────────

class PrimaryMetric(BaseModel):
    """The single KPI the experiment is designed to move."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    name: Annotated[str, Field(min_length=1, max_length=64)]
    baseline: Annotated[
        float,
        Field(gt=0.0, description="Current value of the metric (from data warehouse)"),
    ]
    std_dev: Annotated[
        float,
        Field(gt=0.0, description="Observed standard deviation (for power calculation)"),
    ]
    direction: MetricDirection
    expected_mde_pct: Annotated[
        float,
        Field(gt=0.0, le=100.0, description="Expected relative improvement, %"),
    ]
    unit_value_usd: Annotated[
        float,
        Field(ge=0.0, description="$ value of one unit (for risk/upside in USD)"),
    ] = 1.0
    metric_type: MetricType = MetricType.CONTINUOUS


class GuardrailMetric(BaseModel):
    """A metric that must NOT degrade beyond a specified threshold."""

    model_config = {"frozen": True, "str_strip_whitespace": True}

    name: Annotated[str, Field(min_length=1, max_length=64)]
    baseline: Annotated[
        float,
        Field(gt=0.0, description="Current value of the guardrail metric"),
    ]
    direction: MetricDirection = Field(
        description=(
            "Direction in which the metric SHOULD move. A breach is defined as "
            "movement in the OPPOSITE direction beyond max_degradation_pct."
        )
    )
    max_degradation_pct: Annotated[
        float,
        Field(
            gt=0.0, le=100.0,
            description="Maximum tolerated relative degradation, %",
        ),
    ]
    critical: bool = Field(
        default=False,
        description="If True, breach auto-kills the experiment instead of warning",
    )


class ExperimentHypothesis(BaseModel):
    """
    Full hypothesis submitted before launching an experiment.

    Validated at the API boundary.  Stored as the immutable
    pre-registration record for the experiment.
    """

    model_config = {"frozen": True, "str_strip_whitespace": True}

    experiment_id: Annotated[
        str,
        Field(min_length=3, max_length=128, pattern=r"^[a-z0-9_\-]+$"),
    ]
    hypothesis: Annotated[
        str,
        Field(
            min_length=20, max_length=500,
            description="Plain-text hypothesis, e.g. 'Reducing taker fee from 25 to "
                        "15 bps will increase DTV by 5% without raising churn by more than 1pp.'",
        ),
    ]
    primary: PrimaryMetric
    guardrails: list[GuardrailMetric] = Field(default_factory=list)

    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    power: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.80
    control_share: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.50

    @field_validator("guardrails")
    @classmethod
    def _unique_guardrail_names(cls, v: list[GuardrailMetric]) -> list[GuardrailMetric]:
        names = [g.name for g in v]
        if len(names) != len(set(names)):
            raise ValueError("guardrail names must be unique")
        return v

    @model_validator(mode="after")
    def _primary_not_in_guardrails(self) -> "ExperimentHypothesis":
        guard_names = {g.name for g in self.guardrails}
        if self.primary.name in guard_names:
            raise ValueError(
                f"primary metric '{self.primary.name}' cannot also appear as a guardrail"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Runtime input: observed baseline from warehouse
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineStats:
    """
    Snapshot of current platform state used to populate the preflight report.

    Populated from a warehouse query over the last 30 days (or similar window).
    """
    daily_new_users: int           # eligible users entering the experiment per day
    horizon_days_after_launch: int = 30  # window over which risk/upside is accrued


# ─────────────────────────────────────────────────────────────────────────────
# Output: preflight report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreflightReport:
    """Everything a PM needs to decide GO / NO-GO before launch."""

    # Sizing
    required_n_per_arm: int
    required_n_total:   int
    days_to_complete:   float

    # Business translation — all in USD over horizon_days_after_launch
    expected_upside_usd:         float  # if treatment truly improves by expected_mde_pct
    false_positive_risk_usd:     float  # α × worst-case loss if treatment is secretly −MDE
    false_negative_opportunity:  float  # (1−power) × missed upside if we fail to detect

    # Verdict
    feasible:  bool             # can the test complete within a reasonable window?
    warnings:  list[str]
    verdict:   str              # one-line UI headline


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic: if the test requires more than this many days at current traffic,
# it's too slow — business context changes faster than the test can conclude.
_MAX_REASONABLE_DAYS = 90


def preflight_check(
    hypothesis:      ExperimentHypothesis,
    baseline_stats:  BaselineStats,
) -> PreflightReport:
    """
    Translate a hypothesis into required sample size, duration, and risk in USD.

    The $ figures assume the treatment effect is sustained over
    `baseline_stats.horizon_days_after_launch` days post-launch, at the
    platform's current daily traffic (`baseline_stats.daily_new_users`).
    """
    primary = hypothesis.primary

    # ── Sizing via existing calculator ────────────────────────────────────────
    sizing = required_sample_size(
        baseline=primary.baseline,
        mde_pct=primary.expected_mde_pct,
        alpha=hypothesis.alpha,
        power=hypothesis.power,
        sigma=primary.std_dev,
        metric_type=primary.metric_type.value,
        daily_users=max(baseline_stats.daily_new_users, 1),
    )

    # ── Business translation ──────────────────────────────────────────────────
    # Daily $ impact of a 1×MDE shift in the primary metric across all users:
    daily_users       = max(baseline_stats.daily_new_users, 1)
    mde_fraction      = primary.expected_mde_pct / 100.0
    daily_value_delta = (
        primary.baseline * mde_fraction * daily_users * primary.unit_value_usd
    )
    horizon           = baseline_stats.horizon_days_after_launch

    # Probabilistic outcomes over the post-launch horizon:
    #   upside  ≈ power × (gain per day) × horizon        (correct-launch scenario)
    #   fp_risk ≈ α     × (loss per day) × horizon        (launched a losing variant)
    #   fn_opp  ≈ (1−power) × (gain per day) × horizon    (held back a winning variant)
    expected_upside = hypothesis.power         * daily_value_delta * horizon
    fp_risk         = hypothesis.alpha         * daily_value_delta * horizon
    fn_opportunity  = (1.0 - hypothesis.power) * daily_value_delta * horizon

    # ── Feasibility & warnings ────────────────────────────────────────────────
    warnings: list[str] = []
    feasible = sizing.days_to_complete <= _MAX_REASONABLE_DAYS

    if not feasible:
        warnings.append(
            f"Test needs {sizing.days_to_complete:.0f} days at current traffic "
            f"({daily_users:,}/day) — exceeds {_MAX_REASONABLE_DAYS}-day cap. "
            "Reduce MDE, relax α, or increase traffic allocation."
        )

    if not hypothesis.guardrails:
        warnings.append(
            "No guardrails defined. Tests without guardrails can silently "
            "damage retention, margin, or other critical metrics."
        )

    if primary.unit_value_usd == 1.0:
        warnings.append(
            "unit_value_usd=1.0 (default). Set a realistic $-per-unit value "
            "to get meaningful risk/upside estimates."
        )

    if sizing.days_to_complete < 7:
        warnings.append(
            f"Test completes in < 7 days ({sizing.days_to_complete:.1f}). "
            "Short runs risk weekday/weekend seasonality bias — consider a "
            "minimum 7-day floor."
        )

    # ── Headline verdict ──────────────────────────────────────────────────────
    if not feasible:
        verdict = (
            f"NOT FEASIBLE — {sizing.days_to_complete:.0f} days required, "
            f"cap is {_MAX_REASONABLE_DAYS}."
        )
    else:
        verdict = (
            f"Test requires {sizing.n_total:,} users over "
            f"{sizing.days_to_complete:.0f} days. "
            f"Expected upside: ${expected_upside:,.0f} over {horizon}d. "
            f"False-positive risk: ${fp_risk:,.0f}."
        )

    return PreflightReport(
        required_n_per_arm=sizing.n_per_variant,
        required_n_total=sizing.n_total,
        days_to_complete=sizing.days_to_complete,
        expected_upside_usd=float(expected_upside),
        false_positive_risk_usd=float(fp_risk),
        false_negative_opportunity=float(fn_opportunity),
        feasible=feasible,
        warnings=warnings,
        verdict=verdict,
    )
