"""
cryptoflow.guardrails — Runtime monitoring of guardrail metrics.

Given the hypothesis (declared before launch) and the current observed values
of each guardrail metric, decide which are within tolerance, which are warning,
and which have breached hard enough to require killing the experiment.

Public API:
  GuardrailStatus   Enum (OK / WARNING / BREACHED / CRITICAL)
  GuardrailBreach   per-metric evaluation
  evaluate_guardrails(...) → list[GuardrailBreach]
  breach_verdict(breaches) → ExperimentVerdict  (aggregate decision)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from preflight import ExperimentHypothesis, GuardrailMetric, MetricDirection


# ─────────────────────────────────────────────────────────────────────────────
# Status levels
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailStatus(str, Enum):
    OK       = "ok"        # observed is better-than-or-equal-to baseline
    WARNING  = "warning"   # degraded but under max_degradation_pct
    BREACHED = "breached"  # degraded >= max_degradation_pct (non-critical)
    CRITICAL = "critical"  # breached AND guardrail.critical=True → kill switch


class ExperimentVerdict(str, Enum):
    HEALTHY        = "healthy"          # all guardrails OK or WARNING
    REVIEW         = "review"           # at least one non-critical BREACHED
    KILL           = "kill"             # at least one CRITICAL breach


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailBreach:
    """Evaluation of a single guardrail against its observed value."""
    name:                 str
    status:               GuardrailStatus
    baseline:             float
    observed:             float
    degradation_pct:      float   # relative % degradation (signed — negative means improvement)
    threshold_pct:        float   # max tolerated degradation before BREACHED
    critical:             bool
    direction:            MetricDirection
    message:              str     # human-readable summary


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _classify(
    guardrail: GuardrailMetric,
    observed:  float,
) -> GuardrailBreach:
    """
    Classify a single guardrail.

    The "degradation" is measured as relative movement in the BAD direction:
      HIGHER_IS_BETTER: bad = observed drops below baseline
      LOWER_IS_BETTER : bad = observed rises above baseline

    degradation_pct > 0  ⟹  metric moved the wrong way
    degradation_pct ≤ 0  ⟹  metric moved the right way (improvement)
    """
    if guardrail.baseline == 0:
        raise ValueError(
            f"guardrail '{guardrail.name}' has baseline=0; cannot compute relative degradation"
        )

    if guardrail.direction == MetricDirection.HIGHER_IS_BETTER:
        # Drop below baseline is bad
        degradation_pct = (guardrail.baseline - observed) / guardrail.baseline * 100.0
    else:
        # Rise above baseline is bad
        degradation_pct = (observed - guardrail.baseline) / guardrail.baseline * 100.0

    # Classify
    if degradation_pct <= 0:
        status  = GuardrailStatus.OK
        message = (
            f"{guardrail.name}: observed {observed:,.4f} vs baseline {guardrail.baseline:,.4f} "
            f"— within tolerance (improvement of {-degradation_pct:.2f}%)."
        )
    elif degradation_pct < guardrail.max_degradation_pct:
        status  = GuardrailStatus.WARNING
        message = (
            f"{guardrail.name}: degraded by {degradation_pct:.2f}% "
            f"(threshold {guardrail.max_degradation_pct:.2f}%). Monitor closely."
        )
    elif guardrail.critical:
        status  = GuardrailStatus.CRITICAL
        message = (
            f"CRITICAL BREACH on {guardrail.name}: degraded by {degradation_pct:.2f}% "
            f"(threshold {guardrail.max_degradation_pct:.2f}%). "
            "Experiment should be stopped immediately."
        )
    else:
        status  = GuardrailStatus.BREACHED
        message = (
            f"BREACH on {guardrail.name}: degraded by {degradation_pct:.2f}% "
            f"(threshold {guardrail.max_degradation_pct:.2f}%). "
            "Review required before continuing."
        )

    return GuardrailBreach(
        name=guardrail.name,
        status=status,
        baseline=guardrail.baseline,
        observed=observed,
        degradation_pct=float(degradation_pct),
        threshold_pct=guardrail.max_degradation_pct,
        critical=guardrail.critical,
        direction=guardrail.direction,
        message=message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_guardrails(
    hypothesis:      ExperimentHypothesis,
    observed_values: dict[str, float],
) -> list[GuardrailBreach]:
    """
    Evaluate each guardrail declared in the hypothesis against its observed value.

    Parameters
    ----------
    hypothesis:
        Pre-registered ExperimentHypothesis defining the guardrails.
    observed_values:
        {metric_name: current_value} — a snapshot of each guardrail's current
        value in the treatment arm. Missing keys yield a WARNING breach so
        that silent missing-metric pipelines do not hide regressions.
    """
    results: list[GuardrailBreach] = []

    for g in hypothesis.guardrails:
        if g.name not in observed_values:
            results.append(GuardrailBreach(
                name=g.name,
                status=GuardrailStatus.WARNING,
                baseline=g.baseline,
                observed=float("nan"),
                degradation_pct=float("nan"),
                threshold_pct=g.max_degradation_pct,
                critical=g.critical,
                direction=g.direction,
                message=(
                    f"{g.name}: no observed value supplied. "
                    "Metric pipeline may be broken — cannot assess guardrail."
                ),
            ))
            continue

        results.append(_classify(g, float(observed_values[g.name])))

    return results


def breach_verdict(breaches: list[GuardrailBreach]) -> ExperimentVerdict:
    """
    Aggregate per-metric breaches into a single experiment-level decision.

      CRITICAL    → KILL
      BREACHED    → REVIEW
      WARNING/OK  → HEALTHY
    """
    if any(b.status == GuardrailStatus.CRITICAL for b in breaches):
        return ExperimentVerdict.KILL
    if any(b.status == GuardrailStatus.BREACHED for b in breaches):
        return ExperimentVerdict.REVIEW
    return ExperimentVerdict.HEALTHY
