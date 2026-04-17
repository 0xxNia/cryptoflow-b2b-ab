"""Sample size calculations for A/B experiments."""
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class SampleSizeResult:
    n_per_variant: int
    n_total: int
    baseline: float
    treatment_value: float
    mde_pct: float
    alpha: float
    power: float
    days_to_complete: float


def required_sample_size(
    baseline: float,
    mde_pct: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sigma: float | None = None,
    metric_type: str = "continuous",
    daily_users: int = 100,
) -> SampleSizeResult:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    delta = abs(baseline * mde_pct / 100)

    if delta < 1e-12:
        raise ValueError("MDE cannot be zero")

    if metric_type == "proportion":
        p1 = float(np.clip(baseline, 1e-6, 1 - 1e-6))
        p2 = float(np.clip(baseline + delta, 1e-6, 1 - 1e-6))
        p_bar = (p1 + p2) / 2
        n_per_variant = int(np.ceil(
            (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
             + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            / delta ** 2
        ))
    else:
        sigma_used = sigma if sigma is not None else baseline  # CV=1 default
        n_per_variant = int(np.ceil(
            2 * sigma_used ** 2 * (z_alpha + z_beta) ** 2 / delta ** 2
        ))

    n_total = n_per_variant * 2
    days_to_complete = n_total / max(daily_users, 1)

    return SampleSizeResult(
        n_per_variant=n_per_variant,
        n_total=n_total,
        baseline=baseline,
        treatment_value=baseline * (1 + mde_pct / 100),
        mde_pct=mde_pct,
        alpha=alpha,
        power=power,
        days_to_complete=days_to_complete,
    )


def sensitivity_curve(
    baseline: float,
    alpha: float = 0.05,
    power: float = 0.80,
    sigma: float | None = None,
    metric_type: str = "continuous",
    daily_users: int = 100,
    mde_min: float = 2.0,
    mde_max: float = 50.0,
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mde_values_pct, n_per_variant, days_to_complete) arrays."""
    mdes = np.linspace(mde_min, mde_max, n_points)
    ns, days = [], []
    for mde in mdes:
        r = required_sample_size(baseline, mde, alpha, power, sigma, metric_type, daily_users)
        ns.append(r.n_per_variant)
        days.append(r.days_to_complete)
    return mdes, np.array(ns), np.array(days)
