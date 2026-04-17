"""
Professional A/B testing statistical engine.

Three methods:
  CUPED       — variance reduction via pre-experiment covariate
  mSPRT       — always-valid sequential test (solves peeking problem)
  Bayesian AB — Expected Loss in $, decision framework
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats as spstats


# ═══════════════════════════════════════════════════════════════════════════════
# CUPED — Controlled-experiment Using Pre-Experiment Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CUPEDResult:
    theta: float                    # covariate coefficient (Cov(Y,X)/Var(X))
    pre_post_corr: float            # correlation between covariate and outcome
    variance_reduction_pct: float   # % variance reduction achieved

    # Naive (unadjusted) test
    naive_effect: float
    naive_se: float
    naive_p: float
    naive_ci: tuple[float, float]

    # CUPED-adjusted test
    adj_effect: float
    adj_se: float
    adj_p: float
    adj_ci: tuple[float, float]

    control_adj: np.ndarray
    treatment_adj: np.ndarray


def cuped(
    control_cov: np.ndarray,    # pre-experiment covariate, control group
    treatment_cov: np.ndarray,  # pre-experiment covariate, treatment group
    control_post: np.ndarray,   # post-experiment metric, control group
    treatment_post: np.ndarray, # post-experiment metric, treatment group
) -> CUPEDResult:
    """
    Adjust outcomes using pre-experiment covariate to reduce variance.

    Y_adj = Y - theta * (X - E[X])
    where theta = Cov(Y, X) / Var(X)
    """
    all_cov  = np.concatenate([control_cov, treatment_cov])
    all_post = np.concatenate([control_post, treatment_post])

    var_x = np.var(all_cov, ddof=1)
    theta = np.cov(all_post, all_cov, ddof=1)[0, 1] / var_x if var_x > 0 else 0.0
    mean_cov = np.mean(all_cov)

    control_adj   = control_post   - theta * (control_cov   - mean_cov)
    treatment_adj = treatment_post - theta * (treatment_cov - mean_cov)

    # Variance reduction
    var_raw = np.var(all_post, ddof=1)
    var_adj = np.var(np.concatenate([control_adj, treatment_adj]), ddof=1)
    var_reduction = (1 - var_adj / var_raw) * 100 if var_raw > 0 else 0.0
    corr = np.corrcoef(all_cov, all_post)[0, 1]

    def _test(c, t):
        eff = np.mean(t) - np.mean(c)
        se  = np.sqrt(np.var(c, ddof=1) / len(c) + np.var(t, ddof=1) / len(t))
        _, p = spstats.ttest_ind(t, c, equal_var=False)
        ci = (eff - 1.96 * se, eff + 1.96 * se)
        return eff, se, float(p), ci

    naive_eff, naive_se, naive_p, naive_ci = _test(control_post, treatment_post)
    adj_eff,   adj_se,   adj_p,   adj_ci   = _test(control_adj, treatment_adj)

    return CUPEDResult(
        theta=float(theta),
        pre_post_corr=float(corr),
        variance_reduction_pct=float(var_reduction),
        naive_effect=naive_eff, naive_se=naive_se, naive_p=naive_p, naive_ci=naive_ci,
        adj_effect=adj_eff,     adj_se=adj_se,     adj_p=adj_p,     adj_ci=adj_ci,
        control_adj=control_adj,
        treatment_adj=treatment_adj,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# mSPRT — mixture Sequential Probability Ratio Test
# (Johari et al., 2015 — "Always Valid Inference")
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class mSPRTResult:
    # Per-checkpoint arrays (length = number of checkpoints)
    lambda_series: np.ndarray      # Lambda_t — likelihood ratio
    p_series: np.ndarray           # always-valid p-value = min(1, 1/Lambda_t)
    naive_p_series: np.ndarray     # classical t-test p at same checkpoints (shows inflation)
    n_c_series: np.ndarray         # cumulative control n
    n_t_series: np.ndarray         # cumulative treatment n

    # Summary
    alpha: float
    tau: float                     # prior std used
    stopping_point: int | None     # first checkpoint where p <= alpha (None = no early stop)
    final_p: float
    final_lambda: float
    significant: bool


def msprt(
    snapshots: list[tuple[int, float, int, float]],  # (n_c, mean_c, n_t, mean_t) per checkpoint
    sigma: float,
    alpha: float = 0.05,
    tau: float | None = None,
) -> mSPRTResult:
    """
    Always-valid sequential test using a normal mixture prior.

    snapshots: cumulative (n_control, mean_control, n_treatment, mean_treatment)
               at each checkpoint — must be strictly increasing in n_c, n_t.
    sigma:     estimated pooled standard deviation of per-user metric.
    tau:       prior std for effect size (default = 0.1 * sigma, ~10% effect expected).
    """
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be a positive finite number")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if tau is None:
        tau = sigma * 0.1
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be a positive finite number")

    lambdas, naive_ps, n_cs, n_ts = [], [], [], []

    for n_c, mean_c, n_t, mean_t in snapshots:
        n_cs.append(n_c)
        n_ts.append(n_t)

        if n_c < 2 or n_t < 2:
            lambdas.append(1.0)
            naive_ps.append(1.0)
            continue

        # Variance of the difference-in-means estimator
        V = sigma ** 2 * (1.0 / n_c + 1.0 / n_t)
        effect = mean_t - mean_c

        # mSPRT log-likelihood ratio (normal mixture)
        # log Λ_t = -½ log(1 + τ²/V) + τ²·effect² / (2V·(V+τ²))
        log_lam = (
            -0.5 * np.log(1.0 + tau ** 2 / V)
            + tau ** 2 * effect ** 2 / (2.0 * V * (V + tau ** 2))
        )
        lambdas.append(float(np.exp(np.clip(log_lam, -40, 40))))

        # Classical t-test at same checkpoint (to show peeking inflation)
        t_stat = effect / np.sqrt(V)
        df = n_c + n_t - 2
        naive_ps.append(float(2 * spstats.t.sf(abs(t_stat), df=df)))

    lam_arr = np.array(lambdas)
    p_arr   = np.minimum(1.0 / np.maximum(lam_arr, 1e-10), 1.0)
    naive_arr = np.array(naive_ps)

    threshold = 1.0 / alpha
    stopping = None
    for i, lam in enumerate(lam_arr):
        if lam >= threshold:
            stopping = i
            break

    return mSPRTResult(
        lambda_series=lam_arr,
        p_series=p_arr,
        naive_p_series=naive_arr,
        n_c_series=np.array(n_cs),
        n_t_series=np.array(n_ts),
        alpha=alpha,
        tau=float(tau),
        stopping_point=stopping,
        final_p=float(p_arr[-1]),
        final_lambda=float(lam_arr[-1]),
        significant=stopping is not None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian A/B — Expected Loss framework
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BayesianResult:
    # Posterior on effect (mu_t - mu_c)
    effect_mean: float
    effect_std: float
    credible_interval_95: tuple[float, float]

    # Decision metrics
    prob_treatment_wins: float      # P(μ_t > μ_c)
    expected_loss_launch: float     # E[loss | launch treatment] in metric units
    expected_loss_hold: float       # E[loss | keep control]   in metric units
    expected_loss_launch_usd: float # same, in $
    expected_loss_hold_usd: float

    # Recommendation
    decision: str                   # "launch" | "reject" | "gather_more_data"
    threshold_pct: float


def bayesian_ab(
    control_obs: np.ndarray,
    treatment_obs: np.ndarray,
    revenue_per_unit: float = 1.0,
    threshold_pct: float = 95.0,
) -> BayesianResult:
    """
    Bayesian A/B test with Expected Loss in $.

    Uses Normal-Normal conjugate model.
    Expected Loss = E[max(0, delta)] under the wrong decision.
    Converts to $ via revenue_per_unit (e.g., avg_fee_per_trade).
    """
    n_c, mu_c = len(control_obs), np.mean(control_obs)
    n_t, mu_t = len(treatment_obs), np.mean(treatment_obs)
    if n_c < 2 or n_t < 2:
        raise ValueError("control_obs and treatment_obs must each contain at least 2 observations")
    if not (0 < threshold_pct < 100):
        raise ValueError("threshold_pct must be in (0, 100)")
    var_c = np.var(control_obs,   ddof=1) / n_c
    var_t = np.var(treatment_obs, ddof=1) / n_t

    effect = mu_t - mu_c
    sigma_d = np.sqrt(var_c + var_t)
    if not np.isfinite(sigma_d):
        raise ValueError("posterior std is not finite; check input values")

    # P(treatment wins)
    if sigma_d == 0:
        prob_win = 1.0 if effect > 0 else 0.0 if effect < 0 else 0.5
    else:
        prob_win = float(1 - spstats.norm.cdf(0, loc=effect, scale=sigma_d))

    # Expected Loss: E[max(0, X)] where X ~ N(delta, sigma_d)
    # = sigma_d * phi(delta/sigma_d) - delta * (1 - Phi(delta/sigma_d))
    def _el(delta: float, sigma: float) -> float:
        z = delta / sigma if sigma > 0 else np.sign(delta) * 1e6
        return sigma * spstats.norm.pdf(z) - delta * (1 - spstats.norm.cdf(z))

    el_launch = _el(effect,  sigma_d)   # regret if we launch and treatment is worse
    el_hold   = _el(-effect, sigma_d)   # regret if we hold and treatment was better

    if sigma_d == 0:
        ci = (float(effect), float(effect))
    else:
        ci = (
            float(spstats.norm.ppf(0.025, loc=effect, scale=sigma_d)),
            float(spstats.norm.ppf(0.975, loc=effect, scale=sigma_d)),
        )

    threshold = threshold_pct / 100.0
    if prob_win >= threshold:
        decision = "launch"
    elif (1 - prob_win) >= threshold:
        decision = "reject"
    else:
        decision = "gather_more_data"

    return BayesianResult(
        effect_mean=float(effect),
        effect_std=float(sigma_d),
        credible_interval_95=ci,
        prob_treatment_wins=prob_win,
        expected_loss_launch=float(el_launch),
        expected_loss_hold=float(el_hold),
        expected_loss_launch_usd=float(el_launch * revenue_per_unit),
        expected_loss_hold_usd=float(el_hold * revenue_per_unit),
        decision=decision,
        threshold_pct=threshold_pct,
    )
