import sys
from pathlib import Path

# Ensure imports work regardless of working directory
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st
import duckdb
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as spstats

from calculator import required_sample_size, sensitivity_curve
from agents import ROBOTS, SCENARIOS, simulate
from stats import cuped, msprt, bayesian_ab

st.set_page_config(page_title="CryptoFlow Analytics", page_icon="📊", layout="wide")

_DB_PATH = str(_HERE / "data" / "cryptoflow.duckdb")

@st.cache_resource
def get_connection():
    return duckdb.connect(_DB_PATH, read_only=True)

def _safe_float(value, default=0.0) -> float:
    if value is None:
        return float(default)
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


try:
    con = get_connection()
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

st.title("📊 CryptoFlow Exchange — Product Analytics")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")

try:
    style_opts  = ["All"] + con.execute("SELECT DISTINCT trading_style FROM users ORDER BY 1").df()["trading_style"].tolist()
    wallet_opts = ["All"] + con.execute("SELECT DISTINCT wallet_size FROM users ORDER BY 1").df()["wallet_size"].tolist()
    churn_opts  = ["All"] + con.execute("SELECT DISTINCT churn_sensitivity FROM users ORDER BY 1").df()["churn_sensitivity"].tolist()
except Exception as e:
    st.error(f"Failed to load filter options from database: {e}")
    st.stop()

sel_style  = st.sidebar.selectbox("Trading Style", style_opts)
sel_wallet = st.sidebar.selectbox("Wallet Size", wallet_opts)
sel_churn  = st.sidebar.selectbox("Churn Sensitivity", churn_opts)

conditions = []
if sel_style  != "All": conditions.append(f"u.trading_style = '{_sql_literal(sel_style)}'")
if sel_wallet != "All": conditions.append(f"u.wallet_size = '{_sql_literal(sel_wallet)}'")
if sel_churn  != "All": conditions.append(f"u.churn_sensitivity = '{_sql_literal(sel_churn)}'")
where   = ("WHERE " + " AND ".join(conditions)) if conditions else ""
where_u = ("WHERE " + " AND ".join([c.replace("u.", "") for c in conditions])) if conditions else ""

# ── KPI cards ─────────────────────────────────────────────────────────────────
kpis = con.execute(f"""
    SELECT
        COUNT(DISTINCT u.user_id)                                    AS total_users,
        COUNT(DISTINCT t.user_id)                                    AS trading_users,
        ROUND(SUM(t.fee_usd), 0)                                     AS total_revenue,
        ROUND(COUNT(DISTINCT t.user_id)*100.0/COUNT(DISTINCT u.user_id),1) AS activation_rate,
        ROUND(SUM(t.fee_usd)/COUNT(DISTINCT t.user_id), 1)          AS rev_per_user
    FROM users u
    LEFT JOIN trades t ON u.user_id = t.user_id
    {where}
""").df()

c1, c2, c3, c4, c5 = st.columns(5)
total_users = int(_safe_float(kpis["total_users"][0], default=0))
trading_users = int(_safe_float(kpis["trading_users"][0], default=0))
total_revenue = _safe_float(kpis["total_revenue"][0], default=0.0)
activation_rate = _safe_float(kpis["activation_rate"][0], default=0.0)
rev_per_user = _safe_float(kpis["rev_per_user"][0], default=0.0)

c1.metric("Total Users",     f"{total_users:,}")
c2.metric("Trading Users",   f"{trading_users:,}")
c3.metric("Fee Revenue",     f"${int(total_revenue):,}")
c4.metric("Activation Rate", f"{activation_rate:.1f}%")
c5.metric("Revenue / User",  f"${rev_per_user:,.0f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_seg, tab_ret, tab_ab, tab_churn, tab_calc, tab_sim = st.tabs([
    "💰 Сегменты", "📈 Retention", "🧪 A/B Тест", "⚠️ Churn Risk",
    "🧮 Калькулятор", "🤖 Симуляция",
])

# ── Tab 1: Revenue segments ───────────────────────────────────────────────────
with tab_seg:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Revenue by Trading Style × Wallet")
        seg = con.execute(f"""
            SELECT t.trading_style, t.wallet_size,
                   ROUND(SUM(t.fee_usd), 0) AS revenue
            FROM trades t JOIN users u ON t.user_id = u.user_id
            {where}
            GROUP BY t.trading_style, t.wallet_size
            ORDER BY revenue DESC
        """).df()
        fig1 = px.bar(seg, x="trading_style", y="revenue", color="wallet_size",
                      barmode="group",
                      color_discrete_map={"whale": "#00C896", "dolphin": "#4A90D9", "minnow": "#E8724A"},
                      text_auto=".2s")
        fig1.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                           font_color="white", yaxis_title="Fee Revenue ($)",
                           legend=dict(bgcolor="#1a1d27"))
        st.plotly_chart(fig1, width="stretch")

    with col_r:
        st.subheader("Revenue per User by Persona")
        rpu = con.execute(f"""
            SELECT u.trading_style, u.churn_sensitivity,
                   ROUND(SUM(t.fee_usd)/COUNT(DISTINCT t.user_id), 1) AS rev_per_user
            FROM trades t JOIN users u ON t.user_id = u.user_id
            {where}
            GROUP BY u.trading_style, u.churn_sensitivity
        """).df()
        pivot = rpu.pivot(index="trading_style", columns="churn_sensitivity", values="rev_per_user")
        fig2 = px.imshow(pivot, color_continuous_scale="teal", text_auto=True, aspect="auto")
        fig2.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                           font_color="white", coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

# ── Tab 2: Retention ──────────────────────────────────────────────────────────
with tab_ret:
    st.subheader("Retention Curves by Trading Style")
    ret = con.execute(f"""
        WITH first_trade AS (
            SELECT t.user_id, u.trading_style, MIN(t.traded_at) AS ft
            FROM trades t JOIN users u ON t.user_id = u.user_id
            {where}
            GROUP BY t.user_id, u.trading_style
        ),
        uw AS (
            SELECT f.user_id, f.trading_style,
                   FLOOR(DATEDIFF('day', f.ft, t.traded_at)/7) AS wk
            FROM first_trade f JOIN trades t ON f.user_id = t.user_id
            WHERE FLOOR(DATEDIFF('day', f.ft, t.traded_at)/7) <= 8
        ),
        base AS (
            SELECT trading_style, COUNT(DISTINCT user_id) AS n
            FROM uw WHERE wk = 0 GROUP BY trading_style
        ),
        weekly AS (
            SELECT trading_style, wk, COUNT(DISTINCT user_id) AS active
            FROM uw GROUP BY trading_style, wk
        )
        SELECT w.trading_style, w.wk,
               ROUND(w.active * 100.0 / b.n, 1) AS retention_pct
        FROM weekly w JOIN base b ON w.trading_style = b.trading_style
        ORDER BY w.trading_style, w.wk
    """).df()

    colors = {"hodler": "#00C896", "swing": "#4A90D9", "scalper": "#E8724A"}
    fig3 = go.Figure()
    for style in ret["trading_style"].unique():
        d = ret[ret["trading_style"] == style]
        fig3.add_trace(go.Scatter(
            x=d["wk"], y=d["retention_pct"],
            name=style.capitalize(), mode="lines+markers",
            line=dict(color=colors.get(style, "#aaa"), width=3),
            marker=dict(size=8),
        ))
    fig3.update_layout(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", font_color="white",
        yaxis_title="Retention %", xaxis_title="Week since first trade",
        legend=dict(bgcolor="#1a1d27"),
    )
    st.plotly_chart(fig3, width="stretch")

# ── Tab 3: A/B test — advanced statistical engine ────────────────────────────
with tab_ab:
    st.subheader("🧪 Fee Reduction Experiment — Advanced Statistical Engine")
    st.caption("Experiment: Q3–Q4 2023 · Fee −40% in treatment group")

    @st.cache_data
    def load_ab_data():
        # Per-user: pre-experiment covariate + post-experiment metric
        df = con.execute("""
            WITH pre AS (
                SELECT a.user_id, a.ab_group,
                       COUNT(t.trade_id) AS pre_trades,
                       COALESCE(SUM(t.fee_usd), 0) AS pre_fee
                FROM ab_assignments a
                LEFT JOIN trades t ON a.user_id = t.user_id
                    AND t.traded_at < '2023-07-01'
                GROUP BY a.user_id, a.ab_group
            ),
            post AS (
                SELECT a.user_id,
                       COUNT(t.trade_id) AS post_trades,
                       COALESCE(SUM(t.fee_usd), 0) AS post_fee
                FROM ab_assignments a
                LEFT JOIN trades t ON a.user_id = t.user_id
                    AND t.traded_at >= '2023-07-01'
                    AND t.traded_at < '2023-11-01'
                GROUP BY a.user_id
            )
            SELECT p.user_id, p.ab_group,
                   p.pre_trades, p.pre_fee,
                   COALESCE(po.post_trades, 0) AS post_trades,
                   COALESCE(po.post_fee, 0)    AS post_fee
            FROM pre p LEFT JOIN post po ON p.user_id = po.user_id
        """).df()

        # Weekly snapshots for mSPRT (cumulative users by assignment week)
        weekly = con.execute("""
            WITH user_exp AS (
                SELECT a.user_id, a.ab_group,
                       DATEDIFF('week', '2023-07-01'::DATE, a.assigned_at::DATE) + 1 AS assign_week,
                       COUNT(t.trade_id) AS exp_trades,
                       COALESCE(SUM(t.fee_usd), 0) AS exp_fee
                FROM ab_assignments a
                LEFT JOIN trades t ON a.user_id = t.user_id
                    AND t.traded_at >= '2023-07-01'
                    AND t.traded_at < '2023-11-01'
                GROUP BY a.user_id, a.ab_group, a.assigned_at
            )
            SELECT assign_week, ab_group,
                   COUNT(*) AS n_new,
                   AVG(exp_trades) AS mean_trades
            FROM user_exp
            GROUP BY assign_week, ab_group
            ORDER BY assign_week, ab_group
        """).df()

        # Average fee per trade in the AB period (for $ Expected Loss)
        avg_fee = con.execute("""
            SELECT AVG(t.fee_usd) AS avg_fee_per_trade
            FROM ab_assignments a JOIN trades t ON a.user_id = t.user_id
            WHERE t.traded_at >= '2023-07-01' AND t.traded_at < '2023-11-01'
        """).df()["avg_fee_per_trade"][0]

        return df, weekly, _safe_float(avg_fee, default=0.0)

    ab_df, weekly_df, avg_fee_per_trade = load_ab_data()

    ctrl_df = ab_df[ab_df["ab_group"] == "control"]
    trt_df  = ab_df[ab_df["ab_group"] == "treatment"]

    # ── Section 1: Naive vs CUPED ─────────────────────────────────────────────
    st.markdown("### 1️⃣ Naive vs CUPED (Variance Reduction)")
    st.caption(
        "CUPED adjusts outcomes using pre-experiment trading history as covariate. "
        "Higher correlation → greater variance reduction → smaller required sample size."
    )

    cuped_res = cuped(
        control_cov=ctrl_df["pre_trades"].values.astype(float),
        treatment_cov=trt_df["pre_trades"].values.astype(float),
        control_post=ctrl_df["post_trades"].values.astype(float),
        treatment_post=trt_df["post_trades"].values.astype(float),
    )

    col_n, col_c = st.columns(2)
    _DARK = dict(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", font_color="white")

    with col_n:
        st.markdown("**Naive t-test**")
        sig_n = "✅ Significant" if cuped_res.naive_p < 0.05 else "❌ Not significant"
        m1, m2, m3 = st.columns(3)
        m1.metric("Effect",  f"{cuped_res.naive_effect:+.2f} trades")
        m2.metric("p-value", f"{cuped_res.naive_p:.4f}", delta=sig_n, delta_color="off")
        m3.metric("95% CI",  f"[{cuped_res.naive_ci[0]:+.2f}, {cuped_res.naive_ci[1]:+.2f}]")

    with col_c:
        st.markdown(f"**CUPED-adjusted** (θ = {cuped_res.theta:.3f})")
        sig_c = "✅ Significant" if cuped_res.adj_p < 0.05 else "❌ Not significant"
        m1, m2, m3 = st.columns(3)
        m1.metric("Effect",  f"{cuped_res.adj_effect:+.2f} trades")
        m2.metric("p-value", f"{cuped_res.adj_p:.4f}", delta=sig_c, delta_color="off")
        m3.metric("95% CI",  f"[{cuped_res.adj_ci[0]:+.2f}, {cuped_res.adj_ci[1]:+.2f}]")

    # Variance reduction banner
    vr_color = "#00C896" if cuped_res.variance_reduction_pct > 30 else "#F5A623"
    st.markdown(
        f"<div style='background:{vr_color}22;border-left:4px solid {vr_color};"
        f"padding:10px 16px;border-radius:4px;margin:8px 0'>"
        f"Pre/post correlation: <b>{cuped_res.pre_post_corr:.3f}</b> · "
        f"Variance reduced by <b>{cuped_res.variance_reduction_pct:.1f}%</b> · "
        f"Equivalent to <b>{1/(1-cuped_res.variance_reduction_pct/100):.1f}× more data</b> for free"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Distribution plot: control vs treatment (adjusted)
    fig_dist = go.Figure()
    _bins = dict(size=2)
    fig_dist.add_trace(go.Histogram(
        x=cuped_res.control_adj, name="Control (adj)", nbinsx=60,
        marker_color="#4A90D9", opacity=0.6,
    ))
    fig_dist.add_trace(go.Histogram(
        x=cuped_res.treatment_adj, name="Treatment (adj)", nbinsx=60,
        marker_color="#00C896", opacity=0.6,
    ))
    fig_dist.update_layout(
        **_DARK, barmode="overlay", title="CUPED-adjusted Trade Distribution",
        xaxis_title="Adjusted trades/user", yaxis_title="Users",
        legend=dict(bgcolor="#1a1d27"), height=280,
    )
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("---")

    # ── Section 2: mSPRT Sequential Test ─────────────────────────────────────
    st.markdown("### 2️⃣ Sequential Testing — mSPRT (Solves Peeking Problem)")
    st.caption(
        "Classical t-test p-value inflates with every look (peeking problem). "
        "mSPRT provides always-valid p-values: stop anytime without inflating Type I error."
    )

    # Build weekly cumulative snapshots
    sigma_est = float(np.std(ab_df["post_trades"].values, ddof=1)) if len(ab_df) > 1 else 0.0
    if not np.isfinite(sigma_est) or sigma_est <= 0:
        sigma_est = 1e-6
    snapshots = []
    cum_nc, cum_nt = 0, 0
    cum_sc, cum_st = 0.0, 0.0

    for wk in sorted(weekly_df["assign_week"].unique()):
        wk_df = weekly_df[weekly_df["assign_week"] == wk]
        c_row  = wk_df[wk_df["ab_group"] == "control"]
        t_row  = wk_df[wk_df["ab_group"] == "treatment"]
        if len(c_row):
            cum_nc += int(c_row["n_new"].values[0])
            cum_sc += float(c_row["n_new"].values[0] * c_row["mean_trades"].values[0])
        if len(t_row):
            cum_nt += int(t_row["n_new"].values[0])
            cum_st += float(t_row["n_new"].values[0] * t_row["mean_trades"].values[0])
        if cum_nc > 0 and cum_nt > 0:
            snapshots.append((cum_nc, cum_sc / cum_nc, cum_nt, cum_st / cum_nt))

    col_alpha, col_tau = st.columns([1, 3])
    with col_alpha:
        seq_alpha = st.select_slider("α", [0.01, 0.05, 0.10], value=0.05)
    with col_tau:
        tau_pct = st.slider("Prior expected effect (% of σ)", 5, 30, 10, 1,
                            help="τ = σ × this%. Conservative → harder to stop early.")
    tau_val = sigma_est * tau_pct / 100

    if snapshots:
        msprt_res = msprt(snapshots, sigma=sigma_est, alpha=seq_alpha, tau=tau_val)
    else:
        st.warning("Not enough weekly snapshots for sequential analysis.")
        msprt_res = None
    weeks_axis = list(range(1, len(snapshots) + 1))

    fig_seq = go.Figure()
    if msprt_res is not None:
        fig_seq.add_trace(go.Scatter(
            x=weeks_axis, y=msprt_res.p_series,
            name="mSPRT p (always valid)", mode="lines+markers",
            line=dict(color="#00C896", width=2), marker=dict(size=5),
        ))
        fig_seq.add_trace(go.Scatter(
            x=weeks_axis, y=msprt_res.naive_p_series,
            name="Classical t-test p (inflated by peeking)", mode="lines",
            line=dict(color="#E8724A", width=2, dash="dash"),
        ))
    fig_seq.add_hline(
        y=seq_alpha, line_color="white", line_dash="dot", line_width=1,
        annotation_text=f"α = {seq_alpha}", annotation_position="right",
    )
    if msprt_res is not None and msprt_res.stopping_point is not None:
        stop_wk = weeks_axis[msprt_res.stopping_point]
        fig_seq.add_vline(
            x=stop_wk, line_color="#F5A623", line_dash="dash",
            annotation_text=f"Early stop: week {stop_wk}",
            annotation_position="top left",
        )
    fig_seq.update_layout(
        **_DARK, title="Sequential p-values over experiment duration",
        xaxis_title="Week", yaxis_title="p-value",
        yaxis=dict(range=[0, 0.5]), legend=dict(bgcolor="#1a1d27"), height=340,
    )
    st.plotly_chart(fig_seq, width="stretch")

    sp_col1, sp_col2, sp_col3 = st.columns(3)
    if msprt_res is not None:
        sp_col1.metric(
            "Early stopping",
            f"Week {msprt_res.stopping_point + 1}" if msprt_res.stopping_point is not None else "No",
            delta="Significant" if msprt_res.significant else "Not significant",
            delta_color="normal" if msprt_res.significant else "off",
        )
        sp_col2.metric("mSPRT final p", f"{msprt_res.final_p:.4f}")
        sp_col3.metric("Prior τ", f"{msprt_res.tau:.2f} trades (σ×{tau_pct}%)")
    else:
        sp_col1.metric("Early stopping", "N/A")
        sp_col2.metric("mSPRT final p", "N/A")
        sp_col3.metric("Prior τ", "N/A")

    st.markdown("---")

    # ── Section 3: Bayesian Expected Loss ────────────────────────────────────
    st.markdown("### 3️⃣ Bayesian Decision — Expected Loss in $")
    st.caption(
        "Answers the business question: *How much $ do we lose by making the wrong decision?* "
        "Expected Loss replaces abstract p-values with actionable $ risk."
    )

    bayes_metric = st.radio(
        "Metric for Bayesian analysis",
        ["trades/user", "fee/user ($)"],
        horizontal=True,
    )
    threshold_pct = st.slider("Decision threshold P(treatment wins) %", 80, 99, 95, 1)

    if bayes_metric == "trades/user":
        c_obs = ctrl_df["post_trades"].values.astype(float)
        t_obs = trt_df["post_trades"].values.astype(float)
        rev_per_unit = avg_fee_per_trade  # convert trades → $
        unit_label = "trades/user"
    else:
        c_obs = ctrl_df["post_fee"].values.astype(float)
        t_obs = trt_df["post_fee"].values.astype(float)
        rev_per_unit = 1.0
        unit_label = "$/user"

    if len(c_obs) < 2 or len(t_obs) < 2:
        st.warning("Not enough observations for Bayesian analysis.")
        st.stop()

    bayes_res = bayesian_ab(c_obs, t_obs, revenue_per_unit=rev_per_unit,
                            threshold_pct=float(threshold_pct))

    DECISION_STYLE = {
        "launch":           ("#00C896", "🚀 LAUNCH TREATMENT"),
        "reject":           ("#E74C3C", "❌ REJECT TREATMENT"),
        "gather_more_data": ("#F5A623", "⏳ GATHER MORE DATA"),
    }
    d_color, d_label = DECISION_STYLE[bayes_res.decision]

    st.markdown(
        f"<div style='background:{d_color}33;border:2px solid {d_color};"
        f"padding:14px 20px;border-radius:6px;text-align:center;"
        f"font-size:1.3rem;font-weight:bold;margin:12px 0'>{d_label}</div>",
        unsafe_allow_html=True,
    )

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("P(treatment wins)",  f"{bayes_res.prob_treatment_wins:.1%}")
    bc2.metric("Effect mean",        f"{bayes_res.effect_mean:+.2f} {unit_label}")
    bc3.metric("95% Credible Interval",
               f"[{bayes_res.credible_interval_95[0]:+.2f}, "
               f"{bayes_res.credible_interval_95[1]:+.2f}]")
    bc4.metric("Effect std (σ_δ)",   f"{bayes_res.effect_std:.3f}")

    bl1, bl2 = st.columns(2)
    bl1.metric(
        "💸 Expected Loss if we LAUNCH",
        f"${bayes_res.expected_loss_launch_usd:,.2f} / user",
        delta=f"Risk of launching a worse variant",
        delta_color="off",
    )
    bl2.metric(
        "💸 Expected Loss if we HOLD",
        f"${bayes_res.expected_loss_hold_usd:,.4f} / user",
        delta="Risk of missing a winning variant",
        delta_color="off",
    )

    # Posterior distribution chart
    effect_range = np.linspace(
        bayes_res.effect_mean - 4 * bayes_res.effect_std,
        bayes_res.effect_mean + 4 * bayes_res.effect_std,
        300,
    )
    if bayes_res.effect_std == 0:
        posterior = np.zeros_like(effect_range)
        posterior[np.argmin(np.abs(effect_range - bayes_res.effect_mean))] = 1.0
    else:
        posterior = spstats.norm.pdf(effect_range, bayes_res.effect_mean, bayes_res.effect_std)
    zero_idx  = np.searchsorted(effect_range, 0)

    fig_bay = go.Figure()
    fig_bay.add_trace(go.Scatter(
        x=effect_range[:zero_idx + 1],
        y=posterior[:zero_idx + 1],
        fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
        line=dict(color="#E74C3C", width=0), name="Treatment worse",
    ))
    fig_bay.add_trace(go.Scatter(
        x=effect_range[zero_idx:],
        y=posterior[zero_idx:],
        fill="tozeroy", fillcolor="rgba(0,200,150,0.25)",
        line=dict(color="#00C896", width=0), name="Treatment better",
    ))
    fig_bay.add_trace(go.Scatter(
        x=effect_range, y=posterior,
        line=dict(color="white", width=2), name="Posterior P(δ)", showlegend=False,
    ))
    fig_bay.add_vline(x=0, line_color="#aaa", line_dash="dot")
    fig_bay.add_vline(
        x=bayes_res.effect_mean,
        line_color="#F5A623", line_dash="dash",
        annotation_text=f"E[δ] = {bayes_res.effect_mean:+.2f}",
        annotation_position="top right",
    )
    fig_bay.update_layout(
        **_DARK,
        title=f"Posterior distribution of treatment effect ({unit_label})",
        xaxis_title=f"δ = μ_treatment − μ_control ({unit_label})",
        yaxis_title="Density", legend=dict(bgcolor="#1a1d27"), height=320,
    )
    st.plotly_chart(fig_bay, width="stretch")

# ── Tab 4: Churn risk ─────────────────────────────────────────────────────────
with tab_churn:
    st.subheader("⚠️ Churn Risk by Segment")
    churn_seg = con.execute(f"""
        SELECT u.trading_style, u.wallet_size, u.churn_sensitivity,
               COUNT(DISTINCT u.user_id) AS users,
               ROUND(AVG(u.churn_per_month)*100, 1) AS avg_monthly_churn_pct,
               ROUND(SUM(t.fee_usd)/COUNT(DISTINCT u.user_id), 0) AS rev_per_user
        FROM users u
        LEFT JOIN trades t ON u.user_id = t.user_id
        {where_u.replace("u.trading_style","trading_style").replace("u.wallet_size","wallet_size").replace("u.churn_sensitivity","churn_sensitivity") if where_u else ""}
        GROUP BY u.trading_style, u.wallet_size, u.churn_sensitivity
        HAVING COUNT(DISTINCT u.user_id) > 50
        ORDER BY avg_monthly_churn_pct DESC
        LIMIT 10
    """).df()
    try:
        import matplotlib  # noqa: F401

        st.dataframe(
            churn_seg.style
                .background_gradient(subset=["avg_monthly_churn_pct"], cmap="RdYlGn_r")
                .background_gradient(subset=["rev_per_user"], cmap="Greens"),
            width="stretch",
        )
    except Exception:
        st.dataframe(churn_seg, width="stretch")

# ── Tab 5: Sample size calculator ────────────────────────────────────────────
with tab_calc:
    st.subheader("🧮 Калькулятор размера эксперимента")

    @st.cache_data
    def load_baselines():
        rows = con.execute("""
            SELECT
                AVG(tpu) AS avg_trades, STDDEV(tpu) AS std_trades,
                AVG(fpu) AS avg_fee,    STDDEV(fpu) AS std_fee
            FROM (
                SELECT u.user_id,
                       COUNT(t.trade_id)      AS tpu,
                       COALESCE(SUM(t.fee_usd), 0) AS fpu
                FROM users u LEFT JOIN trades t ON u.user_id = t.user_id
                GROUP BY u.user_id
            )
        """).df()
        act = con.execute("""
            SELECT COUNT(DISTINCT t.user_id)*1.0/COUNT(DISTINCT u.user_id) AS activation
            FROM users u LEFT JOIN trades t ON u.user_id = t.user_id
        """).df()
        daily = con.execute("""
            SELECT COUNT(*) * 1.0 / DATEDIFF('day', MIN(registered_at), MAX(registered_at))
            AS daily_reg FROM users
        """).df()
        return rows, act, daily

    baselines_df, act_df, daily_df = load_baselines()

    METRICS = {
        "Сделки / пользователь":       ("continuous",  float(baselines_df["avg_trades"][0]),  float(baselines_df["std_trades"][0])),
        "Комиссии / пользователь ($)":  ("continuous",  float(baselines_df["avg_fee"][0]),     float(baselines_df["std_fee"][0])),
        "Activation Rate (%)":          ("proportion",  float(act_df["activation"][0]),         None),
    }
    daily_users = max(int(daily_df["daily_reg"][0]), 1)

    col_form, col_result = st.columns([1, 2])

    with col_form:
        metric_label = st.selectbox("Метрика", list(METRICS.keys()))
        metric_type, baseline_val, sigma_val = METRICS[metric_label]
        mde_pct   = st.slider("MDE (%)", min_value=1, max_value=50, value=10, step=1,
                              help="Минимальный обнаруживаемый эффект")
        alpha     = st.radio("Уровень значимости (α)", [0.01, 0.05, 0.10], index=1,
                             format_func=lambda x: f"{x} ({int((1-x)*100)}% уверенность)")
        power     = st.radio("Мощность (1−β)", [0.80, 0.90, 0.95], index=0,
                             format_func=lambda x: f"{x:.0%}")
        st.caption(f"Baseline: **{baseline_val:,.2f}** · Дневной трафик: **~{daily_users:,}** пользователей/день")

    with col_result:
        try:
            res = required_sample_size(
                baseline=baseline_val,
                mde_pct=mde_pct,
                alpha=float(alpha),
                power=float(power),
                sigma=sigma_val,
                metric_type=metric_type,
                daily_users=daily_users,
            )
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("На вариант",     f"{res.n_per_variant:,}")
            r2.metric("Всего",          f"{res.n_total:,}")
            r3.metric("Дней",           f"{res.days_to_complete:.0f}")
            r4.metric("Treatment value", f"{res.treatment_value:,.2f}")

            # Sensitivity curve
            mdes, ns, days_arr = sensitivity_curve(
                baseline=baseline_val,
                alpha=float(alpha),
                power=float(power),
                sigma=sigma_val,
                metric_type=metric_type,
                daily_users=daily_users,
            )
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=mdes, y=ns, name="N на вариант",
                line=dict(color="#4A90D9", width=2), mode="lines",
            ))
            fig_c.add_vline(x=mde_pct, line_dash="dash", line_color="#E8724A",
                            annotation_text=f"MDE = {mde_pct}%", annotation_position="top right")
            fig_c.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", font_color="white",
                xaxis_title="MDE (%)", yaxis_title="N на вариант",
                title="Размер выборки vs Минимальный эффект",
                legend=dict(bgcolor="#1a1d27"), height=320,
            )
            st.plotly_chart(fig_c, width="stretch")

            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=mdes, y=days_arr, name="Дней до завершения",
                line=dict(color="#00C896", width=2), fill="tozeroy",
                fillcolor="rgba(0,200,150,0.1)", mode="lines",
            ))
            fig_d.add_vline(x=mde_pct, line_dash="dash", line_color="#E8724A")
            fig_d.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", font_color="white",
                xaxis_title="MDE (%)", yaxis_title="Дней",
                title="Время до завершения vs Минимальный эффект",
                height=260,
            )
            st.plotly_chart(fig_d, width="stretch")

        except Exception as e:
            st.error(f"Ошибка расчёта: {e}")

# ── Tab 6: Robot A/B simulation ───────────────────────────────────────────────
with tab_sim:
    st.subheader("🤖 Симуляция A/B эксперимента — Роботы-персонажи")
    st.caption("Роботы принимают решения на основе поведенческих параметров из данных биржи. "
               "Результаты детерминированы (без API-вызовов).")

    # Robot info cards
    with st.expander("👥 Персонажи роботов", expanded=False):
        cols = st.columns(3)
        for i, robot in enumerate(ROBOTS):
            with cols[i % 3]:
                st.markdown(
                    f"**{robot.emoji} {robot.name}**  \n"
                    f"`{robot.style}` · `{robot.risk}` · `{robot.wallet}` · `{robot.churn_sens}`  \n"
                    f"_{robot.description}_  \n"
                    f"Сделок/нед: **{robot.trades_per_week:.1f}** · "
                    f"Эластичность: **{robot.fee_elasticity:.2f}** · "
                    f"Churn/мес: **{robot.monthly_churn:.1%}**"
                )

    scenario_labels = [s.label for s in SCENARIOS]
    sel_label = st.selectbox("Сценарий", scenario_labels)
    sel_scenario = next(s for s in SCENARIOS if s.label == sel_label)

    # Custom scenario sliders
    if sel_scenario.id == "custom":
        st.markdown("**Параметры сценария:**")
        c1s, c2s = st.columns(2)
        with c1s:
            fee_chg = st.slider("Изменение комиссий (%)", -50, 50, 0, 1)
            ui_spd  = st.slider("Улучшение UI (%)", 0, 300, 0, 10)
        with c2s:
            compliance = st.slider("Compliance-нагрузка (%)", 0, 100, 0, 5)
            incentive  = st.slider("Реферальный бонус (%)", 0, 30, 0, 1)
        params = {
            "fee_change_pct": float(fee_chg),
            "ui_speed_pct": float(ui_spd),
            "compliance_friction": float(compliance),
            "incentive_pct": float(incentive),
        }
        scenario_id = "custom"
    else:
        st.info(sel_scenario.description)
        params = sel_scenario.params
        scenario_id = sel_scenario.id

    if st.button("▶ Запустить симуляцию", type="primary"):
        st.session_state["sim_results"] = simulate(params, scenario_id)
        st.session_state["sim_scenario_label"] = sel_label

    results = st.session_state.get("sim_results")
    if results:
        st.markdown(f"**Результаты: {st.session_state.get('sim_scenario_label', '')}**")

        # Build summary dataframe (treatment only, show delta vs control)
        ctrl_map = {r.robot.name: r for r in results if r.group == "control"}
        trt_list  = [r for r in results if r.group == "treatment"]

        rows = []
        for r in trt_list:
            ctrl = ctrl_map.get(r.robot.name)
            if ctrl is None:
                continue
            delta_t = r.trades_change_mean - ctrl.trades_change_mean
            delta_c = r.churn_change_mean  - ctrl.churn_change_mean
            rows.append({
                "Робот":            f"{r.robot.emoji} {r.robot.name}",
                "∆ Сделки (%)":    round(delta_t, 1),
                "95% CI":          f"[{r.trades_change_ci_low:.1f}, {r.trades_change_ci_high:.1f}]",
                "∆ Churn (%)":     round(delta_c, 1),
                "Инсайт":          r.insight,
            })
        df_res = pd.DataFrame(rows)

        # Color code deltas
        def color_delta(val):
            if not isinstance(val, (int, float)):
                return ""
            color = "#00C896" if val > 0 else "#E8724A" if val < 0 else "white"
            return f"color: {color}"

        styled = df_res.style.applymap(color_delta, subset=["∆ Сделки (%)", "∆ Churn (%)"])
        st.dataframe(styled, width="stretch", hide_index=True)

        # Bar chart: trades delta per robot
        fig_bar = go.Figure()
        for r in trt_list:
            ctrl = ctrl_map.get(r.robot.name)
            if ctrl is None:
                continue
            delta = r.trades_change_mean - ctrl.trades_change_mean
            err_low  = delta - (r.trades_change_ci_low  - ctrl.trades_change_mean)
            err_high = (r.trades_change_ci_high - ctrl.trades_change_mean) - delta
            fig_bar.add_trace(go.Bar(
                name=f"{r.robot.emoji} {r.robot.name}",
                x=[f"{r.robot.emoji} {r.robot.name}"],
                y=[delta],
                error_y=dict(type="data", symmetric=False,
                             array=[max(err_high, 0)], arrayminus=[max(err_low, 0)]),
                marker_color=r.robot.color,
            ))
        fig_bar.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)
        fig_bar.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117", font_color="white",
            title="Изменение частоты сделок (treatment vs control, %)",
            yaxis_title="∆ Сделки (%)", showlegend=False, height=380,
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Aggregate summary
        paired_trt = [r for r in trt_list if r.robot.name in ctrl_map]
        if paired_trt:
            avg_delta_t = float(np.mean([r.trades_change_mean - ctrl_map[r.robot.name].trades_change_mean
                                         for r in paired_trt]))
            avg_delta_c = float(np.mean([r.churn_change_mean  - ctrl_map[r.robot.name].churn_change_mean
                                         for r in paired_trt]))
        else:
            avg_delta_t = 0.0
            avg_delta_c = 0.0
        m1, m2 = st.columns(2)
        m1.metric("Средний ∆ Сделки (все роботы)",
                  f"{avg_delta_t:+.1f}%",
                  delta=f"{'▲' if avg_delta_t > 0 else '▼'} treatment vs control")
        m2.metric("Средний ∆ Churn (все роботы)",
                  f"{avg_delta_c:+.1f}%",
                  delta_color="inverse")
