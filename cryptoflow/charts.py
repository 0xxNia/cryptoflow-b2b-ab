import duckdb
import plotly.graph_objects as go

con = duckdb.connect("data/cryptoflow.duckdb", read_only=True)

df = con.execute("""
    WITH first_trade AS (
        SELECT u.user_id, u.tier,
               MIN(t.traded_at) AS first_trade_at
        FROM users u
        JOIN trades t ON u.user_id = t.user_id
        GROUP BY u.user_id, u.tier
    ),
    user_weeks AS (
        SELECT f.user_id, f.tier,
               FLOOR(DATEDIFF('day', f.first_trade_at, t.traded_at) / 7) AS week_num
        FROM first_trade f
        JOIN trades t ON f.user_id = t.user_id
        WHERE FLOOR(DATEDIFF('day', f.first_trade_at, t.traded_at) / 7) <= 6
    ),
    base AS (
        SELECT tier, COUNT(DISTINCT user_id) AS base_users
        FROM user_weeks WHERE week_num = 0
        GROUP BY tier
    ),
    weekly AS (
        SELECT tier, week_num, COUNT(DISTINCT user_id) AS active_users
        FROM user_weeks
        GROUP BY tier, week_num
    )
    SELECT w.tier, w.week_num,
           ROUND(w.active_users * 100.0 / b.base_users, 1) AS retention_pct
    FROM weekly w
    JOIN base b ON w.tier = b.tier
    ORDER BY w.tier, w.week_num
""").df()

con.close()

colors = {"pro": "#00C896", "basic": "#4A90D9", "free": "#E8724A"}

fig = go.Figure()

for tier in ["pro", "basic", "free"]:
    d = df[df["tier"] == tier]
    fig.add_trace(go.Scatter(
        x=d["week_num"],
        y=d["retention_pct"],
        name=tier.capitalize(),
        mode="lines+markers",
        line=dict(color=colors[tier], width=3),
        marker=dict(size=8),
    ))

fig.update_layout(
    title="CryptoFlow — Retention Curves by Tier",
    xaxis_title="Week since first trade",
    yaxis_title="Retention %",
    yaxis=dict(range=[0, 105]),
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font=dict(color="white"),
    legend=dict(bgcolor="#1a1d27"),
)

fig.show()