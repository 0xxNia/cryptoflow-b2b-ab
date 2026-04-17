import { loadDashboardPayload } from "../lib/abSummary";

const REPO_URL = "https://github.com/0xxNia/cryptoflow-b2b-ab";
const STREAMLIT_HINT =
  process.env.NEXT_PUBLIC_STREAMLIT_URL ||
  "";

function Card({ title, value, subtitle }) {
  return (
    <div
      style={{
        padding: 16,
        border: "1px solid #24314f",
        borderRadius: 12,
        background: "#131b33"
      }}
    >
      <div style={{ color: "#93a4c7", fontSize: 12 }}>{title}</div>
      <div style={{ fontSize: 28, fontWeight: 700, marginTop: 6 }}>{value}</div>
      {subtitle ? (
        <div style={{ color: "#9fb2d8", marginTop: 4 }}>{subtitle}</div>
      ) : null}
    </div>
  );
}

export default function HomePage() {
  const { kpi, ab, regimes } = loadDashboardPayload();

  return (
    <main style={{ maxWidth: 1080, margin: "0 auto", padding: "24px 16px 48px" }}>
      <h1 style={{ marginBottom: 8 }}>CryptoFlow Analytics (Vercel)</h1>
      <p style={{ marginTop: 0, color: "#9fb2d8" }}>
        Это не полный Streamlit-дашборд — это публичный Vercel-слой (Next.js) для KPI,
        краткого A/B и integrity (SRM). Полный продукт с графиками CUPED / mSPRT /
        Bayesian и вкладками запускается локально через Streamlit в репозитории.
      </p>

      <div
        style={{
          marginTop: 14,
          padding: "12px 16px",
          borderRadius: 10,
          border: "1px solid #2a3f6f",
          background: "#121a30",
          color: "#cfe0ff",
          fontSize: 14,
          lineHeight: 1.5
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Где «всё остальное»?</div>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>
            Код и Streamlit:{" "}
            <a href={REPO_URL} style={{ color: "#7dd3fc" }}>
              {REPO_URL}
            </a>
          </li>
          <li>
            Локально:{" "}
            <code style={{ background: "#0f172a", padding: "2px 6px", borderRadius: 4 }}>
              pip install -e .
            </code>{" "}
            затем{" "}
            <code style={{ background: "#0f172a", padding: "2px 6px", borderRadius: 4 }}>
              streamlit run cryptoflow/dashboard.py
            </code>
          </li>
          {STREAMLIT_HINT ? (
            <li>
              Публичный Streamlit (если поднят):{" "}
              <a href={STREAMLIT_HINT} style={{ color: "#7dd3fc" }}>
                {STREAMLIT_HINT}
              </a>
            </li>
          ) : null}
        </ul>
      </div>

      <section
        style={{
          marginTop: 22,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 12
        }}
      >
        <Card title="Total Users" value={kpi.total_users.toLocaleString()} />
        <Card title="Trading Users" value={kpi.trading_users.toLocaleString()} />
        <Card title="Fee Revenue" value={`$${kpi.total_revenue.toLocaleString()}`} />
        <Card title="Activation Rate" value={`${kpi.activation_rate}%`} />
        <Card title="Revenue / User" value={`$${Number(kpi.rev_per_user).toFixed(1)}`} />
      </section>

      <section style={{ marginTop: 28 }}>
        <h2 style={{ marginBottom: 10 }}>A/B Summary</h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <Card
            title="Control Mean Trades"
            value={ab.control.mean_trades.toFixed(2)}
            subtitle={`n=${ab.control.n}`}
          />
          <Card
            title="Treatment Mean Trades"
            value={ab.treatment.mean_trades.toFixed(2)}
            subtitle={`n=${ab.treatment.n}`}
          />
          <Card
            title="Trades Delta"
            value={`${ab.deltas.trades_pct > 0 ? "+" : ""}${ab.deltas.trades_pct}%`}
            subtitle="treatment vs control"
          />
          <Card
            title="Fee Delta"
            value={`${ab.deltas.fee_pct > 0 ? "+" : ""}${ab.deltas.fee_pct}%`}
            subtitle={`decision: ${ab.decision_hint}`}
          />
        </div>
        <div style={{ marginTop: 14, color: "#9fb2d8", fontSize: 14 }}>
          <strong style={{ color: "#e2e8f0" }}>SRM guard:</strong> p ={" "}
          {ab.integrity.srm.p_value.toFixed(6)}, significant ={" "}
          {String(ab.integrity.srm.significant)}
          {ab.integrity.alerts.length > 0 ? (
            <span style={{ color: "#fca5a5" }}>
              {" "}
              — {ab.integrity.alerts[0].reason}
            </span>
          ) : null}
        </div>
      </section>

      <section style={{ marginTop: 28 }}>
        <h2 style={{ marginBottom: 10 }}>Market Regimes (snapshot)</h2>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {regimes.map((r) => (
            <div
              key={r.name}
              style={{
                background: "#152040",
                border: "1px solid #2a3a61",
                borderRadius: 999,
                padding: "8px 14px"
              }}
            >
              {r.name}: {r.share_pct}%
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
