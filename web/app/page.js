"use client";

import { useEffect, useState } from "react";

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
      {subtitle ? <div style={{ color: "#9fb2d8", marginTop: 4 }}>{subtitle}</div> : null}
    </div>
  );
}

export default function HomePage() {
  const [kpi, setKpi] = useState(null);
  const [ab, setAb] = useState(null);
  const [regimes, setRegimes] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const [kpiRes, abRes, regimesRes] = await Promise.all([
          fetch("/api/kpi"),
          fetch("/api/ab-summary"),
          fetch("/api/regimes")
        ]);
        if (!kpiRes.ok || !abRes.ok || !regimesRes.ok) {
          throw new Error("API request failed");
        }
        setKpi(await kpiRes.json());
        setAb(await abRes.json());
        setRegimes(await regimesRes.json());
      } catch (e) {
        setError(e.message);
      }
    }
    load();
  }, []);

  return (
    <main style={{ maxWidth: 1080, margin: "0 auto", padding: "24px 16px 48px" }}>
      <h1 style={{ marginBottom: 8 }}>CryptoFlow Analytics (Vercel)</h1>
      <p style={{ marginTop: 0, color: "#9fb2d8" }}>
        Vercel-native surface for KPI, A/B deltas and market regime context.
      </p>

      {error ? (
        <div style={{ color: "#ffb4b4", marginTop: 12 }}>Error: {error}</div>
      ) : null}

      {kpi ? (
        <section
          style={{
            marginTop: 18,
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
      ) : (
        <p>Loading KPI...</p>
      )}

      {ab ? (
        <section style={{ marginTop: 28 }}>
          <h2>A/B Summary</h2>
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
        </section>
      ) : (
        <p>Loading A/B...</p>
      )}

      <section style={{ marginTop: 28 }}>
        <h2>Market Regimes</h2>
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
