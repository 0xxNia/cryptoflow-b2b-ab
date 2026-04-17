import { readSummary } from "./data.js";

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const y =
    1 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax));
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

export function buildAbSummary(summary) {
  const control = summary.ab_summary.control;
  const treatment = summary.ab_summary.treatment;

  const deltaTradesPct =
    ((treatment.mean_trades - control.mean_trades) / control.mean_trades) * 100;
  const deltaFeePct =
    ((treatment.mean_fee - control.mean_fee) / control.mean_fee) * 100;

  const total = control.n + treatment.n;
  const expected = total * 0.5;
  const chi2 =
    ((control.n - expected) ** 2) / expected +
    ((treatment.n - expected) ** 2) / expected;
  const srmPValue = 2 * (1 - normalCdf(Math.sqrt(chi2)));
  const srmSignificant = srmPValue <= 0.001;

  return {
    control,
    treatment,
    deltas: {
      trades_pct: Number(deltaTradesPct.toFixed(2)),
      fee_pct: Number(deltaFeePct.toFixed(2))
    },
    decision_hint:
      deltaTradesPct > 0 && deltaFeePct > 0 ? "launch_candidate" : "needs_review",
    integrity: {
      srm: {
        p_value: Number(srmPValue.toFixed(6)),
        significant: srmSignificant
      },
      alerts: srmSignificant
        ? [
            {
              check_name: "srm_guard",
              severity: "critical",
              reason:
                "Sample Ratio Mismatch detected against expected 50/50 split."
            }
          ]
        : []
    }
  };
}

export function loadDashboardPayload() {
  const summary = readSummary();
  return {
    kpi: summary.kpi,
    ab: buildAbSummary(summary),
    regimes: summary.regimes
  };
}
