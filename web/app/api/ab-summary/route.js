import { NextResponse } from "next/server";
import { readSummary } from "../../../lib/data";

export async function GET() {
  try {
    const summary = readSummary();
    const control = summary.ab_summary.control;
    const treatment = summary.ab_summary.treatment;

    const deltaTradesPct =
      ((treatment.mean_trades - control.mean_trades) / control.mean_trades) * 100;
    const deltaFeePct =
      ((treatment.mean_fee - control.mean_fee) / control.mean_fee) * 100;

    return NextResponse.json({
      control,
      treatment,
      deltas: {
        trades_pct: Number(deltaTradesPct.toFixed(2)),
        fee_pct: Number(deltaFeePct.toFixed(2))
      },
      decision_hint:
        deltaTradesPct > 0 && deltaFeePct > 0 ? "launch_candidate" : "needs_review"
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read AB snapshot: ${error.message}` },
      { status: 500 }
    );
  }
}
