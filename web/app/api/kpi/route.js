import { NextResponse } from "next/server";
import { readSummary } from "../../../lib/data";

export async function GET() {
  try {
    const summary = readSummary();
    return NextResponse.json(summary.kpi);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read KPI snapshot: ${error.message}` },
      { status: 500 }
    );
  }
}
