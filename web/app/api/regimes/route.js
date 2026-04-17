import { NextResponse } from "next/server";
import { readSummary } from "../../../lib/data";

export async function GET() {
  try {
    const summary = readSummary();
    return NextResponse.json(summary.regimes);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read market regimes: ${error.message}` },
      { status: 500 }
    );
  }
}
