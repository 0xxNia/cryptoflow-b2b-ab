import { NextResponse } from "next/server";
import { readSummary } from "../../../lib/data";
import { buildAbSummary } from "../../../lib/abSummary";

export async function GET() {
  try {
    const summary = readSummary();
    return NextResponse.json(buildAbSummary(summary));
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read AB snapshot: ${error.message}` },
      { status: 500 }
    );
  }
}
