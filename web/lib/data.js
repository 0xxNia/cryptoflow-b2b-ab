import fs from "node:fs";
import path from "node:path";

const SUMMARY_PATH = path.join(process.cwd(), "data", "summary.json");

export function readSummary() {
  const raw = fs.readFileSync(SUMMARY_PATH, "utf-8");
  return JSON.parse(raw);
}
