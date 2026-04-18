"""Ensure demo DuckDB exists (local + cloud: DB is gitignored)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_demo_database(cryptoflow_dir: Path) -> None:
    """Create cryptoflow/data/cryptoflow.duckdb by running generate_data.py if missing."""
    db_path = cryptoflow_dir / "data" / "cryptoflow.duckdb"
    if db_path.is_file():
        return
    data_dir = cryptoflow_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    script = cryptoflow_dir / "generate_data.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cryptoflow_dir),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or "no output"
        raise RuntimeError(f"generate_data.py failed ({proc.returncode}): {msg}")
