"""Ensure demo DuckDB exists (local + cloud: DB is gitignored)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Streamlit Cloud: full 50k seed often OOMs or times out; override with CRYPTOFLOW_SEED_USERS.
_DEFAULT_CLOUD_SEED = "12000"


def ensure_demo_database(cryptoflow_dir: Path) -> None:
    """Create cryptoflow/data/cryptoflow.duckdb by running generate_data.py if missing."""
    db_path = cryptoflow_dir / "data" / "cryptoflow.duckdb"
    if db_path.is_file():
        return
    data_dir = cryptoflow_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    script = cryptoflow_dir / "generate_data.py"
    env = os.environ.copy()
    if "CRYPTOFLOW_SEED_USERS" not in env:
        env["CRYPTOFLOW_SEED_USERS"] = _DEFAULT_CLOUD_SEED
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cryptoflow_dir),
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or "no output"
        raise RuntimeError(f"generate_data.py failed ({proc.returncode}): {msg}")
