# CryptoFlow — experimentation & analytics skeleton

Python package `cryptoflow` plus SQL marts and event schemas for a B2B-style stack: assignment, market regimes, statistical inference (CUPED / mSPRT / Bayesian), and warehouse-ready exports.

## Layout

- `cryptoflow/` — application code (`stats.py` engine, dashboard, new modules under subpackages).
- `sql/clickhouse/` — DDL templates for exposure and events (adapt for Snowflake).
- `schemas/` — JSON Schema for product analytics tools (e.g. Amplitude/Mixpanel).

## Run dashboard

From repo root:

```bash
pip install -e .
streamlit run cryptoflow/dashboard.py
```

If you prefer not to install the package, add the repo root to `PYTHONPATH` and run Streamlit from `cryptoflow/` as before.
