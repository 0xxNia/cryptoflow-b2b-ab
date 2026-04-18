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

## Deploy full dashboard (Streamlit + DuckDB)

The demo database is **not** committed (see `.gitignore`). On first start in a clean environment, `cryptoflow/data_bootstrap.py` runs `generate_data.py` once to create `cryptoflow/data/cryptoflow.duckdb` (can take a few minutes for 50k users).

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), **New app** → pick the repo.
3. Main file path: **`streamlit_app.py`** (repo root, default in Cloud) or `cryptoflow/dashboard.py`, branch: `main`, Python 3.10+.
4. Cloud uses root `requirements.txt`. After the app URL is live, optionally set in Vercel (project `web`): `NEXT_PUBLIC_STREAMLIT_URL` = that URL so the Next.js landing page links to the full UI.

### Docker (Fly.io, Railway, GCP, etc.)

```bash
docker build -t cryptoflow-dashboard .
docker run -p 8501:8501 cryptoflow-dashboard
```

The image runs `generate_data.py` at **build** time so the container starts with data already present.

### Vercel (`web/`)

The `web/` app is a thin Next.js layer only. Deploy it separately (`cd web && npx vercel deploy --prod`); it does not host Streamlit.
