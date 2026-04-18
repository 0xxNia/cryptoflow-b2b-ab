FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cryptoflow ./cryptoflow/

RUN pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir -e .

WORKDIR /app/cryptoflow
RUN python generate_data.py

WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "cryptoflow/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
