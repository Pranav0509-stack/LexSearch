FROM python:3.11-slim

WORKDIR /app

# System deps for pyarrow + reasonable defaults; kept minimal.
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bootstrap a small BM25 index at build time so a fresh deploy can
# answer immediately. The nightly cron / `--top-up` runs grow it.
RUN python scripts/ingest_github_data.py --source hk_cuthchow_csv || true

# Railway / Render set $PORT at runtime; 8080 is just the local default.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
