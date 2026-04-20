FROM python:3.11-slim

# System deps for WeasyPrint (Indic typography) + psycopg2 build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
    libcairo2 libgdk-pixbuf-2.0-0 libffi-dev \
    libjpeg-dev zlib1g-dev \
    fonts-noto fonts-noto-cjk fonts-noto-color-emoji \
    gcc libpq-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
