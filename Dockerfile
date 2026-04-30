FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY server.py auth.py brief_service.py vault_service.py workflows.py ./
COPY llm/ llm/
COPY validators/ validators/
COPY assets/ assets/
COPY *.html *.js *.css ./

# Port is set by Railway/Render via $PORT env var
ENV PORT=8080
ENV INDIA_COURTS_DB=/data/india_courts.db

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--timeout-keep-alive", "30"]
