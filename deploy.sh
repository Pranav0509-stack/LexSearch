#!/bin/bash
# Sanhita Deployment Script
# Usage: ./deploy.sh [hetzner|railway]
#
# Prerequisites:
#   - For Hetzner: ssh access to server, rsync installed
#   - For Railway: `railway` CLI installed (npm i -g @railway/cli)

set -euo pipefail

MODE="${1:-help}"
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$BACKEND_DIR/sanhita-react/web"
DB_PATH="${INDIA_COURTS_DB:-$BACKEND_DIR/india_courts.db}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

case "$MODE" in

# ─── Option 1: Hetzner VPS (recommended for 81GB DB) ───────────────────
hetzner)
    HETZNER_HOST="${HETZNER_HOST:?Set HETZNER_HOST=user@ip}"
    REMOTE_DIR="/opt/sanhita"
    REMOTE_DB="/data/india_courts.db"

    log "Deploying to Hetzner VPS: $HETZNER_HOST"

    # 1. Copy backend files
    log "Syncing backend code..."
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='node_modules' \
        --exclude='__pycache__' \
        --exclude='.env' \
        --exclude='*.db' \
        --exclude='*.db-wal' \
        --exclude='*.db-shm' \
        --exclude='sanhita-react' \
        --exclude='eval' \
        --exclude='legal-site' \
        --exclude='ingest' \
        --exclude='.claude' \
        "$BACKEND_DIR/" "$HETZNER_HOST:$REMOTE_DIR/"

    # 2. Upload DB (only if not already there — 81GB takes hours)
    log "Checking if DB exists on server..."
    if ssh "$HETZNER_HOST" "test -f $REMOTE_DB"; then
        warn "DB already exists on server ($REMOTE_DB). Skipping upload."
        warn "To force re-upload: ssh $HETZNER_HOST rm $REMOTE_DB && re-run"
    else
        log "Uploading database (81GB — this will take a while)..."
        ssh "$HETZNER_HOST" "mkdir -p /data"
        rsync -avz --progress "$DB_PATH" "$HETZNER_HOST:$REMOTE_DB"
    fi

    # 3. Set up systemd service
    log "Setting up systemd service..."
    ssh "$HETZNER_HOST" bash <<'REMOTE_SETUP'
set -e
cd /opt/sanhita

# Install Python deps
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv
python3 -m venv /opt/sanhita/venv 2>/dev/null || true
source /opt/sanhita/venv/bin/activate
pip install --quiet -r requirements.txt

# Create systemd service
cat > /etc/systemd/system/sanhita.service <<'SVC'
[Unit]
Description=Sanhita Legal AI Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sanhita
EnvironmentFile=/opt/sanhita/.env
ExecStart=/opt/sanhita/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8080 --workers 2 --timeout-keep-alive 30
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable sanhita
systemctl restart sanhita
echo "✅ Sanhita backend running on port 8080"
REMOTE_SETUP

    # 4. Set up nginx reverse proxy (if nginx installed)
    log "Checking nginx..."
    ssh "$HETZNER_HOST" bash <<'NGINX_SETUP' || warn "Nginx not installed — backend exposed on :8080 directly"
if command -v nginx &>/dev/null; then
    cat > /etc/nginx/sites-available/sanhita <<'CONF'
server {
    listen 80;
    server_name _;
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
CONF
    ln -sf /etc/nginx/sites-available/sanhita /etc/nginx/sites-enabled/
    nginx -t && systemctl reload nginx
    echo "✅ Nginx configured"
fi
NGINX_SETUP

    log "✅ Backend deployed to Hetzner!"
    log "Next steps:"
    log "  1. SSH in and create /opt/sanhita/.env with API keys"
    log "  2. Point your domain A record to the server IP"
    log "  3. Run: certbot --nginx -d sanhita.law"
    log "  4. Deploy frontend to Vercel with BACKEND_ORIGIN=https://sanhita.law"
    ;;

# ─── Option 2: Railway ─────────────────────────────────────────────────
railway)
    log "Deploying to Railway..."

    if ! command -v railway &>/dev/null; then
        err "Railway CLI not installed. Run: npm i -g @railway/cli"
    fi

    cd "$BACKEND_DIR"
    railway up --detach
    log "✅ Backend deploying to Railway"
    log "Note: You need to upload the 81GB DB to the persistent volume manually"
    log "  railway volume cp $DB_PATH india_courts_db:/india_courts.db"
    ;;

# ─── Option 3: Deploy frontend to Vercel ───────────────────────────────
vercel)
    log "Deploying frontend to Vercel..."

    if ! command -v vercel &>/dev/null; then
        err "Vercel CLI not installed. Run: npm i -g vercel"
    fi

    BACKEND_ORIGIN="${BACKEND_ORIGIN:?Set BACKEND_ORIGIN=https://your-backend-url}"

    cd "$FRONTEND_DIR"
    log "Building Next.js..."
    BACKEND_ORIGIN="$BACKEND_ORIGIN" npx next build

    log "Deploying to Vercel..."
    vercel --prod -e BACKEND_ORIGIN="$BACKEND_ORIGIN"
    log "✅ Frontend deployed to Vercel"
    ;;

# ─── Local demo mode ───────────────────────────────────────────────────
local)
    log "Starting local demo (backend + frontend)..."

    # Check DB
    if [ ! -f "$DB_PATH" ] && [ ! -L "$DB_PATH" ]; then
        err "Database not found at $DB_PATH"
    fi

    # Start backend
    log "Starting backend on :8080..."
    cd "$BACKEND_DIR"
    source .env 2>/dev/null || true
    uvicorn server:app --host 0.0.0.0 --port 8080 --workers 2 &
    BACKEND_PID=$!

    # Wait for backend
    for i in $(seq 1 30); do
        if curl -s http://localhost:8080/health >/dev/null 2>&1; then
            log "Backend ready!"
            break
        fi
        sleep 1
    done

    # Start frontend
    log "Starting frontend on :3000..."
    cd "$FRONTEND_DIR"
    npm run dev &
    FRONTEND_PID=$!

    log "✅ Sanhita running locally!"
    log "  Frontend: http://localhost:3000"
    log "  Backend:  http://localhost:8080"
    log "  Login code: SNHT-DEMO-2026"
    log ""
    log "Press Ctrl+C to stop"

    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
    wait
    ;;

help|*)
    echo "Sanhita Deployment"
    echo ""
    echo "Usage: ./deploy.sh <mode>"
    echo ""
    echo "Modes:"
    echo "  local    — Start backend + frontend locally for demo"
    echo "  hetzner  — Deploy to Hetzner VPS (recommended for 81GB DB)"
    echo "  railway  — Deploy to Railway.app"
    echo "  vercel   — Deploy frontend to Vercel"
    echo ""
    echo "Environment variables:"
    echo "  HETZNER_HOST    — user@ip for Hetzner deployment"
    echo "  BACKEND_ORIGIN  — Backend URL for Vercel frontend"
    echo "  INDIA_COURTS_DB — Path to india_courts.db (default: ./india_courts.db)"
    echo ""
    echo "Recommended deployment:"
    echo "  1. ./deploy.sh hetzner   (backend + 81GB DB)"
    echo "  2. BACKEND_ORIGIN=https://api.sanhita.law ./deploy.sh vercel  (frontend)"
    ;;
esac
