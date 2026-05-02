#!/bin/bash
# Sanhita — YC Demo Launcher
# Run this to start everything for the demo.
# Access code: SNHT-DEMO-2026
#
# Usage: ./start-demo.sh
# Then open http://localhost:3000/login

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║           S A N H I T A                       ║"
echo "  ║     India's Largest AI Legal Research          ║"
echo "  ║     31.9M Court Records · 25 High Courts      ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Kill any existing processes on our ports
echo "[1/4] Clearing ports..."
lsof -ti:8080 2>/dev/null | xargs kill 2>/dev/null || true
lsof -ti:3000 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

# Start FastAPI backend
echo "[2/4] Starting backend (16.9M judgments)..."
cd "$SCRIPT_DIR"
uvicorn server:app --host 0.0.0.0 --port 8080 --workers 1 &
BACKEND_PID=$!
echo "       Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "       Waiting for FTS5 index to load..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8080/health | grep -q '"status":"ok"' 2>/dev/null; then
        echo "       Backend ready!"
        break
    fi
    sleep 1
done

# Start Next.js frontend
echo "[3/4] Starting frontend..."
cd "$SCRIPT_DIR/sanhita-react/web"
npx next dev --port 3000 &
FRONTEND_PID=$!
echo "       Frontend PID: $FRONTEND_PID"
sleep 5

echo "[4/4] Starting Cloudflare tunnel..."
cloudflared tunnel --url http://localhost:3000 2>&1 | grep -E "INF|ERR" &
TUNNEL_PID=$!
sleep 8

echo ""
echo "  ══════════════════════════════════════════════"
echo ""
echo "  LOCAL:  http://localhost:3000/login"
echo "  CODE:   SNHT-DEMO-2026"
echo ""
echo "  Backend:  http://localhost:8080/health"
echo "  Vercel:   https://web-zeta-beryl-32.vercel.app"
echo ""
echo "  PIDs: backend=$BACKEND_PID frontend=$FRONTEND_PID tunnel=$TUNNEL_PID"
echo "  To stop: kill $BACKEND_PID $FRONTEND_PID $TUNNEL_PID"
echo ""
echo "  ══════════════════════════════════════════════"
echo ""

# Keep running
wait
