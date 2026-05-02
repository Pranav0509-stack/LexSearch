// Realtime client — singleton socket.io connection used by the
// dashboard and any future live surface.
//
// Same-origin in dev (Next.js rewrites /socket.io/* to the FastAPI
// backend) and in prod (Railway colocates them). We connect lazily
// on first subscribe so we don't open a socket on pages that never
// listen for events.

import { io, Socket } from "socket.io-client";

let socket: Socket | null = null;

// Resolve the socket.io endpoint. Default is same-origin so a single
// public URL (Cloudflare Tunnel, Railway, etc.) covers both the HTTP
// API and /socket.io. The realtime client uses `transports: ["polling",
// "websocket"]` which lets us tolerate proxy hops that don't honour
// the WebSocket upgrade (Next dev being one of them).
//
// Override with NEXT_PUBLIC_REALTIME_ORIGIN if the realtime layer
// lives on a separate domain (e.g. Fly.io for socket fan-out).
function realtimeOrigin(): string | undefined {
  const env = process.env.NEXT_PUBLIC_REALTIME_ORIGIN;
  if (env) return env;
  // Returning undefined tells socket.io-client to use window.location.
  return undefined;
}

export function getSocket(): Socket {
  if (socket) return socket;
  const origin = realtimeOrigin();
  // Polling first so the tunnel/proxy (which may not honour WebSocket
  // upgrade) still delivers events. socket.io upgrades to WS when
  // available.
  const opts = {
    path: "/socket.io",
    transports: ["polling", "websocket"] as ("polling" | "websocket")[],
    withCredentials: false,
    autoConnect: true,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
  };
  socket = origin ? io(origin, opts) : io(opts);
  return socket;
}

// Tiny typed helper so callers don't have to import Socket directly.
export function on<T = unknown>(event: string, handler: (data: T) => void): () => void {
  const s = getSocket();
  s.on(event, handler as (...args: unknown[]) => void);
  return () => {
    s.off(event, handler as (...args: unknown[]) => void);
  };
}

export function disconnect(): void {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
}
