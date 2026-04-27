// Realtime client — singleton socket.io connection used by the
// dashboard and any future live surface.
//
// Same-origin in dev (Next.js rewrites /socket.io/* to the FastAPI
// backend) and in prod (Railway colocates them). We connect lazily
// on first subscribe so we don't open a socket on pages that never
// listen for events.

import { io, Socket } from "socket.io-client";

let socket: Socket | null = null;

// Resolve the socket.io endpoint:
//   • If NEXT_PUBLIC_REALTIME_ORIGIN is set, use it (prod / staging override).
//   • In the browser, default to <currentScheme>://<currentHost>:8080 — i.e.
//     hit the FastAPI backend directly. The Next.js dev rewrite handles
//     polling but mangles the WebSocket upgrade, so going direct in dev is
//     more reliable. Cookies still flow because the API origin set them.
function realtimeOrigin(): string {
  const env = process.env.NEXT_PUBLIC_REALTIME_ORIGIN;
  if (env) return env;
  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8080`;
  }
  return "http://localhost:8080";
}

export function getSocket(): Socket {
  if (socket) return socket;
  socket = io(realtimeOrigin(), {
    path: "/socket.io",
    transports: ["websocket", "polling"],
    // false because the server uses cors_allowed_origins="*" which
    // implicitly disables credentials; mismatched flags cause the
    // browser to drop the polling response.
    withCredentials: false,
    autoConnect: true,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
  });
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
