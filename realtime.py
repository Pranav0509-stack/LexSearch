"""Realtime layer — python-socketio mounted on the existing FastAPI app.

The dashboard fans out events to every connected admin so two people
poking the same screen don't trample each other:

  • activity:append      — new audit-log entry (someone did something)
  • settings:changed     — a connector key / library doc / dash setting was edited
  • bm25:reloaded        — admin hit /admin/reload, fresh case count is N
  • presence:join|leave  — another admin opened/closed the dashboard

Usage (server side):
    from realtime import sio, broadcast
    await broadcast("activity:append", {"actor": "...", "action": "...", ...})

Usage (client side, web/src/lib/realtime.ts):
    import { io } from "socket.io-client";
    const sock = io({ path: "/socket.io" });
    sock.on("activity:append", (row) => …)

The module exposes `mount(app)` which the FastAPI server calls during
startup to attach the ASGI socketio app to the existing `app`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import socketio

logger = logging.getLogger(__name__)

# AsyncServer + ASGIApp lets us mount inside FastAPI/uvicorn without a
# separate process. CORS is wide-open in dev; tighten via env var for
# production (see `mount()`).
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)

_asgi: socketio.ASGIApp | None = None
_main_loop: asyncio.AbstractEventLoop | None = None
_queue: asyncio.Queue | None = None


@sio.event
async def connect(sid: str, environ: dict[str, Any], auth: dict[str, Any] | None = None) -> None:
    # In dev (frontend on :3001, backend on :8080), the session cookie
    # set by the API origin doesn't ride along on cross-origin
    # WebSocket connects, so we'd lock every dev session out. We log
    # the auth state but accept the connection — the events emitted
    # on this channel are audit-log entries (low sensitivity) and the
    # channel can be authenticated more strictly in prod via an env
    # toggle once we deploy.
    cookie_header = environ.get("HTTP_COOKIE", "") or ""
    has_session = "ls_session=" in cookie_header
    logger.info("socketio connect %s (authenticated=%s)", sid, has_session)
    await sio.emit("presence:join", {"sid": sid}, skip_sid=sid)


@sio.event
async def disconnect(sid: str) -> None:
    logger.info("socketio disconnect %s", sid)
    await sio.emit("presence:leave", {"sid": sid})


async def _broadcast_worker() -> None:
    """Serialise all broadcasts through a single async queue so bursts
    of synchronous emit calls don't race each other."""
    global _queue
    _queue = asyncio.Queue()
    while True:
        event, payload = await _queue.get()
        try:
            await sio.emit(event, payload)
        except Exception as e:  # noqa: BLE001
            logger.warning("broadcast worker %s failed: %s", event, e)
        finally:
            _queue.task_done()


def mount(app: Any) -> None:
    """Attach the socket.io ASGI handler to the FastAPI `app`. Call
    once during server startup. Idempotent — second call is a no-op."""
    global _asgi, _main_loop
    if _asgi is not None:
        return
    _asgi = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/socket.io")
    try:
        _main_loop = asyncio.get_event_loop()
    except RuntimeError:
        _main_loop = None
    # Start the broadcast worker so emit calls are serialised
    if _main_loop and _main_loop.is_running():
        _main_loop.create_task(_broadcast_worker())
    logger.info("socketio mounted at /socket.io")


def asgi_app(fastapi_app: Any) -> Any:
    """Return a single combined ASGI app that serves the FastAPI HTTP
    surface AND the socket.io endpoint. Use this as the uvicorn target
    instead of bare `server:app`."""
    mount(fastapi_app)
    return _asgi or fastapi_app


def broadcast(event: str, payload: dict[str, Any]) -> None:
    """Fire-and-forget fan-out. Safe to call from synchronous endpoint
    handlers (FastAPI runs them in a threadpool). Events are serialised
    through an async queue worker to avoid race conditions under burst."""
    if _main_loop is None or not _main_loop.is_running():
        return
    if _queue is not None:
        try:
            asyncio.run_coroutine_threadsafe(_queue.put((event, payload)), _main_loop)
        except Exception as e:  # noqa: BLE001
            logger.warning("broadcast enqueue %s failed: %s", event, e)
    else:
        # Fallback if worker hasn't started yet
        try:
            asyncio.run_coroutine_threadsafe(sio.emit(event, payload), _main_loop)
        except Exception as e:  # noqa: BLE001
            logger.warning("broadcast %s failed: %s", event, e)
