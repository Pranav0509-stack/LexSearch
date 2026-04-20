"""
Plivo Media Streams WebSocket handler — the realtime voice pipe.

Frame format: Plivo sends JSON messages; the `media.payload` field is a base64
string of 8kHz μ-law PCM. We decode, stream to Sarvam Saaras ASR, route to the
dialog orchestrator, generate TTS via Sarvam Bulbul, and stream frames back.

This file is the hardest in the codebase — barge-in, cancellation, backpressure.
Current form is a scaffolding skeleton; real implementation lands in Month 1–2.
"""

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.voice.session import CallSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/media/{call_uuid}")
async def media_stream(websocket: WebSocket, call_uuid: str) -> None:
    await websocket.accept()
    session = await CallSession.load_or_create(call_uuid)

    # Tasks: inbound audio pump, outbound TTS pump, orchestrator.
    stop = asyncio.Event()

    try:
        async for raw in websocket.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                logger.info("media start call_uuid=%s", call_uuid)

            elif event == "media":
                payload_b64 = msg.get("media", {}).get("payload", "")
                frame = base64.b64decode(payload_b64)
                # TODO: push frame to ASR stream + VAD detector.
                _ = frame

            elif event == "stop":
                logger.info("media stop call_uuid=%s", call_uuid)
                break

    except WebSocketDisconnect:
        logger.info("ws disconnected call_uuid=%s", call_uuid)
    finally:
        stop.set()
        await session.persist_turn_log()


async def _send_tts_frame(ws: WebSocket, pcm_mulaw_8khz: bytes) -> None:
    """Base64-encode and send a μ-law frame back to Plivo."""
    payload = base64.b64encode(pcm_mulaw_8khz).decode()
    await ws.send_text(json.dumps({
        "event": "playAudio",
        "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": payload},
    }))


async def _send_clear(ws: WebSocket) -> None:
    """Instruct Plivo to drop any buffered TTS (used on barge-in)."""
    await ws.send_text(json.dumps({"event": "clearAudio"}))
