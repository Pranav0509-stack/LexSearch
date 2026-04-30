"""
Plivo answer / hangup / events webhooks.

/voice/plivo/answer   — returns XML (PlivoXML) that instructs Plivo to open a
                        Media Streams WebSocket to /voice/plivo/media/{call_uuid}.
/voice/plivo/hangup   — enqueues post-call Celery job.
/voice/plivo/events   — call progress (ringing, answered, completed).
"""

import uuid as uuidlib
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from app.config import get_settings

router = APIRouter()


def _plivo_answer_xml(ws_url: str) -> str:
    """
    PlivoXML that:
      1. Plays the pre-recorded disclaimer (language is chosen by caller's
         ANI-lookup preference; default Hindi).
      2. Opens a bidirectional Media Streams WebSocket to our server.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream
    bidirectional="true"
    keepCallAlive="true"
    contentType="audio/x-mulaw;rate=8000"
    audioTrack="inbound"
  >{ws_url}</Stream>
</Response>"""


@router.post("/answer")
async def answer(
    request: Request,
    From: Annotated[str, Form()] = "",
    To: Annotated[str, Form()] = "",
    CallUUID: Annotated[str, Form()] = "",
) -> Response:
    settings = get_settings()
    call_uuid = CallUUID or str(uuidlib.uuid4())

    # ws_url: Plivo connects back to us over WebSocket for streaming audio.
    host = request.headers.get("host", "localhost:8080")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{scheme}://{host}/voice/plivo/media/{call_uuid}"

    # TODO: create User + Call row; enforce per-phone rate limit.
    _ = settings

    xml = _plivo_answer_xml(ws_url)
    return Response(content=xml, media_type="application/xml")


@router.post("/hangup")
async def hangup(
    CallUUID: Annotated[str, Form()] = "",
    Duration: Annotated[str, Form()] = "0",
    HangupCause: Annotated[str, Form()] = "",
) -> dict:
    # TODO: enqueue post_call Celery task.
    return {"ok": True, "call_uuid": CallUUID}


@router.post("/events")
async def events(
    Event: Annotated[str, Form()] = "",
    CallUUID: Annotated[str, Form()] = "",
) -> dict:
    # TODO: update call state (ringing | answered | completed | failed).
    return {"ok": True, "event": Event, "call_uuid": CallUUID}
