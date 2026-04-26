"""
access_hub.py — real-time access-request + WebRTC-signaling hub for NyayaSathi.

Provides:
  * REST endpoints:
      GET    /access/requests
      POST   /access/requests                 -> {status:"pending"}
      POST   /access/requests/{id}/approve    -> {status:"approved", code:"NYS-..."}
      POST   /access/requests/{id}/decline    -> {status:"declined"}
      POST   /access/webhooks                 -> subscribe URL for server-to-server delivery
      DELETE /access/webhooks/{url}           -> unsubscribe

  * WebSocket endpoint:
      WS     /ws/access          unified broadcast channel. All tabs receive the
                                 same events; relays WebRTC signaling too.

Events emitted (JSON):
  {"type":"snapshot",        "requests":[...]}          on connect
  {"type":"request.created", "request":{...}}
  {"type":"request.approved","request":{...}}
  {"type":"request.declined","request":{...}}
  {"type":"rtc.join",        "code":"NYS-XXXX-XXX","peer":"abc"}
  {"type":"rtc.leave",       ...}
  {"type":"rtc.offer",       "code":"...","from":"...","sdp":"..."}
  {"type":"rtc.answer",      ...}
  {"type":"rtc.ice",         "code":"...","from":"...","candidate":{...}}

State is in-memory (intended for single-node pilot). Swap the store for Redis
later without touching the routes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import secrets
import string
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

log = logging.getLogger("access_hub")

Status = Literal["pending", "approved", "declined"]

# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class AccessRequest:
    id: str
    ticket: int
    name: str
    phone: str
    language: str
    category: str
    context: str
    status: Status
    submitted_at: int  # epoch ms
    code: Optional[str] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None or k in {"code", "reason"}}


class SubmitBody(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    phone: str = Field(min_length=6, max_length=32)
    language: str = Field(min_length=1, max_length=32)
    category: str = Field(min_length=1, max_length=48)
    context: str = Field(min_length=1, max_length=600)


class DeclineBody(BaseModel):
    reason: Optional[str] = Field(default="Queue full — try again tomorrow.", max_length=240)


class WebhookBody(BaseModel):
    url: str = Field(min_length=8, max_length=400)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatBody(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=24)
    language: str = Field(default="English", max_length=32)


# --------------------------------------------------------------------------- #
# Store + broadcast hub                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class AccessStore:
    requests: dict[str, AccessRequest] = field(default_factory=dict)
    next_ticket: int = 42  # carries on from the demo seeds
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def seed(self) -> None:
        now = int(time.time() * 1000)
        seeds = [
            AccessRequest(
                id=str(uuid.uuid4()),
                ticket=40,
                name="Ramesh Gupta",
                phone="+91 98211 *****",
                language="Hindi",
                category="Bail",
                context=(
                    "Brother in Tihar 8 months. FIR under Section 420. Family lawyer "
                    "non-responsive. Need guidance on bail bond procedure."
                ),
                status="approved",
                submitted_at=now - 1000 * 60 * 42,
                code="NYS-T7KM-4XP",
            ),
            AccessRequest(
                id=str(uuid.uuid4()),
                ticket=41,
                name="Lakshmi Iyer",
                phone="+91 91764 *****",
                language="Tamil",
                category="Maintenance",
                context=(
                    "Separated 2 years. Husband denying maintenance despite court order. "
                    "Section 125 CrPC enforcement query."
                ),
                status="pending",
                submitted_at=now - 1000 * 60 * 7,
            ),
        ]
        for s in seeds:
            self.requests[s.id] = s

    def list_sorted(self) -> list[AccessRequest]:
        return sorted(self.requests.values(), key=lambda r: r.submitted_at, reverse=True)


class Hub:
    """Fan-out WebSocket broadcaster + webhook dispatcher."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._webhooks: set[str] = set()
        self._http = httpx.AsyncClient(timeout=5.0)

    # ---- WebSocket fan-out ----------------------------------------------- #

    async def register(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: dict, *, skip: Optional[WebSocket] = None) -> None:
        """Send event to every connected client (plus every webhook URL)."""
        payload = json.dumps(event)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            if ws is skip:
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

        # Webhooks only fire for domain events, not RTC signaling noise.
        if event.get("type", "").startswith("request."):
            await self._dispatch_webhooks(event)

    async def _dispatch_webhooks(self, event: dict) -> None:
        if not self._webhooks:
            return
        payload = json.dumps(event)
        for url in list(self._webhooks):
            try:
                await self._http.post(
                    url,
                    content=payload,
                    headers={
                        "content-type": "application/json",
                        "x-nyayasathi-event": event.get("type", ""),
                    },
                )
            except Exception as e:
                log.warning("webhook %s failed: %s", url, e)

    # ---- Webhook registration ------------------------------------------- #

    def add_webhook(self, url: str) -> None:
        self._webhooks.add(url)

    def remove_webhook(self, url: str) -> bool:
        if url in self._webhooks:
            self._webhooks.discard(url)
            return True
        return False

    def list_webhooks(self) -> list[str]:
        return sorted(self._webhooks)


# --------------------------------------------------------------------------- #
# Router factory                                                              #
# --------------------------------------------------------------------------- #


CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no ambiguous chars


def _make_code() -> str:
    pick = lambda n: "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))  # noqa: E731
    return f"NYS-{pick(4)}-{pick(3)}"


# --------------------------------------------------------------------------- #
# Chat engine (Auditor / Educator / Advocate pipeline — scripted fallback)    #
# --------------------------------------------------------------------------- #

CHAT_SYSTEM_PROMPT = (
    "You are NyayaSathi, an AI legal companion for India. You are not a lawyer. "
    "Three modules guide every reply: (1) Auditor — flag procedural issues (limitation periods, "
    "jurisdiction, required filings under CPC/CrPC/BNS/BNSS). (2) Educator — ground answers in Indian "
    "statutes and Supreme Court / High Court precedents, cite section numbers and landmark cases "
    "by name. (3) Advocate — offer next steps the user can take today, and recommend consulting a "
    "verified lawyer when the matter is contested. Keep replies under 180 words, warm and plain, "
    "in the user's language. Never fabricate citations. End with: 'I can match you with a verified "
    "lawyer when you're ready.'"
)

# Small knowledge base for the scripted fallback. Matched by keyword; keeps
# the demo useful even with no LLM key attached.
_TOPIC_TEMPLATES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(bail|anticipatory|custody|remand|tihar|arrest|fir)\b", re.I),
        "Bail in India is governed by Sections 478–482 of the BNSS, 2023 (earlier Sections 436–439 CrPC). "
        "Auditor check: verify (a) the offence is bailable or non-bailable, (b) whether a 15-day remand "
        "window has lapsed, and (c) whether any prior bail application was rejected. "
        "Educator: Arnesh Kumar v. State of Bihar (2014) and Satender Kumar Antil v. CBI (2022) require "
        "police to justify arrest for offences punishable up to 7 years — courts must apply the test strictly. "
        "Advocate: collect the FIR copy, charge-sheet (if filed), and prior bail orders, then file a bail "
        "application before the Sessions Court or the High Court depending on the stage. "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(maintenance|alimony|125\s*crpc|divorce|separat|husband|wife|125)\b", re.I),
        "Maintenance is covered by Section 144 BNSS, 2023 (earlier Section 125 CrPC) and the personal law "
        "that applies to you. Auditor: a wife, child, or parent unable to maintain themselves can claim "
        "from any person of sufficient means — non-payment of a final order is a magistrate-level offence. "
        "Educator: Rajnesh v. Neha (2020) lays down the standard Affidavit of Disclosure of Assets — most "
        "magistrates will order it at the first hearing. "
        "Advocate: gather your Aadhaar, income proof, details of the respondent's income/assets, and file "
        "before the Family Court or Magistrate with territorial jurisdiction. "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(evict|tenant|landlord|rent|lease|possession|section\s*106)\b", re.I),
        "Eviction of a tenant without a written lease requires a notice under Section 106 of the Transfer "
        "of Property Act, 1882 (15 days for monthly tenancies). Auditor: check (a) whether state rent control "
        "applies — it overrides TPA, (b) whether the notice was served correctly, and (c) the limitation period "
        "(3 years under Article 67, Limitation Act, 1963). "
        "Educator: the Supreme Court in V. Dhanapal Chettiar v. Yesodai Ammal (1979) held that a Section 106 "
        "notice is not needed where Rent Control legislation governs the tenancy. "
        "Advocate: file a suit for possession and mesne profits before the civil court of pecuniary jurisdiction. "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(domestic violence|dv|498a|dowry|harass)\b", re.I),
        "The Protection of Women from Domestic Violence Act, 2005 (PWDVA) and Section 85 BNS, 2023 "
        "(earlier Section 498A IPC) both apply. Auditor: PWDVA is civil and allows protection, residence, "
        "monetary and custody orders; 498A/BNS 85 is criminal and non-bailable. "
        "Educator: Arnesh Kumar v. State of Bihar (2014) requires a Section 41A CrPC notice before arrest "
        "in 498A cases; D. Velusamy v. D. Patchaiammal (2010) clarifies who is a 'relationship in the nature of marriage'. "
        "Advocate: contact the nearest Protection Officer or Mahila Thana, preserve medical records and "
        "messages, and file a DIR (Domestic Incident Report). "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(consumer|refund|defect|service|e-?commerce|flipkart|amazon)\b", re.I),
        "Consumer Protection Act, 2019 gives you three forums based on value: District (up to ₹50L), "
        "State (₹50L–₹2Cr), National (above ₹2Cr). Auditor: limitation is 2 years from the date of cause "
        "of action; a 15-day written complaint to the seller is best practice before filing. "
        "Educator: the CPA applies to e-commerce platforms under the Consumer Protection (E-Commerce) Rules, "
        "2020 — liability flows to the marketplace for listing misrepresentations. "
        "Advocate: preserve order confirmations, invoices, and chat logs; file online at https://edaakhil.nic.in "
        "or in person. Court fees are nominal. "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(property|land|title|partition|will|inherit|succession|father|mother)\b", re.I),
        "For ancestral or self-acquired property disputes, start with the Hindu Succession Act, 1956 "
        "(as amended in 2005 — daughters are coparceners by birth) or the personal law that applies to you. "
        "Auditor: verify title through mutation records, sale deeds, and encumbrance certificates; the "
        "limitation for partition is 12 years from ouster. "
        "Educator: Vineeta Sharma v. Rakesh Sharma (2020) confirmed daughters' equal coparcenary rights "
        "irrespective of whether the father was alive on 09-Sept-2005. "
        "Advocate: a partition suit is filed before the civil court; consider a family settlement first "
        "— registered settlements save stamp duty and years of litigation. "
        "I can match you with a verified lawyer when you're ready."
    ),
    (
        re.compile(r"\b(cheque|138|ni\s*act|bounce|dishonou?r)\b", re.I),
        "Cheque bounce is governed by Section 138 of the Negotiable Instruments Act, 1881. Auditor: "
        "send a statutory demand notice within 30 days of the cheque return memo; complaint must be filed "
        "within 1 month of the 15-day notice period expiring. "
        "Educator: Dashrath Rupsingh Rathod v. State of Maharashtra (2014) and the 2015 amendment fix "
        "jurisdiction at the payee's bank branch. "
        "Advocate: keep the original cheque, bank return memo, delivery proof of the notice, and file "
        "a private complaint before the Judicial Magistrate. "
        "I can match you with a verified lawyer when you're ready."
    ),
]

_DEFAULT_REPLY = (
    "Thanks for sharing that. I'm NyayaSathi — I can walk you through three checks: "
    "(1) Auditor — whether any procedural deadlines apply to your matter, "
    "(2) Educator — the sections and precedents most relevant to it, and "
    "(3) Advocate — the document trail and next filing you should prepare. "
    "Tell me a bit more: is this a criminal, family, property, consumer, or contract matter? "
    "I can match you with a verified lawyer when you're ready."
)


def _scripted_reply(user_text: str) -> str:
    for pattern, template in _TOPIC_TEMPLATES:
        if pattern.search(user_text):
            return template
    return _DEFAULT_REPLY


async def _claude_reply(messages: list[ChatMessage], language: str) -> Optional[str]:
    """Try Anthropic's Messages API. Returns None on any failure so caller can fall back."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    model = os.environ.get("NYAYASATHI_CHAT_MODEL", "claude-haiku-4-5-20251001")
    system = CHAT_SYSTEM_PROMPT
    if language and language.lower() != "english":
        system += f"\n\nRespond in {language}."
    payload = {
        "model": model,
        "max_tokens": 400,
        "system": system,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                f"{base}/v1/messages",
                json=payload,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            if r.status_code >= 300:
                log.info("anthropic chat fallback: status=%s body=%s", r.status_code, r.text[:200])
                return None
            data = r.json()
            parts = data.get("content") or []
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text") or None
    except Exception as e:
        log.info("anthropic chat fallback: exception=%s", e)
        return None


def build_router() -> tuple[APIRouter, AccessStore, Hub]:
    router = APIRouter()
    store = AccessStore()
    store.seed()
    hub = Hub()

    # -------- REST ------------------------------------------------------- #

    @router.get("/access/requests")
    async def list_requests():
        return {"requests": [r.as_dict() for r in store.list_sorted()]}

    @router.post("/access/requests")
    async def submit(body: SubmitBody):
        async with store.lock:
            req = AccessRequest(
                id=str(uuid.uuid4()),
                ticket=store.next_ticket,
                name=body.name.strip(),
                phone=body.phone.strip(),
                language=body.language,
                category=body.category,
                context=body.context.strip(),
                status="pending",
                submitted_at=int(time.time() * 1000),
            )
            store.requests[req.id] = req
            store.next_ticket += 1
        await hub.broadcast({"type": "request.created", "request": req.as_dict()})
        return req.as_dict()

    @router.post("/access/requests/{rid}/approve")
    async def approve(rid: str):
        async with store.lock:
            req = store.requests.get(rid)
            if not req:
                raise HTTPException(status_code=404, detail="request not found")
            if req.status != "pending":
                # Idempotent: return current state without re-broadcasting.
                return req.as_dict()
            req.status = "approved"
            req.code = _make_code()
        await hub.broadcast({"type": "request.approved", "request": req.as_dict()})
        return req.as_dict()

    @router.post("/access/requests/{rid}/decline")
    async def decline(rid: str, body: DeclineBody):
        async with store.lock:
            req = store.requests.get(rid)
            if not req:
                raise HTTPException(status_code=404, detail="request not found")
            if req.status != "pending":
                return req.as_dict()
            req.status = "declined"
            req.reason = body.reason
        await hub.broadcast({"type": "request.declined", "request": req.as_dict()})
        return req.as_dict()

    # -------- Chat ------------------------------------------------------- #

    @router.post("/access/chat")
    async def chat(body: ChatBody):
        last_user = next(
            (m.content for m in reversed(body.messages) if m.role == "user"), ""
        )
        reply = await _claude_reply(body.messages, body.language)
        source = "claude"
        if not reply:
            reply = _scripted_reply(last_user)
            source = "scripted"
        return {"reply": reply, "source": source}

    # -------- Webhooks --------------------------------------------------- #

    @router.post("/access/webhooks")
    async def webhook_register(body: WebhookBody):
        hub.add_webhook(body.url)
        return {"ok": True, "count": len(hub.list_webhooks())}

    @router.get("/access/webhooks")
    async def webhook_list():
        return {"webhooks": hub.list_webhooks()}

    @router.delete("/access/webhooks")
    async def webhook_delete(body: WebhookBody):
        removed = hub.remove_webhook(body.url)
        return {"ok": removed}

    # -------- WebSocket -------------------------------------------------- #

    RTC_EVENT_TYPES = {"rtc.join", "rtc.leave", "rtc.offer", "rtc.answer", "rtc.ice"}

    @router.websocket("/ws/access")
    async def ws_access(websocket: WebSocket) -> None:
        await websocket.accept()
        await hub.register(websocket)
        try:
            # Send snapshot on connect.
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "snapshot",
                        "requests": [r.as_dict() for r in store.list_sorted()],
                    }
                )
            )

            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                t = msg.get("type")
                if t in RTC_EVENT_TYPES and isinstance(msg.get("code"), str):
                    # Relay WebRTC signaling to all other clients. Keep it narrow —
                    # we don't trust arbitrary client-pushed events beyond signaling.
                    await hub.broadcast(msg, skip=websocket)
                # Ignore unknown messages silently.
        except WebSocketDisconnect:
            pass
        finally:
            await hub.unregister(websocket)

    return router, store, hub
