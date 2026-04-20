"""Redis-backed per-call session state (see plan §1.2)."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import redis.asyncio as aioredis

from app.deps import get_redis

CALL_TTL_SECONDS = 4 * 60 * 60


@dataclass
class CallSession:
    call_uuid: str
    phone_e164: str = ""
    language: str = "hi-IN"
    disclaimer_ack: bool = False
    recording_consent: bool = False
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)
    slot_status: dict[str, str] = field(default_factory=dict)
    turn_log: list[dict[str, Any]] = field(default_factory=list)
    rag_citations: list[dict[str, Any]] = field(default_factory=list)
    state: str = "greeting"  # greeting | slot_filling | drafting | confirm | handoff

    @classmethod
    async def load_or_create(cls, call_uuid: str) -> "CallSession":
        r = get_redis()
        raw = await r.hgetall(cls._key(call_uuid))
        if not raw:
            session = cls(call_uuid=call_uuid)
            await session.save()
            return session
        return cls(
            call_uuid=call_uuid,
            phone_e164=raw.get("phone_e164", ""),
            language=raw.get("language", "hi-IN"),
            disclaimer_ack=raw.get("disclaimer_ack", "0") == "1",
            recording_consent=raw.get("recording_consent", "0") == "1",
            intent=raw.get("intent") or None,
            intent_confidence=float(raw.get("intent_confidence", 0.0) or 0.0),
            slots=json.loads(raw.get("slots", "{}")),
            slot_status=json.loads(raw.get("slot_status", "{}")),
            turn_log=json.loads(raw.get("turn_log", "[]")),
            rag_citations=json.loads(raw.get("rag_citations", "[]")),
            state=raw.get("state", "greeting"),
        )

    async def save(self) -> None:
        r: aioredis.Redis = get_redis()
        key = self._key(self.call_uuid)
        mapping = {
            "phone_e164": self.phone_e164,
            "language": self.language,
            "disclaimer_ack": "1" if self.disclaimer_ack else "0",
            "recording_consent": "1" if self.recording_consent else "0",
            "intent": self.intent or "",
            "intent_confidence": str(self.intent_confidence),
            "slots": json.dumps(self.slots),
            "slot_status": json.dumps(self.slot_status),
            "turn_log": json.dumps(self.turn_log),
            "rag_citations": json.dumps(self.rag_citations),
            "state": self.state,
        }
        await r.hset(key, mapping=mapping)
        await r.expire(key, CALL_TTL_SECONDS)

    async def append_turn(self, role: str, text: str) -> None:
        self.turn_log.append({"role": role, "text": text, "ts": time.time()})
        await self.save()

    async def persist_turn_log(self) -> None:
        """Called on hangup — post_call Celery job migrates this into Postgres."""
        await self.save()

    @staticmethod
    def _key(call_uuid: str) -> str:
        return f"call:{call_uuid}"
