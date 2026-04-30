"""
Dialog orchestrator — guided-slot FSM (plan §2.2).

Per-turn protocol:
  1. Receive user utterance from ASR.
  2. Run safety classifier in parallel.
  3. If unsafe / refusal → emit safe reply + escalate.
  4. Else call LLM with system prompt + slot schema + turn log.
     LLM returns JSON: {reply_text, slots_update, next_action, tool_calls?}
  5. Merge slots_update into session; handle any tool_calls (judgment_search).
  6. Pick next action: ask | confirm | draft | handoff.
  7. Stream reply_text through TTS.
"""

import json
from dataclasses import dataclass
from typing import Optional

from app.voice.dialog.intents import Intent, TEMPLATE_FOR_INTENT
from app.voice.dialog.safety import SafetyVerdict, classify_safety
from app.voice.dialog.tools import execute_tool_call
from app.voice.sarvam_client import get_llm
from app.voice.session import CallSession


@dataclass
class TurnResponse:
    reply_text: str
    next_action: str  # 'ask' | 'confirm' | 'draft' | 'handoff' | 'end'
    should_barge_in_cancel: bool = False
    safety: Optional[SafetyVerdict] = None


SYSTEM_PROMPT = """You are NyayaSathi, an AI legal-information and document-drafting assistant for India.
You are NOT a lawyer. You do not give legal advice. You provide legal information + draft documents.
Always open-disclaim; when asked to judge or predict, defer to a human lawyer.

Speak in the user's language. Keep replies under 2 sentences unless drafting.

Return STRICT JSON:
{"reply_text": "...", "slots_update": {...}, "next_action": "ask|confirm|draft|handoff|end", "tool_calls": []}
"""


async def run_turn(session: CallSession, user_text: str) -> TurnResponse:
    # 1. Safety check (parallel in real impl; sequential here for stub).
    verdict = await classify_safety(user_text, session.language)
    if verdict.must_refuse:
        await session.append_turn("user", user_text)
        await session.append_turn("assistant", verdict.safe_reply)
        return TurnResponse(
            reply_text=verdict.safe_reply,
            next_action="handoff" if verdict.escalate else "ask",
            safety=verdict,
        )

    # 2. LLM turn.
    llm = get_llm()
    messages = [*[
        {"role": t["role"], "content": t["text"]} for t in session.turn_log[-8:]
    ], {"role": "user", "content": user_text}]
    try:
        response = await llm.complete_json(SYSTEM_PROMPT, messages)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception:
        # Fallback graceful message.
        return TurnResponse(
            reply_text="Main samajh nahi paaya. Kripya dobara boliye.",
            next_action="ask",
        )

    # 3. Merge slots + run any tool calls (e.g., judgment_search).
    if parsed.get("slots_update"):
        session.slots.update(parsed["slots_update"])
    for call in parsed.get("tool_calls", []):
        result = await execute_tool_call(call, session)
        if result:
            session.rag_citations.extend(result.get("citations", []))

    # 4. Persist + return.
    reply = parsed.get("reply_text", "")
    next_action = parsed.get("next_action", "ask")
    await session.append_turn("user", user_text)
    await session.append_turn("assistant", reply)

    return TurnResponse(reply_text=reply, next_action=next_action)


def resolve_template_for_intent(intent: Intent) -> Optional[str]:
    return TEMPLATE_FOR_INTENT.get(intent)
