"""
Sarvam AI wrappers: Saaras (streaming ASR), Bulbul (streaming TTS), Sarvam-M (LLM).

Exposes provider-abstraction interfaces (`ASRClient`, `TTSClient`, `LLMClient`)
so we can swap to Google/OpenAI/ElevenLabs fallbacks on rate-limit without
refactoring callers. See plan §8 graceful-degradation.
"""

from typing import AsyncIterator, Optional, Protocol

import httpx

from app.config import get_settings


class ASRClient(Protocol):
    async def transcribe_stream(
        self, frames: AsyncIterator[bytes], language: str
    ) -> AsyncIterator[dict]:
        """Yields {'text': str, 'is_final': bool, 'confidence': float}."""
        ...


class TTSClient(Protocol):
    async def synthesize_stream(
        self, text: str, language: str, voice: str = "default"
    ) -> AsyncIterator[bytes]:
        """Yields 8kHz μ-law frames."""
        ...


class LLMClient(Protocol):
    async def complete_json(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
    ) -> dict:
        """Returns parsed JSON per the system-prompt contract."""
        ...


# ---------------------------------------------------------------------------
# Sarvam concrete implementations — HTTP streaming via their SDK or raw httpx.
# The real streaming endpoints differ per product; this is the interface shim.
# ---------------------------------------------------------------------------


class SarvamASR:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.endpoint = "https://api.sarvam.ai/speech-to-text-streaming"

    async def transcribe_stream(
        self, frames: AsyncIterator[bytes], language: str
    ) -> AsyncIterator[dict]:
        # TODO: real Saaras streaming websocket + partial/final handling.
        raise NotImplementedError


class SarvamTTS:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.endpoint = "https://api.sarvam.ai/text-to-speech"

    async def synthesize_stream(
        self, text: str, language: str, voice: str = "default"
    ) -> AsyncIterator[bytes]:
        # TODO: real Bulbul streaming synthesis; yield 8kHz μ-law frames.
        raise NotImplementedError


class SarvamLLM:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.endpoint = "https://api.sarvam.ai/v1/chat/completions"

    async def complete_json(
        self,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.2,
    ) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.sarvam_api_key}"}
        payload = {
            "model": self.settings.sarvam_llm_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if tools:
            payload["tools"] = tools
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        # Caller is responsible for json.loads on content.
        return data


# Factory: returns primary provider; fallback wiring in month 3+.
def get_asr() -> ASRClient:
    return SarvamASR()


def get_tts() -> TTSClient:
    return SarvamTTS()


def get_llm() -> LLMClient:
    return SarvamLLM()
