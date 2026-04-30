"""
Voice-activity detection for barge-in. Uses Silero VAD (tiny ONNX model,
~20ms per 30ms frame on CPU). Month-1 stub; real wiring in month 2.
"""

from dataclasses import dataclass


@dataclass
class VADResult:
    is_speech: bool
    probability: float


class BargeInDetector:
    """
    Stateful detector: reports barge-in when speech probability stays above
    threshold for `trigger_ms` while we are emitting outbound TTS.
    """

    def __init__(self, threshold: float = 0.6, trigger_ms: int = 300) -> None:
        self.threshold = threshold
        self.trigger_ms = trigger_ms
        self._speech_ms = 0

    def feed(self, frame_ms: int, prob: float) -> bool:
        if prob >= self.threshold:
            self._speech_ms += frame_ms
        else:
            self._speech_ms = 0
        return self._speech_ms >= self.trigger_ms

    def reset(self) -> None:
        self._speech_ms = 0


def probe(_pcm_8k_mulaw: bytes) -> VADResult:
    # TODO: real Silero inference on a 30ms 8kHz frame.
    return VADResult(is_speech=False, probability=0.0)
