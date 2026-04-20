"""Prometheus metrics helpers."""

from prometheus_client import Counter, Histogram

CALLS_STARTED = Counter("nyayasathi_calls_started_total", "Calls answered", ["language"])
CALLS_COMPLETED = Counter(
    "nyayasathi_calls_completed_total", "Calls completed", ["outcome", "intent"]
)
TURN_LATENCY = Histogram(
    "nyayasathi_turn_latency_seconds",
    "Round-trip latency user-speech-end → TTS-first-frame",
    buckets=(0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
)
DOCUMENTS_GENERATED = Counter(
    "nyayasathi_documents_generated_total", "Documents rendered", ["template_key", "language"]
)
HANDOFFS_PROPOSED = Counter(
    "nyayasathi_handoffs_proposed_total", "Lawyer handoffs proposed", ["partner", "domain"]
)
HANDOFFS_PAID = Counter(
    "nyayasathi_handoffs_paid_total", "Paid bookings", ["partner", "domain"]
)
