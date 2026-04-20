"""OpenTelemetry tracing bootstrap — no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set."""

import os

from opentelemetry import trace


def setup_tracing(service_name: str = "nyayasathi") -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    # TODO: wire OTLP exporter + resource attributes on real deploy.
    _ = trace.get_tracer_provider()
    _ = service_name
