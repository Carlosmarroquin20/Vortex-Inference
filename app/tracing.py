"""
Distributed tracing with OpenTelemetry.

Tracing is configured only when OTEL_EXPORTER_OTLP_ENDPOINT (or
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) is set, through the standard OpenTelemetry
environment variables; spans are then exported over OTLP/HTTP in batches.
Without it the OpenTelemetry API is a no-op, so the instrumentation below
costs nothing in a deployment that does not collect traces.

What a trace contains:

    HTTP server span      one per request, continuing an inbound traceparent
    generation span       one per backend call, with the GenAI semantic
                          conventions: model, token usage, finish reason
    publish span          the API queueing a job in RabbitMQ
    process span          a worker running one delivery of that job

The job's trace context travels in the AMQP message headers, so a job reads
as one trace from the HTTP request that submitted it, through the queue, to
the worker that ran it, retries included.
"""

import logging
import os
from collections.abc import Mapping, MutableMapping

from opentelemetry import propagate, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

logger = logging.getLogger("vortex.tracing")

tracer = trace.get_tracer("vortex-inference")


def configure_tracing(default_service_name: str):
    """
    Installs an exporting tracer provider when an OTLP endpoint is configured.

    Returns the provider, to be passed to shutdown_tracing() on exit, or None
    when tracing is off.
    """
    if not (
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    ):
        return None

    # Imported here: the SDK and exporter are only needed when exporting.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Resource.create() reads OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES;
    # the default name applies only when neither sets one.
    attributes = {}
    if "OTEL_SERVICE_NAME" not in os.environ:
        attributes["service.name"] = default_service_name
    provider = TracerProvider(resource=Resource.create(attributes))
    # The exporter takes its endpoint, headers and timeout from the
    # OTEL_EXPORTER_OTLP_* variables.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info(
        "Tracing enabled.",
        extra={"service_name": provider.resource.attributes.get("service.name")},
    )
    return provider


def shutdown_tracing(provider) -> None:
    """Exports the spans still buffered; called once at process exit."""
    if provider is not None:
        provider.shutdown()


def inject(carrier: MutableMapping[str, str]) -> None:
    """Writes the current trace context into carrier (W3C traceparent)."""
    propagate.inject(carrier)


def extract(carrier: Mapping[str, object] | None):
    """The trace context carried in headers; values may arrive as bytes."""
    headers = {
        key: value.decode("latin-1") if isinstance(value, bytes) else str(value)
        for key, value in (carrier or {}).items()
    }
    return propagate.extract(headers)


def current_trace_id() -> str | None:
    """The active trace's ID as 32 hex digits, for log correlation."""
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def operation_name(prompt: object) -> str:
    """gen_ai.operation.name: a conversation is a chat, text a completion."""
    return "chat" if isinstance(prompt, list) else "text_completion"


def start_generation_span(generation, model: str, provider: str) -> Span:
    """
    A span for one backend call, named "{operation} {model}" as the GenAI
    semantic conventions specify. Ended with end_generation_span().
    """
    operation = operation_name(generation.prompt)
    return tracer.start_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT if provider != "llama_cpp" else SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "gen_ai.request.max_tokens": generation.max_tokens,
            "gen_ai.request.temperature": generation.temperature,
            "gen_ai.request.top_p": generation.top_p,
        },
    )


def end_generation_span(
    span: Span,
    *,
    tokens_prompt: int | None = None,
    tokens_completion: int | None = None,
    finish_reason: str | None = None,
    response_model: str | None = None,
    outcome: str = "success",
    status_code: int | None = None,
) -> None:
    if tokens_prompt is not None:
        span.set_attribute("gen_ai.usage.input_tokens", tokens_prompt)
    if tokens_completion is not None:
        span.set_attribute("gen_ai.usage.output_tokens", tokens_completion)
    if finish_reason is not None:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    if response_model is not None:
        span.set_attribute("gen_ai.response.model", response_model)
    span.set_attribute("vortex.outcome", outcome)
    if outcome != "success":
        span.set_attribute("error.type", str(status_code) if status_code else outcome)
    # Server-side failures only; a rejected prompt or a client that went away
    # is not an error of this service.
    if outcome in ("error", "timeout"):
        span.set_status(Status(StatusCode.ERROR))
    span.end()
