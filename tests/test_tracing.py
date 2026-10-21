"""Tracing: what a request, a stream and a job record, and how context travels."""

import asyncio
import logging

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from app import tracing
from app.logging_config import LogfmtFormatter

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = f"00-{TRACE_ID}-00f067aa0ba902b7-01"


def by_name(spans, name: str):
    matches = [s for s in spans.get_finished_spans() if s.name == name]
    assert matches, [s.name for s in spans.get_finished_spans()]
    return matches[-1]


def trace_id(span) -> str:
    return format(span.context.trace_id, "032x")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_tracing_is_off_without_an_otlp_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    assert tracing.configure_tracing("vortex-inference") is None
    tracing.shutdown_tracing(None)


@pytest.mark.parametrize(
    "service_env, expected", [(None, "vortex-worker"), ("jobs-eu", "jobs-eu")]
)
def test_tracing_exports_when_an_endpoint_is_set(monkeypatch, service_env, expected):
    installed = []
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    if service_env is None:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    else:
        monkeypatch.setenv("OTEL_SERVICE_NAME", service_env)
    # The test session's provider is global and cannot be replaced.
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", installed.append)

    provider = tracing.configure_tracing("vortex-worker")

    assert installed == [provider]
    assert provider.resource.attributes["service.name"] == expected
    tracing.shutdown_tracing(provider)


# ---------------------------------------------------------------------------
# HTTP requests
# ---------------------------------------------------------------------------


def test_request_traces_the_http_call_and_the_generation(client, spans):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "max_tokens": 5},
        headers={"traceparent": TRACEPARENT},
    )
    assert response.status_code == 200

    server = by_name(spans, "POST /v1/completions")
    generation = by_name(spans, "text_completion test-model")

    assert server.kind == SpanKind.SERVER
    # The caller's trace continues through the server and the backend call.
    assert trace_id(server) == trace_id(generation) == TRACE_ID
    assert generation.parent.span_id == server.context.span_id
    assert server.attributes["http.route"] == "/v1/completions"
    assert server.attributes["http.response.status_code"] == 200
    assert server.attributes["vortex.request_id"] == response.headers["x-request-id"]

    attributes = generation.attributes
    assert attributes["gen_ai.operation.name"] == "text_completion"
    assert attributes["gen_ai.provider.name"] == "llama_cpp"
    assert attributes["gen_ai.request.max_tokens"] == 5
    assert (
        attributes["gen_ai.usage.output_tokens"]
        == response.json()["usage"]["completion_tokens"]
    )
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert [e.name for e in generation.events] == ["permit_acquired"]


def test_probes_and_scrapes_are_not_traced(client, spans):
    client.get("/health/ready")
    client.get("/metrics")

    assert spans.get_finished_spans() == ()


def test_server_failures_mark_the_spans_as_errors(client, fake_model, spans):
    fake_model.error = RuntimeError("native crash")

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 500
    generation = by_name(spans, "text_completion test-model")
    assert generation.status.status_code == StatusCode.ERROR
    assert generation.attributes["error.type"] == "500"
    assert by_name(spans, "POST /v1/completions").status.status_code == StatusCode.ERROR


def test_rejected_prompts_are_not_service_errors(client, spans):
    response = client.post("/v1/completions", json={"prompt": "word " * 200})

    assert response.status_code == 400
    generation = by_name(spans, "text_completion test-model")
    assert generation.status.status_code == StatusCode.UNSET
    assert generation.attributes["vortex.outcome"] == "rejected"


def test_stream_span_ends_with_the_stream(client, fake_model, spans):
    fake_model.stream_pieces = ["a", "b", "c"]

    with client.stream(
        "POST", "/v1/completions", json={"prompt": "hi", "stream": True}
    ) as response:
        body = "".join(response.iter_text())
    assert "[DONE]" in body

    generation = by_name(spans, "text_completion test-model")
    assert generation.attributes["gen_ai.usage.output_tokens"] == 3
    assert generation.attributes["vortex.outcome"] == "success"
    assert [e.name for e in generation.events] == ["permit_acquired", "first_token"]


def test_log_lines_inside_a_trace_carry_its_id():
    record = logging.LogRecord("vortex", logging.INFO, "", 0, "hello", None, None)

    assert "trace_id=" not in LogfmtFormatter().format(record)
    with tracing.tracer.start_as_current_span("work") as span:
        line = LogfmtFormatter().format(record)

    assert f"trace_id={format(span.get_span_context().trace_id, '032x')}" in line


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_job_messages_carry_the_current_trace():
    from app.jobs.models import JobMessage
    from app.jobs.queue import job_message

    message = JobMessage(job_id="job_" + "a" * 32, url="/v1/completions", body={})
    with tracing.tracer.start_as_current_span("submit") as span:
        headers = job_message(message).headers

    context = span.get_span_context()
    # version-trace_id-parent_id-flags; the flags depend on the sampler.
    _, trace_hex, parent_hex, _ = headers["traceparent"].split("-")
    assert (trace_hex, parent_hex) == (
        f"{context.trace_id:032x}",
        f"{context.span_id:016x}",
    )


def test_worker_continues_the_trace_from_the_message(spans):
    from app.jobs.processor import Decision
    from tests.test_worker import FakeChannel, FakeMessage, StaticProcessor, make_worker

    worker = make_worker(StaticProcessor(Decision("ack", "succeeded")))
    worker._channel = FakeChannel()
    message = FakeMessage()
    # Header values arrive from the broker as bytes or str.
    message.headers = {"traceparent": TRACEPARENT.encode()}

    asyncio.run(worker._on_message(message))

    process = by_name(spans, "process vortex.jobs")
    assert process.kind == SpanKind.CONSUMER
    assert trace_id(process) == TRACE_ID
    assert process.attributes["messaging.message.id"] == "job_" + "a" * 32
    assert process.attributes["vortex.job.outcome"] == "succeeded"


def test_job_generation_is_a_child_of_its_processing(spans):
    from app.jobs.models import Job
    from app.jobs.processor import JobProcessor
    from tests.fakes import InMemoryJobStore
    from tests.test_worker import JOB_ID, FakeRunner, body

    store = InMemoryJobStore()
    store.jobs[JOB_ID] = Job(id=JOB_ID, url="/v1/completions", status="queued")
    processor = JobProcessor(
        store, FakeRunner(), "test-model", 3, 5, provider="openai_compatible"
    )

    with tracing.tracer.start_as_current_span("process vortex.jobs") as parent:
        asyncio.run(processor.process(body()))

    generation = by_name(spans, "text_completion test-model")
    assert generation.parent.span_id == parent.get_span_context().span_id
    assert generation.kind == SpanKind.CLIENT
    assert generation.attributes["gen_ai.provider.name"] == "openai_compatible"
    assert generation.attributes["gen_ai.usage.input_tokens"] == 3
