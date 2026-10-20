"""
Streamed /v1/completions: wire format, failure handling and resource release.

Most tests use TestClient, which buffers the whole response. Client
disconnects need a real ASGI exchange, so those drive main.app directly with
a receive/send pair that hangs up partway through the stream.
"""

import asyncio
import contextlib
import json

import httpx2
import openai
import pytest
from openai.types import Completion

import app.main as main
from app.backends.openai_compatible import OpenAICompatibleBackend
from app.main import CompletionRequest, create_completion
from tests.helpers import metric
from tests.test_backend_streaming import chat_chunk, sse, sse_response
from tests.test_openai_compatible import remote_settings

REQUESTS = "vortex_inference_requests_total"
TTFT = "vortex_inference_time_to_first_token_seconds_count"
ACTIVE = "vortex_inference_active_requests"


def events(response) -> list:
    """Decoded data payloads of an SSE response, in order."""
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def stream(client, **body):
    return client.post("/v1/completions", json={"prompt": "hi", "stream": True, **body})


async def drained() -> None:
    """Waits until every worker thread has exited and released its permit."""
    for _ in range(200):
        if not main._drain_tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("drain tasks did not finish")


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def test_stream_emits_openai_chunks_then_done(client, fake_model):
    fake_model.stream_pieces = [" Paris", " is", " lovely"]

    response = stream(client)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    payloads = events(response)
    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    # Intermediate chunks carry finish_reason null, which openai.types.Completion
    # rejects on strict validation; the SDK itself builds stream chunks
    # leniently (see test_official_sdk_consumes_the_stream). The final chunk
    # is a complete Completion.
    Completion.model_validate(chunks[-1])
    assert all(c["object"] == "text_completion" for c in chunks)
    assert "".join(c["choices"][0]["text"] for c in chunks) == " Paris is lovely"
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [
        None,
        None,
        None,
        "stop",
    ]
    assert {c["id"] for c in chunks} == {chunks[0]["id"]}
    assert all("usage" not in c for c in chunks)


def test_usage_chunk_is_sent_when_requested(client, fake_model):
    fake_model.stream_pieces = [" one", " two"]

    payloads = events(stream(client, stream_options={"include_usage": True}))

    usage_chunk = Completion.model_validate(payloads[-2])
    assert usage_chunk.choices == []
    assert usage_chunk.usage.model_dump() == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }


def test_completion_id_reuses_the_request_id(client, fake_model):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "stream": True},
        headers={"X-Request-ID": "trace-7"},
    )

    assert events(response)[0]["id"] == "cmpl-trace-7"


def test_official_sdk_consumes_the_stream(client, fake_model):
    fake_model.stream_pieces = [" Hello", " world"]
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="unused", http_client=client
    )

    chunks = list(
        sdk.completions.create(
            model="test-model",
            prompt="hi",
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert "".join(c.choices[0].text for c in chunks if c.choices) == " Hello world"
    assert chunks[-1].usage.completion_tokens == 2


def test_successful_stream_is_measured(client, fake_model):
    successes, ttft = metric(REQUESTS, status="success"), metric(TTFT)

    stream(client)

    assert metric(REQUESTS, status="success") == successes + 1
    assert metric(TTFT) == ttft + 1


def test_stream_without_text_records_no_time_to_first_token(client, fake_model):
    fake_model.stream_pieces = []
    ttft = metric(TTFT)

    payloads = events(stream(client))

    assert payloads[0]["choices"][0]["finish_reason"] == "stop"
    assert metric(TTFT) == ttft


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_failure_before_output_keeps_its_status_code(client, fake_model):
    # The response is committed only after the first event, so these still
    # get a JSON error with the right status instead of a 200 stream.
    response = stream(client, prompt="word " * 64)
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"

    fake_model.error = RuntimeError("weights corrupted at /models/x.gguf")
    response = stream(client)
    assert response.status_code == 500
    assert "/models" not in response.text


def test_failure_mid_stream_ends_with_an_error_event(client, fake_model):
    fake_model.stream_pieces = [" a", " b"]
    fake_model.stream_error = RuntimeError("decode failed at /models/x.gguf")
    errors = metric(REQUESTS, status="error")

    payloads = events(stream(client))

    assert payloads[0]["choices"][0]["text"] == " a"
    assert payloads[-1]["error"]["code"] == 500
    assert "/models" not in json.dumps(payloads[-1])
    assert "[DONE]" not in payloads
    assert metric(REQUESTS, status="error") == errors + 1


def test_deadline_before_first_output_is_a_504(client, fake_model, monkeypatch):
    monkeypatch.setattr(main.settings, "inference_timeout_seconds", 0.2)
    fake_model.stream_pieces = [" late"]
    fake_model.step_delay = 0.5
    timeouts = metric(REQUESTS, status="timeout")

    response = stream(client)

    assert response.status_code == 504
    assert metric(REQUESTS, status="timeout") == timeouts + 1


def test_deadline_mid_stream_ends_with_a_timeout_event(client, fake_model, monkeypatch):
    monkeypatch.setattr(main.settings, "inference_timeout_seconds", 0.3)
    fake_model.stream_pieces = [" a", " b", " c", " d", " e", " f"]
    fake_model.step_delay = 0.1
    timeouts = metric(REQUESTS, status="timeout")

    payloads = events(stream(client))

    assert payloads[0]["choices"][0]["text"] == " a"
    assert payloads[-1]["error"]["code"] == 504
    assert metric(REQUESTS, status="timeout") == timeouts + 1


def test_remote_backend_streams_through_the_api(client, monkeypatch):
    body = sse(
        (None, chat_chunk(content="Hola")),
        (None, chat_chunk(content=" mundo", finish_reason="stop")),
        (None, "[DONE]"),
    )
    backend = OpenAICompatibleBackend(
        remote_settings(),
        transport=httpx2.MockTransport(lambda request: sse_response(body)),
    )
    monkeypatch.setattr(main, "backend", backend)

    payloads = events(stream(client))

    assert "".join(p["choices"][0]["text"] for p in payloads[:-1]) == "Hola mundo"


# ---------------------------------------------------------------------------
# Disconnects and concurrency (real ASGI exchanges)
# ---------------------------------------------------------------------------


async def hang_up_after(chunks_wanted: int, spec_version: str | None) -> list[bytes]:
    """
    Streams a completion and disconnects after receiving some chunks.

    spec_version 2.3 (what uvicorn advertises) makes Starlette listen for the
    disconnect and cancel the response; 2.4 makes it notice only when a send
    fails, which leaves the body generator suspended unless it is closed.
    """
    asgi = {"version": "3.0"}
    if spec_version is not None:
        asgi["spec_version"] = spec_version
    scope = {
        "type": "http",
        "asgi": asgi,
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    body = json.dumps({"prompt": "hi", "stream": True}).encode()
    hung_up = asyncio.Event()
    request_sent = False
    received: list[bytes] = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await hung_up.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            if hung_up.is_set():
                raise OSError("client went away")
            received.append(message["body"])
            if len(received) >= chunks_wanted:
                hung_up.set()

    # On the 2.4 path Starlette surfaces the disconnect as ClientDisconnect.
    with contextlib.suppress(Exception):
        await main.app(scope, receive, send)
    return received


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_stops_generation_and_releases_the_permit(spec_version):
    async def scenario():
        async with main.lifespan(main.app):
            fake = main.backend._model
            fake.stream_pieces = [f" t{i}" for i in range(50)]
            fake.step_delay = 0.02
            cancelled = metric(REQUESTS, status="cancelled")

            received = await hang_up_after(2, spec_version)
            await drained()

            assert len(received) == 2
            assert fake.active_calls == 0
            # Generation stopped well before the 50 pieces it would produce.
            assert fake.steps_run == [] or fake.steps_run[-1] < 50
            assert metric(REQUESTS, status="cancelled") == cancelled + 1
            assert metric(ACTIVE) == 0

            # The permit is free again: the next stream runs to completion.
            fake.stream_pieces, fake.step_delay = [" ok"], 0.0
            response = await create_completion(
                CompletionRequest(prompt="hi", stream=True)
            )
            body = [chunk async for chunk in response.body_iterator]
            assert body[-1] == "data: [DONE]\n\n"

    asyncio.run(scenario())


def test_streams_are_serialized_on_the_local_model():
    async def scenario():
        async with main.lifespan(main.app):
            fake = main.backend._model
            fake.stream_pieces = [" a", " b", " c"]
            fake.step_delay = 0.01

            async def consume():
                response = await create_completion(
                    CompletionRequest(prompt="hi", stream=True)
                )
                return [chunk async for chunk in response.body_iterator]

            bodies = await asyncio.gather(*(consume() for _ in range(3)))
            await drained()

            assert all(body[-1] == "data: [DONE]\n\n" for body in bodies)
            assert fake.max_active_calls == 1
            assert metric(ACTIVE) == 0

    asyncio.run(scenario())


def test_cancel_while_queued_or_before_output_leaks_nothing():
    async def scenario():
        async with main.lifespan(main.app):
            fake = main.backend._model
            fake.stream_pieces = [" slow"]
            fake.step_delay = 0.3
            cancelled = metric(REQUESTS, status="cancelled")

            # The first stream holds the only permit; the second queues.
            holder = asyncio.create_task(
                create_completion(CompletionRequest(prompt="hi", stream=True))
            )
            await asyncio.to_thread(fake.started.wait, 5)
            queued = asyncio.create_task(
                create_completion(CompletionRequest(prompt="hi", stream=True))
            )
            await asyncio.sleep(0.05)
            assert metric(ACTIVE) == 2

            queued.cancel()
            holder.cancel()
            for task in (queued, holder):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await drained()

            # Only the holder had started inference; the queued request never
            # reached the backend and is not counted as an outcome.
            assert metric(REQUESTS, status="cancelled") == cancelled + 1
            assert metric(ACTIVE) == 0
            assert fake.active_calls == 0
            assert main._inference_semaphore._value == 1

    asyncio.run(scenario())


def test_official_sdk_raises_on_a_failure_mid_stream(client, fake_model):
    fake_model.stream_pieces = [" a", " b"]
    fake_model.stream_error = RuntimeError("decode failed")
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="unused", http_client=client
    )

    stream = sdk.completions.create(model="test-model", prompt="hi", stream=True)

    with pytest.raises(openai.APIError, match="internal error"):
        list(stream)
