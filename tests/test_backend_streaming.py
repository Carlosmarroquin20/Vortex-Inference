"""
Backend.stream() for each backend, below the HTTP layer.

The remote backends read real server-sent event bodies served through
httpx2.MockTransport, so SSE framing and parsing are exercised end to end.
"""

import json
import threading

import httpx2
import pytest

import app.main as main
from app.backends import BackendError, PromptTooLongError, StreamEnd, TextDelta
from app.backends.anthropic_api import AnthropicBackend
from app.backends.openai_compatible import OpenAICompatibleBackend
from tests.test_anthropic_backend import MODEL, anthropic_settings
from tests.test_openai_compatible import remote_settings


def collect(events):
    events = list(events)
    deltas = [event.text for event in events if isinstance(event, TextDelta)]
    ends = [event for event in events if isinstance(event, StreamEnd)]
    assert len(ends) == 1 and events[-1] is ends[0], "stream must end with StreamEnd"
    return deltas, ends[0]


def sse(*events: tuple[str | None, object]) -> bytes:
    """Encodes (event name, payload) pairs as a server-sent event body."""
    lines = []
    for name, payload in events:
        if name is not None:
            lines.append(f"event: {name}")
        data = payload if isinstance(payload, str) else json.dumps(payload)
        lines.append(f"data: {data}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def sse_response(body: bytes, status: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status, content=body, headers={"content-type": "text/event-stream"}
    )


class Upstream:
    def __init__(self, handler) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler = handler

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


def test_llama_streams_chunks_then_usage(fake_model):
    fake_model.stream_pieces = [" Paris", " is", " lovely"]

    deltas, end = collect(main.backend.stream("one two three", 8, 0.2, 0.9, ["\n"]))

    assert deltas == [" Paris", " is", " lovely"]
    assert end == StreamEnd("stop", tokens_prompt=3, tokens_completion=3)
    call = fake_model.calls[-1]
    assert call["stream"] is True
    assert (call["max_tokens"], call["temperature"], call["top_p"]) == (8, 0.2, 0.9)
    assert call["stop"] == ["\n"]
    assert main.backend.busy_seconds() == 0.0


def test_llama_empty_chunks_count_but_are_not_forwarded(fake_model):
    # A token that decodes to nothing on its own still consumed a step.
    fake_model.stream_pieces = ["", " hi", ""]

    deltas, end = collect(main.backend.stream("x", 8, 0.7, 0.95, None))

    assert deltas == [" hi"]
    assert end.tokens_completion == 3


def test_llama_prompt_too_long_raises_before_any_event(fake_model):
    prompt = " ".join(["word"] * main.settings.n_ctx)

    with pytest.raises(PromptTooLongError):
        next(main.backend.stream(prompt, 8, 0.7, 0.95, None))
    assert fake_model.calls == []


def test_llama_closing_the_stream_releases_the_model(fake_model):
    fake_model.stream_pieces = [" a", " b", " c"]
    stream = main.backend.stream("x", 8, 0.7, 0.95, None)

    assert next(stream) == TextDelta(" a")
    assert fake_model.active_calls == 1
    assert main.backend.busy_seconds() > 0.0 or main.backend._busy_since is not None

    stream.close()

    assert fake_model.active_calls == 0
    assert main.backend.busy_seconds() == 0.0


def test_llama_stop_event_ends_generation(fake_model):
    fake_model.stream_pieces = [" a", " b", " c"]
    stop_event = threading.Event()
    stream = main.backend.stream("x", 8, 0.7, 0.95, None, stop_event)

    assert next(stream) == TextDelta(" a")
    stop_event.set()
    deltas, end = collect(stream)

    assert deltas == []
    assert end.finish_reason == "stop"
    assert fake_model.steps_run[-1] == 1


def test_llama_failure_mid_stream_propagates(fake_model):
    fake_model.stream_pieces = [" a", " b"]
    fake_model.stream_error = RuntimeError("decode failed")
    stream = main.backend.stream("x", 8, 0.7, 0.95, None)

    assert next(stream) == TextDelta(" a")
    with pytest.raises(RuntimeError, match="decode failed"):
        next(stream)
    assert fake_model.active_calls == 0
    assert main.backend.busy_seconds() == 0.0


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def chat_chunk(content=None, finish_reason=None, usage=None, choices=True):
    chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk"}
    chunk["choices"] = (
        [
            {
                "index": 0,
                "delta": {} if content is None else {"content": content},
                "finish_reason": finish_reason,
            }
        ]
        if choices
        else []
    )
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def openai_backend(upstream, **overrides) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        remote_settings(**overrides), transport=httpx2.MockTransport(upstream)
    )


def test_openai_chat_stream(client):
    upstream = Upstream(
        lambda request: sse_response(
            sse(
                (None, chat_chunk(content="")),
                (None, chat_chunk(content="Par")),
                (None, chat_chunk(content="is.")),
                (None, chat_chunk(finish_reason="stop")),
                (
                    None,
                    chat_chunk(
                        usage={"prompt_tokens": 9, "completion_tokens": 2},
                        choices=False,
                    ),
                ),
                (None, "[DONE]"),
            )
        )
    )

    deltas, end = collect(openai_backend(upstream).stream("hi", 5, 0.7, 0.95, None))

    assert deltas == ["Par", "is."]
    assert end == StreamEnd("stop", tokens_prompt=9, tokens_completion=2)
    body = upstream.last_body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_completions_stream_ignores_non_data_lines(client):
    body = (
        b": keep-alive\n\n"
        b'data: {"choices": [{"text": " Pa", "finish_reason": null}]}\n\n'
        b"event: ping\n\n"
        b'data: {"choices": [{"text": "ris", "finish_reason": "length"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    upstream = Upstream(lambda request: sse_response(body))

    deltas, end = collect(
        openai_backend(upstream, remote_api_style="completions").stream(
            "hi", 5, 0.7, 0.95, None
        )
    )

    assert deltas == [" Pa", "ris"]
    # No usage event: counts stay at zero rather than failing the stream.
    assert end == StreamEnd("length", 0, 0)
    assert upstream.requests[-1].url.path.endswith("/completions")


def test_openai_error_status_raises_before_any_event(client):
    upstream = Upstream(
        lambda request: httpx2.Response(
            429, json={"error": {"message": "slow down"}}, headers={"retry-after": "4"}
        )
    )

    with pytest.raises(BackendError) as excinfo:
        next(openai_backend(upstream).stream("hi", 5, 0.7, 0.95, None))
    assert (excinfo.value.status_code, excinfo.value.retry_after) == (503, "4")


@pytest.mark.parametrize(
    "event",
    [
        '{"error": {"message": "model crashed"}}',
        "not json",
        "[1, 2]",
        '{"choices": ["not an object"]}',
        '{"choices": [{"delta": {"content": 42}}]}',
    ],
)
def test_openai_bad_event_mid_stream_raises(client, event):
    body = sse((None, chat_chunk(content="ok")), (None, event))
    stream = openai_backend(Upstream(lambda request: sse_response(body))).stream(
        "hi", 5, 0.7, 0.95, None
    )

    assert next(stream) == TextDelta("ok")
    with pytest.raises(BackendError):
        next(stream)


def test_openai_stop_event_closes_the_stream(client):
    body = sse(*[(None, chat_chunk(content=f"t{i}")) for i in range(5)])
    stop_event = threading.Event()
    backend = openai_backend(Upstream(lambda request: sse_response(body)))
    stream = backend.stream("hi", 5, 0.7, 0.95, None, stop_event)

    assert next(stream) == TextDelta("t0")
    stop_event.set()

    assert list(stream) == []
    assert backend.busy_seconds() == 0.0


def test_openai_transport_failure_mid_stream(client):
    def broken(request):
        raise httpx2.ReadTimeout("stalled", request=request)

    with pytest.raises(BackendError) as excinfo:
        next(openai_backend(Upstream(broken)).stream("hi", 5, 0.7, 0.95, None))
    assert excinfo.value.status_code == 504


def test_openai_abandoned_stream_is_not_sent(client):
    upstream = Upstream(lambda request: sse_response(b""))
    stop_event = threading.Event()
    stop_event.set()

    deltas, end = collect(
        openai_backend(upstream).stream("hi", 5, 0.7, 0.95, None, stop_event)
    )

    assert (deltas, end, upstream.requests) == ([], StreamEnd(None, 0, 0), [])


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def message_events(*texts, stop_reason="end_turn", model=MODEL, extra=()):
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "hmm"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for text in texts:
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
    events.extend(extra)
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


def anthropic_backend(upstream, **overrides) -> AnthropicBackend:
    return AnthropicBackend(
        anthropic_settings(**overrides),
        http_client=httpx2.Client(transport=httpx2.MockTransport(upstream)),
    )


def test_anthropic_stream_forwards_text_only(client):
    upstream = Upstream(
        lambda request: sse_response(sse(*message_events("Par", "is.")))
    )

    deltas, end = collect(anthropic_backend(upstream).stream("hi", 50, 0.7, 0.95, None))

    assert deltas == ["Par", "is."]
    assert end == StreamEnd("stop", tokens_prompt=12, tokens_completion=7)
    body = upstream.last_body
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in upstream.requests[-1].headers.get(
        "anthropic-beta", ""
    )


def test_anthropic_stream_without_fallback_uses_the_standard_endpoint(client):
    upstream = Upstream(lambda request: sse_response(sse(*message_events("ok"))))

    collect(
        anthropic_backend(upstream, anthropic_refusal_fallback=False).stream(
            "hi", 50, 0.7, 0.95, None
        )
    )

    assert "fallbacks" not in upstream.last_body
    assert "beta" not in str(upstream.requests[-1].url)


def test_anthropic_final_refusal_maps_to_content_filter(client):
    upstream = Upstream(
        lambda request: sse_response(
            sse(*message_events("partial", stop_reason="refusal"))
        )
    )

    deltas, end = collect(anthropic_backend(upstream).stream("hi", 50, 0.7, 0.95, None))

    assert deltas == ["partial"]
    assert end.finish_reason == "content_filter"


def test_anthropic_fallback_continues_the_same_stream(client):
    # A mid-stream decline keeps the partial text, marks the switch with a
    # fallback block, and the fallback model's text follows on the stream.
    marker = [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "fallback"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": " continued"},
            },
        ),
    ]
    upstream = Upstream(
        lambda request: sse_response(sse(*message_events("Begun", extra=marker)))
    )

    deltas, end = collect(anthropic_backend(upstream).stream("hi", 50, 0.7, 0.95, None))

    assert "".join(deltas) == "Begun continued"
    assert end.finish_reason == "stop"


def test_anthropic_error_event_mid_stream_keeps_its_meaning(client):
    events = message_events("ok")[:6] + [
        (
            "error",
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
        )
    ]
    stream = anthropic_backend(
        Upstream(lambda request: sse_response(sse(*events)))
    ).stream("hi", 50, 0.7, 0.95, None)

    assert next(stream) == TextDelta("ok")
    with pytest.raises(BackendError) as excinfo:
        next(stream)
    # Raised by the SDK with the stream's status (200); mapped from the type.
    assert excinfo.value.status_code == 503


def test_anthropic_error_status_raises_before_any_event(client):
    upstream = Upstream(
        lambda request: httpx2.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad stop"},
            },
        )
    )

    with pytest.raises(ValueError, match="bad stop"):
        next(anthropic_backend(upstream).stream("hi", 50, 0.7, 0.95, None))


def test_anthropic_stop_event_closes_the_stream(client):
    upstream = Upstream(
        lambda request: sse_response(sse(*message_events("a", "b", "c")))
    )
    stop_event = threading.Event()
    backend = anthropic_backend(upstream)
    stream = backend.stream("hi", 50, 0.7, 0.95, None, stop_event)

    assert next(stream) == TextDelta("a")
    stop_event.set()

    assert list(stream) == []
    assert backend.busy_seconds() == 0.0


def test_anthropic_abandoned_stream_is_not_sent(client):
    upstream = Upstream(lambda request: sse_response(b""))
    stop_event = threading.Event()
    stop_event.set()

    deltas, end = collect(
        anthropic_backend(upstream).stream("hi", 50, 0.7, 0.95, None, stop_event)
    )

    assert (deltas, end, upstream.requests) == ([], StreamEnd(None, 0, 0), [])
