"""/v1/chat/completions: request validation, response formats, SDK compatibility."""

import json

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import SecretStr

import app.main as main
from app.backends.openai_compatible import OpenAICompatibleBackend
from tests.fakes import FakeChatFormatter
from tests.helpers import metric
from tests.test_openai_compatible import chat_reply, remote_settings

REQUESTS = "vortex_inference_requests_total"
CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Capital of France?"},
]


@pytest.fixture
def chat_model(chat_backend, monkeypatch):
    """Installs the template-carrying llama.cpp backend in the running app."""
    monkeypatch.setattr(main, "backend", chat_backend)
    return chat_backend._model


def chat(client, **body):
    return client.post("/v1/chat/completions", json={"messages": CONVERSATION, **body})


def sse_payloads(response) -> list:
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            data = line.removeprefix("data: ")
            payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


# ---------------------------------------------------------------------------
# Buffered
# ---------------------------------------------------------------------------


def test_reply_matches_the_openai_chat_completion_schema(client, chat_model):
    chat_model.text = "Paris."

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": CONVERSATION},
        headers={"X-Request-ID": "trace-9"},
    )

    assert response.status_code == 200
    completion = ChatCompletion.model_validate(response.json())
    assert completion.id == "chatcmpl-trace-9"
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Paris."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )
    assert set(response.json()["performance"]) == {
        "tokens_per_second",
        "duration_seconds",
    }


def test_messages_are_normalized_for_the_backend(client, chat_model):
    chat(
        client,
        messages=[
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Be "},
                    {"type": "text", "text": "brief."},
                ],
            },
            {"role": "user", "content": "Hi"},
        ],
    )

    (formatter,) = FakeChatFormatter.instances
    assert formatter.rendered[-1] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]


def test_parameters_are_forwarded(client, chat_model):
    chat(
        client,
        max_tokens=50,
        max_completion_tokens=12,
        temperature=0.1,
        top_p=0.5,
        stop="\n",
    )

    call = chat_model.calls[-1]
    # max_completion_tokens wins over max_tokens; a string stop is a list of
    # one, followed by the template's end-of-turn marker.
    assert call["max_tokens"] == 12
    assert (call["temperature"], call["top_p"]) == (0.1, 0.5)
    assert call["stop"] == ["\n", "</s>"]


def test_max_tokens_alone_is_honoured(client, chat_model):
    chat(client, max_tokens=7)

    assert chat_model.calls[-1]["max_tokens"] == 7


def test_model_without_a_chat_template_answers_400(client, fake_model):
    rejected = metric(REQUESTS, status="rejected")

    response = chat(client)

    assert response.status_code == 400
    assert "no chat template" in response.json()["detail"]
    assert metric(REQUESTS, status="rejected") == rejected + 1


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"messages": [{"role": "user", "content": "x", "name": "bob"}]},
        {"messages": [{"role": "user", "content": "x" * 32769}]},
        {"messages": [{"role": "user", "content": "x" * 30000}] * 3},
        {"messages": CONVERSATION, "stop": ["a", "b", "c", "d", "e"]},
        {"messages": CONVERSATION, "stop": ""},
        {"messages": CONVERSATION, "max_completion_tokens": 4096},
        {"messages": CONVERSATION, "tools": []},
        {"messages": CONVERSATION, "n": 2},
        {"messages": CONVERSATION, "stream_options": {"include_usage": True}},
    ],
    ids=[
        "no-messages",
        "unsupported-role",
        "non-text-part",
        "unknown-message-field",
        "message-too-long",
        "conversation-too-long",
        "too-many-stops",
        "empty-stop",
        "max-tokens-above-limit",
        "tools",
        "multiple-choices",
        "stream-options-without-stream",
    ],
)
def test_invalid_requests_are_rejected_before_inference(client, chat_model, body):
    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 422
    assert chat_model.calls == []


def test_chat_requires_the_api_key(client, chat_model, monkeypatch):
    monkeypatch.setattr(main.settings, "api_key", SecretStr("test-api-key"))

    assert chat(client).status_code == 401
    authorized = client.post(
        "/v1/chat/completions",
        json={"messages": CONVERSATION},
        headers={"Authorization": "Bearer test-api-key"},
    )
    assert authorized.status_code == 200


def test_remote_backend_receives_the_conversation(client, monkeypatch):
    requests = []

    def upstream(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=chat_reply(content="Paris."))

    monkeypatch.setattr(
        main,
        "backend",
        OpenAICompatibleBackend(
            remote_settings(), transport=httpx2.MockTransport(upstream)
        ),
    )

    response = chat(client)

    assert response.json()["choices"][0]["message"]["content"] == "Paris."
    assert requests[-1]["messages"] == CONVERSATION


# ---------------------------------------------------------------------------
# Streamed
# ---------------------------------------------------------------------------


def test_stream_follows_openai_chunk_conventions(client, chat_model):
    chat_model.stream_pieces = ["Pa", "ris."]

    payloads = sse_payloads(
        chat(client, stream=True, stream_options={"include_usage": True})
    )

    assert payloads[-1] == "[DONE]"
    chunks = [ChatCompletionChunk.model_validate(p) for p in payloads[:-1]]
    assert all(c.object == "chat.completion.chunk" for c in chunks)
    # Role first with empty content, then text, then an empty delta that
    # carries the finish reason, then usage with no choices.
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content == ""
    assert [c.choices[0].delta.content for c in chunks[1:3]] == ["Pa", "ris."]
    assert chunks[3].choices[0].finish_reason == "stop"
    assert chunks[3].choices[0].delta.content is None
    assert chunks[4].choices == []
    assert chunks[4].usage.completion_tokens == 2


def test_stream_failure_before_output_keeps_its_status(client, fake_model):
    # The template-less model rejects the conversation before any output.
    response = chat(client, stream=True)

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"


def test_official_sdk_round_trip(client, chat_model):
    chat_model.text = "Paris."
    chat_model.stream_pieces = ["Pa", "ris."]
    sdk = openai.OpenAI(
        base_url="http://testserver/v1", api_key="unused", http_client=client
    )

    reply = sdk.chat.completions.create(model="test-model", messages=CONVERSATION)
    streamed = sdk.chat.completions.create(
        model="test-model",
        messages=CONVERSATION,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(streamed)

    assert reply.choices[0].message.content == "Paris."
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "Paris."
    assert chunks[-1].usage.completion_tokens == 2
