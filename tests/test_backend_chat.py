"""Conversations (list-of-messages prompts) in each backend."""

import json
import threading

import httpx2
import pytest

import app.main as main
from app.backends import PromptTooLongError, RequestRejectedError, StreamEnd
from app.backends.anthropic_api import AnthropicBackend
from app.backends.llamacpp import LlamaCppBackend
from app.backends.openai_compatible import OpenAICompatibleBackend
from tests.fakes import EOS_ID, FakeChatFormatter, FakeLlama
from tests.test_anthropic_backend import anthropic_settings, message_reply
from tests.test_backend_streaming import chat_chunk, collect, sse, sse_response
from tests.test_openai_compatible import chat_reply, remote_settings

CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Capital of France?"},
]


@pytest.fixture
def chat_backend(client, monkeypatch) -> LlamaCppBackend:
    """A llama.cpp backend whose model ships a chat template."""
    monkeypatch.setattr(FakeLlama, "chat_template", "{{ messages }}")
    monkeypatch.setattr(FakeChatFormatter, "instances", [])
    monkeypatch.setattr(FakeChatFormatter, "render_error", None)
    return LlamaCppBackend(main.settings)


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


def test_formatter_is_built_from_the_model_template(chat_backend):
    (formatter,) = FakeChatFormatter.instances

    assert formatter.template == "{{ messages }}"
    assert (formatter.bos_token, formatter.eos_token) == ("<s>", "</s>")
    assert formatter.stop_token_ids == [EOS_ID]


def test_conversation_is_rendered_and_sent_as_tokens(chat_backend):
    fake = chat_backend._model

    result = chat_backend.generate(CONVERSATION, 16, 0.2, 0.9, ["\n\n"])

    (formatter,) = FakeChatFormatter.instances
    assert formatter.rendered == [CONVERSATION]
    # Tokenized like llama-cpp-python's chat handler: the template already
    # adds special tokens, so no extra BOS, and special tokens are parsed.
    assert fake.last_tokenize["add_bos"] is False
    assert fake.last_tokenize["special"] is True
    call = fake.calls[-1]
    assert isinstance(call["prompt"], list)
    assert call["stop"] == ["\n\n", "</s>"]
    assert result["tokens_prompt"] == len(call["prompt"])
    assert result["text"] == fake.text


def test_template_and_cancellation_stop_conditions_both_apply(chat_backend):
    fake = chat_backend._model
    fake.steps = 5

    chat_backend.generate(CONVERSATION, 16, 0.7, 0.95, None)
    (formatter,) = FakeChatFormatter.instances
    assert formatter.criteria_calls > 0  # the template's criterion is consulted
    assert fake.steps_run[-1] == 5

    stop_event = threading.Event()
    stop_event.set()
    chat_backend.generate(CONVERSATION, 16, 0.7, 0.95, None, stop_event)
    assert fake.steps_run[-1] == 1  # and the stop event still halts generation


def test_conversation_streams(chat_backend):
    fake = chat_backend._model
    fake.stream_pieces = ["Pa", "ris."]

    deltas, end = collect(chat_backend.stream(CONVERSATION, 16, 0.7, 0.95, None))

    assert deltas == ["Pa", "ris."]
    assert end == StreamEnd("stop", len(fake.calls[-1]["prompt"]), 2)


def test_long_conversation_is_rejected_before_generation(chat_backend):
    words = " ".join(["word"] * main.settings.n_ctx)

    with pytest.raises(PromptTooLongError):
        chat_backend.generate([{"role": "user", "content": words}], 8, 0.7, 0.95, None)
    assert chat_backend._model.calls == []


def test_conversation_the_template_refuses_is_a_client_error(chat_backend):
    FakeChatFormatter.render_error = ValueError(
        "Conversation roles must alternate user/assistant/user/assistant/..."
    )

    with pytest.raises(RequestRejectedError, match="must alternate"):
        chat_backend.generate(CONVERSATION, 8, 0.7, 0.95, None)


def test_model_without_template_rejects_conversations(fake_model):
    # The default test model has no chat template; text prompts still work.
    with pytest.raises(RequestRejectedError, match="no chat template"):
        main.backend.generate(CONVERSATION, 8, 0.7, 0.95, None)
    assert main.backend.generate("plain text", 8, 0.7, 0.95, None)["text"]


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def test_openai_conversation_uses_chat_even_with_completions_style(client):
    upstream = Upstream(lambda request: httpx2.Response(200, json=chat_reply()))
    backend = OpenAICompatibleBackend(
        remote_settings(remote_api_style="completions"),
        transport=httpx2.MockTransport(upstream),
    )

    result = backend.generate(CONVERSATION, 16, 0.7, 0.95, None)

    assert upstream.requests[-1].url.path.endswith("/chat/completions")
    assert upstream.last_body["messages"] == CONVERSATION
    assert "prompt" not in upstream.last_body
    assert result["text"] == "Paris."


def test_openai_conversation_streams(client):
    body = sse(
        (None, chat_chunk(content="Pa")),
        (None, chat_chunk(content="ris.", finish_reason="stop")),
        (None, "[DONE]"),
    )
    upstream = Upstream(lambda request: sse_response(body))
    backend = OpenAICompatibleBackend(
        remote_settings(remote_api_style="completions"),
        transport=httpx2.MockTransport(upstream),
    )

    deltas, end = collect(backend.stream(CONVERSATION, 16, 0.7, 0.95, None))

    assert deltas == ["Pa", "ris."]
    assert upstream.last_body["messages"] == CONVERSATION


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def anthropic_backend(upstream) -> AnthropicBackend:
    return AnthropicBackend(
        anthropic_settings(),
        http_client=httpx2.Client(transport=httpx2.MockTransport(upstream)),
    )


def test_anthropic_system_messages_become_the_system_field(client):
    upstream = Upstream(lambda request: httpx2.Response(200, json=message_reply()))
    conversation = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "system", "content": "Answer in Spanish."},
        {"role": "user", "content": "And of Spain?"},
    ]

    anthropic_backend(upstream).generate(conversation, 16, 0.7, 0.95, None)

    body = upstream.last_body
    assert body["system"] == "Be brief.\n\nAnswer in Spanish."
    assert body["messages"] == [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And of Spain?"},
    ]


def test_anthropic_conversation_without_system_sends_none(client):
    upstream = Upstream(lambda request: httpx2.Response(200, json=message_reply()))

    anthropic_backend(upstream).generate(CONVERSATION[1:], 16, 0.7, 0.95, None)

    assert "system" not in upstream.last_body
    assert upstream.last_body["messages"] == CONVERSATION[1:]
