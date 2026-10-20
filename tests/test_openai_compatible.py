"""
OpenAI-compatible backend against a mocked upstream.

httpx2.MockTransport stands in for the provider, so these tests exercise
the real client, request encoding and error mapping without a network.
"""

import asyncio
import json
import threading
import time

import httpx2
import pytest
from openai.types import Completion
from pydantic import ValidationError

import app.main as main
from app.backends import create_backend
from app.backends.openai_compatible import OpenAICompatibleBackend
from app.config import Settings
from app.main import CompletionRequest, create_completion
from tests.helpers import metric

REQUESTS = "vortex_inference_requests_total"
BASE_URL = "http://upstream.test/v1"


def chat_reply(content="Paris.", finish_reason="stop", usage=None):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }


class Upstream:
    """Records requests and answers with a configurable handler."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler = lambda request: httpx2.Response(200, json=chat_reply())

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


def remote_settings(**overrides) -> Settings:
    values = {
        "backend": "openai_compatible",
        "remote_base_url": BASE_URL,
        "remote_api_key": "upstream-key",
        **overrides,
    }
    return Settings(**values)


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


@pytest.fixture
def use_remote(client, upstream, monkeypatch):
    """Swaps the running application's backend for a remote one."""

    def install(**overrides) -> OpenAICompatibleBackend:
        backend = OpenAICompatibleBackend(
            remote_settings(**overrides), transport=httpx2.MockTransport(upstream)
        )
        monkeypatch.setattr(main, "backend", backend)
        return backend

    return install


# ---------------------------------------------------------------------------
# Request encoding and response mapping
# ---------------------------------------------------------------------------


def test_chat_style_sends_the_prompt_as_a_user_message(client, upstream, use_remote):
    use_remote()

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "The capital of France is",
            "max_tokens": 7,
            "temperature": 0.1,
            "top_p": 0.5,
            "stop": ["\n"],
        },
    )

    assert response.status_code == 200
    request = upstream.requests[-1]
    assert request.url == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer upstream-key"
    assert upstream.last_body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 7,
        "temperature": 0.1,
        "top_p": 0.5,
        "stop": ["\n"],
    }


def test_upstream_reply_is_served_in_the_openai_completion_schema(client, use_remote):
    use_remote()

    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    completion = Completion.model_validate(body)
    assert completion.choices[0].text == "Paris."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.prompt_tokens == 9
    assert completion.usage.completion_tokens == 2


def test_completions_style_forwards_the_raw_prompt(client, upstream, use_remote):
    use_remote(remote_api_style="completions")
    upstream.handler = lambda request: httpx2.Response(
        200,
        json={
            "choices": [{"text": " Paris.\n", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
    )

    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    assert upstream.requests[-1].url == f"{BASE_URL}/completions"
    assert upstream.last_body["prompt"] == "hello"
    assert "messages" not in upstream.last_body
    # Same verbatim guarantee as the local backend.
    assert body["choices"][0]["text"] == " Paris.\n"
    assert body["choices"][0]["finish_reason"] == "length"


def test_max_tokens_field_name_is_configurable(client, upstream, use_remote):
    use_remote(remote_max_tokens_field="max_completion_tokens")

    client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 9})

    assert upstream.last_body["max_completion_tokens"] == 9
    assert "max_tokens" not in upstream.last_body


def test_optional_fields_are_omitted(client, upstream, use_remote):
    use_remote(remote_api_key=None)

    client.post("/v1/completions", json={"prompt": "hello"})

    assert "authorization" not in upstream.requests[-1].headers
    assert "stop" not in upstream.last_body


def test_missing_usage_counts_as_zero_tokens(client, upstream, use_remote):
    use_remote()
    reply = chat_reply()
    del reply["usage"]
    upstream.handler = lambda request: httpx2.Response(200, json=reply)

    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    assert body["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_null_chat_content_becomes_empty_text(client, upstream, use_remote):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(
        200, json=chat_reply(content=None)
    )

    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    assert body["choices"][0]["text"] == ""


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_upstream_rejection_is_a_client_error(client, upstream, use_remote):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(
        400,
        json={
            "error": {
                "message": "This model's maximum context length is 8192 tokens.",
                "code": "context_length_exceeded",
            }
        },
    )
    rejected_before = metric(REQUESTS, status="rejected")
    errors_before = metric(REQUESTS, status="error")

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 400
    assert "maximum context length is 8192" in response.json()["detail"]
    assert metric(REQUESTS, status="rejected") == rejected_before + 1
    assert metric(REQUESTS, status="error") == errors_before


def test_rate_limit_becomes_503_with_retry_after(client, upstream, use_remote):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(
        429, json={"error": "rate limited"}, headers={"Retry-After": "12"}
    )

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "12"


@pytest.mark.parametrize("status", [401, 404, 500, 503])
def test_upstream_failures_do_not_leak_details(client, upstream, use_remote, status):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(
        status, json={"error": {"message": "org-internal-7731 has no access"}}
    )
    errors_before = metric(REQUESTS, status="error")

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 502
    assert "org-internal-7731" not in response.text
    assert metric(REQUESTS, status="error") == errors_before + 1


def test_non_json_error_body_is_handled(client, upstream, use_remote):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(502, text="<html>Bad</html>")

    assert client.post("/v1/completions", json={"prompt": "hi"}).status_code == 502


def test_unreachable_upstream_is_a_bad_gateway(client, upstream, use_remote):
    use_remote()

    def refuse(request):
        raise httpx2.ConnectError("connection refused", request=request)

    upstream.handler = refuse

    assert client.post("/v1/completions", json={"prompt": "hi"}).status_code == 502


@pytest.mark.parametrize("failure", ["read_timeout", "gateway_timeout"])
def test_upstream_timeouts_are_counted_as_timeouts(
    client, upstream, use_remote, failure
):
    use_remote()

    def slow(request):
        if failure == "read_timeout":
            raise httpx2.ReadTimeout("timed out", request=request)
        return httpx2.Response(504, text="upstream timeout")

    upstream.handler = slow
    timeouts_before = metric(REQUESTS, status="timeout")

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 504
    assert metric(REQUESTS, status="timeout") == timeouts_before + 1


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": ["not", "text"]}}]},
        {"unexpected": True},
    ],
)
def test_malformed_reply_is_a_bad_gateway(client, upstream, use_remote, payload):
    use_remote()
    upstream.handler = lambda request: httpx2.Response(200, json=payload)

    assert client.post("/v1/completions", json={"prompt": "hi"}).status_code == 502


def test_redirects_are_not_followed(client, upstream, use_remote):
    # Following one would resend the bearer token to the Location target.
    use_remote()
    upstream.handler = lambda request: httpx2.Response(
        307, headers={"Location": "http://elsewhere.test/v1/chat/completions"}
    )

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 502
    assert len(upstream.requests) == 1


# ---------------------------------------------------------------------------
# Backend contract
# ---------------------------------------------------------------------------


def test_busy_seconds_tracks_the_request_on_the_wire(upstream):
    backend = OpenAICompatibleBackend(
        remote_settings(), transport=httpx2.MockTransport(upstream)
    )
    observed = []

    def record(request):
        # Longer than the coarsest monotonic clock tick (about 16ms on Windows).
        time.sleep(0.05)
        observed.append(backend.busy_seconds())
        return httpx2.Response(200, json=chat_reply())

    upstream.handler = record

    assert backend.is_ready()
    assert backend.busy_seconds() == 0.0
    backend.generate("hello", 5, 0.7, 0.95, None)

    assert observed[0] > 0.0
    assert backend.busy_seconds() == 0.0
    backend.close()


def test_abandoned_request_is_not_sent(upstream):
    backend = OpenAICompatibleBackend(
        remote_settings(), transport=httpx2.MockTransport(upstream)
    )
    stop_event = threading.Event()
    stop_event.set()

    result = backend.generate("hello", 5, 0.7, 0.95, None, stop_event)

    assert result["text"] == ""
    assert upstream.requests == []


def test_factory_builds_the_remote_backend():
    backend = create_backend(remote_settings())

    assert isinstance(backend, OpenAICompatibleBackend)
    backend.close()


def test_backend_requires_a_base_url():
    settings = Settings().model_copy(update={"backend": "openai_compatible"})

    with pytest.raises(ValueError, match="remote_base_url"):
        OpenAICompatibleBackend(settings)


def test_remote_requests_run_concurrently_up_to_the_configured_limit(
    upstream, monkeypatch
):
    # Every request blocks until four are in flight at once. The semaphore
    # and the worker pool must both admit that many, or the barrier breaks.
    barrier = threading.Barrier(4, timeout=5)

    def wait_for_peers(request):
        barrier.wait()
        return httpx2.Response(200, json=chat_reply())

    upstream.handler = wait_for_peers
    monkeypatch.setattr(main.settings, "max_concurrent_requests", 4)
    monkeypatch.setattr(
        main,
        "create_backend",
        lambda settings: OpenAICompatibleBackend(
            remote_settings(), transport=httpx2.MockTransport(upstream)
        ),
    )

    async def scenario():
        async with main.lifespan(main.app):
            requests = [
                create_completion(CompletionRequest(prompt="hi")) for _ in range(4)
            ]
            return await asyncio.gather(*requests)

    responses = asyncio.run(scenario())

    assert [r.choices[0].text for r in responses] == ["Paris."] * 4


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_remote_backend_requires_a_base_url_setting(monkeypatch):
    monkeypatch.setenv("VORTEX_BACKEND", "openai_compatible")

    with pytest.raises(ValidationError, match="VORTEX_REMOTE_BASE_URL is required"):
        Settings()


@pytest.mark.parametrize("url", ["api.openai.com/v1", "ftp://upstream.test"])
def test_base_url_must_be_http(url):
    with pytest.raises(ValidationError, match="http:// or https://"):
        remote_settings(remote_base_url=url)


def test_base_url_trailing_slash_is_dropped():
    assert remote_settings(remote_base_url=f"{BASE_URL}/").remote_base_url == BASE_URL


def test_remote_secrets_stay_out_of_repr():
    settings = remote_settings(remote_api_key="sk-very-secret")

    assert "sk-very-secret" not in repr(settings)


def test_blank_remote_key_means_unauthenticated():
    assert remote_settings(remote_api_key="  ").remote_api_key is None
