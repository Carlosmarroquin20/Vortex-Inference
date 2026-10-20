"""
Anthropic backend against a mocked Messages API.

The real SDK client runs over an httpx2.MockTransport, so request encoding,
headers and the SDK's error classes are exercised without a network.
"""

import json
import threading

import httpx2
import pytest
from openai.types import Completion
from pydantic import ValidationError

import app.main as main
from app.backends import create_backend
from app.backends.anthropic_api import AnthropicBackend
from app.config import Settings
from tests.helpers import metric

REQUESTS = "vortex_inference_requests_total"
MODEL = "claude-opus-5-5"


def message_reply(content=None, stop_reason="end_turn", model=MODEL, **extra):
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content
        if content is not None
        else [{"type": "text", "text": "Paris."}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
        **extra,
    }


def api_error(status, error_type, message, headers=None):
    return httpx2.Response(
        status,
        json={"type": "error", "error": {"type": error_type, "message": message}},
        headers={"request-id": "req_test123", **(headers or {})},
    )


class MessagesAPI:
    """Records requests and answers with a configurable handler."""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler = lambda request: httpx2.Response(200, json=message_reply())

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def last_body(self) -> dict:
        return json.loads(self.requests[-1].content)


def anthropic_settings(**overrides) -> Settings:
    values = {
        "backend": "anthropic",
        "remote_api_key": "sk-ant-test",
        "model_name": MODEL,
        **overrides,
    }
    return Settings(**values)


def build(api: MessagesAPI, **overrides) -> AnthropicBackend:
    return AnthropicBackend(
        anthropic_settings(**overrides),
        http_client=httpx2.Client(transport=httpx2.MockTransport(api)),
    )


@pytest.fixture
def api() -> MessagesAPI:
    return MessagesAPI()


@pytest.fixture
def use_anthropic(client, api, monkeypatch):
    """Swaps the running application's backend for the Anthropic one."""

    def install(**overrides) -> AnthropicBackend:
        backend = build(api, **overrides)
        monkeypatch.setattr(main, "backend", backend)
        return backend

    return install


# ---------------------------------------------------------------------------
# Request encoding
# ---------------------------------------------------------------------------


def test_request_is_a_single_user_turn(client, api, use_anthropic):
    use_anthropic()

    response = client.post(
        "/v1/completions",
        json={"prompt": "The capital of France is", "max_tokens": 7, "stop": ["."]},
    )

    assert response.status_code == 200
    request = api.requests[-1]
    assert request.url.host == "api.anthropic.com"
    assert request.url.path == "/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in request.headers
    body = api.last_body
    assert body["model"] == MODEL
    assert body["max_tokens"] == 7
    assert body["messages"] == [{"role": "user", "content": "The capital of France is"}]
    assert body["stop_sequences"] == ["."]


def test_sampling_parameters_are_not_sent(client, api, use_anthropic):
    # Current Claude models reject temperature and top_p with a 400.
    use_anthropic()

    client.post("/v1/completions", json={"prompt": "hi", "temperature": 0.2})

    assert "temperature" not in api.last_body
    assert "top_p" not in api.last_body


def test_refusal_fallback_is_requested_by_default(client, api, use_anthropic):
    use_anthropic()

    client.post("/v1/completions", json={"prompt": "hi"})

    assert api.last_body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in api.requests[-1].headers.get(
        "anthropic-beta", ""
    )


def test_refusal_fallback_can_be_disabled(client, api, use_anthropic):
    use_anthropic(anthropic_refusal_fallback=False)

    client.post("/v1/completions", json={"prompt": "hi"})

    assert "fallbacks" not in api.last_body
    assert "anthropic-beta" not in api.requests[-1].headers


def test_effort_is_sent_only_when_configured(client, api, use_anthropic):
    use_anthropic()
    client.post("/v1/completions", json={"prompt": "hi"})
    assert "output_config" not in api.last_body

    use_anthropic(anthropic_effort="low")
    client.post("/v1/completions", json={"prompt": "hi"})
    assert api.last_body["output_config"] == {"effort": "low"}


def test_base_url_from_the_environment_is_ignored(api, monkeypatch):
    # The SDK reads ANTHROPIC_BASE_URL on its own; the backend must only ever
    # call the endpoint in its own settings.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://elsewhere.test")

    build(api).generate("hi", 5, 0.7, 0.95, None)

    assert api.requests[-1].url.host == "api.anthropic.com"


def test_custom_base_url_is_used(api):
    build(api, remote_base_url="https://gateway.internal/anthropic").generate(
        "hi", 5, 0.7, 0.95, None
    )

    url = api.requests[-1].url
    assert (url.host, url.path) == ("gateway.internal", "/anthropic/v1/messages")


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


def test_reply_is_served_in_the_openai_completion_schema(client, api, use_anthropic):
    use_anthropic()
    api.handler = lambda request: httpx2.Response(
        200,
        json=message_reply(
            content=[
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Paris"},
                {"type": "text", "text": " is the capital."},
            ]
        ),
    )

    body = client.post("/v1/completions", json={"prompt": "hi"}).json()

    completion = Completion.model_validate(body)
    # Thinking blocks are dropped; text blocks are joined.
    assert completion.choices[0].text == "Paris is the capital."
    assert completion.usage.prompt_tokens == 12
    assert completion.usage.completion_tokens == 3


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "pause_turn"),
    ],
)
def test_stop_reason_maps_to_finish_reason(
    client, api, use_anthropic, stop_reason, finish_reason
):
    use_anthropic()
    api.handler = lambda request: httpx2.Response(
        200, json=message_reply(stop_reason=stop_reason)
    )

    body = client.post("/v1/completions", json={"prompt": "hi"}).json()

    assert body["choices"][0]["finish_reason"] == finish_reason


def test_refusal_is_a_successful_empty_completion(client, api, use_anthropic):
    use_anthropic()
    api.handler = lambda request: httpx2.Response(
        200,
        json=message_reply(
            content=[],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber"},
        ),
    )

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.json()["choices"][0] == {
        "index": 0,
        "text": "",
        "logprobs": None,
        "finish_reason": "content_filter",
    }


def test_completion_served_by_a_fallback_model(client, api, use_anthropic):
    use_anthropic()
    api.handler = lambda request: httpx2.Response(
        200, json=message_reply(model="claude-opus-4-8")
    )

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "Paris."


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_invalid_request_is_a_client_error(client, api, use_anthropic):
    use_anthropic()
    api.handler = lambda request: api_error(
        400, "invalid_request_error", "prompt is too long: 1200000 tokens > 1000000"
    )
    rejected_before = metric(REQUESTS, status="rejected")

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 400
    assert "prompt is too long" in response.json()["detail"]
    assert metric(REQUESTS, status="rejected") == rejected_before + 1


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, "rate_limit_error"), (529, "overloaded_error")],
)
def test_overload_becomes_503_with_retry_after(
    client, api, use_anthropic, status, error_type
):
    use_anthropic()
    api.handler = lambda request: api_error(
        status, error_type, "try later", headers={"retry-after": "30"}
    )

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (500, "api_error"),
    ],
)
def test_upstream_failures_do_not_leak_details(
    client, api, use_anthropic, status, error_type
):
    use_anthropic()
    api.handler = lambda request: api_error(status, error_type, "workspace wrk_7731")
    errors_before = metric(REQUESTS, status="error")

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 502
    assert "wrk_7731" not in response.text
    assert "req_test123" not in response.text
    assert metric(REQUESTS, status="error") == errors_before + 1


def test_request_id_is_logged_on_failure(client, api, use_anthropic, caplog):
    use_anthropic()
    api.handler = lambda request: api_error(500, "api_error", "boom")

    client.post("/v1/completions", json={"prompt": "hi"})

    assert any(
        "req_test123" in str(getattr(record, "reason", "")) for record in caplog.records
    )


def test_timeout_is_counted_as_timeout(client, api, use_anthropic):
    use_anthropic()

    def slow(request):
        raise httpx2.ReadTimeout("timed out", request=request)

    api.handler = slow
    timeouts_before = metric(REQUESTS, status="timeout")

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 504
    assert metric(REQUESTS, status="timeout") == timeouts_before + 1


def test_unreachable_api_is_a_bad_gateway(client, api, use_anthropic):
    use_anthropic()

    def refuse(request):
        raise httpx2.ConnectError("connection refused", request=request)

    api.handler = refuse

    assert client.post("/v1/completions", json={"prompt": "hi"}).status_code == 502


def test_failures_are_not_retried(client, api, use_anthropic):
    # Retries would run inside the request deadline and hold the permit.
    use_anthropic()
    api.handler = lambda request: api_error(529, "overloaded_error", "busy")

    client.post("/v1/completions", json={"prompt": "hi"})

    assert len(api.requests) == 1


def test_redirect_is_a_bad_gateway(client, api, use_anthropic):
    use_anthropic()
    api.handler = lambda request: httpx2.Response(
        307, headers={"Location": "http://elsewhere.test/v1/messages"}
    )

    response = client.post("/v1/completions", json={"prompt": "hi"})

    assert response.status_code == 502
    assert len(api.requests) == 1


def test_default_http_client_does_not_follow_redirects():
    # Following one would resend the API key to the Location target. The SDK's
    # own default client follows redirects, so the backend must override it.
    backend = AnthropicBackend(anthropic_settings())

    assert backend._client._client.follow_redirects is False
    backend.close()


# ---------------------------------------------------------------------------
# Backend contract and settings
# ---------------------------------------------------------------------------


def test_abandoned_request_is_not_sent(api):
    stop_event = threading.Event()
    stop_event.set()

    result = build(api).generate("hi", 5, 0.7, 0.95, None, stop_event)

    assert result["text"] == ""
    assert api.requests == []


def test_busy_seconds_returns_to_zero(api):
    backend = build(api)

    backend.generate("hi", 5, 0.7, 0.95, None)

    assert backend.is_ready()
    assert backend.busy_seconds() == 0.0


def test_factory_builds_the_anthropic_backend():
    backend = create_backend(anthropic_settings())

    assert isinstance(backend, AnthropicBackend)
    backend.close()


def test_backend_requires_an_api_key():
    settings = anthropic_settings().model_copy(update={"remote_api_key": None})

    with pytest.raises(ValueError, match="remote_api_key"):
        AnthropicBackend(settings)


@pytest.mark.parametrize(
    ("missing", "env_var"),
    [("remote_api_key", "VORTEX_REMOTE_API_KEY"), ("model_name", "VORTEX_MODEL_NAME")],
)
def test_settings_require_key_and_model(monkeypatch, missing, env_var):
    # conftest pins VORTEX_MODEL_NAME, which would count as set explicitly.
    monkeypatch.delenv("VORTEX_MODEL_NAME")
    values = {"backend": "anthropic", "remote_api_key": "k", "model_name": MODEL}
    del values[missing]

    with pytest.raises(ValidationError, match=env_var):
        Settings(**values)


def test_invalid_effort_fails_at_startup():
    with pytest.raises(ValidationError):
        anthropic_settings(anthropic_effort="extreme")


def test_blank_effort_keeps_the_model_default():
    assert anthropic_settings(anthropic_effort="").anthropic_effort is None
