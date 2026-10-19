import pytest
from openai.types import Completion


def test_response_matches_the_openai_completion_schema(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "The capital of France is"},
    )

    assert response.status_code == 200
    completion = Completion.model_validate(response.json())
    assert completion.object == "text_completion"
    assert completion.model == "test-model"
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


def test_performance_extension_is_reported(client):
    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    assert set(body["performance"]) == {"tokens_per_second", "duration_seconds"}


def test_text_is_returned_verbatim(client, fake_model):
    # Completion clients concatenate prompt and output; the leading space is
    # part of the continuation and must survive.
    fake_model.text = " Paris.\n"

    body = client.post("/v1/completions", json={"prompt": "hello"}).json()

    assert body["choices"][0]["text"] == " Paris.\n"


def test_completion_id_reuses_the_request_id(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello"},
        headers={"X-Request-ID": "trace-42"},
    )

    assert response.json()["id"] == "cmpl-trace-42"


def test_parameters_are_forwarded_to_the_model(client, fake_model):
    client.post(
        "/v1/completions",
        json={
            "prompt": "hello",
            "max_tokens": 7,
            "temperature": 0.1,
            "top_p": 0.5,
            "stop": ["\n"],
        },
    )

    call = fake_model.calls[-1]
    assert call["max_tokens"] == 7
    assert call["temperature"] == 0.1
    assert call["top_p"] == 0.5
    assert call["stop"] == ["\n"]
    # echo=True would prepend the prompt to the output and skew token counts.
    assert call["echo"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "max_tokens": 4096},
        {"prompt": "hi", "temperature": 3},
        {"prompt": "hi", "top_p": 1.5},
        {"prompt": "hi", "stop": [""]},
        {"prompt": "hi", "stop": ["x" * 65]},
        {"prompt": "hi", "stop": ["a", "b", "c", "d", "e"]},
        {"prompt": "hi", "stream": True},
        {"prompt": "hi", "unknown_field": 1},
    ],
    ids=[
        "empty-prompt",
        "zero-max-tokens",
        "max-tokens-above-limit",
        "temperature-above-limit",
        "top-p-above-limit",
        "empty-stop-sequence",
        "stop-sequence-too-long",
        "too-many-stop-sequences",
        "streaming",
        "unknown-field",
    ],
)
def test_invalid_requests_are_rejected_before_inference(client, fake_model, payload):
    response = client.post("/v1/completions", json=payload)

    assert response.status_code == 422
    assert fake_model.calls == []


def test_prompt_exceeding_context_window_returns_400(client, fake_model):
    # The test context window is 64 tokens and the fake tokenizes per word.
    response = client.post("/v1/completions", json={"prompt": "word " * 64})

    assert response.status_code == 400
    assert "context window is 64 tokens" in response.json()["detail"]
    assert fake_model.calls == []


def test_prompt_just_inside_context_window_is_accepted(client):
    response = client.post("/v1/completions", json={"prompt": "word " * 63})

    assert response.status_code == 200
