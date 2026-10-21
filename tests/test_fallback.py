"""FallbackBackend: when a request moves to the fallback, and the circuit breaker."""

import threading
import time

import pytest

from app.backends import BackendError, PromptTooLongError, RequestRejectedError
from app.backends.base import StreamEnd, TextDelta
from app.backends.fallback import FallbackBackend
from app.config import Settings
from tests.helpers import metric

ARGS = ("hi", 8, 0.0, 1.0, None)


class Scripted:
    """Backend that raises the queued errors in order, then answers."""

    def __init__(self, name: str, *errors: Exception, busy: float = 0.0) -> None:
        self.name = name
        self.errors = list(errors)
        self.calls = 0
        self.closed = False
        self.ready = True
        self.busy = busy
        self.stream_error_after_first: Exception | None = None
        self.stream_closed = False

    def is_ready(self):
        return self.ready

    def busy_seconds(self):
        return self.busy

    def close(self):
        self.closed = True

    def generate(self, prompt, max_tokens, temperature, top_p, stop, stop_event=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {
            "text": f"from {self.name}",
            "finish_reason": "stop",
            "tokens_prompt": 1,
            "tokens_completion": 2,
        }

    def stream(self, prompt, max_tokens, temperature, top_p, stop, stop_event=None):
        self.calls += 1
        try:
            if self.errors:
                raise self.errors.pop(0)
            yield TextDelta(f"from {self.name}")
            if self.stream_error_after_first is not None:
                raise self.stream_error_after_first
            yield StreamEnd("stop", 1, 2)
        finally:
            self.stream_closed = True


def make(primary, fallback, threshold=3, cooldown=30.0):
    return FallbackBackend(
        primary,
        fallback,
        fallback_model="big-model",
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
    )


def fallbacks(reason: str) -> float:
    return metric("vortex_backend_fallbacks_total", reason=reason)


def test_primary_answers_without_touching_the_fallback():
    primary, fallback = Scripted("primary"), Scripted("fallback")

    result = make(primary, fallback).generate(*ARGS)

    assert result["text"] == "from primary"
    assert "model" not in result
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "error, reason",
    [
        (BackendError("429 upstream", status_code=503), "overloaded"),
        (BackendError("connection reset"), "upstream_error"),
        (BackendError("upstream slow", status_code=504), "timeout"),
        (PromptTooLongError(5000, 4096), "prompt_too_long"),
    ],
)
def test_conditions_the_fallback_can_change_move_the_request(error, reason):
    primary, fallback = Scripted("primary", error), Scripted("fallback")
    before = fallbacks(reason)

    result = make(primary, fallback).generate(*ARGS)

    assert result["text"] == "from fallback"
    # The response names the model that produced it.
    assert result["model"] == "big-model"
    assert fallbacks(reason) == before + 1


@pytest.mark.parametrize(
    "error",
    [RequestRejectedError("moderation refusal"), RuntimeError("bug")],
)
def test_other_failures_reach_the_client_unchanged(error):
    primary, fallback = Scripted("primary", error), Scripted("fallback")

    with pytest.raises(type(error)):
        make(primary, fallback).generate(*ARGS)
    assert fallback.calls == 0


def test_an_abandoned_request_is_not_retried():
    stop = threading.Event()
    stop.set()
    primary = Scripted("primary", BackendError("down", status_code=503))
    fallback = Scripted("fallback")

    with pytest.raises(BackendError):
        make(primary, fallback).generate(*ARGS, stop)
    assert fallback.calls == 0


def test_when_both_fail_the_fallback_error_wins_with_the_primary_attached():
    primary_error = BackendError("primary down", status_code=503)
    primary = Scripted("primary", primary_error)
    fallback = Scripted("fallback", BackendError("fallback down"))

    with pytest.raises(BackendError, match="fallback down") as raised:
        make(primary, fallback).generate(*ARGS)
    assert raised.value.__cause__ is primary_error


def test_repeated_failures_open_the_circuit_until_the_cooldown_ends():
    primary = Scripted(
        "primary",
        BackendError("down", status_code=503),
        BackendError("down", status_code=503),
    )
    fallback = Scripted("fallback")
    backend = make(primary, fallback, threshold=2, cooldown=0.2)
    skipped = fallbacks("circuit_open")

    backend.generate(*ARGS)
    backend.generate(*ARGS)  # second failure opens the circuit
    result = backend.generate(*ARGS)

    assert primary.calls == 2  # skipped while open
    assert result["text"] == "from fallback"
    assert fallbacks("circuit_open") == skipped + 1

    time.sleep(0.25)
    # Cooldown over: the primary is tried again, and its success closes it.
    assert backend.generate(*ARGS)["text"] == "from primary"
    assert backend.generate(*ARGS)["text"] == "from primary"
    assert primary.calls == 4


def test_long_prompts_do_not_count_as_primary_failures():
    primary = Scripted("primary", *(PromptTooLongError(5000, 4096) for _ in range(3)))
    backend = make(primary, Scripted("fallback"), threshold=2)

    for _ in range(3):
        backend.generate(*ARGS)

    # A prompt that does not fit says nothing about the primary's health.
    assert primary.calls == 3


def test_stream_falls_back_before_any_output():
    primary = Scripted("primary", BackendError("overloaded", status_code=503))
    fallback = Scripted("fallback")

    events = list(make(primary, fallback).stream(*ARGS))

    assert events[0] == TextDelta("from fallback")
    assert isinstance(events[-1], StreamEnd)
    assert primary.stream_closed


def test_stream_failure_after_output_is_not_retried():
    primary = Scripted("primary")
    primary.stream_error_after_first = BackendError("cut", status_code=503)
    fallback = Scripted("fallback")
    stream = make(primary, fallback).stream(*ARGS)

    assert next(stream) == TextDelta("from primary")
    with pytest.raises(BackendError):
        next(stream)
    assert fallback.calls == 0


def test_stream_goes_straight_to_the_fallback_while_the_circuit_is_open():
    primary = Scripted("primary", BackendError("down", status_code=503))
    fallback = Scripted("fallback")
    backend = make(primary, fallback, threshold=1)
    backend.generate(*ARGS)  # opens the circuit

    events = list(backend.stream(*ARGS))

    assert events[0] == TextDelta("from fallback")
    assert primary.calls == 1


def test_health_and_lifecycle_cover_both_backends():
    primary = Scripted("primary", busy=2.0)
    fallback = Scripted("fallback", busy=5.0)
    backend = make(primary, fallback)

    assert backend.busy_seconds() == 5.0
    primary.ready = False
    assert backend.is_ready()  # the fallback can still serve
    fallback.ready = False
    assert not backend.is_ready()
    backend.close()
    assert primary.closed and fallback.closed


# ---------------------------------------------------------------------------
# Settings and wiring
# ---------------------------------------------------------------------------


def test_fallback_settings_override_only_what_they_name():
    settings = Settings(
        fallback_backend="openai_compatible",
        fallback_model_name="gpt-large",
        fallback_remote_base_url="https://api.example.com/v1",
        fallback_remote_api_key="sk-fallback",
    )

    derived = settings.fallback_settings()

    assert (derived.backend, derived.model_name) == ("openai_compatible", "gpt-large")
    assert derived.remote_base_url == "https://api.example.com/v1"
    assert derived.remote_api_key.get_secret_value() == "sk-fallback"
    assert derived.fallback_backend is None
    assert derived.n_threads == settings.n_threads
    assert Settings().fallback_settings() is None


def test_fallback_needs_a_model_name():
    with pytest.raises(ValueError, match="VORTEX_FALLBACK_MODEL_NAME"):
        Settings(fallback_backend="openai_compatible")


def test_fallback_missing_its_own_requirements_fails_at_startup():
    # anthropic needs an API key, and none is configured for either backend.
    with pytest.raises(ValueError, match="VORTEX_REMOTE_API_KEY"):
        Settings(fallback_backend="anthropic", fallback_model_name="claude-opus-5-5")


def test_factory_wraps_the_primary_when_a_fallback_is_configured():
    from app.backends import create_backend
    from app.backends.llamacpp import LlamaCppBackend
    from app.backends.openai_compatible import OpenAICompatibleBackend

    backend = create_backend(
        Settings(
            fallback_backend="openai_compatible",
            fallback_model_name="gpt-large",
            fallback_remote_base_url="https://api.example.com/v1",
        )
    )

    assert isinstance(backend, FallbackBackend)
    assert isinstance(backend._primary, LlamaCppBackend)
    assert isinstance(backend._fallback, OpenAICompatibleBackend)
    backend.close()


def test_api_response_names_the_fallback_model(client, monkeypatch):
    import app.main as main

    primary = Scripted("primary", PromptTooLongError(5000, 64))
    monkeypatch.setattr(main, "backend", make(primary, Scripted("fallback")))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "a long question"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "big-model"
    assert body["choices"][0]["message"]["content"] == "from fallback"
