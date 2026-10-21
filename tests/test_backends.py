import pytest
from pydantic import ValidationError

import app.main as main
from app.backends import InferenceBackend, create_backend
from app.backends.llamacpp import LlamaCppBackend
from app.config import Settings


def test_default_backend_is_llama_cpp(client):
    assert isinstance(main.backend, LlamaCppBackend)
    # The HTTP layer only relies on the protocol; the llama.cpp backend must
    # keep satisfying it as either side evolves.
    assert isinstance(main.backend, InferenceBackend)


def test_unknown_backend_fails_at_startup(monkeypatch):
    monkeypatch.setenv("VORTEX_BACKEND", "does-not-exist")

    with pytest.raises(ValidationError, match="llama_cpp"):
        Settings()


def test_factory_rejects_a_backend_without_an_implementation():
    # Guards against a name being added to the setting's allowed values
    # without a matching branch in the factory.
    settings = Settings().model_copy(update={"backend": "not-implemented"})

    with pytest.raises(ValueError, match="Unsupported inference backend"):
        create_backend(settings)


def test_prompt_processing_uses_the_generation_thread_count():
    backend = LlamaCppBackend(Settings().model_copy(update={"n_threads": 3}))

    kwargs = backend._model.init_kwargs
    # Not llama-cpp-python's default of os.cpu_count(), which ignores the
    # container's CPU limit.
    assert kwargs["n_threads"] == kwargs["n_threads_batch"] == 3


def test_warm_up_generates_one_token_and_clears_the_context():
    from tests.helpers import metric

    backend = LlamaCppBackend(Settings().model_copy(update={"warmup": True}))

    model = backend._model
    (call,) = model.calls
    assert call["max_tokens"] == 1
    assert model.resets == 1
    assert backend.is_ready()
    assert metric("vortex_model_warmup_duration_seconds") >= 0


def test_warm_up_can_be_turned_off():
    backend = LlamaCppBackend(Settings().model_copy(update={"warmup": False}))

    assert backend._model.calls == []
