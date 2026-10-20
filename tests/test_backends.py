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
