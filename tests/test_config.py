import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_load_from_pinned_environment():
    settings = Settings()

    assert settings.model_name == "test-model"
    assert settings.api_key is None
    assert settings.enable_docs is False


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("VORTEX_MAX_CONCURRENT_REQUESTS", "0"),
        ("VORTEX_N_THREADS", "0"),
        ("VORTEX_N_CTX", "0"),
        ("VORTEX_N_GPU_LAYERS", "-2"),
        ("VORTEX_PORT", "70000"),
        ("VORTEX_INFERENCE_TIMEOUT_SECONDS", "0"),
    ],
)
def test_out_of_range_values_fail_at_startup(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)

    with pytest.raises(ValidationError):
        Settings()


def test_model_path_must_be_gguf(monkeypatch):
    monkeypatch.setenv("VORTEX_MODEL_PATH", "/models/model.bin")

    with pytest.raises(ValidationError, match="gguf"):
        Settings()


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_api_key_disables_auth(monkeypatch, value):
    monkeypatch.setenv("VORTEX_API_KEY", value)

    assert Settings().api_key is None


def test_api_key_is_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("VORTEX_API_KEY", "super-secret-value")

    settings = Settings()

    assert settings.api_key.get_secret_value() == "super-secret-value"
    assert "super-secret-value" not in repr(settings)
