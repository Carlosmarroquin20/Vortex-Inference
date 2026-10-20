"""
Shared test harness.

app.main resolves its settings and builds the FastAPI app at import time, and
imports llama_cpp transitively. Both are handled here before that import:

- llama_cpp is replaced with tests.fakes.FakeLlama, so the suite needs
  neither the native library nor model weights (CI does not even install
  llama-cpp-python).
- Every VORTEX_* setting is pinned through environment variables, which
  take precedence over a developer's local .env file, so results do not
  depend on the machine running them.

Tests that need a different setting patch the module-level settings object;
endpoints read it per request, so the change applies immediately.
"""

import os
import sys
import types

import pytest

from tests.fakes import FakeChatFormatter, FakeLlama

_TEST_ENV = {
    "VORTEX_BACKEND": "llama_cpp",
    "VORTEX_MODEL_PATH": "/models/test-model.gguf",
    "VORTEX_MODEL_NAME": "test-model",
    "VORTEX_N_CTX": "64",
    "VORTEX_N_THREADS": "1",
    "VORTEX_N_GPU_LAYERS": "0",
    "VORTEX_MAX_CONCURRENT_REQUESTS": "1",
    "VORTEX_PORT": "8080",
    "VORTEX_INFERENCE_TIMEOUT_SECONDS": "5",
    "VORTEX_API_KEY": "",
    "VORTEX_ENABLE_DOCS": "false",
}
os.environ.update(_TEST_ENV)

_fake_llama_cpp = types.ModuleType("llama_cpp")
_fake_llama_cpp.Llama = FakeLlama
_fake_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
_fake_chat_format.Jinja2ChatFormatter = FakeChatFormatter
_fake_llama_cpp.llama_chat_format = _fake_chat_format
sys.modules["llama_cpp"] = _fake_llama_cpp
sys.modules["llama_cpp.llama_chat_format"] = _fake_chat_format

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402


@pytest.fixture
def client():
    """TestClient with the application lifespan running (model loaded)."""
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def fake_model(client) -> FakeLlama:
    """The FakeLlama instance owned by the running application."""
    return main.backend._model
